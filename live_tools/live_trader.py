"""
라이브 트레이더 — 5분봉마다 신호 확인 후 Bybit 주문 실행
TRADE_MODE 옵션 (.env):
  real    : 실계좌 주문 실행
  sandbox : 테스트넷 주문 실행
  paper   : 실시세 조회 + 가상 주문 시뮬레이션 (주문 없음, CSV 저장)

STRATEGY 옵션 (.env):
  dl_v17      : TCN+Attention DL 모델 (단일 코인, COIN 환경변수)
  antifragile : AdaptRSI + ATR trailing stop (4종목 자동 25%씩 분할)

실행:
  python live_tools/live_trader.py
  nohup python live_tools/live_trader.py > logs/live.log 2>&1 &
"""
from __future__ import annotations
import sys, os, json, time, logging, csv, copy, fcntl
from logging.handlers import RotatingFileHandler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import torch, joblib

from data_pipeline    import add_technical_indicators, FEATURE_COLS
from signal_extractor import extract_signals_from_df
from expert_models    import load_all_experts
import expert_models as em

from exchange_client import (
    build_exchange, get_usdt_balance, get_position,
    set_leverage, place_market_order, close_position,
    fetch_ohlcv_df, calc_qty, get_symbol, get_min_qty,
    get_closed_pnl_history,
    set_position_stop_loss, cancel_position_stop_loss,
    get_last_closed_price,
)
from telegram_notifier import send_trade_alert, poll_commands, get_credentials

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
PARAMS_FILE = ROOT / "models/production/btc_params.json"
MODEL_DIR   = ROOT / "models/signal_model"
SCALER_PATH = ROOT / "models/production/btc_scaler.pkl"
STATE_FILE  = ROOT / "logs/live_state.json"
ENV_FILE    = ROOT / ".env"

PAPER_CSV_HEADER = [
    "timestamp", "direction", "entry_price", "exit_price",
    "hold_bars", "leverage", "rr",
    "sig_long_entry", "sig_short_entry",
    "pnl", "capital_after", "reason", "forced",
]

# ── 로깅 설정 ──────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

BARS_PER_DAY        = 288
FETCH_LIMIT         = 600    # 신호 롤링(100) + 지표 워밍(200) + 여유
SIG_ROLL_WIN        = 100
INITIAL_CAPITAL     = 10_000.0
STOP_POLL_INTERVAL  = 2.0

# ── 멀티코인 (Antifragile 전용) ────────────────────────────────────────────────
COINS_MULTI      = ["BTC", "ETH", "SOL", "XRP"]
MULTI_COIN_ALLOC = 0.25   # 4종목 균등 배분 (25% each → 2,500 USDT)

def MIN_QTY():
    return get_min_qty()  # COIN 환경변수 기준 동적 최소 수량


# ── 파라미터 로드 ──────────────────────────────────────────────────────────────
def load_params() -> dict:
    with open(PARAMS_FILE) as f:
        p = json.load(f)
    p["tiers"] = [tuple(t) for t in p["tiers"]]
    return p


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_models(device):
    for k in em.EXPERT_CONFIG:
        em.EXPERT_CONFIG[k]["save"] = str(MODEL_DIR / Path(em.EXPERT_CONFIG[k]["save"]).name)
    models = load_all_experts(device)
    scaler = joblib.load(SCALER_PATH)
    return models, scaler


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


# ── 상태 관리 ──────────────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "position":        0,       #  0=없음  1=롱  -1=숏
    "entry_price":     0.0,
    "avg_entry_price": 0.0,
    "entry_qty":       0.0,
    "entry_time":      None,
    "entry_lev":       1.0,
    "entry_rr":        0.0,
    "entry_bar":       0,
    "entry_sig_long":  0.0,     # 진입 시점 신호값 (paper 분석용)
    "entry_sig_short": 0.0,
    "current_bar":     0,
    "last_price":      0.0,
    "last_candle_ts":      None,    # 중복 틱 방지: 마지막 처리 완성봉 timestamp
    "capital":             INITIAL_CAPITAL,
    "initial_capital":     INITIAL_CAPITAL,
    "peak_capital":        INITIAL_CAPITAL,
    "daily_start_capital": INITIAL_CAPITAL,
    "daily_date":          None,
    "daily_halt":          False,
    "tg_update_offset":    0,
    "cooling_left":        0,
    "cb_triggers":     0,
    "sig_long_hist":   [],
    "sig_short_hist":  [],
    "trade_log":       [],
    # ── Antifragile Trailing Stop 전용 상태 ──────────────────────────────
    "af_trail_sl":       0.0,   # 현재 trailing stop 가격
    "af_peak_price":     0.0,   # 진입 후 최고(롱)/최저(숏) 가격
    "af_pyramid_count":  0,     # 피라미딩 추가 횟수
    "af_current_rr":     0.0,   # 현재 자본 위험 비율 (피라미딩으로 증가)
    "af_entry_atr":      0.0,   # 진입 시점 ATR (피라미딩 단계 계산용)
    "af_registered_sl":  0.0,   # 거래소에 등록된 SL 가격 (갱신 여부 판단용)
    "af_partial_taken":  False, # RSI 극단구간 절반 익절 실행 여부 (청산 시 리셋)
}

# ── Antifragile 전략 파라미터 ──────────────────────────────────────────────────
# 기본값 (AF_PARAM_PRESET 미설정 시 사용)
AF_PARAMS = {
    "dt_rsi_lo":       28,    # 하락추세: 롱 진입 RSI 임계값
    "dt_rsi_hi":       60,    # 하락추세: 숏 진입 RSI 임계값
    "rg_rsi_lo":       25,    # 횡보:     롱 진입 RSI 임계값 (BB 구역 적용 시 둔감화)
    "rg_rsi_hi":       75,    # 횡보:     숏 진입 RSI 임계값 (BB 구역 적용 시 둔감화)
    "ut_rsi_lo":       42,    # 상승추세: 롱 진입 RSI 임계값
    "ut_rsi_hi":       75,    # 상승추세: 숏 진입 RSI 임계값
    "bb_sigma":        0.5,   # 횡보 구역 BB 폭 (0=EMA 기준, 0.5=BB 0.5σ 권장)
    "trail_atr_init":  1.8,   # 초기 trailing stop 거리 (ATR 배수)
    "trail_atr_tight": 2.0,   # 피라미딩 후 tight trailing (ATR 배수)
    "rr_base":         0.20,  # 초기 자본 위험 비율
    "rr_add":          0.10,  # 피라미딩 1회당 추가 비율 (base보다 작게 → 평단 이동 억제)
    "add_levels":      3,     # 최대 피라미딩 횟수
    "atr_add_step":    0.5,   # 피라미딩 트리거 (유리방향 X×ATR마다)
    "leverage":        int(os.getenv("LEVERAGE", "5")),  # .env LEVERAGE 우선
    "max_hold_bars":   288,   # 최대 보유봉수 (1일)
}

# 거래소 emergency SL 배수: trail_sl 대신 넓은 SL을 등록해 intrabar 조기 체결 방지
# 소프트웨어 trail_sl은 봉 close 기준으로 별도 체크 (백테스트와 동일)
EMERGENCY_SL_ATR = 4.0

# ── 파라미터 프리셋 (.env AF_PARAM_PRESET으로 선택) ───────────────────────────
# 스윕 검증: temp/scripts/51_bb_trail_sweep.py (2026-06-12)
# 공통: BB0.5σ 횡보 구역 적용 (1h BB 내부 = ranging → rg_rsi 기준 둔감화)
_AF_PRESETS: dict[str, dict] = {
    # stable: 2026 +1108%, MDD 7.2%, hist 40/40 — 스윕 최적화 2026-06-15
    "stable": {
        "dt_rsi_lo": 30, "dt_rsi_hi": 60,
        "rg_rsi_lo": 25, "rg_rsi_hi": 75,
        "ut_rsi_lo": 42, "ut_rsi_hi": 70,
        "bb_sigma": 0.5,
        "trail_atr_init": 1.5, "trail_atr_tight": 2.0,
        "add_levels": 4,
    },
    # aggressive: 2026 +1202%, MDD 6.8%, hist 40/40 — 스윕 최적화 2026-06-15
    "aggressive": {
        "dt_rsi_lo": 25, "dt_rsi_hi": 60,
        "rg_rsi_lo": 25, "rg_rsi_hi": 75,
        "ut_rsi_lo": 42, "ut_rsi_hi": 78,
        "bb_sigma": 0.5,
        "trail_atr_init": 0.8, "trail_atr_tight": 1.5,
        "add_levels": 4,
    },
    # conservative: 2026 +699%, MDD 8.2%, hist 38/40 — 스윕 최적화 2026-06-15
    "conservative": {
        "dt_rsi_lo": 28, "dt_rsi_hi": 70,
        "rg_rsi_lo": 25, "rg_rsi_hi": 75,
        "ut_rsi_lo": 42, "ut_rsi_hi": 78,
        "bb_sigma": 0.5,
        "trail_atr_init": 2.0, "trail_atr_tight": 2.5,
        "add_levels": 3,
    },
}


def fresh_state() -> dict:
    return copy.deepcopy(DEFAULT_STATE)


def _total_tracked_capital(paper_mode: bool) -> float:
    """4종 코인 state 파일에서 총 자본 합산 (비례 동기화용)."""
    prefix = "paper" if paper_mode else "live"
    total = 0.0
    for sfx in ("", "_eth", "_sol", "_xrp"):
        p = ROOT / "logs" / f"{prefix}_state{sfx}.json"
        try:
            total += json.loads(p.read_text()).get("capital", 0.0)
        except Exception:
            pass
    return total


def _clear_entry_fields(state: dict) -> None:
    """진입 실패/취소 시 모든 진입·AF 관련 state 필드를 초기값으로 리셋."""
    for key in (
        "position", "entry_price", "avg_entry_price", "entry_qty",
        "entry_time", "entry_lev", "entry_rr", "entry_bar",
        "entry_sig_long", "entry_sig_short",
        "af_trail_sl", "af_peak_price", "af_pyramid_count",
        "af_current_rr", "af_entry_atr", "af_registered_sl",
        "af_partial_taken",
    ):
        state[key] = copy.deepcopy(DEFAULT_STATE.get(key, 0))


def get_runtime_paths(paper_mode: bool) -> dict:
    coin = os.environ.get("COIN", "BTC").lower()
    prefix = f"_{coin}" if coin != "btc" else ""  # BTC는 기존 파일명 유지
    if paper_mode:
        return {
            "state_file":      ROOT / f"logs/paper_state{prefix}.json",
            "log_file":        ROOT / f"logs/paper{prefix}.log",
            "paper_trade_csv": ROOT / f"logs/paper_trades{prefix}.csv",
        }
    return {
        "state_file":      ROOT / f"logs/live_state{prefix}.json",
        "log_file":        ROOT / f"logs/live{prefix}.log",
        "paper_trade_csv": None,
    }


def configure_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _write_paper_trade_csv(row: dict, paper_trade_csv: Path):
    """paper_trades.csv에 거래 1건 추가."""
    paper_trade_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not paper_trade_csv.exists()
    with open(paper_trade_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PAPER_CSV_HEADER)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PAPER_CSV_HEADER})
        f.flush()
        os.fsync(f.fileno())


def _completed_ohlcv_df(df: pd.DataFrame, timeframe_minutes: int = 5) -> pd.DataFrame:
    """Drop the currently forming candle; ccxt OHLCV usually includes it as the last row."""
    if df.empty:
        return df
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=timeframe_minutes)
    return df[df.index <= cutoff]


def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    tmp = Path(str(path) + f".{os.getpid()}.tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(tmp, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(payload)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp, path)
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if tmp.exists():
                tmp.unlink()
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return fresh_state()

def save_state(state: dict):
    atomic_write_json(STATE_FILE, state)


# ── 신호 생성 ──────────────────────────────────────────────────────────────────
def get_signals(exchange, models, scaler, device) -> pd.DataFrame | None:
    try:
        df_raw = fetch_ohlcv_df(exchange, limit=FETCH_LIMIT)
    except Exception as e:
        log.error(f"OHLCV 조회 실패: {e}")
        return None

    try:
        df_ind = add_technical_indicators(df_raw)
        df_ind.dropna(subset=FEATURE_COLS, inplace=True)
        df_sc = df_ind.copy()
        df_sc[FEATURE_COLS] = scaler.transform(df_ind[FEATURE_COLS])
        df_sg = extract_signals_from_df(df_sc, models, device)
        df_sg.dropna(subset=["signal_long", "signal_short", "signal_context"], inplace=True)
        return df_sg
    except Exception as e:
        log.error(f"신호 계산 실패: {e}")
        return None


# ── v17 티어 조회 ──────────────────────────────────────────────────────────────
def get_tier(sig: float, tiers: list) -> tuple[float, float]:
    for t in tiers:
        if sig >= t[0]:
            return t[1], t[2]
    return 0.0, 0.0


# ── Antifragile: 원시 지표 계산 ────────────────────────────────────────────────
def _compute_af_indicators(df: pd.DataFrame) -> tuple:
    """df의 raw OHLCV에서 ATR, RSI, 1h 추세 방향 계산 (마지막 봉 기준)"""
    close = df["close"]; high = df["high"]; low = df["low"]

    # ATR 14
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

    # RSI 14
    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi = float((100 - 100 / (1 + ag / (al + 1e-9))).iloc[-1])

    # 1h EMA20 + BB 횡보 구역 추세 판별 (마지막 1h 봉 기준)
    # bb_sigma > 0: price > BB upper → trendup, price < BB lower → trenddown, 사이 → ranging
    # bb_sigma = 0: 기존 EMA 기준 (price > EMA → up, price < EMA → down)
    try:
        cl1h   = close.resample("1h").last().ffill()
        ema_1h = cl1h.ewm(span=20, adjust=False).mean()
        bb_sigma = AF_PARAMS.get("bb_sigma", 0.0)
        if bb_sigma > 0:
            std_1h = cl1h.rolling(20).std()
            bb_up  = (ema_1h + bb_sigma * std_1h).iloc[-1]
            bb_lo  = (ema_1h - bb_sigma * std_1h).iloc[-1]
            last   = float(cl1h.iloc[-1])
            trend_up   = bool(last > bb_up)
            trend_down = bool(last < bb_lo)
        else:
            trend_up   = bool(cl1h.iloc[-1] > ema_1h.iloc[-1])
            trend_down = bool(cl1h.iloc[-1] < ema_1h.iloc[-1])
    except Exception:
        trend_up = trend_down = False

    return atr, rsi, trend_up, trend_down


# ── Antifragile: 한 틱 처리 ────────────────────────────────────────────────────
def process_tick_af(exchange, df: pd.DataFrame, state: dict, price: float,
                    now_str: str, paper_mode: bool,
                    paper_trade_csv: Optional[Path]) -> dict:
    """Antifragile Trailing Stop 전략: AdaptRSI 진입 + ATR trailing stop + 피라미딩"""
    coin = os.environ.get("COIN", "BTC").upper()
    TRADING_FEE = 0.0005 + 0.0002

    atr, rsi, trend_up, trend_down = _compute_af_indicators(df)

    p       = AF_PARAMS
    rsi_lo  = p["dt_rsi_lo"] if trend_down else (p["ut_rsi_lo"] if trend_up else p["rg_rsi_lo"])
    rsi_hi  = p["dt_rsi_hi"] if trend_down else (p["ut_rsi_hi"] if trend_up else p["rg_rsi_hi"])
    pos     = state["position"]
    lev     = p["leverage"]
    trend_str = "DN" if trend_down else ("UP" if trend_up else "RG")
    prev_trend = state.get("last_trend", trend_str)
    state["last_trend"] = trend_str
    long_ok_now = rsi <= rsi_lo
    short_ok_now = rsi >= rsi_hi and not long_ok_now
    candle_ts = df.index[-1] if len(df.index) else "n/a"
    log.info(
        f"[{coin}/AF 신호] ts={candle_ts} price={price:,.4f} "
        f"RSI={rsi:.1f} trend={trend_str} lo={rsi_lo} hi={rsi_hi} "
        f"long={long_ok_now} short={short_ok_now} pos={pos} halt={state.get('daily_halt', False)}"
    )

    # ── AF 상태 안전 초기화 (구버전 state 로드 또는 첫 실행 시) ───────────────────
    if pos != 0 and not state.get("af_trail_sl"):
        log.warning(f"[{coin}/AF] AF 상태 없음 → 현재가 기준 trail_sl 재초기화")
        state["af_peak_price"]    = price
        state["af_current_rr"]    = state.get("entry_rr", p["rr_base"])
        state["af_pyramid_count"] = 0
        state["af_entry_atr"]     = atr
        state["af_trail_sl"]      = (price - p["trail_atr_init"] * atr) if pos == 1 else \
                                    (price + p["trail_atr_init"] * atr)

    # ── 청산 체크 (trailing stop) ─────────────────────────────────────────────
    if pos != 0:
        trail_sl  = state["af_trail_sl"]
        hold_bars = state["current_bar"] - state["entry_bar"]
        hit_stop  = (pos ==  1 and price <= trail_sl) or \
                    (pos == -1 and price >= trail_sl)
        timeout   = hold_bars >= p["max_hold_bars"]

        if hit_stop or timeout:
            reason = "trail_SL" if hit_stop else "timeout"
            entry_price = float(state.get("avg_entry_price") or state.get("entry_price") or price)
            dir_str = "LONG 🟢" if pos == 1 else "SHORT 🔴"
            capital_before = state["capital"]
            _close_and_log(exchange, state, price, now_str, forced=False, reason=reason,
                           paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
            capital_after = state["capital"]
            pnl_usdt = capital_after - capital_before
            pnl_pct  = (capital_after / (capital_before + 1e-9) - 1) * 100
            send_trade_alert(
                f"📤 <b>{'[PAPER] ' if paper_mode else ''}[{coin}/AF] 청산</b> [{reason}] {dir_str}\n"
                f"진입가: {entry_price:,.4f} → 청산가: {price:,.4f}\n"
                f"PnL: {pnl_pct:+.2f}% ({pnl_usdt:+.1f} USDT)\n"
                f"자본: {capital_before:,.0f} → {capital_after:,.0f} USDT"
            )
        else:
            # trailing stop 업데이트
            trail_mult = p["trail_atr_tight"] if state["af_pyramid_count"] > 0 else p["trail_atr_init"]
            entry_atr_saved = state.get("af_entry_atr", atr) or atr
            effective_atr = max(atr, entry_atr_saved * 0.6)  # 진입 ATR 60% 미만으로 SL 좁아지지 않도록
            if pos == 1:
                state["af_peak_price"] = max(state["af_peak_price"], price)
                new_trail = state["af_peak_price"] - trail_mult * effective_atr
                state["af_trail_sl"] = max(state["af_trail_sl"], new_trail)
            else:
                state["af_peak_price"] = min(state["af_peak_price"], price)
                new_trail = state["af_peak_price"] + trail_mult * effective_atr
                state["af_trail_sl"] = min(state["af_trail_sl"], new_trail)

            # ── 절반 익절 (RSI 극단구간) ────────────────────────────────────────
            if not state.get("af_partial_taken", False):
                partial_ok = (pos == 1 and rsi >= rsi_hi) or (pos == -1 and rsi <= rsi_lo)
                if partial_ok:
                    _dec    = {"BTC": 3, "ETH": 2, "SOL": 1, "XRP": 0}.get(coin, 3)
                    half_qty = round(state["entry_qty"] / 2, _dec)
                    min_q   = MIN_QTY()
                    if half_qty >= min_q and (state["entry_qty"] - half_qty) >= min_q:
                        avg_e = float(state.get("avg_entry_price") or state.get("entry_price") or price)
                        prr   = state["entry_rr"] * 0.5
                        ppnl  = max(pos * (price - avg_e) / (avg_e + 1e-9) * lev * prr, -prr)
                        cap_b = state["capital"]
                        log.info(
                            f"[{coin}/AF 절반익절] RSI={rsi:.1f} "
                            f"qty {state['entry_qty']} → {state['entry_qty'] - half_qty:.{_dec}f} "
                            f"PnL={ppnl:+.4f}"
                        )
                        success = True
                        if not paper_mode:
                            try:
                                place_market_order(exchange, "sell" if pos == 1 else "buy", half_qty, reduce_only=True)
                            except Exception as _pe:
                                log.error(f"[{coin}/AF 절반익절 실패] {_pe}")
                                success = False
                        if success:
                            state["capital"]         *= (1 + ppnl)
                            state["entry_qty"]        -= half_qty
                            state["entry_rr"]         *= 0.5
                            state["af_current_rr"]     = state["entry_rr"]
                            state["af_partial_taken"]  = True
                            send_trade_alert(
                                f"✂️ <b>[{coin}/AF] 절반 익절</b>\n"
                                f"RSI={rsi:.1f} | 가격={price:,.4f}\n"
                                f"익절qty={half_qty} | PnL={ppnl:+.4f}\n"
                                f"자본: {cap_b:,.0f} → {state['capital']:,.0f} USDT"
                            )
                    else:
                        log.info(f"[{coin}/AF 절반익절 스킵] half_qty={half_qty} < min_qty={min_q}")

            # 피라미딩 체크 (유리방향 N×ATR마다 추가)
            entry_atr = state.get("af_entry_atr", atr) or atr
            base_entry = float(state.get("avg_entry_price") or state.get("entry_price") or price)
            favorable = pos * (price - base_entry) / (entry_atr + 1e-9)
            next_lvl  = (state["af_pyramid_count"] + 1) * p["atr_add_step"]
            rsi_ok_pyr = (pos == 1 and rsi < rsi_hi) or (pos == -1 and rsi > rsi_lo)
            if state["af_pyramid_count"] < p["add_levels"] and favorable >= next_lvl and not rsi_ok_pyr:
                log.info(f"[{coin}/AF 피라미딩 RSI차단] RSI={rsi:.1f} 극단구간 스킵")
            if state["af_pyramid_count"] < p["add_levels"] and favorable >= next_lvl and rsi_ok_pyr:
                pyramid_snapshot = {
                    "entry_price":       state.get("entry_price", 0.0),
                    "avg_entry_price":   state.get("avg_entry_price", 0.0),
                    "entry_qty":         state.get("entry_qty", 0.0),
                    "entry_rr":          state.get("entry_rr", 0.0),
                    "af_trail_sl":       state.get("af_trail_sl", 0.0),
                    "af_pyramid_count":  state.get("af_pyramid_count", 0),
                    "af_current_rr":     state.get("af_current_rr", 0.0),
                    "af_registered_sl":  state.get("af_registered_sl", 0.0),
                }
                add_qty = calc_qty(state["capital"], p["rr_add"], lev, price)
                state["af_pyramid_count"] += 1
                state["af_current_rr"]    += p["rr_add"]
                state["entry_rr"]          = state["af_current_rr"]
                old_qty = float(state.get("entry_qty") or 0.0)
                old_avg = float(state.get("avg_entry_price") or state.get("entry_price") or price)
                state["entry_qty"] = old_qty + add_qty
                state["avg_entry_price"] = (
                    ((old_avg * old_qty) + (price * add_qty)) / state["entry_qty"]
                    if state["entry_qty"] > 0 else price
                )
                state["entry_price"] = state["avg_entry_price"]
                if pos == 1:
                    state["af_trail_sl"] = max(state["af_trail_sl"], price - p["trail_atr_tight"] * atr)
                else:
                    state["af_trail_sl"] = min(state["af_trail_sl"], price + p["trail_atr_tight"] * atr)

                log.info(f"[{coin}/AF 피라미딩 #{state['af_pyramid_count']}] "
                         f"favorable={favorable:.2f}ATR | rr={state['af_current_rr']:.2f} | "
                         f"trail_sl={state['af_trail_sl']:,.4f}")

                if not paper_mode:
                    try:
                        if add_qty < MIN_QTY():
                            raise ValueError(f"최소수량 미달: {add_qty} < {MIN_QTY()}")
                        side = "buy" if pos == 1 else "sell"
                        place_market_order(exchange, side, add_qty)
                    except Exception as e:
                        state.update(pyramid_snapshot)
                        log.error(f"[{coin}/AF] 피라미딩 주문 실패: {e}")
                    else:
                        send_trade_alert(
                            f"➕ <b>[{coin}/AF] 피라미딩 #{state['af_pyramid_count']}</b>\n"
                            f"가격: {price:,.4f} | 추가수량: {add_qty}"
                        )

            # trail_sl은 소프트웨어(봉 close 기준)로만 체크 — 거래소 SL 업데이트 없음
            # emergency SL은 진입 시 1회만 등록 (intrabar 조기 체결 방지)

    # ── 신규 진입 ─────────────────────────────────────────────────────────────
    if state["position"] == 0:
        trend_stable = (prev_trend == trend_str)
        if not trend_stable and (long_ok_now or short_ok_now):
            log.info(f"[{coin}/AF 진입 스킵] 트렌드 첫 전환 봉 {prev_trend}→{trend_str} — 안정화 대기")
        long_ok  = long_ok_now and trend_stable
        short_ok = short_ok_now and trend_stable

        direction = 1 if long_ok else (-1 if short_ok else 0)
        if direction != 0 and atr < price * 0.0015:
            log.info(f"[{coin}/AF 진입 스킵] ATR={atr:.4f} < 가격×0.15% ({price * 0.0015:.4f}) — 횡보 구간")
            direction = 0
        if direction != 0:
            dir_str    = "LONG 🟢" if direction == 1 else "SHORT 🔴"
            init_trail = (price - p["trail_atr_init"] * atr) if direction == 1 else \
                         (price + p["trail_atr_init"] * atr)

            state["position"]          = direction
            state["entry_price"]       = price
            state["avg_entry_price"]   = price
            state["entry_qty"]         = calc_qty(state["capital"], p["rr_base"], lev, price)
            state["entry_time"]        = now_str
            state["entry_lev"]         = lev
            state["entry_rr"]          = p["rr_base"]
            state["entry_bar"]         = state["current_bar"]
            state["entry_sig_long"]    = rsi
            state["entry_sig_short"]   = atr
            state["af_trail_sl"]       = init_trail
            state["af_peak_price"]     = price
            state["af_pyramid_count"]  = 0
            state["af_current_rr"]     = p["rr_base"]
            state["af_entry_atr"]      = atr

            log.info(f"[{coin}/AF {'PAPER ' if paper_mode else ''}진입] {dir_str} | "
                     f"가격={price:,.4f} | RSI={rsi:.1f}({trend_str}) | "
                     f"ATR={atr:.4f} | trail_sl={init_trail:,.4f}")
            send_trade_alert(
                f"📥 <b>{'[PAPER] ' if paper_mode else ''}[{coin}/AF] 진입</b> {dir_str}\n"
                f"가격: {price:,.4f} | RSI: {rsi:.1f}({trend_str}) | ATR: {atr:.4f}\n"
                f"trail_SL: {init_trail:,.4f} | 자본: {state['capital']:,.0f} USDT"
            )

            if not paper_mode:
                try:
                    set_leverage(exchange, lev)
                    qty = calc_qty(state["capital"], p["rr_base"], lev, price)
                    if qty < MIN_QTY():
                        log.warning(f"[{coin}/AF] 최소수량 미달: {qty} — 진입 취소")
                        _clear_entry_fields(state)
                    else:
                        side = "buy" if direction == 1 else "sell"
                        place_market_order(exchange, side, qty)
                        time.sleep(1)
                        try:
                            filled = get_position(exchange)
                            if filled["side"] and filled["entry_price"] > 0:
                                actual_px = float(filled["entry_price"])
                                slip_pct  = (actual_px - price) / price * 100 * direction
                                state["entry_price"]           = actual_px
                                state["avg_entry_price"]       = actual_px
                                state["entry_qty"]             = float(filled.get("size") or state.get("entry_qty") or qty)
                                state["_entry_signal_price"]   = price
                                state["_entry_slippage_pct"]   = round(slip_pct, 4)
                                log.info(f"[{coin}/AF] 체결확인 | 신호={price:,.4f} 체결={actual_px:,.4f} 슬리피지={slip_pct:+.3f}%")
                        except Exception as e2:
                            log.warning(f"[{coin}/AF] 체결확인 실패: {e2}")
                        # 진입 직후 거래소 emergency SL 등록
                        # trail_sl 대신 넓은 emergency SL — intrabar 조기 체결 방지
                        # 소프트웨어 trail_sl 체크는 봉 close 기준으로 별도 동작
                        try:
                            entry_px = state.get("avg_entry_price") or price
                            entry_atr_val = state.get("af_entry_atr") or atr
                            if direction == 1:
                                emergency_sl = entry_px - EMERGENCY_SL_ATR * entry_atr_val
                            else:
                                emergency_sl = entry_px + EMERGENCY_SL_ATR * entry_atr_val
                            set_position_stop_loss(exchange, emergency_sl)
                            state["af_registered_sl"] = emergency_sl
                            log.info(f"[{coin}/AF emergency SL등록] {emergency_sl:,.4f} (trail={state['af_trail_sl']:,.4f})")
                        except Exception as e_sl:
                            log.warning(f"[{coin}/AF SL등록 실패] {e_sl}")
                except Exception as e:
                    log.error(f"[{coin}/AF] 진입 주문 실패: {e}")
                    # 주문이 거래소에 수락됐을 수 있으므로 실제 포지션 확인 후 결정
                    try:
                        orphan = get_position(exchange)
                        if orphan["side"] and orphan["size"] > 0:
                            log.warning(f"[{coin}/AF] 주문 예외 후 거래소 포지션 감지 — state 복원 (orphan)")
                            state["entry_qty"]       = orphan["size"]
                            state["avg_entry_price"] = float(orphan["entry_price"] or state.get("entry_price", 0))
                        else:
                            _clear_entry_fields(state)
                    except Exception as e2:
                        log.warning(f"[{coin}/AF] orphan 포지션 확인 실패: {e2}")
                        _clear_entry_fields(state)

    return state


# ── 멀티코인 AF 전용: 단일 코인 틱 처리 ───────────────────────────────────────
def _run_coin_tick_af(exchange, coin: str, state: dict, paper_mode: bool,
                      paper_trade_csv: Optional[Path],
                      total_tracked_snapshot: float | None = None) -> dict:
    """4종목 멀티코인 모드 — 코인별 OHLCV fetch → 중복 방지 → process_tick_af 호출."""
    os.environ["COIN"] = coin

    try:
        df = fetch_ohlcv_df(exchange, limit=FETCH_LIMIT)
    except Exception as e:
        log.error(f"[{coin}] OHLCV 조회 실패: {e}")
        return state
    df = _completed_ohlcv_df(df)

    if len(df) < SIG_ROLL_WIN + 10:
        log.warning(f"[{coin}/AF] 완성봉 부족: {len(df)}")
        return state

    row = df.iloc[-1]
    candle_ts = str(row.name)
    if state.get("last_candle_ts") == candle_ts:
        log.info(f"[{coin}/AF 스킵] 중복 완성봉 ts={candle_ts}")
        return state
    state["last_candle_ts"] = candle_ts

    # 거래소 포지션 동기화: 봉 사이에 거래소 SL이 체결된 경우 state 업데이트
    if state.get("position", 0) != 0 and not paper_mode:
        try:
            ex_pos = get_position(exchange)
            if ex_pos["side"] is None or ex_pos["size"] == 0:
                log.warning(f"[{coin}/AF] 거래소 포지션 없음 — SL 자동체결 감지, state 동기화")

                # 실체결가 조회 (거래소 실제 체결가 우선, 실패 시 trail_sl 추정)
                actual_exit = get_last_closed_price(exchange)
                sl_estimate = state.get("af_trail_sl", 0.0)
                exit_price  = actual_exit if actual_exit > 0 else sl_estimate
                entry_price = float(state.get("avg_entry_price") or state.get("entry_price") or 0.0)
                entry_lev   = float(state.get("entry_lev") or 0.0)
                entry_rr    = float(state.get("entry_rr") or 0.0)
                pos         = state["position"]

                slip_exit_pct = 0.0
                if sl_estimate > 0 and actual_exit > 0:
                    slip_exit_pct = round((actual_exit - sl_estimate) / sl_estimate * 100 * pos * -1, 4)

                capital_before_sl = state["capital"]
                # 심볼별 실현 PnL 조회 (잔고 델타 대신 symbol-specific closed PnL 사용)
                # 동시 다중 SL 발생 시 잔고 델타 귀속 오류를 방지
                realized_applied = False
                try:
                    closed = get_closed_pnl_history(exchange, limit=1)
                    if closed:
                        realized_pnl_usdt = float(closed[0].get("realized_pnl", 0.0))
                        if realized_pnl_usdt != 0.0:
                            state["capital"] = capital_before_sl + realized_pnl_usdt
                            log.info(
                                f"[{coin}/AF EX_SL 자본동기화] "
                                f"realized_pnl={realized_pnl_usdt:+.4f} USDT → {state['capital']:,.2f} USDT"
                            )
                            realized_applied = True
                except Exception:
                    pass

                if not realized_applied and entry_price > 0 and exit_price > 0:
                    pnl_raw = pos * (exit_price - entry_price) / entry_price
                    pnl_est = max(pnl_raw * entry_lev * entry_rr, -entry_rr)
                    state["capital"] = capital_before_sl * (1 + pnl_est)
                    log.info(
                        f"[{coin}/AF EX_SL 자본동기화(추정)] "
                        f"exit={exit_price:.4f} pnl={pnl_est:+.4f} → {state['capital']:,.2f} USDT"
                    )

                pnl_recorded = (state["capital"] - capital_before_sl) / (capital_before_sl + 1e-9)
                state["peak_capital"] = max(state.get("peak_capital", state["capital"]), state["capital"])

                now_str_sync = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                trade_row = {
                    "time":               now_str_sync,
                    "direction":          pos,
                    "entry":              entry_price,
                    "exit":               exit_price,
                    "pnl":                round(pnl_recorded, 6),
                    "capital":            round(state["capital"], 2),
                    "forced":             False,
                    "reason":             "EX_SL",
                    "slippage_entry_pct": round(state.pop("_entry_slippage_pct", 0.0), 4),
                    "slippage_exit_pct":  slip_exit_pct,
                    "exit_mark":          round(exit_price, 4),
                }
                state["trade_log"].append(trade_row)
                src = "실체결" if actual_exit > 0 else "trail_sl추정"
                log.info(f"[{coin}/AF] SL 자동체결({src}) exit={exit_price:.4f} 자본: {capital_before_sl:,.2f} → {state['capital']:,.2f}")

                _clear_entry_fields(state)
                send_trade_alert(
                    f"⚡ <b>[{coin}/AF] 거래소 SL 자동체결</b>\n"
                    f"봉 사이에 trail_SL 도달 → 포지션 종료됨\n"
                    f"실행가격(추정): {exit_price:,.4f}"
                )
        except Exception as e:
            log.warning(f"[{coin}/AF] 포지션 동기화 조회 실패: {e}")

    price = float(row["close"])
    state["last_price"] = price
    state["current_bar"] += 1
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 일일 손실 한도 리셋 + 거래소 잔고 절대 동기화
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("daily_date") != today:
        if not paper_mode:
            try:
                actual_balance = get_usdt_balance(exchange)
                total_tracked  = total_tracked_snapshot or _total_tracked_capital(paper_mode=False)
                if actual_balance > 0 and total_tracked > 0:
                    proportion  = state["capital"] / total_tracked
                    new_capital = actual_balance * proportion
                    log.info(
                        f"[{coin} 일일동기화] 거래소총잔고={actual_balance:,.2f} USDT "
                        f"× {proportion:.4f} → {state['capital']:,.2f} → {new_capital:,.2f} USDT"
                    )
                    state["capital"] = new_capital
            except Exception as _se:
                log.warning(f"[{coin} 일일동기화 실패] {_se}")
            # 거래소 closed PnL 히스토리 최근 5건 로그 출력
            try:
                history = get_closed_pnl_history(exchange, limit=50)
                if history:
                    log.info(f"[{coin} 거래소 최근청산 {len(history)}건]")
                    for h in history:
                        log.info(
                            f"  {h['side']} qty={h['qty']} "
                            f"entry={h['entry_price']:.4f} exit={h['exit_price']:.4f} "
                            f"realizedPnL={h['realized_pnl']:+.4f} USDT  [{h['updated_time']}]"
                        )
            except Exception as _he:
                log.warning(f"[{coin} PnL히스토리 조회실패] {_he}")
        state["daily_start_capital"] = state["capital"]
        state["daily_date"]  = today
        state["daily_halt"]  = False
        log.info(f"[{coin} 일일리셋] 시작자본={state['capital']:.0f} USDT")

    if not paper_mode and state.get("position", 0) == 0:
        bars_chk = state.get("_bars_since_balance_sync", 0) + 1
        state["_bars_since_balance_sync"] = bars_chk
        if bars_chk >= 12:
            state["_bars_since_balance_sync"] = 0
            try:
                actual_balance = get_usdt_balance(exchange)
                total_tracked  = total_tracked_snapshot or _total_tracked_capital(paper_mode=False)
                if actual_balance > 0 and total_tracked > 0:
                    proportion = state["capital"] / total_tracked
                    expected   = actual_balance * proportion
                    log.info(
                        f"[{coin} 잔고동기화] {state['capital']:,.2f} → {expected:,.2f} USDT"
                    )
                    state["capital"] = expected
            except Exception:
                pass
    else:
        state["_bars_since_balance_sync"] = 0  # 포지션 보유 중엔 카운터 리셋

    _halt_enabled = os.getenv("DAILY_HALT_ENABLED", "false").lower() in ("1", "true", "yes")
    if _halt_enabled and state.get("daily_halt"):
        log.info(f"[{coin}/AF 스킵] daily_halt=True")
        return state

    daily_start    = state.get("daily_start_capital", state["capital"])
    daily_loss_pct = (state["capital"] - daily_start) / (daily_start + 1e-9)
    if _halt_enabled and daily_loss_pct <= -0.02:
        log.warning(f"[{coin}] 일일 손실 한도 {daily_loss_pct:.2%} → 금일 거래 중단")
        if state["position"] != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, reason="일일한도",
                           paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
        state["daily_halt"] = True
        return state

    return process_tick_af(exchange, df, state, price, now_str, paper_mode, paper_trade_csv)


# ── 메인 루프 한 틱 처리 (DL v17 단일코인용) ──────────────────────────────────
def process_tick(exchange, models, scaler, device, params: dict, state: dict,
                 paper_mode: bool = False, paper_trade_csv: Optional[Path] = None) -> dict:
    df = get_signals(exchange, models, scaler, device)
    if df is None or len(df) < SIG_ROLL_WIN + 10:
        return state

    row        = df.iloc[-1]
    candle_ts  = str(row.name)
    if state.get("last_candle_ts") == candle_ts:
        return state
    state["last_candle_ts"] = candle_ts

    price      = float(row["close"])
    state["last_price"] = price
    sig_long   = float(row.get("signal_long",   0.0))
    sig_short  = float(row.get("signal_short",  0.0))
    sig_ctx    = float(row.get("signal_context", 1.0))
    now_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    state["sig_long_hist"].append(sig_long)
    state["sig_short_hist"].append(sig_short)
    state["sig_long_hist"]  = state["sig_long_hist"][-SIG_ROLL_WIN:]
    state["sig_short_hist"] = state["sig_short_hist"][-SIG_ROLL_WIN:]

    roll_l = float(np.mean(state["sig_long_hist"]))
    roll_s = float(np.mean(state["sig_short_hist"]))
    overconfident = ((roll_l + roll_s) / 2.0) > params.get("sig_upper_thr", 1.0)

    state["current_bar"] += 1
    pos     = state["position"]
    capital = state["capital"]
    peak    = state["peak_capital"]

    # ── 일일 손실 한도 (2%) ────────────────────────────────────────────────────
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("daily_date") != today:
        state["daily_start_capital"] = capital
        state["daily_date"]  = today
        state["daily_halt"]  = False
        log.info(f"[일일리셋] 시작자본={capital:.0f} USDT")

    _halt_enabled = os.getenv("DAILY_HALT_ENABLED", "false").lower() in ("1", "true", "yes")
    if _halt_enabled and state.get("daily_halt"):
        return state

    daily_start    = state.get("daily_start_capital", capital)
    daily_loss_pct = (capital - daily_start) / (daily_start + 1e-9)
    if _halt_enabled and daily_loss_pct <= -0.02:
        log.warning(f"[일일한도] 손실 {daily_loss_pct:.2%} → 금일 거래 중단")
        if pos != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, reason="일일한도",
                           paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
        state["daily_halt"] = True
        send_trade_alert(
            f"🚫 <b>일일 손실 한도 도달</b>\n"
            f"손실: {daily_loss_pct:.2%} | 자본: {state['capital']:,.0f} USDT\n"
            f"금일 거래 중단"
        )
        return state

    # ── 미실현 equity / CB ────────────────────────────────────────────────────
    if pos != 0:
        avg_entry = float(state.get("avg_entry_price") or state.get("entry_price") or price)
        pnl_raw = pos * (price - avg_entry) / (avg_entry + 1e-9)
        equity  = capital * (1 + pnl_raw * state["entry_lev"] * state["entry_rr"])
    else:
        equity = capital

    peak = max(peak, equity)
    state["peak_capital"] = peak
    dd = (peak - equity) / (peak + 1e-9)

    if dd > params.get("max_dd_cb", 1.0) and state["cooling_left"] == 0:
        state["cooling_left"]  = params.get("cooling_bars", 0)
        state["cb_triggers"]  += 1
        log.warning(f"[CB] 서킷브레이커 발동 #{state['cb_triggers']} | dd={dd:.2%}")
        if pos != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, paper_mode=paper_mode,
                           paper_trade_csv=paper_trade_csv)
            send_trade_alert(f"⚠️ <b>서킷브레이커</b> #{state['cb_triggers']}\n"
                             f"DD={dd:.2%} → 쿨링 {params.get('cooling_bars',0)}봉")
        return state

    if state["cooling_left"] > 0:
        state["cooling_left"] -= 1
        if pos != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, paper_mode=paper_mode,
                           paper_trade_csv=paper_trade_csv)
        return state

    # ── 청산 체크 (DL v17) ────────────────────────────────────────────────────
    if pos != 0:
        avg_entry = float(state.get("avg_entry_price") or state.get("entry_price") or price)
        pnl_raw  = pos * (price - avg_entry) / (avg_entry + 1e-9)
        pnl_lev  = pnl_raw * state["entry_lev"]
        hold_bars = state["current_bar"] - state["entry_bar"]
        sl_pnl   = params.get("price_sl", 0.02) * state["entry_lev"]
        tp_pnl   = params.get("price_tp", 0.06) * state["entry_lev"]
        tiers    = params.get("tiers", [])

        reverse = hold_bars >= params.get("min_hold_bars", 0) and (
            (pos ==  1 and sig_short >= tiers[-1][0] if tiers else False) or
            (pos == -1 and sig_long  >= tiers[-1][0] if tiers else False)
        )

        if pnl_lev <= -0.9 or pnl_lev <= -sl_pnl or pnl_lev >= tp_pnl or reverse:
            reason = ("TP" if pnl_lev >= tp_pnl else
                      "SL" if pnl_lev <= -sl_pnl else
                      "역신호" if reverse else "강제청산")
            _close_and_log(exchange, state, price, now_str, forced=False, reason=reason,
                           paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
            send_trade_alert(
                f"📤 <b>청산</b> [{reason}]\n"
                f"가격: {price:,.1f} | PnL(lev): {pnl_lev:+.2%}\n"
                f"자본: {state['capital']:,.0f} USDT"
            )

    # ── 신규 진입 (DL v17) ────────────────────────────────────────────────────
    if state["position"] == 0 and sig_ctx >= params.get("context_filter_thr", 0.0):
        tiers  = params.get("tiers", [])
        rr_cap = params.get("rr_cap", 0.5)
        lev, rr = get_tier(sig_long, tiers)

        direction = 0
        if lev > 0:
            direction = 1
        else:
            lev, rr = get_tier(sig_short, tiers)
            if lev > 0:
                direction = -1

        if direction != 0:
            if overconfident:
                src_sig = sig_long if direction == 1 else sig_short
                ti = next((i for i, t in enumerate(tiers) if src_sig >= t[0]), None)
                if ti is not None and ti + 1 < len(tiers):
                    _, lev, rr = tiers[ti + 1]
                else:
                    lev, rr = lev * 0.5, rr * 0.8
            rr = min(rr, rr_cap)
            lev_int = max(1, round(lev))
            dir_str = "LONG 🟢" if direction == 1 else "SHORT 🔴"

            if paper_mode:
                state["position"]        = direction
                state["entry_price"]     = price
                state["avg_entry_price"] = price
                state["entry_qty"]       = calc_qty(capital, rr, lev_int, price)
                state["entry_time"]      = now_str
                state["entry_lev"]       = lev_int
                state["entry_rr"]        = rr
                state["entry_bar"]       = state["current_bar"]
                state["entry_sig_long"]  = sig_long
                state["entry_sig_short"] = sig_short
                log.info(f"[PAPER 진입] {dir_str} | 가격={price:,.1f} | lev={lev_int}x | rr={rr:.2f}")
                send_trade_alert(
                    f"📥 <b>[PAPER] 진입</b> {dir_str}\n"
                    f"가격: {price:,.1f} | 레버: {lev_int}x | rr: {rr:.2f}\n"
                    f"자본(가상): {capital:,.0f} USDT"
                )
            else:
                try:
                    set_leverage(exchange, lev_int)
                    qty = calc_qty(capital, rr, lev_int, price)
                    if qty < MIN_QTY():
                        coin = os.environ.get("COIN", "BTC").upper()
                        log.warning(f"[스킵] 최소수량 미달: {qty:.4f} {coin} < {MIN_QTY()} (자본 {capital:.2f} USDT)")
                    if qty >= MIN_QTY():
                        side = "buy" if direction == 1 else "sell"
                        place_market_order(exchange, side, qty)
                        state["position"]        = direction
                        state["entry_price"]     = price
                        state["avg_entry_price"] = price
                        state["entry_qty"]       = qty
                        state["entry_time"]      = now_str
                        state["entry_lev"]       = lev_int
                        state["entry_rr"]        = rr
                        state["entry_bar"]       = state["current_bar"]
                        state["entry_sig_long"]  = sig_long
                        state["entry_sig_short"] = sig_short
                        log.info(f"[진입] {dir_str} | 가격={price:,.1f} | lev={lev_int}x | rr={rr:.2f} | qty={qty}")
                        send_trade_alert(
                            f"📥 <b>진입</b> {dir_str}\n"
                            f"가격: {price:,.1f} | 레버: {lev_int}x | rr: {rr:.2f}\n"
                            f"수량: {qty} {os.environ.get('COIN','BTC').upper()} | 자본: {capital:,.0f} USDT"
                        )
                except Exception as e:
                    log.error(f"주문 실패: {e}")

    return state


def _close_and_log(exchange, state, price, now_str, forced=False, reason="",
                   paper_mode=False, paper_trade_csv: Optional[Path] = None):
    pos = state["position"]
    if pos == 0:
        return
    entry_price = float(state.get("avg_entry_price") or state.get("entry_price") or 0.0)
    entry_lev = float(state.get("entry_lev") or 0.0)
    entry_rr = float(state.get("entry_rr") or 0.0)

    mark_px = price
    balance_before = 0.0
    if not paper_mode:
        try:
            balance_before = get_usdt_balance(exchange)
            ex_pos = get_position(exchange)
            if entry_price <= 0 and float(ex_pos.get("entry_price") or 0.0) > 0:
                entry_price = float(ex_pos["entry_price"])
            mark_px = float(ex_pos.get("mark_price") or price)
            close_position(exchange, ex_pos)
            time.sleep(1)
            # 실체결가 조회
            try:
                actual_fill = get_last_closed_price(exchange)
                if actual_fill > 0:
                    mark_px = actual_fill
            except Exception:
                pass
            try:
                ex_after = get_position(exchange)
                if ex_after["side"] is None or ex_after["size"] == 0:
                    log.info(f"[청산확인] 포지션 정상 청산 실체결={mark_px:.4f}")
                    cancel_position_stop_loss(exchange)
                else:
                    log.warning(f"[청산경고] 잔여포지션: side={ex_after['side']} size={ex_after['size']}")
            except Exception as e2:
                log.warning(f"[청산확인] 조회실패: {e2}")
            slip_exit_pct = round((mark_px - price) / (price + 1e-9) * 100 * pos, 4)
            if abs(slip_exit_pct) > 0.03:
                log.info(f"[슬리피지] 신호={price:,.4f} 마크={mark_px:,.4f} ({slip_exit_pct:+.3f}%)")
        except Exception as e:
            log.error(f"청산 실패: {e}")
            return
    slip_exit_pct = round((mark_px - price) / (price + 1e-9) * 100 * pos, 4)

    if entry_price <= 0 or entry_lev <= 0 or entry_rr <= 0:
        log.error(
            f"[청산 PnL 오류] invalid entry state: entry={entry_price}, "
            f"lev={entry_lev}, rr={entry_rr}, exit={mark_px}, pos={pos}"
        )
        pnl_raw = 0.0
        pnl = 0.0
    else:
        pnl_raw = pos * (mark_px - entry_price) / entry_price  # 신호가 아닌 실체결가 사용
        pnl = max(pnl_raw * entry_lev * entry_rr, -entry_rr)
    hold_bars = state["current_bar"] - state["entry_bar"]
    capital_before = state["capital"]
    state["capital"] *= (1 + pnl)

    # 심볼별 실현 PnL로 자본 동기화 (수수료 흡수, 다중 동시청산 귀속 오류 방지)
    if not paper_mode:
        try:
            closed = get_closed_pnl_history(exchange, limit=1)
            if closed:
                realized_pnl_usdt = float(closed[0].get("realized_pnl", 0.0))
                if realized_pnl_usdt != 0.0:
                    state["capital"] = capital_before + realized_pnl_usdt
                    log.info(f"[자본 동기화] 계산={capital_before*(1+pnl):,.2f} → realized={state['capital']:,.2f} USDT")
            elif balance_before > 0:
                balance_after = get_usdt_balance(exchange)
                if balance_after > 0:
                    state["capital"] = capital_before + (balance_after - balance_before)
                    log.info(f"[자본 동기화] 계산={capital_before*(1+pnl):,.2f} → 실잔고={state['capital']:,.2f} USDT")
        except Exception:
            pass  # 조회 실패 시 계산값 유지

    state["peak_capital"] = max(state.get("peak_capital", state["capital"]), state["capital"])

    trade_row = {
        "time":               now_str,
        "direction":          pos,
        "entry":              entry_price,
        "exit":               round(mark_px, 6),  # 실체결가 기록
        "pnl":                round(pnl, 6),
        "capital":            round(state["capital"], 2),
        "forced":             forced,
        "reason":             reason,
        "slippage_entry_pct": round(state.pop("_entry_slippage_pct", 0.0), 4),
        "slippage_exit_pct":  slip_exit_pct if not paper_mode else 0.0,
        "exit_mark":          round(mark_px, 4) if not paper_mode else round(price, 4),
    }
    state["trade_log"].append(trade_row)

    if paper_mode:
        if paper_trade_csv is None:
            raise ValueError("paper_trade_csv is required in paper mode")
        _write_paper_trade_csv({
            "timestamp":       now_str,
            "direction":       "long" if pos == 1 else "short",
            "entry_price":     round(entry_price, 4),
            "exit_price":      round(price, 4),
            "hold_bars":       hold_bars,
            "leverage":        entry_lev,
            "rr":              round(entry_rr, 4),
            "sig_long_entry":  round(state.get("entry_sig_long",  0.0), 4),
            "sig_short_entry": round(state.get("entry_sig_short", 0.0), 4),
            "pnl":             round(pnl, 6),
            "capital_after":   round(state["capital"], 2),
            "reason":          reason,
            "forced":          forced,
        }, paper_trade_csv)
        log.info(f"[PAPER 청산] {reason} | 가격={price:,.4f} | PnL={pnl:+.4f} | 자본={state['capital']:,.0f}")
    else:
        log.info(f"[청산] {reason} | 가격={price:,.4f} | PnL={pnl:+.4f} | 자본={state['capital']:,.0f}")

    _clear_entry_fields(state)


REPORT_INTERVAL = 12   # 12 × 5분 = 1시간


# ── 단일코인 1시간 상태 보고 ──────────────────────────────────────────────────
def build_hourly_report(exchange, state: dict, mode: str, paper_mode: bool = False) -> str:
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🕐 <b>1시간 상태 보고</b> [{mode.upper()}]", f"⏰ {now}\n"]

    if paper_mode:
        lines.append("💰 <b>지갑 잔고</b>: paper 모드 미조회")
    else:
        try:
            bal = get_usdt_balance(exchange)
            lines.append(f"💰 <b>지갑 잔고</b>: {bal:,.2f} USDT")
        except Exception:
            lines.append("💰 <b>지갑 잔고</b>: 조회 실패")

    pos = state["position"]
    if pos == 0:
        lines.append("📊 <b>포지션</b>: 없음 (관망)")
    else:
        dir_str   = "LONG 🟢" if pos == 1 else "SHORT 🔴"
        entry     = float(state.get("avg_entry_price") or state.get("entry_price") or 0.0)
        if paper_mode:
            cur_price = state.get("last_price") or entry
        else:
            try:
                ex_pos = get_position(exchange)
                cur_price = ex_pos.get("mark_price") or entry
            except Exception:
                cur_price = entry
        pnl_raw   = pos * (cur_price - entry) / (entry + 1e-9)
        pnl_lev   = pnl_raw * state["entry_lev"]
        hold_bars = state["current_bar"] - state["entry_bar"]
        lines.append(
            f"📊 <b>포지션</b>: {dir_str}\n"
            f"   진입가: {entry:,.4f} | 레버: {state['entry_lev']}x\n"
            f"   보유: {hold_bars}봉 ({hold_bars*5//60}h {hold_bars*5%60}m)\n"
            f"   미실현 PnL(lev): {pnl_lev:+.2%}"
        )

    capital = state["capital"]
    ret_pct = (capital / INITIAL_CAPITAL - 1) * 100
    peak    = state["peak_capital"]
    dd      = max(0.0, (peak - capital) / (peak + 1e-9) * 100)
    lines.append(
        f"\n📈 <b>수익률</b>: {ret_pct:+.2f}%"
        f"  (자본 {capital:,.0f} USDT)\n"
        f"   MDD: {dd:.2f}% | CB: {state['cb_triggers']}회"
    )

    trades   = state.get("trade_log", [])
    n_trades = len(trades)
    n_win    = sum(1 for t in trades if t.get("pnl", 0) > 0)
    wr       = n_win / n_trades * 100 if n_trades else 0
    cooling  = state.get("cooling_left", 0)
    lines.append(
        f"\n🔢 <b>거래 통계</b>: 총 {n_trades}건 | WR {wr:.1f}%"
        + (f"\n⏸ 쿨링 중: {cooling}봉 남음" if cooling > 0 else "")
    )

    return "\n".join(lines)


# ── 멀티코인 1시간 상태 보고 ──────────────────────────────────────────────────
def build_hourly_report_multi(all_states: dict, mode: str, paper_mode: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🕐 <b>1시간 보고 [MULTI-AF {mode.upper()}]</b>", f"⏰ {now}\n"]

    total_capital = sum(s["capital"] for s in all_states.values())
    total_initial = sum(s.get("initial_capital", INITIAL_CAPITAL * MULTI_COIN_ALLOC) for s in all_states.values())
    total_ret     = (total_capital / (total_initial + 1e-9) - 1) * 100
    total_trades  = sum(len(s.get("trade_log", [])) for s in all_states.values())
    total_wins    = sum(sum(1 for t in s.get("trade_log", []) if t.get("pnl", 0) > 0) for s in all_states.values())
    total_wr_str  = f" · WR {total_wins/total_trades*100:.0f}%" if total_trades else ""
    lines.append(f"💰 <b>총 자본</b>: {total_capital:,.2f} USDT  ({total_ret:+.2f}%)")
    lines.append(f"📊 총 {total_trades}건{total_wr_str}\n")

    for coin, state in all_states.items():
        pos          = state["position"]
        capital      = state["capital"]
        coin_initial = state.get("initial_capital", INITIAL_CAPITAL * MULTI_COIN_ALLOC)
        ret          = (capital / (coin_initial + 1e-9) - 1) * 100
        peak    = state.get("peak_capital", capital)
        dd      = max(0.0, (peak - capital) / (peak + 1e-9) * 100)
        pos_str = "없음" if pos == 0 else ("LONG 🟢" if pos == 1 else "SHORT 🔴")
        trades  = state.get("trade_log", [])
        n_win   = sum(1 for t in trades if t.get("pnl", 0) > 0)
        wr      = n_win / len(trades) * 100 if trades else 0.0

        trail_str = ""
        if pos != 0:
            trail_sl  = state.get("af_trail_sl", 0)
            entry_p   = state.get("avg_entry_price") or state.get("entry_price", 0)
            pyr       = state.get("af_pyramid_count", 0)
            last_px   = state.get("last_price", 0)
            unreal_str = ""
            if last_px > 0 and entry_p > 0:
                pnl_raw = pos * (last_px - entry_p) / (entry_p + 1e-9)
                pnl_lev = pnl_raw * state.get("entry_lev", 1) * state.get("entry_rr", 0.1)
                unreal_str = f" | 미실현 {pnl_lev:+.2%}"
            trail_dist = ""
            if last_px > 0 and trail_sl > 0:
                dist_pct = abs(last_px - trail_sl) / last_px * 100
                trail_dist = f" (-{dist_pct:.1f}%)"
            trail_str = (
                f"\n  ↳ 진입가={entry_p:,.4f}{unreal_str} · pyr={pyr}/3"
                f"\n  ↳ trail청산={trail_sl:,.4f}{trail_dist}"
            )

        lines.append(
            f"<b>{coin}</b>: {capital:,.0f} USDT ({ret:+.2f}%) | {pos_str}{trail_str}\n"
            f"  거래 {len(trades)}건 · WR {wr:.0f}% · MDD {dd:.1f}%"
        )

    return "\n".join(lines)


# ── 멀티코인 KST 09:00 일일 보고 ──────────────────────────────────────────────
def build_daily_report_multi(all_states: dict, mode: str, paper_mode: bool) -> str:
    """UTC 00:00 (= KST 09:00) 날짜 변경 직전 상태로 전일 결과 보고."""
    kst_now  = datetime.now(timezone.utc) + timedelta(hours=9)
    kst_str  = kst_now.strftime("%Y-%m-%d %H:%M KST")
    lines = [f"📅 <b>일일 보고 [MULTI-AF {mode.upper()}]</b>", f"⏰ {kst_str}\n"]

    total_capital     = sum(s["capital"] for s in all_states.values())
    total_initial     = sum(s.get("initial_capital", INITIAL_CAPITAL * MULTI_COIN_ALLOC) for s in all_states.values())
    total_daily_start = sum(s.get("daily_start_capital", s.get("initial_capital", INITIAL_CAPITAL * MULTI_COIN_ALLOC)) for s in all_states.values())
    daily_ret  = (total_capital / (total_daily_start + 1e-9) - 1) * 100
    daily_usdt = total_capital - total_daily_start
    total_ret  = (total_capital / (total_initial + 1e-9) - 1) * 100
    lines.append(
        f"💰 <b>총 자본</b>: {total_capital:,.2f} USDT\n"
        f"   전일 대비: {daily_ret:+.2f}% ({daily_usdt:+.2f} USDT) | 누적: {total_ret:+.2f}%\n"
    )

    for coin, state in all_states.items():
        capital      = state["capital"]
        coin_initial = state.get("initial_capital", INITIAL_CAPITAL * MULTI_COIN_ALLOC)
        d_start      = state.get("daily_start_capital", coin_initial)
        d_ret        = (capital / (d_start + 1e-9) - 1) * 100
        d_usdt       = capital - d_start
        tot_ret      = (capital / (coin_initial + 1e-9) - 1) * 100
        trades   = state.get("trade_log", [])
        n        = len(trades)
        n_win    = sum(1 for t in trades if t.get("pnl", 0) > 0)
        wr       = n_win / n * 100 if n else 0
        peak     = state.get("peak_capital", capital)
        dd       = max(0.0, (peak - capital) / (peak + 1e-9) * 100)
        lines.append(
            f"<b>{coin}</b>: {capital:,.0f} USDT  전일 {d_ret:+.2f}% ({d_usdt:+.0f} USDT)\n"
            f"  누적 {tot_ret:+.2f}% · {n}건 WR {wr:.0f}% · MDD {dd:.1f}%"
        )

    return "\n".join(lines)


# ── 다음 5분봉까지 대기 ────────────────────────────────────────────────────────
def sleep_until_next_candle(on_wait_tick=None, poll_interval: float = STOP_POLL_INTERVAL) -> bool:
    now  = time.time()
    wait = 300 - (now % 300) + 5   # 5초 여유
    log.info(f"다음 캔들까지 {wait:.0f}초 대기...")
    deadline = now + wait

    while True:
        remain = deadline - time.time()
        if remain <= 0:
            return False
        if on_wait_tick is not None and on_wait_tick():
            return True
        time.sleep(min(poll_interval, remain))


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    load_env_file()
    preset_name = os.getenv("AF_PARAM_PRESET", "").lower()
    if preset_name in _AF_PRESETS:
        AF_PARAMS.update(_AF_PRESETS[preset_name])
        log.info(f"[AF] 프리셋 적용: {preset_name}")
    AF_PARAMS["leverage"] = int(os.getenv("LEVERAGE", str(AF_PARAMS["leverage"])))
    strategy   = os.environ.get("STRATEGY", "dl_v17")   # 반드시 최상단에서 정의
    trade_mode = os.getenv("TRADE_MODE", "sandbox").lower()
    paper_mode = (trade_mode == "paper")
    multi_mode = (strategy == "antifragile")

    # 로그 파일: 멀티코인은 공유 로그, 단일코인은 코인별
    if multi_mode:
        log_fname = "paper_multi.log" if paper_mode else "live_multi.log"
        configure_logging(ROOT / "logs" / log_fname)
    else:
        paths = get_runtime_paths(paper_mode)
        configure_logging(paths["log_file"])

    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    if multi_mode:
        params = {}
        models = None
        scaler = None
        log.info("[AF] DL 모델 로드 스킵 (Antifragile 전략 rule-based)")
    else:
        params = load_params()
        models, scaler = load_models(device)

    exchange, mode = build_exchange(trade_mode)

    try:
        tg_token, tg_chat_id = get_credentials()
    except Exception:
        tg_token, tg_chat_id = "", ""

    log.info("=" * 60)
    log.info("  ConnectAI Trade Bot 시작")
    if multi_mode:
        log.info(f"  코인:     {' / '.join(COINS_MULTI)} (멀티 4종목)")
        log.info(f"  배분:     각 {MULTI_COIN_ALLOC*100:.0f}%")
    else:
        coin = os.environ.get("COIN", "BTC").upper()
        log.info(f"  코인:     {coin}/USDT ({get_symbol()})")
    log.info(f"  전략:     {strategy}")
    log.info(f"  모드:     {trade_mode.upper()}")
    log.info(f"  디바이스: {device}")
    log.info("=" * 60)

    # ══════════════════════════════════════════════════════════════
    #  멀티코인 Antifragile 경로
    # ══════════════════════════════════════════════════════════════
    if multi_mode:
        seed      = float(os.getenv("PAPER_SEED", "10000"))
        coin_seed = seed * MULTI_COIN_ALLOC   # 2,500 per coin

        # real 모드: 실계좌 잔고 조회 → 종목당 자본 계산
        real_coin_seed: float | None = None
        if mode == "real":
            try:
                real_bal = get_usdt_balance(exchange)
                if real_bal > 0:
                    real_coin_seed = real_bal * MULTI_COIN_ALLOC
                    log.info(f"[실계좌] 잔고 조회: {real_bal:.2f} USDT → 종목당 {real_coin_seed:.2f} USDT")
            except Exception as e:
                log.warning(f"[실계좌] 잔고 조회 실패, INITIAL_CAPITAL 사용: {e}")

        all_states: dict[str, dict] = {}
        all_paths:  dict[str, dict] = {}

        for c in COINS_MULTI:
            os.environ["COIN"] = c
            cpaths = get_runtime_paths(paper_mode)
            all_paths[c] = cpaths
            sf = cpaths["state_file"]

            if sf.exists():
                try:
                    st = json.loads(sf.read_text())
                    # AF 필드 누락 시 마이그레이션
                    for k, v in DEFAULT_STATE.items():
                        st.setdefault(k, copy.deepcopy(v))
                    st["peak_capital"] = max(st.get("peak_capital", st["capital"]), st["capital"])
                    # real 모드 시작 시 포지션 동기화 (오프라인 SL 감지 + entry_qty 복원)
                    if mode == "real" and st.get("position", 0) != 0:
                        try:
                            os.environ["COIN"] = c
                            ex_pos = get_position(exchange)
                            if ex_pos["side"] is None or ex_pos["size"] == 0:
                                # 오프라인 중 SL/TP 체결 — realized PnL로 자본 복원 후 state 초기화
                                log.warning(f"[{c}] 시작 시 포지션 불일치 — 오프라인 체결 감지, 자본 복원")
                                capital_before_rc = st["capital"]
                                entry_price_rc = float(st.get("avg_entry_price") or st.get("entry_price") or 0.0)
                                pos_rc         = st["position"]
                                realized_applied = False
                                try:
                                    closed = get_closed_pnl_history(exchange, limit=1)
                                    if closed:
                                        realized_pnl_usdt = float(closed[0].get("realized_pnl", 0.0))
                                        exit_price_rc     = float(closed[0].get("exit_price") or 0.0)
                                        if realized_pnl_usdt != 0.0:
                                            st["capital"] = capital_before_rc + realized_pnl_usdt
                                            st["peak_capital"] = max(st.get("peak_capital", st["capital"]), st["capital"])
                                            now_s = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                                            st["trade_log"].append({
                                                "time":               now_s,
                                                "direction":          pos_rc,
                                                "entry":              entry_price_rc,
                                                "exit":               exit_price_rc,
                                                "pnl":                round(realized_pnl_usdt / (capital_before_rc + 1e-9), 6),
                                                "capital":            round(st["capital"], 2),
                                                "forced":             False,
                                                "reason":             "STARTUP_RECONCILE",
                                                "slippage_entry_pct": 0.0,
                                                "slippage_exit_pct":  0.0,
                                                "exit_mark":          round(exit_price_rc, 4),
                                            })
                                            log.info(
                                                f"[{c}] 시작 자본복원 완료: "
                                                f"realized_pnl={realized_pnl_usdt:+.4f} USDT "
                                                f"→ {st['capital']:,.2f} USDT"
                                            )
                                            realized_applied = True
                                except Exception as _re:
                                    log.warning(f"[{c}] realized_pnl 조회 실패: {_re}")
                                if not realized_applied:
                                    log.warning(f"[{c}] 자본복원 실패 — 기존값 유지 (수동 확인 필요)")
                                _clear_entry_fields(st)
                                atomic_write_json(sf, st)  # 대시보드 즉시 반영
                            else:
                                # 포지션 여전히 열려있음 — 수량·진입가 동기화
                                st["entry_qty"]       = float(ex_pos["size"])
                                st["avg_entry_price"] = float(ex_pos["entry_price"] or st.get("entry_price", 0.0))
                                log.info(f"[{c}] 포지션 확인: {ex_pos['side']} {ex_pos['size']} @ {st['avg_entry_price']:.4f}")
                                atomic_write_json(sf, st)  # 대시보드 즉시 반영
                        except Exception as _em:
                            log.warning(f"[{c}] 시작 포지션 동기화 실패: {_em}")
                    all_states[c] = st
                except Exception:
                    all_states[c] = fresh_state()
            else:
                st = fresh_state()
                if paper_mode:
                    st["capital"]          = coin_seed
                    st["initial_capital"]  = coin_seed
                    st["peak_capital"]     = coin_seed
                    st["daily_start_capital"] = coin_seed
                    atomic_write_json(sf, st)
                    log.info(f"[PAPER-{c}] 가상 시드 초기화: {coin_seed:,.0f} USDT")
                elif real_coin_seed is not None:
                    st["capital"]          = real_coin_seed
                    st["initial_capital"]  = real_coin_seed
                    st["peak_capital"]     = real_coin_seed
                    st["daily_start_capital"] = real_coin_seed
                    atomic_write_json(sf, st)
                    log.info(f"[REAL-{c}] 잔고 동기화: {real_coin_seed:.2f} USDT")
                all_states[c] = st

        total_cap = sum(s["capital"] for s in all_states.values())
        log.info(f"총 자본: {total_cap:,.2f} USDT | 종목당: {total_cap/len(all_states):.2f} USDT")

        # 시작 시 exchange 잔고 기준 자본 비례 보정 (과거 잘못된 동기화 누적 교정)
        if mode == "real" and total_cap > 0:
            try:
                actual_balance = get_usdt_balance(exchange)
                if actual_balance > 0:
                    for c in COINS_MULTI:
                        s = all_states[c]
                        proportion = s["capital"] / total_cap
                        new_cap    = actual_balance * proportion
                        log.info(
                            f"[{c} startup동기화] {s['capital']:,.2f} → {new_cap:,.2f} USDT "
                            f"(비율={proportion:.4f})"
                        )
                        s["capital"]      = new_cap
                        s["peak_capital"] = max(s.get("peak_capital", new_cap), new_cap)
                        atomic_write_json(all_paths[c]["state_file"], s)
                    total_cap = actual_balance
                    log.info(f"[startup동기화 완료] exchange={actual_balance:,.2f} USDT → 4코인 비례 보정")
            except Exception as _ss:
                log.warning(f"[startup 자본동기화 실패] {_ss}")

        per_coin_cap = total_cap / len(all_states)
        send_trade_alert(
            f"🚀 <b>멀티코인 트레이딩 봇 시작</b> [{mode.upper()}]\n"
            f"전략: Antifragile | 종목: {' / '.join(COINS_MULTI)}\n"
            f"총 자본: {total_cap:,.2f} USDT (종목당 {per_coin_cap:,.2f})"
            + ("\n📄 거래 기록 저장 중" if paper_mode else "")
        )

        def _save_all():
            for c in COINS_MULTI:
                atomic_write_json(all_paths[c]["state_file"], all_states[c])

        def _poll_stop_multi() -> bool:
            if not tg_token:
                return False
            try:
                offset = all_states["BTC"].get("tg_update_offset", 0)
                cmds, new_offset = poll_commands(tg_token, tg_chat_id, offset)
                if new_offset != offset:
                    for c in COINS_MULTI:
                        all_states[c]["tg_update_offset"] = new_offset
                if "/account" in cmds:
                    report = build_hourly_report_multi(all_states, mode, paper_mode)
                    send_trade_alert(report)
                    log.info("[텔레그램] /account 처리 완료")
                if "/stop" not in cmds:
                    return False
                log.warning("[텔레그램] /stop 수신 → 봇 종료")
                now_s = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                for c in COINS_MULTI:
                    os.environ["COIN"] = c
                    s = all_states[c]
                    if s["position"] != 0 and s.get("last_price", 0) > 0:
                        _close_and_log(exchange, s, s["last_price"], now_s,
                                       forced=True, reason="텔레그램 /stop",
                                       paper_mode=paper_mode,
                                       paper_trade_csv=all_paths[c]["paper_trade_csv"])
                _save_all()
                send_trade_alert("🛑 <b>봇 종료</b> (텔레그램 /stop)")
                return True
            except Exception as e:
                log.warning(f"텔레그램 폴링 실패: {e}")
                return False

        last_report_hour = -1
        last_daily_date  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        while True:
            try:
                if _poll_stop_multi():
                    break

                # 일일 보고: UTC 00:00 (KST 09:00) 날짜 변경 시, tick 처리 전에 발송
                today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today_utc != last_daily_date:
                    send_trade_alert(build_daily_report_multi(all_states, mode, paper_mode))
                    log.info(f"[일일 보고] {last_daily_date} 결산 → 텔레그램 발송")
                    last_daily_date = today_utc

                total_tracked_snapshot = sum(s.get("capital", 0.0) for s in all_states.values())
                for c in COINS_MULTI:
                    try:
                        all_states[c] = _run_coin_tick_af(
                            exchange, c, all_states[c], paper_mode,
                            all_paths[c]["paper_trade_csv"],
                            total_tracked_snapshot
                        )
                        atomic_write_json(all_paths[c]["state_file"], all_states[c])
                    except Exception as e:
                        log.error(f"[{c}] 틱 처리 오류: {e}", exc_info=True)

                # 1시간 보고: UTC 정각 단위 (시작 시간 무관)
                cur_hour = datetime.now(timezone.utc).hour
                if cur_hour != last_report_hour:
                    last_report_hour = cur_hour
                    report = build_hourly_report_multi(all_states, mode, paper_mode)
                    send_trade_alert(report)
                    log.info("[1시간 보고] 텔레그램 발송")

            except KeyboardInterrupt:
                log.info("수동 중지")
                _save_all()
                send_trade_alert("🛑 <b>봇 수동 중지</b>")
                break
            except Exception as e:
                log.error(f"틱 처리 오류: {e}", exc_info=True)

            if sleep_until_next_candle(on_wait_tick=_poll_stop_multi):
                break

    # ══════════════════════════════════════════════════════════════
    #  단일코인 DL v17 경로
    # ══════════════════════════════════════════════════════════════
    else:
        coin       = os.environ.get("COIN", "BTC").upper()
        state_file = paths["state_file"]
        paper_trade_csv = paths["paper_trade_csv"]

        state_existed = state_file.exists()
        state = (json.loads(state_file.read_text()) if state_existed else fresh_state())
        for k, v in DEFAULT_STATE.items():
            state.setdefault(k, copy.deepcopy(v))
        state["peak_capital"] = max(state.get("peak_capital", state["capital"]), state["capital"])

        if paper_mode and not state_existed:
            seed = float(os.getenv("PAPER_SEED", "10000"))
            state["capital"]      = seed
            state["peak_capital"] = seed
            atomic_write_json(state_file, state)
            log.info(f"[PAPER] 가상 시드 초기화: {seed:,.0f} USDT")

        if mode == "real" and len(state.get("trade_log", [])) == 0:
            try:
                real_bal = get_usdt_balance(exchange)
                if real_bal > 0:
                    state["capital"]      = real_bal
                    state["peak_capital"] = real_bal
                    atomic_write_json(state_file, state)
                    log.info(f"[실계좌] 잔고 동기화: {real_bal:.2f} USDT")
            except Exception as e:
                log.warning(f"잔고 조회 실패, 기본값 사용: {e}")

        log.info("=" * 55)
        log.info(f"라이브 트레이더 시작 | 모드: {mode.upper()} | 디바이스: {device}")
        log.info(f"rr_cap={params.get('rr_cap','N/A')} | dd_cb={params.get('max_dd_cb','N/A')} | hold={params.get('min_hold_bars','N/A')}")
        log.info(f"자본: {state['capital']:,.0f} USDT | 포지션: {state['position']}")
        if paper_mode:
            log.info(f"[PAPER] 거래 기록 → {paper_trade_csv}")
        log.info("=" * 55)

        send_trade_alert(
            f"🚀 <b>트레이딩 봇 시작</b> [{mode.upper()}]\n"
            f"rr_cap={params.get('rr_cap','N/A')} | dd_cb={params.get('max_dd_cb','N/A')}\n"
            f"자본: {state['capital']:,.0f} USDT"
            + ("\n📄 거래 기록 저장 중" if paper_mode else "")
        )

        def _save(s):
            atomic_write_json(state_file, s)

        def _poll_stop_command() -> bool:
            nonlocal state
            if not tg_token:
                return False
            try:
                cmds, new_offset = poll_commands(
                    tg_token, tg_chat_id, state.get("tg_update_offset", 0)
                )
                if new_offset != state.get("tg_update_offset", 0):
                    state["tg_update_offset"] = new_offset
                    _save(state)
                if "/stop" not in cmds:
                    return False
                log.warning("[텔레그램] /stop 수신 → 봇 종료")
                last_price = state.get("last_price", 0.0)
                now_s = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                if state["position"] != 0 and last_price > 0:
                    _close_and_log(exchange, state, last_price, now_s,
                                   forced=True, reason="텔레그램 /stop",
                                   paper_mode=paper_mode,
                                   paper_trade_csv=paper_trade_csv)
                _save(state)
                send_trade_alert("🛑 <b>봇 종료</b> (텔레그램 /stop)")
                return True
            except Exception as e:
                log.warning(f"텔레그램 폴링 실패: {e}")
                return False

        last_report_hour = -1
        last_daily_date  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        while True:
            try:
                if _poll_stop_command():
                    break

                today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today_utc != last_daily_date:
                    last_daily_date = today_utc

                state = process_tick(exchange, models, scaler, device, params, state,
                                     paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
                _save(state)

                cur_hour = datetime.now(timezone.utc).hour
                if cur_hour != last_report_hour:
                    last_report_hour = cur_hour
                    report = build_hourly_report(exchange, state, mode, paper_mode=paper_mode)
                    send_trade_alert(report)
                    log.info("[1시간 보고] 텔레그램 발송")
            except KeyboardInterrupt:
                log.info("수동 중지")
                _save(state)
                send_trade_alert("🛑 <b>봇 수동 중지</b>")
                break
            except Exception as e:
                log.error(f"틱 처리 오류: {e}", exc_info=True)
            if sleep_until_next_candle(on_wait_tick=_poll_stop_command):
                break


if __name__ == "__main__":
    main()
