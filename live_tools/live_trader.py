"""
라이브 트레이더 (NAS 경량화 버전) — Antifragile 전략 전용
TRADE_MODE 옵션 (.env):
  real    : 실계좌 주문 실행
  sandbox : 테스트넷 주문 실행
  paper   : 실시세 조회 + 가상 주문 시뮬레이션 (주문 없음, CSV 저장)

STRATEGY: antifragile (AdaptRSI + ATR trailing stop, 4종목 25%씩 분할)

실행:
  python live_tools/live_trader.py
  nohup python live_tools/live_trader.py > logs/live.log 2>&1 &
"""
from __future__ import annotations
import sys, os, json, time, logging, csv, copy
from logging.handlers import RotatingFileHandler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data_pipeline    import add_technical_indicators, FEATURE_COLS

from exchange_client import (
    build_exchange, get_usdt_balance, get_position,
    set_leverage, place_market_order, close_position,
    fetch_ohlcv_df, calc_qty, get_symbol, get_min_qty,
    set_position_stop_loss, cancel_position_stop_loss,
)
from telegram_notifier import send_trade_alert, poll_commands, get_credentials

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
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
    "af_registered_sl":  0.0,   # 거래소에 등록된 SL 가격 (중복 갱신 방지)
}

# ── Antifragile 전략 파라미터 ──────────────────────────────────────────────────
# 검증: BTC 9/10 +419%/3개월, ETH 10/10 +678%, SOL 10/10 +2944%, XRP 10/10 +10985%
# 2026-06-10: ut_rsi_lo 35→40, ut_rsi_hi 78→85 (E.UT-NoShrt, 4종목 전수 개선)
# 2026-06-10: leverage 3→5 (레버리지 스윕 — 4종목 hist 9~10/10 유지, MDD ≤5.6%)
AF_PARAMS = {
    "dt_rsi_lo":       22,    # 하락추세: 롱 진입 RSI 임계값
    "dt_rsi_hi":       65,    # 하락추세: 숏 진입 RSI 임계값
    "rg_rsi_lo":       30,    # 횡보:     롱 진입 RSI 임계값
    "rg_rsi_hi":       70,    # 횡보:     숏 진입 RSI 임계값
    "ut_rsi_lo":       40,    # 상승추세: 롱 진입 RSI 임계값 (35→40, 상승장 롱 완화)
    "ut_rsi_hi":       85,    # 상승추세: 숏 진입 RSI 임계값 (78→85, 상승장 숏 억제)
    "trail_atr_init":  1.0,   # 초기 trailing stop 거리 (ATR 배수)
    "trail_atr_tight": 1.5,   # 피라미딩 후 tight trailing (ATR 배수)
    "rr_base":         0.10,  # 초기 자본 위험 비율
    "rr_add":          0.15,  # 피라미딩 1회당 추가 비율
    "add_levels":      3,     # 최대 피라미딩 횟수
    "atr_add_step":    0.5,   # 피라미딩 트리거 (유리방향 X×ATR마다)
    "leverage":        5,     # 레버리지 (3→5, 2026-06-10 레버리지 스윕 검증)
    "max_hold_bars":   288,   # 최대 보유봉수 (1일)
}


def fresh_state() -> dict:
    return copy.deepcopy(DEFAULT_STATE)


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
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return fresh_state()

def save_state(state: dict):
    atomic_write_json(STATE_FILE, state)



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

    # 1h EMA20 추세 (마지막 1h 봉 기준)
    try:
        cl1h   = close.resample("1h").last().ffill()
        ema_1h = cl1h.ewm(span=20, adjust=False).mean()
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
            entry_price = float(state.get("entry_price") or price)
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
            if pos == 1:
                state["af_peak_price"] = max(state["af_peak_price"], price)
                new_trail = state["af_peak_price"] - trail_mult * atr
                state["af_trail_sl"] = max(state["af_trail_sl"], new_trail)
            else:
                state["af_peak_price"] = min(state["af_peak_price"], price)
                new_trail = state["af_peak_price"] + trail_mult * atr
                state["af_trail_sl"] = min(state["af_trail_sl"], new_trail)

            # 피라미딩 체크 (유리방향 N×ATR마다 추가)
            entry_atr = state.get("af_entry_atr", atr) or atr
            favorable = pos * (price - state["entry_price"]) / (entry_atr + 1e-9)
            next_lvl  = (state["af_pyramid_count"] + 1) * p["atr_add_step"]
            if state["af_pyramid_count"] < p["add_levels"] and favorable >= next_lvl:
                state["af_pyramid_count"] += 1
                state["af_current_rr"]    += p["rr_add"]
                state["entry_rr"]          = state["af_current_rr"]
                if pos == 1:
                    state["af_trail_sl"] = max(state["af_trail_sl"], price - p["trail_atr_tight"] * atr)
                else:
                    state["af_trail_sl"] = min(state["af_trail_sl"], price + p["trail_atr_tight"] * atr)

                log.info(f"[{coin}/AF 피라미딩 #{state['af_pyramid_count']}] "
                         f"favorable={favorable:.2f}ATR | rr={state['af_current_rr']:.2f} | "
                         f"trail_sl={state['af_trail_sl']:,.4f}")

                if not paper_mode:
                    try:
                        add_qty = calc_qty(state["capital"], p["rr_add"], lev, price)
                        if add_qty >= MIN_QTY():
                            side = "buy" if pos == 1 else "sell"
                            place_market_order(exchange, side, add_qty)
                            send_trade_alert(
                                f"➕ <b>[{coin}/AF] 피라미딩 #{state['af_pyramid_count']}</b>\n"
                                f"가격: {price:,.4f} | 추가수량: {add_qty}"
                            )
                    except Exception as e:
                        log.error(f"[{coin}/AF] 피라미딩 주문 실패: {e}")

            # trail_sl 변경 시 거래소 SL 갱신 (real 모드)
            if not paper_mode:
                new_sl = state["af_trail_sl"]
                if new_sl != state.get("af_registered_sl", 0.0):
                    try:
                        set_position_stop_loss(exchange, new_sl)
                        state["af_registered_sl"] = new_sl
                        log.info(f"[{coin}/AF SL갱신] {new_sl:,.4f}")
                    except Exception as e_sl:
                        log.warning(f"[{coin}/AF SL갱신 실패] {e_sl}")

    # ── 신규 진입 ─────────────────────────────────────────────────────────────
    if state["position"] == 0:
        long_ok  = long_ok_now
        short_ok = short_ok_now

        direction = 1 if long_ok else (-1 if short_ok else 0)
        if direction != 0:
            dir_str    = "LONG 🟢" if direction == 1 else "SHORT 🔴"
            init_trail = (price - p["trail_atr_init"] * atr) if direction == 1 else \
                         (price + p["trail_atr_init"] * atr)

            state["position"]          = direction
            state["entry_price"]       = price
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
                        state["position"] = 0
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
                                state["_entry_signal_price"]   = price
                                state["_entry_slippage_pct"]   = round(slip_pct, 4)
                                log.info(f"[{coin}/AF] 체결확인 | 신호={price:,.4f} 체결={actual_px:,.4f} 슬리피지={slip_pct:+.3f}%")
                        except Exception as e2:
                            log.warning(f"[{coin}/AF] 체결확인 실패: {e2}")
                        # 진입 직후 거래소 SL 등록
                        try:
                            set_position_stop_loss(exchange, state["af_trail_sl"])
                            state["af_registered_sl"] = state["af_trail_sl"]
                            log.info(f"[{coin}/AF SL등록] {state['af_trail_sl']:,.4f}")
                        except Exception as e_sl:
                            log.warning(f"[{coin}/AF SL등록 실패] {e_sl}")
                except Exception as e:
                    log.error(f"[{coin}/AF] 진입 주문 실패: {e}")
                    state["position"] = 0

    return state


# ── 멀티코인 AF 전용: 단일 코인 틱 처리 ───────────────────────────────────────
def _run_coin_tick_af(exchange, coin: str, state: dict, paper_mode: bool,
                      paper_trade_csv: Optional[Path]) -> dict:
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
                state["position"]          = 0
                state["af_trail_sl"]       = 0.0
                state["af_registered_sl"]  = 0.0
                state["af_peak_price"]     = 0.0
                state["af_pyramid_count"]  = 0
                state["af_current_rr"]     = 0.0
                send_trade_alert(
                    f"⚡ <b>[{coin}/AF] 거래소 SL 자동체결</b>\n"
                    f"봉 사이에 trail_SL 도달 → 포지션 종료됨"
                )
        except Exception as e:
            log.warning(f"[{coin}/AF] 포지션 동기화 조회 실패: {e}")

    price = float(row["close"])
    state["last_price"] = price
    state["current_bar"] += 1
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 일일 손실 한도 리셋 (2%)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("daily_date") != today:
        state["daily_start_capital"] = state["capital"]
        state["daily_date"]  = today
        state["daily_halt"]  = False
        log.info(f"[{coin} 일일리셋] 시작자본={state['capital']:.0f} USDT")

    if state.get("daily_halt"):
        log.info(f"[{coin}/AF 스킵] daily_halt=True")
        return state

    daily_start    = state.get("daily_start_capital", state["capital"])
    daily_loss_pct = (state["capital"] - daily_start) / (daily_start + 1e-9)
    if daily_loss_pct <= -0.02:
        log.warning(f"[{coin}] 일일 손실 한도 {daily_loss_pct:.2%} → 금일 거래 중단")
        if state["position"] != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, reason="일일한도",
                           paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
        state["daily_halt"] = True
        return state

    return process_tick_af(exchange, df, state, price, now_str, paper_mode, paper_trade_csv)


def _close_and_log(exchange, state, price, now_str, forced=False, reason="",
                   paper_mode=False, paper_trade_csv: Optional[Path] = None):
    pos = state["position"]
    if pos == 0:
        return
    entry_price = float(state.get("entry_price") or 0.0)
    entry_lev = float(state.get("entry_lev") or 0.0)
    entry_rr = float(state.get("entry_rr") or 0.0)

    mark_px = price
    if not paper_mode:
        try:
            ex_pos = get_position(exchange)
            if entry_price <= 0 and float(ex_pos.get("entry_price") or 0.0) > 0:
                entry_price = float(ex_pos["entry_price"])
            mark_px = float(ex_pos.get("mark_price") or price)
            close_position(exchange, ex_pos)
            time.sleep(1)
            try:
                ex_after = get_position(exchange)
                if ex_after["side"] is None or ex_after["size"] == 0:
                    log.info(f"[청산확인] 포지션 정상 청산")
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
            f"lev={entry_lev}, rr={entry_rr}, exit={price}, pos={pos}"
        )
        pnl_raw = 0.0
        pnl = 0.0
    else:
        pnl_raw = pos * (price - entry_price) / entry_price
        pnl = max(pnl_raw * entry_lev * entry_rr, -entry_rr)
    hold_bars = state["current_bar"] - state["entry_bar"]
    state["capital"] *= (1 + pnl)
    state["peak_capital"] = max(state.get("peak_capital", state["capital"]), state["capital"])

    trade_row = {
        "time":               now_str,
        "direction":          pos,
        "entry":              entry_price,
        "exit":               price,
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

    state["position"]          = 0
    state["entry_price"]       = 0.0
    state["entry_lev"]         = 1.0
    state["entry_rr"]          = 0.0
    state["entry_sig_long"]    = 0.0
    state["entry_sig_short"]   = 0.0
    state["af_trail_sl"]       = 0.0
    state["af_peak_price"]     = 0.0
    state["af_pyramid_count"]  = 0
    state["af_current_rr"]     = 0.0
    state["af_entry_atr"]      = 0.0


REPORT_INTERVAL = 12   # 12 × 5분 = 1시간


# ── 멀티코인 1시간 상태 보고 ──────────────────────────────────────────────────
def build_hourly_report_multi(all_states: dict, mode: str, paper_mode: bool) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🕐 <b>1시간 보고 [MULTI-AF {mode.upper()}]</b>", f"⏰ {now}\n"]

    coin_initial  = INITIAL_CAPITAL * MULTI_COIN_ALLOC   # 2,500 per coin
    total_capital = sum(s["capital"] for s in all_states.values())
    total_ret     = (total_capital / INITIAL_CAPITAL - 1) * 100
    total_trades  = sum(len(s.get("trade_log", [])) for s in all_states.values())
    total_wins    = sum(sum(1 for t in s.get("trade_log", []) if t.get("pnl", 0) > 0) for s in all_states.values())
    total_wr_str  = f" · WR {total_wins/total_trades*100:.0f}%" if total_trades else ""
    lines.append(f"💰 <b>총 자본</b>: {total_capital:,.0f} USDT  ({total_ret:+.2f}%)")
    lines.append(f"📊 총 {total_trades}건{total_wr_str}\n")

    for coin, state in all_states.items():
        pos     = state["position"]
        capital = state["capital"]
        ret     = (capital / coin_initial - 1) * 100
        peak    = state.get("peak_capital", capital)
        dd      = max(0.0, (peak - capital) / (peak + 1e-9) * 100)
        pos_str = "없음" if pos == 0 else ("LONG 🟢" if pos == 1 else "SHORT 🔴")
        trades  = state.get("trade_log", [])
        n_win   = sum(1 for t in trades if t.get("pnl", 0) > 0)
        wr      = n_win / len(trades) * 100 if trades else 0.0

        trail_str = ""
        if pos != 0:
            trail_sl  = state.get("af_trail_sl", 0)
            entry_p   = state.get("entry_price", 0)
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

    coin_initial      = INITIAL_CAPITAL * MULTI_COIN_ALLOC
    total_capital     = sum(s["capital"] for s in all_states.values())
    total_daily_start = sum(s.get("daily_start_capital", coin_initial) for s in all_states.values())
    daily_ret  = (total_capital / (total_daily_start + 1e-9) - 1) * 100
    daily_usdt = total_capital - total_daily_start
    total_ret  = (total_capital / INITIAL_CAPITAL - 1) * 100
    lines.append(
        f"💰 <b>총 자본</b>: {total_capital:,.0f} USDT\n"
        f"   전일 대비: {daily_ret:+.2f}% ({daily_usdt:+.0f} USDT) | 누적: {total_ret:+.2f}%\n"
    )

    for coin, state in all_states.items():
        capital  = state["capital"]
        d_start  = state.get("daily_start_capital", coin_initial)
        d_ret    = (capital / (d_start + 1e-9) - 1) * 100
        d_usdt   = capital - d_start
        tot_ret  = (capital / coin_initial - 1) * 100
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

    params = {}

    exchange, mode = build_exchange(trade_mode)

    try:
        tg_token, tg_chat_id = get_credentials()
    except Exception:
        tg_token, tg_chat_id = "", ""

    log.info("=" * 60)
    log.info("  ConnectAI Trade Bot 시작")
    if multi_mode:
        log.info(f"  코인:     {' / '.join(COINS_MULTI)} (멀티 4종목)")
        log.info(f"  배분:     각 {MULTI_COIN_ALLOC*100:.0f}% (2,500 USDT/종목)")
    else:
        coin = os.environ.get("COIN", "BTC").upper()
        log.info(f"  코인:     {coin}/USDT ({get_symbol()})")
    log.info(f"  전략:     {strategy}")
    log.info(f"  모드:     {trade_mode.upper()}")
    log.info("=" * 60)

    # ══════════════════════════════════════════════════════════════
    #  멀티코인 Antifragile 경로
    # ══════════════════════════════════════════════════════════════
    if multi_mode:
        seed      = float(os.getenv("PAPER_SEED", "10000"))
        coin_seed = seed * MULTI_COIN_ALLOC   # 2,500 per coin

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
                    all_states[c] = st
                except Exception:
                    all_states[c] = fresh_state()
            else:
                st = fresh_state()
                if paper_mode:
                    st["capital"]      = coin_seed
                    st["peak_capital"] = coin_seed
                    st["daily_start_capital"] = coin_seed
                    atomic_write_json(sf, st)
                    log.info(f"[PAPER-{c}] 가상 시드 초기화: {coin_seed:,.0f} USDT")
                all_states[c] = st

        total_cap = sum(s["capital"] for s in all_states.values())
        log.info(f"총 자본: {total_cap:,.0f} USDT | 종목당: {coin_seed:,.0f} USDT")

        send_trade_alert(
            f"🚀 <b>멀티코인 트레이딩 봇 시작</b> [{mode.upper()}]\n"
            f"전략: Antifragile | 종목: {' / '.join(COINS_MULTI)}\n"
            f"총 자본: {total_cap:,.0f} USDT (종목당 {coin_seed:,.0f})"
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

                for c in COINS_MULTI:
                    try:
                        all_states[c] = _run_coin_tick_af(
                            exchange, c, all_states[c], paper_mode,
                            all_paths[c]["paper_trade_csv"]
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



if __name__ == "__main__":
    main()
