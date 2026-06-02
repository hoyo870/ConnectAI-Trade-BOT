"""
라이브 트레이더 — 5분봉마다 신호 확인 후 Bybit 주문 실행
TRADE_MODE 옵션 (.env):
  real    : 실계좌 주문 실행
  sandbox : 테스트넷 주문 실행
  paper   : 실시세 조회 + 가상 주문 시뮬레이션 (주문 없음, CSV 저장)

실행:
  python src/live_trader.py
  nohup python src/live_trader.py > logs/live.log 2>&1 &
"""
import sys, os, json, time, logging, csv, copy
sys.path.insert(0, "src")

from pathlib import Path
from datetime import datetime, timezone
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
    fetch_ohlcv_df, calc_qty,
)
from telegram_notifier import send_trade_alert

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

BARS_PER_DAY   = 288
FETCH_LIMIT    = 600    # 신호 롤링(100) + 지표 워밍(200) + 여유
SIG_ROLL_WIN   = 100
INITIAL_CAPITAL = 10_000.0
MIN_QTY_BTC    = 0.001   # Bybit BTC/USDT 최소 주문 수량


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
    "entry_time":      None,
    "entry_lev":       1.0,
    "entry_rr":        0.0,
    "entry_bar":       0,
    "entry_sig_long":  0.0,     # 진입 시점 신호값 (paper 분석용)
    "entry_sig_short": 0.0,
    "current_bar":     0,
    "last_price":      0.0,
    "capital":         INITIAL_CAPITAL,
    "peak_capital":    INITIAL_CAPITAL,
    "cooling_left":    0,
    "cb_triggers":     0,
    "sig_long_hist":   [],
    "sig_short_hist":  [],
    "trade_log":       [],
}


def fresh_state() -> dict:
    return copy.deepcopy(DEFAULT_STATE)


def get_runtime_paths(paper_mode: bool) -> dict:
    if paper_mode:
        return {
            "state_file": ROOT / "logs/paper_state.json",
            "log_file": ROOT / "logs/paper.log",
            "paper_trade_csv": ROOT / "logs/paper_trades.csv",
        }
    return {
        "state_file": STATE_FILE,
        "log_file": ROOT / "logs/live.log",
        "paper_trade_csv": None,
    }


def configure_logging(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
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

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return fresh_state()

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))


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


# ── 메인 루프 한 틱 처리 ──────────────────────────────────────────────────────
def process_tick(exchange, models, scaler, device, params: dict, state: dict,
                 paper_mode: bool = False, paper_trade_csv: Optional[Path] = None) -> dict:
    df = get_signals(exchange, models, scaler, device)
    if df is None or len(df) < SIG_ROLL_WIN + 10:
        return state

    row        = df.iloc[-1]
    price      = float(row["close"])
    state["last_price"] = price
    sig_long   = float(row.get("signal_long",   0.0))
    sig_short  = float(row.get("signal_short",  0.0))
    sig_ctx    = float(row.get("signal_context", 1.0))
    now_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 롤링 신호 히스토리 업데이트
    state["sig_long_hist"].append(sig_long)
    state["sig_short_hist"].append(sig_short)
    state["sig_long_hist"]  = state["sig_long_hist"][-SIG_ROLL_WIN:]
    state["sig_short_hist"] = state["sig_short_hist"][-SIG_ROLL_WIN:]

    roll_l = float(np.mean(state["sig_long_hist"]))
    roll_s = float(np.mean(state["sig_short_hist"]))
    overconfident = ((roll_l + roll_s) / 2.0) > params["sig_upper_thr"]

    state["current_bar"] += 1
    pos      = state["position"]
    capital  = state["capital"]
    peak     = state["peak_capital"]

    # ── 미실현 equity / CB ────────────────────────────────────────────────────
    if pos != 0:
        pnl_raw = pos * (price - state["entry_price"]) / (state["entry_price"] + 1e-9)
        equity  = capital * (1 + pnl_raw * state["entry_lev"] * state["entry_rr"])
    else:
        equity = capital

    peak = max(peak, equity)
    state["peak_capital"] = peak
    dd = (peak - equity) / (peak + 1e-9)

    if dd > params["max_dd_cb"] and state["cooling_left"] == 0:
        state["cooling_left"]  = params["cooling_bars"]
        state["cb_triggers"]  += 1
        log.warning(f"[CB] 서킷브레이커 발동 #{state['cb_triggers']} | dd={dd:.2%} | equity={equity:.0f}")
        if pos != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, paper_mode=paper_mode,
                           paper_trade_csv=paper_trade_csv)
            send_trade_alert(f"⚠️ <b>서킷브레이커</b> #{state['cb_triggers']}\n"
                             f"DD={dd:.2%} → 쿨링 {params['cooling_bars']}봉")
        return state

    # ── 쿨링 중 ───────────────────────────────────────────────────────────────
    if state["cooling_left"] > 0:
        state["cooling_left"] -= 1
        if pos != 0:
            _close_and_log(exchange, state, price, now_str, forced=True, paper_mode=paper_mode,
                           paper_trade_csv=paper_trade_csv)
        return state

    # ── 청산 체크 ─────────────────────────────────────────────────────────────
    if pos != 0:
        pnl_raw  = pos * (price - state["entry_price"]) / (state["entry_price"] + 1e-9)
        pnl_lev  = pnl_raw * state["entry_lev"]
        hold_bars = state["current_bar"] - state["entry_bar"]
        sl_pnl   = params["price_sl"] * state["entry_lev"]
        tp_pnl   = params["price_tp"] * state["entry_lev"]
        tiers    = params["tiers"]

        reverse = hold_bars >= params["min_hold_bars"] and (
            (pos ==  1 and sig_short >= tiers[-1][0]) or
            (pos == -1 and sig_long  >= tiers[-1][0])
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

    # ── 신규 진입 ─────────────────────────────────────────────────────────────
    if state["position"] == 0 and sig_ctx >= params.get("context_filter_thr", 0.0):
        tiers  = params["tiers"]
        rr_cap = params["rr_cap"]
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
                # 가상 진입: 주문 없이 상태만 기록
                state["position"]        = direction
                state["entry_price"]     = price
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
                    if qty < MIN_QTY_BTC:
                        log.warning(f"[스킵] 최소수량 미달: {qty:.4f} BTC < {MIN_QTY_BTC} (자본 {capital:.2f} USDT)")
                    if qty >= MIN_QTY_BTC:
                        side = "buy" if direction == 1 else "sell"
                        place_market_order(exchange, side, qty)
                        state["position"]        = direction
                        state["entry_price"]     = price
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
                            f"수량: {qty} BTC | 자본: {capital:,.0f} USDT"
                        )
                except Exception as e:
                    log.error(f"주문 실패: {e}")

    return state


def _close_and_log(exchange, state, price, now_str, forced=False, reason="",
                   paper_mode=False, paper_trade_csv: Optional[Path] = None):
    pos = state["position"]
    if pos == 0:
        return

    if not paper_mode:
        try:
            ex_pos = get_position(exchange)
            close_position(exchange, ex_pos)
        except Exception as e:
            log.error(f"청산 실패: {e}")
            return

    pnl_raw   = pos * (price - state["entry_price"]) / (state["entry_price"] + 1e-9)
    pnl       = max(pnl_raw * state["entry_lev"] * state["entry_rr"], -state["entry_rr"])
    hold_bars = state["current_bar"] - state["entry_bar"]
    state["capital"] *= (1 + pnl)

    trade_row = {
        "time":       now_str,
        "direction":  pos,
        "entry":      state["entry_price"],
        "exit":       price,
        "pnl":        round(pnl, 4),
        "capital":    round(state["capital"], 2),
        "forced":     forced,
        "reason":     reason,
    }
    state["trade_log"].append(trade_row)

    if paper_mode:
        if paper_trade_csv is None:
            raise ValueError("paper_trade_csv is required in paper mode")
        _write_paper_trade_csv({
            "timestamp":       now_str,
            "direction":       "long" if pos == 1 else "short",
            "entry_price":     round(state["entry_price"], 2),
            "exit_price":      round(price, 2),
            "hold_bars":       hold_bars,
            "leverage":        state["entry_lev"],
            "rr":              round(state["entry_rr"], 4),
            "sig_long_entry":  round(state.get("entry_sig_long",  0.0), 4),
            "sig_short_entry": round(state.get("entry_sig_short", 0.0), 4),
            "pnl":             round(pnl, 4),
            "capital_after":   round(state["capital"], 2),
            "reason":          reason,
            "forced":          forced,
        })
        log.info(f"[PAPER 청산] {reason} | 가격={price:,.1f} | PnL={pnl:+.4f} | 자본={state['capital']:,.0f}")
    else:
        log.info(f"[청산] {reason} | 가격={price:,.1f} | PnL={pnl:+.4f} | 자본={state['capital']:,.0f}")

    state["position"]        = 0
    state["entry_price"]     = 0.0
    state["entry_lev"]       = 1.0
    state["entry_rr"]        = 0.0
    state["entry_sig_long"]  = 0.0
    state["entry_sig_short"] = 0.0


REPORT_INTERVAL = 12   # 12 × 5분 = 1시간


# ── 1시간 상태 보고 ────────────────────────────────────────────────────────────
def build_hourly_report(exchange, state: dict, mode: str, paper_mode: bool = False) -> str:
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"🕐 <b>1시간 상태 보고</b> [{mode.upper()}]", f"⏰ {now}\n"]

    # 잔고
    if paper_mode:
        lines.append("💰 <b>지갑 잔고</b>: paper 모드 미조회")
    else:
        try:
            bal = get_usdt_balance(exchange)
            lines.append(f"💰 <b>지갑 잔고</b>: {bal:,.2f} USDT")
        except Exception:
            lines.append("💰 <b>지갑 잔고</b>: 조회 실패")

    # 포지션
    pos = state["position"]
    if pos == 0:
        lines.append("📊 <b>포지션</b>: 없음 (관망)")
    else:
        dir_str    = "LONG 🟢" if pos == 1 else "SHORT 🔴"
        entry      = state["entry_price"]
        if paper_mode:
            cur_price = state.get("last_price") or entry
        else:
            try:
                ex_pos = get_position(exchange)
                cur_price = ex_pos.get("markPrice") or entry
            except Exception:
                cur_price = entry
        pnl_raw = pos * (cur_price - entry) / (entry + 1e-9)
        pnl_lev = pnl_raw * state["entry_lev"]
        hold_bars = state["current_bar"] - state["entry_bar"]
        lines.append(
            f"📊 <b>포지션</b>: {dir_str}\n"
            f"   진입가: {entry:,.1f} | 레버: {state['entry_lev']}x\n"
            f"   보유: {hold_bars}봉 ({hold_bars*5//60}h {hold_bars*5%60}m)\n"
            f"   미실현 PnL(lev): {pnl_lev:+.2%}"
        )

    # 시뮬 자본 & 수익률
    capital  = state["capital"]
    ret_pct  = (capital / INITIAL_CAPITAL - 1) * 100
    peak     = state["peak_capital"]
    dd       = (peak - capital) / (peak + 1e-9) * 100
    lines.append(
        f"\n📈 <b>수익률</b>: {ret_pct:+.2f}%"
        f"  (자본 {capital:,.0f} USDT)\n"
        f"   MDD: {dd:.2f}% | CB: {state['cb_triggers']}회"
    )

    # 거래 통계
    trades     = state.get("trade_log", [])
    n_trades   = len(trades)
    n_win      = sum(1 for t in trades if t.get("pnl", 0) > 0)
    wr         = n_win / n_trades * 100 if n_trades else 0
    cooling    = state.get("cooling_left", 0)
    lines.append(
        f"\n🔢 <b>거래 통계</b>: 총 {n_trades}건 | WR {wr:.1f}%"
        + (f"\n⏸ 쿨링 중: {cooling}봉 남음" if cooling > 0 else "")
    )

    return "\n".join(lines)


# ── 다음 5분봉까지 대기 ────────────────────────────────────────────────────────
def sleep_until_next_candle():
    now  = time.time()
    wait = 300 - (now % 300) + 5   # 5초 여유
    log.info(f"다음 캔들까지 {wait:.0f}초 대기...")
    time.sleep(wait)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    load_env_file()
    trade_mode = os.getenv("TRADE_MODE", "sandbox").lower()
    paper_mode = (trade_mode == "paper")

    paths = get_runtime_paths(paper_mode)
    configure_logging(paths["log_file"])

    params = load_params()
    device = torch.device("mps"  if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    models, scaler = load_models(device)
    exchange, mode = build_exchange(trade_mode)

    state_file = paths["state_file"]
    paper_trade_csv = paths["paper_trade_csv"]

    state = (json.loads(state_file.read_text())
             if state_file.exists() else fresh_state())

    # paper 모드: 가상 시드로 초기화 (이력 없을 때만)
    if paper_mode and len(state.get("trade_log", [])) == 0:
        seed = float(os.getenv("PAPER_SEED", "10000"))
        state["capital"]      = seed
        state["peak_capital"] = seed
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))
        log.info(f"[PAPER] 가상 시드 초기화: {seed:,.0f} USDT")

    # 실계좌: 첫 실행 시 실잔고 동기화
    if mode == "real" and len(state.get("trade_log", [])) == 0:
        try:
            real_bal = get_usdt_balance(exchange)
            if real_bal > 0:
                state["capital"]      = real_bal
                state["peak_capital"] = real_bal
                state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))
                log.info(f"[실계좌] 잔고 동기화: {real_bal:.2f} USDT")
        except Exception as e:
            log.warning(f"잔고 조회 실패, 기본값 사용: {e}")

    log.info(f"{'='*55}")
    log.info(f"라이브 트레이더 시작 | 모드: {mode.upper()} | 디바이스: {device}")
    log.info(f"rr_cap={params['rr_cap']} | dd_cb={params['max_dd_cb']} | hold={params['min_hold_bars']}")
    log.info(f"자본: {state['capital']:,.0f} USDT | 포지션: {state['position']}")
    if paper_mode:
        log.info(f"[PAPER] 거래 기록 → {paper_trade_csv}")
    log.info(f"{'='*55}")

    send_trade_alert(
        f"🚀 <b>트레이딩 봇 시작</b> [{mode.upper()}]\n"
        f"rr_cap={params['rr_cap']} | dd_cb={params['max_dd_cb']}\n"
        f"자본: {state['capital']:,.0f} USDT"
        + ("\n📄 거래 기록 저장 중" if paper_mode else "")
    )

    def _save(s):
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str))

    tick = 0
    while True:
        try:
            state = process_tick(exchange, models, scaler, device, params, state,
                                 paper_mode=paper_mode, paper_trade_csv=paper_trade_csv)
            _save(state)
            tick += 1
            if tick % REPORT_INTERVAL == 0:
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
        sleep_until_next_candle()


if __name__ == "__main__":
    main()
