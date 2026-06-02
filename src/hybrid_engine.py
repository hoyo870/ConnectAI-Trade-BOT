"""
v17: rr_cap 파라미터 추가 (backtest_base 기반)
  핵심 변경: entry_rr = min(rr, rr_cap)  — 기본값 0.75로 상향
  목적: rr 상향으로 per-trade 수익 확대 → ret=150% 달성
  WR은 가격 기반 TP/SL로 결정되므로 rr 변경과 무관
"""
import numpy as np
import pandas as pd

TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002


def _fee(price: float) -> float:
    return price * (TRADING_FEE + SLIPPAGE)


def _get_tier(sig: float, tiers: list) -> tuple:
    for t in tiers:
        if sig >= t[0]:
            return t[1], t[2]
    return 0.0, 0.0


def compute_metrics(
    equity_curve: list,
    trade_log: list,
    bars_per_year: int = 105_120,
) -> dict:
    """equity_curve 와 trade_log 로 공통 성과 지표 계산."""
    eq   = np.array(equity_curve, dtype=float)
    rets = np.diff(eq) / (eq[:-1] + 1e-9)

    total_return = (eq[-1] - eq[0]) / eq[0]
    years        = len(eq) / bars_per_year
    cagr         = (eq[-1] / eq[0]) ** (1 / max(years, 1e-9)) - 1

    peak      = np.maximum.accumulate(eq)
    drawdown  = (peak - eq) / (peak + 1e-9)
    mdd       = float(drawdown.max())

    ann_factor = np.sqrt(bars_per_year)
    sharpe     = float(rets.mean() / (rets.std() + 1e-9) * ann_factor)
    neg_rets   = rets[rets < 0]
    sortino    = float(rets.mean() / (neg_rets.std() + 1e-9) * ann_factor)

    pnls          = [t["pnl"] for t in trade_log] if trade_log else []
    n_trades      = len(pnls)
    win_rate      = float(np.mean([p > 0 for p in pnls])) if pnls else 0.0
    avg_win       = float(np.mean([p for p in pnls if p > 0])) if any(p > 0 for p in pnls) else 0.0
    avg_loss      = float(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0.0
    profit_factor = abs(avg_win / (avg_loss + 1e-9))
    avg_hold      = float(np.mean([t.get("hold_steps", 0) for t in trade_log])) if trade_log else 0.0

    return {
        "total_return":  round(total_return * 100, 2),
        "cagr":          round(cagr * 100, 2),
        "mdd":           round(mdd * 100, 2),
        "sharpe":        round(sharpe, 3),
        "sortino":       round(sortino, 3),
        "n_trades":      n_trades,
        "win_rate":      round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_hold_bars": round(avg_hold, 1),
    }


DEFAULT_V17_TIERS = [
    (0.72, 10.0, 0.50),
    (0.62,  4.0, 0.40),
    (0.55,  3.5, 0.30),
    (0.48,  2.5, 0.25),
]


def run_v17_backtest(
    df: pd.DataFrame,
    initial_capital: float    = 10_000.0,
    tiers: list               = None,
    price_sl: float           = 0.030,
    price_tp: float           = 0.060,
    min_hold_bars: int        = 120,
    context_filter_thr: float = 0.0,
    sig_roll_window: int      = 100,
    sig_upper_thr: float      = 0.65,
    max_dd_cb: float          = 0.35,
    cooling_bars: int         = 200,
    rr_cap: float             = 0.75,
    atr_filter_thr: float     = None,
    gate_long_col: str        = None,
    gate_short_col: str       = None,
) -> dict:
    if tiers is None:
        tiers = DEFAULT_V17_TIERS

    rows = df.reset_index(drop=True)

    sig_long_roll  = rows["signal_long"].rolling(sig_roll_window).mean()
    sig_short_roll = rows["signal_short"].rolling(sig_roll_window).mean()

    capital      = initial_capital
    position     = 0
    entry_price  = 0.0
    entry_step   = 0
    entry_lev    = 1.0
    entry_rr     = 0.2
    peak_capital = initial_capital
    cooling_left = 0
    cb_triggers  = 0

    equity_curve = [capital]
    trade_log    = []

    for idx in range(len(rows)):
        row       = rows.iloc[idx]
        price     = float(row["close"])
        sig_long  = float(row.get("signal_long",  0.0))
        sig_short = float(row.get("signal_short", 0.0))
        sig_ctx   = float(row.get("signal_context", 1.0))

        roll_l = sig_long_roll.iloc[idx]
        roll_s = sig_short_roll.iloc[idx]
        roll_l = roll_l if not np.isnan(roll_l) else sig_long
        roll_s = roll_s if not np.isnan(roll_s) else sig_short
        overconfident = ((roll_l + roll_s) / 2.0) > sig_upper_thr

        # ── 미실현 equity & CB ────────────────────────────────────────────────
        if position != 0:
            pnl_raw = position * (price - entry_price) / (entry_price + 1e-9)
            equity  = capital * (1 + pnl_raw * entry_lev * entry_rr)
        else:
            equity  = capital
        peak_capital = max(peak_capital, equity)
        dd = (peak_capital - equity) / (peak_capital + 1e-9)

        if dd > max_dd_cb and cooling_left == 0:
            cooling_left = cooling_bars
            cb_triggers += 1

        if cooling_left > 0:
            cooling_left -= 1
            if position != 0:
                cp  = price - _fee(price) * position
                raw = position * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * entry_lev * entry_rr, -entry_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": idx - entry_step,
                                   "lev": entry_lev, "rr": entry_rr, "forced": True,
                                   "direction": position})
                position = 0; entry_price = 0.0
            equity_curve.append(capital)
            continue

        # ── 포지션 청산 ───────────────────────────────────────────────────────
        if position != 0:
            pnl_raw   = position * (price - entry_price) / (entry_price + 1e-9)
            pnl_lev   = pnl_raw * entry_lev
            hold_bars = idx - entry_step
            sl_pnl    = price_sl * entry_lev
            tp_pnl    = price_tp * entry_lev

            reverse = hold_bars >= min_hold_bars and (
                (position ==  1 and sig_short >= tiers[-1][0]) or
                (position == -1 and sig_long  >= tiers[-1][0])
            )
            if pnl_lev <= -0.9 or pnl_lev <= -sl_pnl or pnl_lev >= tp_pnl or reverse:
                cp  = price - _fee(price) * position
                raw = position * (cp - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * entry_lev * entry_rr, -entry_rr)
                capital *= (1 + pnl)
                trade_log.append({"pnl": pnl, "hold_steps": hold_bars,
                                   "lev": entry_lev, "rr": entry_rr, "forced": False,
                                   "direction": position})
                position = 0; entry_price = 0.0

        # ── 신규 진입 ─────────────────────────────────────────────────────────
        atr_pct = float(row.get("atr_pct", 0.0))
        atr_blocked = (atr_filter_thr is not None) and (atr_pct > atr_filter_thr)
        gate_long_ok  = (gate_long_col  is None) or (float(row.get(gate_long_col,  0.0)) > 0)
        gate_short_ok = (gate_short_col is None) or (float(row.get(gate_short_col, 0.0)) > 0)
        if position == 0 and sig_ctx >= context_filter_thr and not atr_blocked:
            lev, rr = _get_tier(sig_long, tiers) if gate_long_ok else (0.0, 0.0)
            if lev > 0:
                if overconfident:
                    ti = next((i for i,t in enumerate(tiers) if sig_long >= t[0]), None)
                    if ti is not None and ti + 1 < len(tiers):
                        _, lev, rr = tiers[ti + 1]
                    else:
                        lev, rr = lev * 0.5, rr * 0.8
                entry_price = price + _fee(price)
                position    = 1
                entry_lev   = lev
                entry_rr    = min(rr, rr_cap)
                entry_step  = idx
            else:
                lev, rr = _get_tier(sig_short, tiers) if gate_short_ok else (0.0, 0.0)
                if lev > 0:
                    if overconfident:
                        ti = next((i for i,t in enumerate(tiers) if sig_short >= t[0]), None)
                        if ti is not None and ti + 1 < len(tiers):
                            _, lev, rr = tiers[ti + 1]
                        else:
                            lev, rr = lev * 0.5, rr * 0.8
                    entry_price = price - _fee(price)
                    position    = -1
                    entry_lev   = lev
                    entry_rr    = min(rr, rr_cap)
                    entry_step  = idx

        equity_curve.append(capital)

    metrics = compute_metrics(equity_curve, trade_log)
    metrics["cb_triggers"] = cb_triggers

    if trade_log:
        lev_used = [t["lev"] for t in trade_log]
        metrics["avg_lev"]        = round(float(np.mean(lev_used)), 2)
        metrics["max_lev"]        = round(float(np.max(lev_used)), 1)
        total_bars                = len(rows)
        metrics["trades_per_day"] = round(len(trade_log) / (total_bars / 288), 2)
        metrics["long_cnt"]  = sum(1 for t in trade_log if t.get("direction", 1) == 1)
        metrics["short_cnt"] = sum(1 for t in trade_log if t.get("direction", 1) == -1)
    return {"metrics": metrics, "equity_curve": equity_curve, "trade_log": trade_log}
