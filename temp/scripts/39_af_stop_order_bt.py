"""
Antifragile stop-order 시뮬레이션 백테스트
세 가지 청산 방식 비교:
  Baseline   : 봉 close로 trail_sl 체크, close 가격에 청산 (현재 라이브)
  Variant A  : 항상 high/low로 trail_sl 체크, trail_sl 가격에 청산 (거래소 stop-order 항상)
  Variant B  : 손실구간 → Baseline, 수익구간(trail_sl이 진입가 넘을 때) → high/low + trail_sl 가격 청산

Usage:
  python temp/scripts/39_af_stop_order_bt.py
  python temp/scripts/39_af_stop_order_bt.py --coin all --mode both
"""
import sys, argparse, random
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics
from backtest_antifragile import load_coin_full, remove_top_n, COIN_CONFIG

TRADING_FEE = 0.0005
SLIPPAGE    = 0.0002
FEE_TOTAL   = TRADING_FEE + SLIPPAGE

AF = {
    "dt_rsi_lo": 22, "dt_rsi_hi": 65,
    "rg_rsi_lo": 30, "rg_rsi_hi": 70,
    "ut_rsi_lo": 40, "ut_rsi_hi": 85,
    "trail_atr_init":  1.0,
    "trail_atr_tight": 1.5,
    "rr_base":    0.10,
    "rr_add":     0.15,
    "add_levels":    3,
    "atr_add_step": 0.5,
    "leverage":      5,
    "max_hold_bars": 288,
}

VARIANTS = ["baseline", "stop_order", "profit_locked"]


def run_variant(df, variant: str, initial_capital=10_000.0):
    p = AF
    capital   = initial_capital
    peak_cap  = initial_capital
    pos       = 0
    entry_price = 0.0
    current_rr  = 0.0
    add_count   = 0
    trail_sl    = 0.0
    peak_price  = 0.0
    entry_bar   = 0
    lev         = p["leverage"]

    equity_curve = [capital]
    trade_log    = []

    for idx in range(1, len(df)):
        row   = df.iloc[idx]
        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        rsi   = float(row["_rsi"])
        atr   = float(row["_atr"])
        tup   = int(row.get("_trend_up", 0))
        tdn   = int(row.get("_trend_down", 0))

        rsi_lo = p["dt_rsi_lo"] if tdn else (p["ut_rsi_lo"] if tup else p["rg_rsi_lo"])
        rsi_hi = p["dt_rsi_hi"] if tdn else (p["ut_rsi_hi"] if tup else p["rg_rsi_hi"])

        if pos != 0:
            hold = idx - entry_bar
            in_profit = (pos == 1 and trail_sl > entry_price) or \
                        (pos == -1 and trail_sl < entry_price)

            # trail_sl 업데이트
            if pos == 1:
                peak_price = max(peak_price, close)
                trail_mult = p["trail_atr_tight"] if add_count > 0 else p["trail_atr_init"]
                trail_sl   = max(trail_sl, peak_price - trail_mult * atr)
            else:
                peak_price = min(peak_price, close)
                trail_mult = p["trail_atr_tight"] if add_count > 0 else p["trail_atr_init"]
                trail_sl   = min(trail_sl, peak_price + trail_mult * atr)

            # ── 청산 체크 ─────────────────────────────────────────────
            if variant == "baseline":
                hit_stop = (pos == 1 and close <= trail_sl) or \
                           (pos == -1 and close >= trail_sl)
                exit_price = close

            elif variant == "stop_order":
                # 항상 high/low로 체크, trail_sl 가격에 청산
                hit_stop = (pos == 1 and low  <= trail_sl) or \
                           (pos == -1 and high >= trail_sl)
                exit_price = trail_sl if hit_stop else close

            else:  # profit_locked
                if in_profit:
                    # 수익 구간: high/low 체크 + trail_sl 가격 청산
                    hit_stop = (pos == 1 and low  <= trail_sl) or \
                               (pos == -1 and high >= trail_sl)
                    exit_price = trail_sl if hit_stop else close
                else:
                    # 손실 구간: baseline 동일
                    hit_stop = (pos == 1 and close <= trail_sl) or \
                               (pos == -1 and close >= trail_sl)
                    exit_price = close

            if hit_stop or hold >= p["max_hold_bars"]:
                if hold >= p["max_hold_bars"] and not hit_stop:
                    exit_price = close
                ep  = exit_price * (1 - FEE_TOTAL * pos)
                raw = pos * (ep - entry_price) / (entry_price + 1e-9)
                pnl = max(raw * lev * current_rr, -current_rr)
                capital *= (1 + pnl)
                trade_log.append({
                    "pnl": pnl, "hold_steps": hold,
                    "lev": lev, "rr": current_rr,
                    "forced": hold >= p["max_hold_bars"],
                    "direction": pos,
                    "exit_at_trail": hit_stop and variant != "baseline",
                })
                pos = 0; add_count = 0; current_rr = 0.0
            else:
                # 피라미딩
                favorable = pos * (close - entry_price) / (atr + 1e-9)
                next_lvl  = (add_count + 1) * p["atr_add_step"]
                if add_count < p["add_levels"] and favorable >= next_lvl:
                    current_rr += p["rr_add"]; add_count += 1
                    if pos == 1:
                        trail_sl = max(trail_sl, close - p["trail_atr_tight"] * atr)
                    else:
                        trail_sl = min(trail_sl, close + p["trail_atr_tight"] * atr)

        # 신규 진입
        if pos == 0:
            if rsi <= rsi_lo:
                ep = close * (1 + FEE_TOTAL)
                entry_price = ep; current_rr = p["rr_base"]; add_count = 0
                trail_sl = ep - p["trail_atr_init"] * atr; peak_price = ep
                pos = 1; entry_bar = idx
            elif rsi >= rsi_hi:
                ep = close * (1 - FEE_TOTAL)
                entry_price = ep; current_rr = p["rr_base"]; add_count = 0
                trail_sl = ep + p["trail_atr_init"] * atr; peak_price = ep
                pos = -1; entry_bar = idx

        peak_cap = max(peak_cap, capital)
        equity_curve.append(capital)

    m = compute_metrics(equity_curve, trade_log)
    if trade_log:
        days = len(df) / 288
        m["tpd"] = round(len(trade_log) / days, 2)
        m["exit_at_trail_cnt"] = sum(1 for t in trade_log if t.get("exit_at_trail"))
    else:
        m["tpd"] = 0.0
        m["exit_at_trail_cnt"] = 0
    return m, trade_log


def print_comparison(results: dict, title=""):
    print(f"\n{'='*80}")
    if title:
        print(f"  {title}")
    hdr = f"  {'Variant':<14} {'수익':>9} {'MDD':>6} {'WR':>6} {'PF':>6} {'TPD':>5} {'top5':>8} {'stop청산':>8}"
    print(hdr)
    print(f"  {'─'*72}")
    base_ret = results["baseline"][0].get("total_return", 0)
    for v in VARIANTS:
        m, tl = results[v]
        r5   = remove_top_n(tl, 5)
        diff = m["total_return"] - base_ret
        sign = "+" if diff >= 0 else ""
        mark = " ◀ 기준" if v == "baseline" else f"  ({sign}{diff:.1f}%p)"
        print(f"  {v:<14} {m['total_return']:>+8.1f}% "
              f"{m['mdd']:>5.1f}% {m['win_rate']:>5.1f}% "
              f"{m.get('profit_factor',0):>5.2f} {m['tpd']:>4.2f} "
              f"{r5:>+7.1f}%  {m['exit_at_trail_cnt']:>5}건{mark}")
    print(f"{'='*80}")


def run_hist(all_df, coin_label, seed=42, windows=10, window_days=91, hist_start=None):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng = random.Random(seed)
    possible = all_df[
        (all_df.index >= hist_start) &
        (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))
    ].index
    chosen = sorted(rng.choices(list(possible), k=windows))

    agg = {v: [] for v in VARIANTS}
    for sd in chosen:
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500:
            continue
        for v in VARIANTS:
            m, tl = run_variant(seg, v)
            r5 = remove_top_n(tl, 5)
            ok = sum([m["total_return"] > 0, m["tpd"] >= 1.5, r5 > 0])
            agg[v].append({"ret": m["total_return"], "mdd": m["mdd"], "ok": ok})

    print(f"\n  hist {windows}창 검증 ({window_days}일, seed={seed}) — {coin_label}")
    print(f"  {'Variant':<14} {'avg수익':>9} {'avgMDD':>7} {'3/3':>6} {'비고'}")
    print(f"  {'─'*48}")
    base_avg = np.mean([x["ret"] for x in agg["baseline"]]) if agg["baseline"] else 0
    for v in VARIANTS:
        lst = agg[v]
        if not lst: continue
        avg_r = np.mean([x["ret"] for x in lst])
        avg_m = np.mean([x["mdd"] for x in lst])
        p3    = sum(x["ok"] == 3 for x in lst)
        mark  = "✅" if p3 >= len(lst)*0.8 else ("⚠️" if p3 >= len(lst)*0.5 else "❌")
        diff  = f"  ({avg_r - base_avg:+.1f}%p)" if v != "baseline" else "  ◀ 기준"
        print(f"  {v:<14} {avg_r:>+8.1f}%  {avg_m:>5.1f}%  {p3:>2}/{len(lst)}  {mark}{diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",        default="btc", choices=["btc","eth","sol","xrp","all"])
    parser.add_argument("--mode",        default="both", choices=["2026","random","both"])
    parser.add_argument("--windows",     type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    args = parser.parse_args()

    coins = ["btc","eth","sol","xrp"] if args.coin == "all" else [args.coin]

    for coin in coins:
        cfg = COIN_CONFIG[coin]
        print(f"\n{'█'*60}")
        print(f"  {cfg['label']}/USDT — stop-order 시뮬레이션 비교")
        print(f"{'█'*60}")

        all_df = load_coin_full(coin)

        if args.mode in ("2026", "both"):
            seg26 = all_df[all_df.index >= "2026-01-01"].copy()
            days  = len(seg26) / 288
            print(f"\n▶ 2026 OOS ({len(seg26):,}봉 / {days:.0f}일)")
            results = {}
            for v in VARIANTS:
                results[v] = run_variant(seg26, v)
            print_comparison(results, f"{cfg['label']} 2026 OOS")

        if args.mode in ("random", "both"):
            run_hist(all_df, cfg["label"],
                     seed=args.seed, windows=args.windows,
                     window_days=args.window_days,
                     hist_start=cfg["hist_start"])


if __name__ == "__main__":
    main()
