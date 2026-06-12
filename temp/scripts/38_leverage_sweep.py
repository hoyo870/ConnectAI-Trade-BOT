"""
Antifragile 레버리지 스윕
leverage = [1, 2, 3, 5, 7, 10] 비교 (현재 라이브 = 3)

Usage:
  python temp/scripts/38_leverage_sweep.py
  python temp/scripts/38_leverage_sweep.py --coin all --mode both
"""
import sys, argparse, random
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics
from backtest_antifragile import load_coin_full, remove_top_n, COIN_CONFIG, run_antifragile

LEVERAGES = [1, 2, 3, 5, 7, 10]
CURRENT_LEV = 3


def run_lev_sweep(df, days_total):
    rows = []
    for lev in LEVERAGES:
        res = run_antifragile(df, leverage=lev)
        m   = res["metrics"]
        tl  = res["trade_log"]
        tpd = m.get("tpd", round(len(tl) / max(days_total, 1), 2))
        r5  = remove_top_n(tl, 5)
        ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
        rows.append({
            "lev":  lev,
            "n":    m["n_trades"],
            "wr":   round(m["win_rate"], 1),
            "tpd":  tpd,
            "ret":  round(m["total_return"], 1),
            "mdd":  round(m["mdd"], 1),
            "pf":   round(m.get("profit_factor", 0), 3),
            "top5": round(r5, 1),
            "ok":   ok,
        })
    return rows


def print_table(rows, title=""):
    print(f"\n{'='*80}")
    if title:
        print(f"  {title}")
    print(f"  {'레버리지':>8} {'n':>6} {'WR':>6} {'TPD':>5} {'수익':>10} {'MDD':>6} {'PF':>6} {'Top5':>8} {'판정':>4}")
    print(f"  {'─'*74}")
    cur = next(r for r in rows if r["lev"] == CURRENT_LEV)
    for r in rows:
        mark = "✅" if r["ok"] == 3 else ("⚠️" if r["ok"] >= 2 else "❌")
        flag = " ◀ 현재" if r["lev"] == CURRENT_LEV else ""
        d_ret = r["ret"] - cur["ret"]
        d_mdd = r["mdd"] - cur["mdd"]
        sign  = "+" if d_ret >= 0 else ""
        diff  = f"  ({sign}{d_ret:.0f}% ret, {'+' if d_mdd>=0 else ''}{d_mdd:.1f}% mdd)" if r["lev"] != CURRENT_LEV else ""
        print(f"  {r['lev']:>7}x  {r['n']:>6} {r['wr']:>5.1f}% "
              f"{r['tpd']:>4.2f} {r['ret']:>+9.1f}% {r['mdd']:>5.1f}% "
              f"{r['pf']:>5.3f} {r['top5']:>+7.1f}%  {mark}{flag}{diff}")
    print(f"{'='*80}")

    # 레버리지별 Sharpe-like ratio
    print(f"\n  수익/MDD 비율 (레버리지별 효율):")
    print(f"  {'레버리지':>8} {'수익/MDD':>10} {'수익×WR/MDD':>14}")
    print(f"  {'─'*38}")
    for r in rows:
        ratio = r["ret"] / max(r["mdd"], 0.1)
        adj   = r["ret"] * r["wr"] / 100 / max(r["mdd"], 0.1)
        flag  = " ◀ 현재" if r["lev"] == CURRENT_LEV else ""
        print(f"  {r['lev']:>7}x  {ratio:>9.1f}x  {adj:>13.1f}x{flag}")


def run_hist_validation(all_df, coin_label, seed, windows, window_days, hist_start):
    all_df = all_df.dropna(subset=["_rsi", "_atr"])
    rng = random.Random(seed)
    possible = all_df[
        (all_df.index >= hist_start) &
        (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))
    ].index
    chosen = sorted(rng.choices(list(possible), k=windows))

    print(f"\n랜덤 {windows}창 검증 ({window_days}일 윈도우, seed={seed}) — {coin_label}")
    agg = {lev: [] for lev in LEVERAGES}

    for sd in chosen:
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500:
            continue
        days = len(seg) / 288
        for lev in LEVERAGES:
            res = run_antifragile(seg, leverage=lev)
            m   = res["metrics"]
            tl  = res["trade_log"]
            tpd = m.get("tpd", round(len(tl) / max(days, 1), 2))
            r5  = remove_top_n(tl, 5)
            ok  = sum([m["total_return"] > 0, tpd >= 1.5, r5 > 0])
            agg[lev].append({"ret": m["total_return"], "mdd": m["mdd"], "ok": ok})

    cur_avg = np.mean([x["ret"] for x in agg[CURRENT_LEV]]) if agg[CURRENT_LEV] else 0
    print(f"\n  {'레버리지':>8} {'avg수익':>10} {'avg MDD':>8} {'3/3':>6} {'ret/MDD':>9} {'비고'}")
    print(f"  {'─'*55}")
    for lev in LEVERAGES:
        lst = agg[lev]
        if not lst:
            continue
        avg_r   = np.mean([x["ret"] for x in lst])
        avg_m   = np.mean([x["mdd"] for x in lst])
        p3      = sum(x["ok"] == 3 for x in lst)
        ratio   = avg_r / max(avg_m, 0.1)
        diff    = f"  ({avg_r - cur_avg:+.1f}%)" if lev != CURRENT_LEV else ""
        mark    = "✅" if p3 >= len(lst)*0.8 else ("⚠️" if p3 >= len(lst)*0.5 else "❌")
        flag    = " ◀ 현재" if lev == CURRENT_LEV else ""
        print(f"  {lev:>7}x  {avg_r:>+8.1f}%  {avg_m:>6.1f}%  {p3:>2}/{len(lst)}  "
              f"{ratio:>7.1f}x  {mark}{diff}{flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin",        default="btc", choices=["btc","eth","sol","xrp","both","all"])
    parser.add_argument("--mode",        default="both", choices=["2026","random","both"])
    parser.add_argument("--windows",     type=int, default=10)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--window-days", type=int, default=91)
    args = parser.parse_args()

    coin_map = {"both": ["btc","eth"], "all": ["btc","eth","sol","xrp"]}
    coins = coin_map.get(args.coin, [args.coin])

    for coin in coins:
        cfg = COIN_CONFIG[coin]
        print(f"\n{'█'*60}")
        print(f"  {cfg['label']}/USDT — 레버리지 스윕")
        print(f"{'█'*60}")

        all_df = load_coin_full(coin)

        if args.mode in ("2026", "both"):
            seg26 = all_df[all_df.index >= "2026-01-01"].copy()
            days  = len(seg26) / 288
            print(f"\n▶ 2026 OOS ({len(seg26):,}봉 / {days:.0f}일)")
            rows = run_lev_sweep(seg26, days)
            print_table(rows, f"{cfg['label']} 2026 OOS — 레버리지 스윕")

        if args.mode in ("random", "both"):
            run_hist_validation(
                all_df, cfg["label"],
                seed=args.seed, windows=args.windows,
                window_days=args.window_days,
                hist_start=cfg["hist_start"],
            )


if __name__ == "__main__":
    main()
