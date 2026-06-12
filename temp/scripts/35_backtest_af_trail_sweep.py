"""
Antifragile intra-trail 파라미터 스윕
trail_atr_init 를 넓혀가며 intra_trail vs Baseline 비교.

현재 라이브 파라미터: trail_init=1.0 / trail_tight=1.5
이전 실험(34)에서 intra_trail(0.5/0.8) → WR 9.8%, -30% 확인 → 파라미터 재탐색

Usage:
  python temp/scripts/35_backtest_af_trail_sweep.py
  python temp/scripts/35_backtest_af_trail_sweep.py --coin eth
  python temp/scripts/35_backtest_af_trail_sweep.py --coin all --mode both
"""
import sys, argparse, random
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "temp/scripts")

import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_engine import compute_metrics
from backtest_antifragile import load_coin_full, remove_top_n, COIN_CONFIG

# 34번 스크립트 importlib으로 로드 (숫자 시작 파일명 대응)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "backtest_af_intra",
    Path(__file__).parent / "34_backtest_af_intra.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
run_af_extended = _mod.run_af_extended
add_intra_rsi   = _mod.add_intra_rsi

# 스윕 설정: (trail_atr_init, trail_atr_tight)
# tight = init × 1.5 비율 유지 (라이브 파라미터 비율 동일)
SWEEP = [
    (0.5,  0.8),   # 기존 백테스트 기본값 (참고용)
    (1.0,  1.5),   # 현재 라이브 파라미터
    (1.5,  2.0),
    (2.0,  2.5),
    (2.5,  3.0),
    (3.0,  4.0),
    (4.0,  5.0),
]

ROOT = Path(__file__).parent.parent.parent


def run_sweep_on_df(df, days_total):
    """한 구간에 대해 Baseline / intra_trail 을 모든 trail_atr 값으로 실행."""
    rows = []
    for init, tight in SWEEP:
        kwargs = dict(trail_atr_init=init, trail_atr_tight=tight)

        base = run_af_extended(df, **kwargs,
                               reverse_rsi=False, intra_trail=False, intra_rsi=False)
        intr = run_af_extended(df, **kwargs,
                               reverse_rsi=False, intra_trail=True,  intra_rsi=False)

        for label, res in [("Base", base), ("Intra", intr)]:
            m  = res["metrics"]
            tl = res["trade_log"]
            tpd = m.get("tpd", round(len(tl)/max(days_total,1), 2))
            r5  = remove_top_n(tl, 5)
            ok  = sum([m["total_return"]>0, tpd>=1.5, r5>0])
            rows.append({
                "init":  init,
                "tight": tight,
                "mode":  label,
                "n":     m["n_trades"],
                "wr":    round(m["win_rate"], 1),
                "tpd":   tpd,
                "ret":   round(m["total_return"], 1),
                "mdd":   round(m["mdd"], 1),
                "pf":    round(m.get("profit_factor", 0), 3),
                "top5":  round(r5, 1),
                "ok":    ok,
            })
    return rows


def print_sweep_table(rows, title=""):
    print(f"\n{'='*90}")
    if title:
        print(f"  {title}")
    print(f"  {'init/tight':<12} {'모드':<6} {'n':>5} {'WR':>6} {'TPD':>5} "
          f"{'수익':>8} {'MDD':>5} {'PF':>6} {'Top5':>7} {'판정':>4}")
    print(f"  {'─'*84}")

    prev_init = None
    for r in rows:
        if r["init"] != prev_init:
            if prev_init is not None:
                print(f"  {'─'*84}")
            prev_init = r["init"]

        mark = "✅" if r["ok"]==3 else ("⚠️" if r["ok"]>=2 else "❌")
        label = f"{r['init']:.1f}/{r['tight']:.1f}"
        mode_color = ">>> " if r["mode"]=="Intra" else "    "
        print(f"  {mode_color}{label:<8} {r['mode']:<6} {r['n']:>5} {r['wr']:>5.1f}% "
              f"{r['tpd']:>4.2f} {r['ret']:>+7.1f}% {r['mdd']:>4.1f}% "
              f"{r['pf']:>5.3f} {r['top5']:>+6.1f}%  {mark}({r['ok']}/3)")
    print(f"{'='*90}")

    # intra_trail 이 baseline 대비 얼마나 개선/악화됐는지 요약
    print(f"\n  intra_trail vs Baseline 비교:")
    print(f"  {'init/tight':<12} {'수익 차이':>10} {'MDD 차이':>10} {'WR 차이':>8} {'intra 판정':>10}")
    print(f"  {'─'*55}")
    base_map = {r["init"]: r for r in rows if r["mode"]=="Base"}
    intr_map = {r["init"]: r for r in rows if r["mode"]=="Intra"}
    for init in sorted(base_map):
        b = base_map[init]; i_r = intr_map.get(init)
        if not i_r:
            continue
        d_ret = i_r["ret"] - b["ret"]
        d_mdd = i_r["mdd"] - b["mdd"]
        d_wr  = i_r["wr"]  - b["wr"]
        mark = "✅" if i_r["ok"]==3 else ("⚠️" if i_r["ok"]>=2 else "❌")
        sign_ret = "+" if d_ret >= 0 else ""
        sign_mdd = "+" if d_mdd >= 0 else ""
        sign_wr  = "+" if d_wr  >= 0 else ""
        print(f"  {init:.1f}/{intr_map[init]['tight']:.1f}       "
              f"{sign_ret}{d_ret:>6.1f}%    {sign_mdd}{d_mdd:>5.1f}%    "
              f"{sign_wr}{d_wr:>4.1f}%p    {mark}({i_r['ok']}/3)")


def run_random_validation(all_df, coin_label, seed, windows, window_days, hist_start):
    all_df.dropna(subset=["_rsi", "_atr"], inplace=True)
    rng = random.Random(seed)
    possible = all_df[(all_df.index >= hist_start) &
                      (all_df.index <= all_df.index[-1] - pd.Timedelta(days=window_days))].index
    chosen = sorted(rng.choices(possible, k=windows))

    print(f"\n랜덤 {windows}창 검증 ({window_days}일 윈도우, seed={seed}) — {coin_label}")

    # trail_atr_init 별로 집계
    agg = {(init, tight): {"base": [], "intr": []} for init, tight in SWEEP}

    for i, sd in enumerate(chosen):
        ed  = sd + pd.Timedelta(days=window_days)
        seg = all_df[(all_df.index >= sd) & (all_df.index < ed)].copy()
        if len(seg) < 500:
            continue
        days = len(seg) / 288

        for init, tight in SWEEP:
            kwargs = dict(trail_atr_init=init, trail_atr_tight=tight)
            base = run_af_extended(seg, **kwargs,
                                   reverse_rsi=False, intra_trail=False, intra_rsi=False)
            intr = run_af_extended(seg, **kwargs,
                                   reverse_rsi=False, intra_trail=True, intra_rsi=False)
            for tag, res in [("base", base), ("intr", intr)]:
                m  = res["metrics"]
                tl = res["trade_log"]
                tpd = m.get("tpd", round(len(tl)/max(days,1), 2))
                r5  = remove_top_n(tl, 5)
                ok  = sum([m["total_return"]>0, tpd>=1.5, r5>0])
                agg[(init, tight)][tag].append(
                    {"ret": m["total_return"], "ok": ok, "tpd": tpd}
                )

    print(f"\n  {'init/tight':<12} {'Base avg':>9} {'Base 3/3':>9} {'Intr avg':>9} {'Intr 3/3':>9} {'차이':>8}")
    print(f"  {'─'*62}")
    for init, tight in SWEEP:
        blist = agg[(init, tight)]["base"]
        ilist = agg[(init, tight)]["intr"]
        if not blist or not ilist:
            continue
        b_avg = np.mean([x["ret"] for x in blist])
        i_avg = np.mean([x["ret"] for x in ilist])
        b_p3  = sum(x["ok"]==3 for x in blist)
        i_p3  = sum(x["ok"]==3 for x in ilist)
        diff  = i_avg - b_avg
        sign  = "+" if diff >= 0 else ""
        n     = len(blist)
        print(f"  {init:.1f}/{tight:.1f}       "
              f"{b_avg:>+7.1f}%  {b_p3:>3}/{n}     "
              f"{i_avg:>+7.1f}%  {i_p3:>3}/{n}   {sign}{diff:>+6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="intra-trail trail_atr 파라미터 스윕")
    parser.add_argument("--coin",       default="btc",
                        choices=["btc","eth","sol","xrp","both","all"])
    parser.add_argument("--mode",       default="2026", choices=["2026","random","both"])
    parser.add_argument("--windows",    type=int, default=10)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--window-days",type=int, default=91)
    args = parser.parse_args()

    coin_map = {"both": ["btc","eth"], "all": ["btc","eth","sol","xrp"]}
    coins = coin_map.get(args.coin, [args.coin])

    for coin in coins:
        cfg = COIN_CONFIG[coin]
        print(f"\n{'█'*66}")
        print(f"  {cfg['label']}/USDT — intra-trail trail_atr 스윕")
        print(f"{'█'*66}")

        all_df = load_coin_full(coin)
        all_df = add_intra_rsi(all_df)

        if args.mode in ("2026", "both"):
            seg26 = all_df[all_df.index >= "2026-01-01"].copy()
            days  = len(seg26) / 288
            print(f"\n▶ 2026 OOS ({len(seg26):,}봉 / {days:.0f}일)")
            rows = run_sweep_on_df(seg26, days)
            print_sweep_table(rows, f"{cfg['label']} 2026 OOS")

        if args.mode in ("random", "both"):
            run_random_validation(
                all_df, cfg["label"],
                seed=args.seed, windows=args.windows,
                window_days=args.window_days,
                hist_start=cfg["hist_start"],
            )


if __name__ == "__main__":
    main()
