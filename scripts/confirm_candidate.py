"""
scripts/confirm_candidate.py — 후보 설정 고정 다중 fold 확인 (3x vs 10x).

스윕에서 4코인 공통으로 떠오른 후보(δ=10 선별강화 + trail_atr_init=2.0 + ML θ=0.45,
add_levels=0 단일진입)를 *고정*하고, 여러 disjoint 시간 fold에서 평가한다. fold마다 재튜닝
하지 않으므로 누출 없음 — 시간 강건성(특히 최근 fold)과 레버리지별 MDD(생존성)를 본다.

Usage:
  .venv/bin/python scripts/confirm_candidate.py --coin all
  .venv/bin/python scripts/confirm_candidate.py --coin btc --fold-days 120
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import DEFAULT_PARAMS

# 후보 설정 (스윕 4코인 공통 최선 영역)
CAND_RSI_STRICT = 10
CAND_TRAIL_INIT = 2.0
CAND_ML_THETA   = 0.45
LEVS            = [3, 10]
_LO = ["dt_rsi_lo", "rg_rsi_lo", "ut_rsi_lo"]
_HI = ["dt_rsi_hi", "rg_rsi_hi", "ut_rsi_hi"]


def candidate_params(leverage):
    p = {**DEFAULT_PARAMS, "leverage": leverage, "add_levels": 0,
         "trail_atr_init": CAND_TRAIL_INIT}
    for k in _LO: p[k] = max(5, DEFAULT_PARAMS[k] - CAND_RSI_STRICT)
    for k in _HI: p[k] = min(95, DEFAULT_PARAMS[k] + CAND_RSI_STRICT)
    return p


def confirm_coin(runner, coin, fold_days, hist_start):
    df, dfml = runner.load_coin(coin)
    runner.ensemble.threshold = CAND_ML_THETA
    t0 = max(pd.Timestamp(hist_start), df.index[0])
    tN = df.index[-1]
    folds = []
    cur = t0
    while cur + pd.Timedelta(days=fold_days) <= tN:
        lo, hi = cur, cur + pd.Timedelta(days=fold_days)
        seg   = df[(df.index >= lo) & (df.index < hi)].copy()
        segml = dfml[(dfml.index >= lo) & (dfml.index < hi)].copy()
        if len(seg) < 300:
            cur += pd.Timedelta(days=fold_days); continue
        res = {}
        for lev in LEVS:
            runner.params = candidate_params(lev)
            res[lev] = runner.run(seg, segml)["metrics"]
        folds.append((lo.date(), hi.date(), res))
        cur += pd.Timedelta(days=fold_days)
    return folds


def _print_coin(coin, folds):
    print(f"\n{'='*82}")
    print(f"  {coin.upper()}  후보설정 고정 (δ={CAND_RSI_STRICT} trail={CAND_TRAIL_INIT} θ={CAND_ML_THETA})")
    print(f"{'='*82}")
    print(f"  {'fold':<22} | {'ret@3x':>8} {'MDD@3x':>7} | {'ret@10x':>9} {'MDD@10x':>8} | {'WR':>5}")
    print(f"  {'─'*76}")
    for lo, hi, res in folds:
        m3, m10 = res[3], res[10]
        print(f"  {str(lo)+'~'+str(hi):<22} | {m3['total_return']:>+7.1f}% {m3['mdd']:>6.1f}% | "
              f"{m10['total_return']:>+8.1f}% {m10['mdd']:>7.1f}% | {m3['win_rate']:>4.1f}%")
    if not folds:
        return None
    r3  = [res[3]['total_return'] for _, _, res in folds]
    r10 = [res[10]['total_return'] for _, _, res in folds]
    d3  = [res[3]['mdd'] for _, _, res in folds]
    d10 = [res[10]['mdd'] for _, _, res in folds]
    print(f"  {'─'*76}")
    print(f"  fold {len(folds)} | 3x: 양수 {sum(r>0 for r in r3)}/{len(r3)} "
          f"중앙값 {np.median(r3):+.1f}% 최악MDD {max(d3):.1f}% | "
          f"10x: 양수 {sum(r>0 for r in r10)}/{len(r10)} 중앙값 {np.median(r10):+.1f}% 최악MDD {max(d10):.1f}%")
    return {"r3": r3, "r10": r10, "d3": d3, "d10": d10}


def main():
    ap = argparse.ArgumentParser(description="후보설정 다중 fold 확인 (3x vs 10x)")
    ap.add_argument("--coin", default="all", choices=["btc", "eth", "sol", "xrp", "all"])
    ap.add_argument("--fold-days", type=int, default=120)
    ap.add_argument("--hist-start", default="2022-01-01")
    ap.add_argument("--model", default=str(ROOT / "models/af_ensemble/saved"))
    args = ap.parse_args()

    runner = AntifragileBacktestRunner.from_saved(args.model)
    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]
    print(f"[confirm] 후보 δ={CAND_RSI_STRICT} trail={CAND_TRAIL_INIT} θ={CAND_ML_THETA} | "
          f"fold={args.fold_days}d from {args.hist_start} | lev {LEVS}")
    agg = {}
    for coin in coins:
        agg[coin] = _print_coin(coin, confirm_coin(runner, coin, args.fold_days, args.hist_start))

    print(f"\n{'#'*82}\n  종합 (후보설정 다중 fold)\n{'#'*82}")
    for coin, s in agg.items():
        if not s: continue
        print(f"  {coin.upper():4} | 3x 양수 {sum(r>0 for r in s['r3'])}/{len(s['r3'])} "
              f"최악MDD {max(s['d3']):.0f}% | 10x 양수 {sum(r>0 for r in s['r10'])}/{len(s['r10'])} "
              f"최악MDD {max(s['d10']):.0f}%")


if __name__ == "__main__":
    main()
