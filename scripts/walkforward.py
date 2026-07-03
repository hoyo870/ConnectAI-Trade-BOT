"""
scripts/walkforward.py — 정직한 walk-forward 검증 (튜닝/평가 시간 분리).

프리셋(파라미터)을 in-sample(과거 train 구간)에서만 선택하고, 직후 out-of-sample
(미래 test 구간)에서 평가한다. 평가 구간에서는 절대 튜닝하지 않으므로 과적합·미래정보
누출을 차단한다. 기존 단일 구간 튜닝(2026에 140+ 조합 평가)의 핵심 결함을 교정.

엔진: strategies.backtest_engine.AntifragileBacktestRunner (재사용, 신규 엔진 없음).

Usage:
  .venv/bin/python scripts/walkforward.py --coin btc
  .venv/bin/python scripts/walkforward.py --coin all --train-days 180 --test-days 90
  .venv/bin/python scripts/walkforward.py --coin btc --max-folds 3   # 빠른 검증
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

import numpy as np
import pandas as pd

from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import PRESETS, get_preset

_DEFAULT_MODEL = str(ROOT / "models/af_ensemble/saved")


def walkforward_coin(runner, coin, presets, leverage, train_days, test_days,
                     cost_mult, max_folds=0):
    """coin 전체 히스토리에 대해 walk-forward fold 리스트 반환."""
    df_all, dfml_all = runner.load_coin(coin)
    t0, tN = df_all.index[0], df_all.index[-1]
    cur = t0 + pd.Timedelta(days=train_days)
    folds = []
    while cur + pd.Timedelta(days=test_days) <= tN:
        is_lo, is_hi   = cur - pd.Timedelta(days=train_days), cur
        oos_lo, oos_hi = cur, cur + pd.Timedelta(days=test_days)
        is_df  = df_all[(df_all.index >= is_lo) & (df_all.index < is_hi)]
        is_ml  = dfml_all[(dfml_all.index >= is_lo) & (dfml_all.index < is_hi)]
        oos_df = df_all[(df_all.index >= oos_lo) & (df_all.index < oos_hi)]
        oos_ml = dfml_all[(dfml_all.index >= oos_lo) & (dfml_all.index < oos_hi)]
        if len(is_df) < 500 or len(oos_df) < 200:
            cur += pd.Timedelta(days=test_days)
            continue

        # ── in-sample에서만 프리셋 선택 (수익률 기준) ──
        best_name, best_score = None, -1e18
        for name in presets:
            runner.params = {**get_preset(name), "leverage": leverage}
            m = runner.run(is_df.copy(), is_ml.copy(), cost_mult=cost_mult)["metrics"]
            if m["total_return"] > best_score:
                best_name, best_score = name, m["total_return"]

        # ── out-of-sample 평가 (선택된 프리셋 + 비교용 prod) ──
        runner.params = {**get_preset(best_name), "leverage": leverage}
        oos = runner.run(oos_df.copy(), oos_ml.copy(), cost_mult=cost_mult)["metrics"]
        runner.params = {**get_preset("prod"), "leverage": leverage}
        oos_prod = runner.run(oos_df.copy(), oos_ml.copy(), cost_mult=cost_mult)["metrics"]

        folds.append({
            "oos_lo": oos_lo.date(), "oos_hi": oos_hi.date(),
            "preset": best_name, "oos": oos, "oos_prod": oos_prod,
        })
        if max_folds and len(folds) >= max_folds:
            break
        cur += pd.Timedelta(days=test_days)
    return folds


def _print_coin(coin, folds):
    print(f"\n{'='*78}")
    print(f"  {coin.upper()} walk-forward  (선택 프리셋 = in-sample, 평가 = out-of-sample)")
    print(f"{'='*78}")
    print(f"  {'OOS 구간':<24} {'프리셋':<12} {'수익':>9} {'WR':>6} {'MDD':>6} {'PF':>6} {'prod수익':>9}")
    print(f"  {'─'*72}")
    for f in folds:
        m, mp = f["oos"], f["oos_prod"]
        print(f"  {str(f['oos_lo'])+'~'+str(f['oos_hi']):<24} {f['preset']:<12} "
              f"{m['total_return']:>+8.1f}% {m['win_rate']:>5.1f}% {m['mdd']:>5.1f}% "
              f"{m.get('profit_factor',0):>5.2f} {mp['total_return']:>+8.1f}%")
    if not folds:
        print("  (fold 없음 — 데이터 부족)")
        return None
    rets  = [f["oos"]["total_return"] for f in folds]
    prods = [f["oos_prod"]["total_return"] for f in folds]
    pos   = sum(r > 0 for r in rets)
    print(f"  {'─'*72}")
    print(f"  OOS fold {len(folds)}개 | 양수 {pos}/{len(folds)} | "
          f"평균 {np.mean(rets):+.1f}% | 중앙값 {np.median(rets):+.1f}% | "
          f"prod평균 {np.mean(prods):+.1f}%")
    return {"n": len(folds), "pos": pos, "mean": float(np.mean(rets)),
            "median": float(np.median(rets))}


def main():
    ap = argparse.ArgumentParser(description="walk-forward 검증 (튜닝/평가 시간분리)")
    ap.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp", "all"])
    ap.add_argument("--train-days", type=int, default=180)
    ap.add_argument("--test-days",  type=int, default=90)
    ap.add_argument("--leverage",   type=int, default=int(os.getenv("LEVERAGE", "10")))
    ap.add_argument("--cost-mult",  type=float, default=1.0)
    ap.add_argument("--max-folds",  type=int, default=0, help="0=전체, N=처음 N fold만(빠른 검증)")
    ap.add_argument("--presets",    default="prod,stable,aggressive,conservative")
    ap.add_argument("--model",      default=_DEFAULT_MODEL)
    args = ap.parse_args()

    presets = [s.strip() for s in args.presets.split(",") if s.strip() in PRESETS]
    runner  = AntifragileBacktestRunner.from_saved(args.model)
    print(f"[walk-forward] train={args.train_days}d test={args.test_days}d "
          f"lev={args.leverage} cost_mult={args.cost_mult} presets={presets}")

    coins = ["btc", "eth", "sol", "xrp"] if args.coin == "all" else [args.coin]
    summary = {}
    for coin in coins:
        folds = walkforward_coin(runner, coin, presets, args.leverage,
                                 args.train_days, args.test_days, args.cost_mult,
                                 max_folds=args.max_folds)
        summary[coin] = _print_coin(coin, folds)

    print(f"\n{'#'*78}\n  종합 (out-of-sample)\n{'#'*78}")
    for coin, s in summary.items():
        if s:
            print(f"  {coin.upper():4} | fold {s['n']:2d} | 양수 {s['pos']}/{s['n']} | "
                  f"평균 {s['mean']:+.1f}% | 중앙값 {s['median']:+.1f}%")


if __name__ == "__main__":
    main()
