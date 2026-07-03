"""
temp/scripts/eval_retrain.py
재학습 모델 vs 기존 모델 — 2026 OOS 백테스트 정직 비교. (ML-4)

후보 설정(δ=10 RSI 선별 + trail_atr_init=2.0 + add_levels=0, 단일진입)을 고정하고 ML 앙상블만
교체하여 2026-01~06 4코인 백테스트. 각 모델은 자기 calibrated theta(meta.json) 사용(배포 그대로),
--theta로 강제 매칭도 가능. 비용 1/2배 + robust 지표.

Usage:
  .venv/bin/python temp/scripts/eval_retrain.py \
      --old models/af_ensemble/saved --new models/af_ensemble/retrain_2025
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "src")

from strategies.backtest_engine import AntifragileBacktestRunner, robust_metrics
from config.af_params import DEFAULT_PARAMS

_LO = ["dt_rsi_lo", "rg_rsi_lo", "ut_rsi_lo"]
_HI = ["dt_rsi_hi", "rg_rsi_hi", "ut_rsi_hi"]


def candidate_params(leverage, rsi_strict=10, trail_init=2.0):
    p = {**DEFAULT_PARAMS, "leverage": leverage, "add_levels": 0, "trail_atr_init": trail_init}
    for k in _LO:
        p[k] = max(5, DEFAULT_PARAMS[k] - rsi_strict)
    for k in _HI:
        p[k] = min(95, DEFAULT_PARAMS[k] + rsi_strict)
    return p


def eval_model(model_dir, leverage, theta, coins, start, end):
    runner = AntifragileBacktestRunner.from_saved(model_dir, params=candidate_params(leverage))
    if theta is not None:
        runner.ensemble.threshold = theta
    th = runner.ensemble.threshold
    out = {}
    for coin in coins:
        df, dfml = runner.load_coin(coin, start=start, end=end)
        m1 = runner.run(df, dfml, cost_mult=1.0)["metrics"]
        res2 = runner.run(df, dfml, cost_mult=2.0)
        m2 = res2["metrics"]
        out[coin] = (m1, m2["total_return"])
    return th, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="models/af_ensemble/saved")
    ap.add_argument("--new", default="models/af_ensemble/retrain_2025")
    ap.add_argument("--leverage", type=int, default=10)
    ap.add_argument("--theta", type=float, default=None, help="강제 theta(미지정=각 모델 calibrated)")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    args = ap.parse_args()
    coins = ["btc", "eth", "sol", "xrp"]

    print(f"[eval] 2026 OOS {args.start}~{args.end} | 후보설정 δ=10 trail=2.0 lev={args.leverage}")
    rows = {}
    for tag, mdir in [("OLD", args.old), ("NEW", args.new)]:
        th, out = eval_model(mdir, args.leverage, args.theta, coins, args.start, args.end)
        rows[tag] = (th, out)
        print(f"\n  ── {tag} ({mdir}) theta={th:.3f} ──")
        print(f"  {'coin':<5} {'수익':>9} {'수익@2x':>9} {'WR':>5} {'MDD':>6} {'PF':>6} {'Sharpe':>7} {'n':>5}")
        for coin in coins:
            m1, r2 = out[coin]
            sh = m1.get("sharpe", 0.0)
            print(f"  {coin.upper():<5} {m1['total_return']:>+8.1f}% {r2:>+8.1f}% {m1['win_rate']:>4.0f}% "
                  f"{m1['mdd']:>5.1f}% {m1.get('profit_factor',0):>5.2f} {sh:>+6.2f} {m1['n_trades']:>5}")

    # 요약 비교
    print(f"\n  ── 비교 (2026 OOS 합산 수익률, cost×1) ──")
    for tag in ("OLD", "NEW"):
        th, out = rows[tag]
        tot = sum(out[c][0]["total_return"] for c in coins)
        pos = sum(out[c][0]["total_return"] > 0 for c in coins)
        print(f"  {tag}: 합산 {tot:+.1f}% | 양수 {pos}/4 | theta {th:.3f}")
    print("\n  → 판정: NEW 가 OLD 대비 OOS에서 더 나은가? (수익·Sharpe·양수코인수 종합) 수동 결론.")


if __name__ == "__main__":
    main()
