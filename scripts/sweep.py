"""
scripts/sweep.py — 과적합 방지 파라미터 스윕.

⚠️ 핵심 원칙: in-sample 최댓값을 "최적"이라 부르지 않는다. 과거(TRAIN)에서 후보를 보고,
별개의 미래(TEST, 최근 약세 레짐)에서 평가한 뒤 robust 지표 + 비용 스트레스로 판정한다.
(2026 단일구간 140+ 조합 튜닝이 +212% 환상을 만든 원죄를 반복하지 않기 위함.)

스윕 knob (add_levels=0 단일진입 기준 실제 엣지 변수):
  - rsi_strict : 진입 선별 강도(δ). 롱 임계 -δ, 숏 임계 +δ (클수록 선별적 = 저빈도/고WR 기대)
  - trail_atr_init : 청산 거리(ATR 배수)
  - ml_theta : ensemble.threshold (클수록 ML 필터 엄격)

Usage:
  .venv/bin/python scripts/sweep.py --coin btc
  .venv/bin/python scripts/sweep.py --coin btc --train 2023-01-01:2025-01-01 --test 2025-01-01:2026-06-24
"""
import os, sys, argparse, itertools
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from strategies.backtest_engine import AntifragileBacktestRunner
from config.af_params import DEFAULT_PARAMS

# ── 스윕 그리드 (필요시 수정) ─────────────────────────────────────────────────
GRID = {
    "rsi_strict":      [0, 5, 10],          # 진입 선별 강도 δ
    "trail_atr_init":  [1.0, 1.5, 2.0, 2.5],
    "ml_theta":        [0.30, 0.45, 0.60],
}
_LO = ["dt_rsi_lo", "rg_rsi_lo", "ut_rsi_lo"]
_HI = ["dt_rsi_hi", "rg_rsi_hi", "ut_rsi_hi"]


def _params(rsi_strict, trail_init, leverage):
    p = {**DEFAULT_PARAMS, "leverage": leverage, "add_levels": 0,
         "trail_atr_init": trail_init}
    for k in _LO: p[k] = max(5, DEFAULT_PARAMS[k] - rsi_strict)
    for k in _HI: p[k] = min(95, DEFAULT_PARAMS[k] + rsi_strict)
    return p


def _robust_ok(m, tpd_min=0.3):
    """held-out 통과 기준: 양수 + Sharpe>0 + 아웃라이어 비의존 + 거래 충분."""
    return (m["total_return"] > 0 and m["sharpe"] > 0
            and 0 < m["pct_from_top10"] <= 100 and m["n_trades"] >= 20)


def _slice(df, dfml, lo, hi):
    a = df[(df.index >= lo) & (df.index < hi)].copy()
    b = dfml[(dfml.index >= lo) & (dfml.index < hi)].copy()
    return a, b


def main():
    ap = argparse.ArgumentParser(description="과적합 방지 파라미터 스윕")
    ap.add_argument("--coin", default="btc", choices=["btc", "eth", "sol", "xrp"])
    ap.add_argument("--train", default="2023-01-01:2025-01-01")
    ap.add_argument("--test",  default="2025-01-01:2026-06-24")
    ap.add_argument("--leverage", type=int, default=int(os.getenv("LEVERAGE", "10")))
    ap.add_argument("--model", default=str(ROOT / "models/af_ensemble/saved"))
    args = ap.parse_args()

    tr_lo, tr_hi = args.train.split(":"); te_lo, te_hi = args.test.split(":")
    runner = AntifragileBacktestRunner.from_saved(args.model)
    df, dfml = runner.load_coin(args.coin)
    tr_df, tr_ml = _slice(df, dfml, tr_lo, tr_hi)
    te_df, te_ml = _slice(df, dfml, te_lo, te_hi)
    print(f"[sweep] {args.coin.upper()} lev={args.leverage} | TRAIN {tr_lo}~{tr_hi} ({len(tr_df)}) | "
          f"TEST {te_lo}~{te_hi} ({len(te_df)}) | 조합 {np.prod([len(v) for v in GRID.values()])}개")

    rows = []
    for rs, ti, th in itertools.product(GRID["rsi_strict"], GRID["trail_atr_init"], GRID["ml_theta"]):
        p = _params(rs, ti, args.leverage)
        runner.params = p
        runner.ensemble.threshold = th
        tr = runner.run(tr_df, tr_ml)["metrics"]
        te = runner.run(te_df, te_ml)["metrics"]
        te2 = runner.run(te_df, te_ml, cost_mult=2.0)["metrics"]   # 비용 2배 스트레스
        rows.append({"rs": rs, "ti": ti, "th": th, "tr": tr, "te": te, "te2": te2})

    # ── TRAIN 수익 상위 (in-sample "최적"이 OOS에서 어떻게 되는지) ──
    rows_tr = sorted(rows, key=lambda r: r["tr"]["total_return"], reverse=True)
    print(f"\n  TRAIN 수익 상위 8 → 그 조합의 held-out TEST 성적")
    print(f"  {'δ':>2} {'trail':>5} {'θ':>5} | {'TRAIN수익':>9} | "
          f"{'TEST수익':>9} {'Shrp':>6} {'top10%':>7} {'TEST@2x':>9}  {'판정'}")
    print(f"  {'─'*78}")
    for r in rows_tr[:8]:
        te, te2 = r["te"], r["te2"]
        ok = _robust_ok(te) and te2["total_return"] > 0
        print(f"  {r['rs']:>2} {r['ti']:>5.1f} {r['th']:>5.2f} | {r['tr']['total_return']:>+8.1f}% | "
              f"{te['total_return']:>+8.1f}% {te['sharpe']:>+5.2f} {te['pct_from_top10']:>6.0f}% "
              f"{te2['total_return']:>+8.1f}%  {'✅' if ok else '❌'}")

    # ── held-out TEST 기준 최선 (진짜 일반화 후보) ──
    cand = [r for r in rows if _robust_ok(r["te"]) and r["te2"]["total_return"] > 0]
    print(f"\n  held-out TEST robust 통과(+비용2x 생존): {len(cand)}/{len(rows)} 조합")
    if cand:
        best = max(cand, key=lambda r: r["te"]["sharpe"])
        b = best["te"]
        print(f"  ★ 최선 일반화 후보: δ={best['rs']} trail={best['ti']} θ={best['th']} | "
              f"TEST {b['total_return']:+.1f}% Sharpe {b['sharpe']:+.2f} "
              f"top10기여 {b['pct_from_top10']:.0f}% | TEST@2x {best['te2']['total_return']:+.1f}%")
    else:
        print(f"  ★ 결론: held-out + 비용2x 를 통과하는 조합 없음 → 현재 신호는 강건한 엣지 없음(정직한 答).")


if __name__ == "__main__":
    main()
