"""
Top-N 거래 제거 후 수익률 분석: +180%가 outlier-driven인지 검증
- 복리 재계산 방식 (단순 빼기 X)
- BTC 2026 백테스트 기준
"""
import sys, json
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path
from hybrid_engine import run_v17_backtest

ROOT = Path(__file__).parent.parent


def remove_top_n(trades: list, n: int, base_capital: float = 10_000.0) -> dict:
    """상위 n개 수익 거래 제거 후 복리 수익률 재계산."""
    top_n_idx = set(sorted(range(len(trades)),
                           key=lambda i: trades[i]["pnl"], reverse=True)[:n])
    capital = base_capital
    kept = 0
    wins = 0
    for i, t in enumerate(trades):
        if i in top_n_idx:
            continue
        capital *= 1 + t["pnl"]
        kept += 1
        if t["pnl"] > 0:
            wins += 1

    ret = (capital / base_capital - 1) * 100
    wr  = wins / kept * 100 if kept else 0
    top_pnl = sum(trades[i]["pnl"] for i in top_n_idx)
    return {"n": n, "ret": ret, "kept": kept, "wr": wr,
            "top_n_pnl_sum": round(top_pnl, 4)}


def main():
    btc_params_path = ROOT / "models/production/btc_params.json"
    btc_params = json.loads(btc_params_path.read_text())
    btc_params["tiers"] = [tuple(t) for t in btc_params["tiers"]]

    df = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")

    result = run_v17_backtest(df, **{k: v for k, v in btc_params.items()
                                     if k not in ("version", "rr_scale", "phase",
                                                  "notes", "metrics_2026", "metrics_hist")})
    trades    = result["trade_log"]
    metrics   = result["metrics"]
    base_ret  = metrics["total_return"]
    n_trades  = metrics["n_trades"]

    print(f"\n{'='*60}")
    print(f"  BTC 2026 Top-N 거래 제거 분석")
    print(f"{'='*60}")
    print(f"  전체 거래수:    {n_trades}")
    print(f"  기본 수익률:    {base_ret:+.2f}%")
    print(f"  기본 WR:        {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor:  {metrics.get('profit_factor', 'N/A')}")
    print(f"\n  {'제거':<8} {'수익률':>10} {'변화':>10} {'유지거래':>8} {'WR':>8} {'판정'}")
    print(f"  {'-'*56}")

    for n in [1, 3, 5, 10]:
        r = remove_top_n(trades, n)
        verdict = "✅ edge 있음" if r["ret"] > 0 else "❌ outlier 의존"
        delta = r["ret"] - base_ret
        print(f"  Top-{n:<4} {r['ret']:>+9.2f}% {delta:>+9.2f}%p"
              f"  {r['kept']:>6}건  {r['wr']:>6.1f}%  {verdict}")

    # 거래별 수익 분포
    pnls = sorted([t["pnl"] for t in trades], reverse=True)
    top5_sum  = sum(pnls[:5])
    total_pos = sum(p for p in pnls if p > 0)
    top5_share = top5_sum / total_pos * 100 if total_pos > 0 else 0

    print(f"\n  상위 5거래 기여도: 전체 수익의 {top5_share:.1f}%")
    print(f"  상위 5거래 pnl 합: {top5_sum:+.4f}")
    print(f"  최대 단일 거래:    {pnls[0]:+.4f}")
    print(f"  중간값(pnl):       {np.median(pnls):+.4f}")

    print(f"\n{'='*60}")
    if remove_top_n(trades, 5)["ret"] > 0:
        print("  결론: ✅ Top-5 제거 후에도 양수 — 전략에 실질적 edge 존재")
    else:
        print("  결론: ❌ Top-5 제거 시 음수 — outlier 의존, 임계값 상향 검토 필요")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
