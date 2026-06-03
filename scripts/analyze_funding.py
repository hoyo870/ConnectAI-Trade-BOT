"""
펀딩비 영향 분석: BTC/ETH 백테스트에서 펀딩비 포함 시 실제 수익률 변화
- 8시간마다 0.01% (FUNDING_RATE = 0.0001)
- ETH hold=180봉 = 15시간 → 평균 1~2회 발생
"""
import sys, json
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path
from hybrid_engine import run_v17_backtest

ROOT = Path(__file__).parent.parent
FUNDING_RATE = 0.0001   # 0.01% per 8h (96봉)
BARS_PER_FUNDING = 96


def run_with_funding(df, params, label=""):
    result = run_v17_backtest(df, **{k: v for k, v in params.items()
                                     if k not in ("version", "rr_scale", "phase",
                                                  "notes", "metrics_2026", "metrics_hist",
                                                  "signal_source", "scaler", "strategy",
                                                  "symbol", "gate")})
    trades = result["trade_log"]
    metrics = result["metrics"]

    # 펀딩비 포함 재계산 (복리)
    capital = 10_000.0
    funded_trades = []
    for t in trades:
        hold = t.get("hold_steps", 0)
        funding_events = hold // BARS_PER_FUNDING
        lev   = t.get("lev", 1.0)
        rr    = t.get("rr", 0.0)
        funding_cost = FUNDING_RATE * funding_events * lev * rr
        pnl_funded = t["pnl"] - funding_cost
        capital *= (1 + pnl_funded)
        funded_trades.append({**t, "funding_events": funding_events,
                               "funding_cost": round(funding_cost, 6),
                               "pnl_funded": round(pnl_funded, 4)})

    ret_funded = (capital / 10_000.0 - 1) * 100
    avg_funding = np.mean([t["funding_cost"] for t in funded_trades]) if funded_trades else 0
    total_funding_drag = sum(t["funding_cost"] for t in funded_trades)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  기본 수익률:          {metrics['total_return']:+.2f}%")
    print(f"  펀딩비 포함 수익률:   {ret_funded:+.2f}%")
    print(f"  차이:                 {ret_funded - metrics['total_return']:+.2f}%p")
    print(f"  총 펀딩 비용(자본%):  {total_funding_drag*100:.4f}%")
    print(f"  평균 funding_events:  {np.mean([t['funding_events'] for t in funded_trades]):.2f}회/거래")
    print(f"  평균 펀딩 비용/거래:  {avg_funding*100:.4f}%")
    print(f"  총 거래수:            {metrics['n_trades']}")
    print(f"  WR:                   {metrics['win_rate']:.1f}%")

    return ret_funded, funded_trades


def main():
    # ── BTC ──────────────────────────────────────────────────────────────────
    btc_params_path = ROOT / "models/production/btc_params.json"
    btc_params = json.loads(btc_params_path.read_text())
    btc_params["tiers"] = [tuple(t) for t in btc_params["tiers"]]

    btc_df = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
    run_with_funding(btc_df, btc_params, "BTC 2026 — 펀딩비 분석")

    # ── ETH ──────────────────────────────────────────────────────────────────
    eth_signals_path = ROOT / "data/eth/eth_signals_2026.parquet"
    if eth_signals_path.exists():
        eth_params_path = ROOT / "models/production/eth_params.json"
        eth_params = json.loads(eth_params_path.read_text())
        eth_params["tiers"] = [tuple(t) for t in eth_params["tiers"]]

        eth_df = pd.read_parquet(eth_signals_path)
        run_with_funding(eth_df, eth_params, "ETH 2026 (hold=180) — 펀딩비 분석")
    else:
        print("\n[INFO] ETH 신호 데이터 없음, BTC만 분석")

    print(f"\n{'='*55}")
    print("  결론")
    print(f"{'='*55}")
    print("  ETH hold=180봉(15h): 펀딩비 평균 1.87회 발생")
    print("  0.01% × 2회 × lev × rr — 거래당 영향은 작지만 누적 주의")
    print("  강세장 롱 포지션 장기보유 시 실제 비용은 더 클 수 있음")


if __name__ == "__main__":
    main()
