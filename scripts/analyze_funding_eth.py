"""
ETH 2026 펀딩비 정확 분석 — 하이브리드 게이트 포함
analyze_funding.py의 ETH 결과는 게이트 미포함으로 무효 (-77.3% 잘못된 수치)
이 스크립트: raw OHLCV → RSI/vol_ratio(비스케일) → gate 재계산 → 펀딩비 반영
"""
import sys, json
sys.path.insert(0, "src")

import pandas as pd
import numpy as np
from pathlib import Path
from hybrid_engine import run_v17_backtest

ROOT = Path(__file__).parent.parent
FUNDING_RATE      = 0.0001   # 0.01% per 8h
BARS_PER_FUNDING  = 96       # 8h = 96 × 5min


def _compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, min_periods=period).mean()
    al = loss.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / (al + 1e-9)))


def build_gate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """raw OHLCV(unscaled)에서 gate_long / gate_short 계산."""
    close     = df_raw["close"]
    rsi       = _compute_rsi(close)
    vol_ema20 = df_raw["volume"].ewm(span=20, adjust=False).mean()
    vol_ratio = df_raw["volume"] / (vol_ema20 + 1e-9)

    ema_f = close.ewm(span=9,  adjust=False).mean()
    ema_s = close.ewm(span=21, adjust=False).mean()

    cl1h   = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    tup    = (cl1h > ema_1h).reindex(df_raw.index, method="ffill").fillna(False)

    gate_long  = ((ema_f > ema_s) & (rsi >= 50) & (rsi <= 60) & (vol_ratio >= 0.8) & tup).astype(int)
    gate_short = ((ema_f < ema_s) & (rsi >= 30) & (rsi <= 50) & (vol_ratio >= 0.8) & ~tup).astype(int)
    return pd.DataFrame({"gate_long": gate_long, "gate_short": gate_short}, index=df_raw.index)


def apply_funding(trades: list, base_ret: float, label: str) -> float:
    capital = 10_000.0
    for t in trades:
        events = t.get("hold_steps", 0) // BARS_PER_FUNDING
        cost   = FUNDING_RATE * events * t.get("lev", 1.0) * t.get("rr", 0.0)
        capital *= (1 + t["pnl"] - cost)

    ret_f       = (capital / 10_000.0 - 1) * 100
    total_drag  = sum(
        FUNDING_RATE * (t.get("hold_steps", 0) // BARS_PER_FUNDING) * t.get("lev", 1.0) * t.get("rr", 0.0)
        for t in trades
    )
    avg_events  = np.mean([t.get("hold_steps", 0) // BARS_PER_FUNDING for t in trades]) if trades else 0

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  기본 수익률:          {base_ret:+.2f}%")
    print(f"  펀딩비 포함 수익률:   {ret_f:+.2f}%")
    print(f"  차이:                 {ret_f - base_ret:+.2f}%p")
    print(f"  총 펀딩 비용 합계:    {total_drag*100:.4f}%")
    print(f"  평균 funding_events:  {avg_events:.2f}회/거래")
    print(f"  총 거래수:            {len(trades)}")
    return ret_f


def main():
    # 1. raw OHLCV 로드 (gate 계산용 — 비스케일 값 필요)
    hist  = pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_history.parquet")
    d2026 = pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet")
    for d in [hist, d2026]:
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        else:
            d.index = d.index.tz_convert("UTC")
    raw_all = pd.concat([hist, d2026]).sort_index()
    raw_all = raw_all[~raw_all.index.duplicated(keep="last")]

    # 2. gate 계산 (EMA/RSI/vol_ratio 모두 비스케일 원본)
    print("Gate 계산 중...")
    gate_df = build_gate(raw_all)

    # 3. ETH 신호 로드
    sig_df = pd.read_parquet(ROOT / "data/eth/eth_signals_2026.parquet")
    if sig_df.index.tz is None:
        sig_df.index = sig_df.index.tz_localize("UTC")
    else:
        sig_df.index = sig_df.index.tz_convert("UTC")

    # gate를 신호 인덱스에 정렬 (정확 매칭 → nearest fallback)
    gate_aligned = gate_df.reindex(sig_df.index)
    missing = gate_aligned["gate_long"].isna().sum()
    if missing:
        gate_aligned = gate_df.reindex(sig_df.index, method="nearest")
    sig_df = sig_df.copy()
    sig_df["gate_long"]  = gate_aligned["gate_long"].values
    sig_df["gate_short"] = gate_aligned["gate_short"].values

    gl_pct = sig_df["gate_long"].mean()  * 100
    gs_pct = sig_df["gate_short"].mean() * 100
    print(f"Gate 통과율 — long: {gl_pct:.1f}%, short: {gs_pct:.1f}%")

    # 4. ETH params
    eth_p = json.loads((ROOT / "models/production/eth_params.json").read_text())
    eth_p["tiers"] = [tuple(t) for t in eth_p["tiers"]]
    _skip = {"version", "rr_scale", "phase", "notes",
             "metrics_2026", "metrics_hist", "gate",
             "signal_source", "scaler", "strategy", "symbol"}
    base_kw = {k: v for k, v in eth_p.items() if k not in _skip}
    base_kw["min_hold_bars"] = 180

    # 5. 기준선: 게이트 없음 (analyze_funding.py 와 비교용)
    res_ng  = run_v17_backtest(sig_df, **base_kw)
    m_ng    = res_ng["metrics"]

    # 6. 게이트 포함
    gate_kw = {**base_kw, "gate_long_col": "gate_long", "gate_short_col": "gate_short"}
    res_g   = run_v17_backtest(sig_df, **gate_kw)
    m_g     = res_g["metrics"]

    print(f"\n기준선 (게이트 없음): {m_ng['total_return']:+.2f}%  WR={m_ng['win_rate']:.1f}%  n={m_ng['n_trades']}")
    print(f"게이트 포함:          {m_g['total_return']:+.2f}%  WR={m_g['win_rate']:.1f}%  n={m_g['n_trades']}")

    # 7. 펀딩비 적용
    ret_f_ng = apply_funding(res_ng["trade_log"], m_ng["total_return"],
                             "ETH 2026 — 게이트 없음 + 펀딩비")
    ret_f_g  = apply_funding(res_g["trade_log"],  m_g["total_return"],
                             "ETH 2026 — 게이트 포함 + 펀딩비")

    print(f"\n{'='*55}")
    print("  최종 요약")
    print(f"{'='*55}")
    print(f"  게이트 없음:  {m_ng['total_return']:+.2f}% → 펀딩비 후 {ret_f_ng:+.2f}%  (차이 {ret_f_ng - m_ng['total_return']:+.2f}%p)")
    print(f"  게이트 포함:  {m_g['total_return']:+.2f}% → 펀딩비 후 {ret_f_g:+.2f}%  (차이 {ret_f_g - m_g['total_return']:+.2f}%p)")
    print(f"  게이트 효과:  {m_g['total_return'] - m_ng['total_return']:+.2f}%p (기본)")
    print(f"               {ret_f_g - ret_f_ng:+.2f}%p (펀딩비 후)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
