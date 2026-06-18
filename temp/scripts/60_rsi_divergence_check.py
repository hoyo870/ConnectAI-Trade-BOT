"""
[US-003] RSI 계산 창 비교 — 600봉 vs 1000봉
- 동일 기간에 대해 두 창 크기의 RSI 값 차이(MAE) 정량화
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "src")
from scripts.backtest_antifragile import load_coin_full
import pandas as pd, numpy as np

def calc_rsi(close_series, com=13, window=None):
    if window:
        close_series = close_series.iloc[-window:]
    delta = close_series.diff()
    ag = delta.clip(lower=0).ewm(com=com, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=com, adjust=False).mean()
    return 100 - 100 / (1 + ag / (al + 1e-9))

START = "2026-06-16"
END   = "2026-06-18 12:00"
coins = ["btc","eth","sol","xrp"]

print("="*65)
print("[US-003] RSI EWM 수렴 검증 — 600봉 vs 1000봉 창")
print(f"분석 기간: {START} ~ {END}")
print("="*65)

total_mae_600 = []; total_mae_1000 = []

for coin in coins:
    df_full = load_coin_full(coin)

    # 분석 기간 + 워밍업용 이전 데이터
    cutoff = pd.Timestamp(END)
    start_idx = df_full.index.searchsorted(pd.Timestamp(START))

    # 1000봉 워밍업 + 분석구간
    slice_1000 = df_full.iloc[max(0, start_idx-1000): df_full.index.searchsorted(cutoff)]
    slice_600  = df_full.iloc[max(0, start_idx-600) : df_full.index.searchsorted(cutoff)]
    # 전체 히스토리 (백테스트 기준 = ground truth)
    slice_full = df_full.iloc[:df_full.index.searchsorted(cutoff)]

    rsi_full  = calc_rsi(slice_full["close"])
    rsi_600   = calc_rsi(slice_600["close"])
    rsi_1000  = calc_rsi(slice_1000["close"])

    # 분석 구간만 비교
    idx_range = df_full[(df_full.index >= START) & (df_full.index < END)].index
    r_full  = rsi_full.reindex(idx_range).dropna()
    r_600   = rsi_600.reindex(idx_range).dropna()
    r_1000  = rsi_1000.reindex(idx_range).dropna()

    common = r_full.index.intersection(r_600.index).intersection(r_1000.index)
    mae_600  = (r_full[common] - r_600[common]).abs().mean()
    mae_1000 = (r_full[common] - r_1000[common]).abs().mean()
    max_600  = (r_full[common] - r_600[common]).abs().max()
    max_1000 = (r_full[common] - r_1000[common]).abs().max()

    total_mae_600.append(mae_600); total_mae_1000.append(mae_1000)

    print(f"\n  [{coin.upper()}] (분석봉수={len(common)})")
    print(f"    600봉 창:  MAE={mae_600:.4f}  Max오차={max_600:.4f}")
    print(f"    1000봉 창: MAE={mae_1000:.4f}  Max오차={max_1000:.4f}")
    imp = (mae_600-mae_1000)/mae_600*100 if mae_600>0 else 0
    print(f"    개선율: {imp:.1f}% ({'✅ 향상' if imp>0 else '❌ 동일/악화'})")

print(f"\n{'='*65}")
avg_600  = np.mean(total_mae_600)
avg_1000 = np.mean(total_mae_1000)
print(f"  [4코인 평균 MAE]  600봉={avg_600:.4f}  1000봉={avg_1000:.4f}")
print(f"  평균 개선율: {(avg_600-avg_1000)/avg_600*100:.1f}%")
print(f"\n  → FETCH_LIMIT 1000으로 실거래 RSI가 백테스트 RSI에 더 근접")
