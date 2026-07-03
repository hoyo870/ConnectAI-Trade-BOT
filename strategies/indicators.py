"""
strategies/indicators.py — BB σ 기반 ATR/RSI/트렌드 지표 계산 단일 소스
"""
import pandas as pd


def add_indicators_af(df: pd.DataFrame, bb_sigma: float = 0.5) -> pd.DataFrame:
    """ATR/RSI/1h 트렌드를 df에 추가. 컬럼: _atr, _rsi, _trend_up, _trend_down."""
    df = df.copy()
    close = df["close"]; high = df["high"]; low = df["low"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df["_atr"] = tr.ewm(span=14, adjust=False).mean()

    delta = close.diff()
    ag = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["_rsi"] = 100 - 100 / (1 + ag / (al + 1e-9))

    cl1h   = close.resample("1h").last()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    # look-ahead 제거 + 백테스트↔live 통일:
    #   기존은 resample('1h').last() 가 현재 형성 중 1h봉에 그 시간대 미래 5m종가(:55)를 넣어
    #   백테스트 trend에 미래참조가 있었다(live는 매 틱 그 시점까지만 봄).
    #   완성된 1h봉 EMA/밴드(shift1)를 현재 5m 종가와 비교 → 인과적. bb_sigma=0에서 live의
    #   봉단위 계산과 수학적으로 동치(현재종가가 EMA 양변 상쇄), 백테스트만 미래참조가 제거됨.
    ema_c = ema_1h.shift(1).reindex(df.index, method="ffill")
    if bb_sigma > 0:
        std_c = cl1h.rolling(20).std().shift(1).reindex(df.index, method="ffill")
        df["_trend_up"]   = (close > ema_c + bb_sigma * std_c).fillna(False).astype(int)
        df["_trend_down"] = (close < ema_c - bb_sigma * std_c).fillna(False).astype(int)
    else:
        df["_trend_up"]   = (close > ema_c).fillna(False).astype(int)
        df["_trend_down"] = (close < ema_c).fillna(False).astype(int)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI + ATR + 2σ BB + bb_sigma=0 트렌드. ML 피처 계산 기반 df 생성용."""
    df = add_indicators_af(df, bb_sigma=0)
    close = df["close"]
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["_bb_upper"] = mid + 2 * std
    df["_bb_lower"] = mid - 2 * std
    return df


def compute_scalar_indicators(df: pd.DataFrame, bb_sigma: float = 0.5) -> tuple:
    """마지막 봉 기준 (atr, rsi, trend_up, trend_down) 스칼라 반환."""
    out = add_indicators_af(df, bb_sigma)
    return (
        float(out["_atr"].iloc[-1]),
        float(out["_rsi"].iloc[-1]),
        bool(out["_trend_up"].iloc[-1]),
        bool(out["_trend_down"].iloc[-1]),
    )
