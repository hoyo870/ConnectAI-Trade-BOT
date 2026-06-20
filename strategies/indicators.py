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

    cl1h   = close.resample("1h").last().ffill()
    ema_1h = cl1h.ewm(span=20, adjust=False).mean()
    if bb_sigma > 0:
        std_1h = cl1h.rolling(20).std()
        bb_up  = ema_1h + bb_sigma * std_1h
        bb_lo  = ema_1h - bb_sigma * std_1h
        df["_trend_up"]   = (cl1h > bb_up).reindex(df.index, method="ffill").fillna(False).astype(int)
        df["_trend_down"] = (cl1h < bb_lo).reindex(df.index, method="ffill").fillna(False).astype(int)
    else:
        df["_trend_up"]   = (cl1h > ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)
        df["_trend_down"] = (cl1h < ema_1h).reindex(df.index, method="ffill").fillna(False).astype(int)

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI + ATR + 2σ BB + bb_sigma=0 트렌드. 기존 backtest_antifragile 호환 단일 소스."""
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
