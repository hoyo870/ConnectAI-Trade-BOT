"""
ML 피처 추출 — Antifragile 진입 필터용

add_ml_features(df)  : 기존 _rsi/_atr/_bb_* 위에 확장 ML 피처 추가
extract_snapshot(...): 진입 시점 스냅샷 피처 dict (LightGBM 입력)
extract_sequence(...): LSTM용 (seq_len, 7) 시퀀스
SNAPSHOT_COLS        : LightGBM 입력 피처 순서 (이 순서 고정)

데이터 누수 방지: extract_* 는 idx 및 그 이전 봉만 사용.
"""
import numpy as np
import pandas as pd


# LightGBM 입력 피처 순서 (고정)
SNAPSHOT_COLS = [
    "rsi_value", "rsi_dist",
    "atr_pct", "atr_percentile",
    "ema20_slope", "macd_hist", "adx",
    "bb_width", "bb_pos",
    "vol_ratio", "obv_slope",
    "close_vs_ema9", "close_vs_ema21", "close_vs_ema50", "close_vs_ema100",
    "hour_sin", "hour_cos", "day_of_week",
    "direction", "trend_regime",
    "ret1", "ret3", "body_ratio",
    "loss_streak",
]


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """기존 지표(add_indicators) 위에 ML 피처를 추가. DatetimeIndex 가정."""
    df = df.copy()
    close = df["close"]

    # EMAs
    for span in [9, 21, 50, 100]:
        df[f"_ema{span}"] = close.ewm(span=span, adjust=False).mean()

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["_macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # ADX (Wilder's 14-period)
    high = df["high"]; low = df["low"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (atr_w + 1e-9)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / (atr_w + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["_adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # BB width and position (needs _bb_upper, _bb_lower)
    mid = close.rolling(20).mean()
    df["_bb_width"] = (df["_bb_upper"] - df["_bb_lower"]) / (mid + 1e-9)
    df["_bb_pos"] = (close - df["_bb_lower"]) / (df["_bb_upper"] - df["_bb_lower"] + 1e-9)

    # Volume features (volume 컬럼 없으면 1.0 처리)
    if "volume" in df.columns:
        volume = df["volume"].fillna(0.0)
    else:
        volume = pd.Series(1.0, index=df.index)
    vol_ma = volume.rolling(20).mean()
    df["_vol_ratio"] = volume / (vol_ma + 1e-9)
    obv = (close.diff().apply(np.sign) * volume).fillna(0).cumsum()
    df["_obv_slope"] = obv.diff(5)

    # ATR features
    df["_atr_pct"] = df["_atr"] / (close + 1e-9)
    df["_atr_percentile"] = df["_atr"].rolling(100, min_periods=20).rank(pct=True)

    # EMA20 slope (uses _ema21 per spec)
    df["_ema20_slope"] = (df["_ema21"] - df["_ema21"].shift(5)) / (df["_ema21"].shift(5) + 1e-9)

    # Returns
    df["_ret1"] = close.pct_change(1)
    df["_ret3"] = close.pct_change(3)

    # Candle body ratio
    body = (close - df["open"]).abs()
    range_ = (df["high"] - df["low"]).clip(lower=1e-9)
    df["_body_ratio"] = body / range_

    return df


def _safe(v, default=0.0) -> float:
    f = float(v)
    if np.isnan(f) or np.isinf(f):
        return default
    return f


def extract_snapshot(df: pd.DataFrame, idx: int, direction: int,
                     rsi_lo: float, rsi_hi: float, loss_streak: int = 0) -> dict:
    """진입 시점 스냅샷 피처 dict 반환. keys == SNAPSHOT_COLS. idx 봉만 사용."""
    row = df.iloc[idx]
    ts = df.index[idx]
    close = float(row["close"])
    rsi = _safe(row["_rsi"], 50.0)

    rsi_dist = (rsi_lo - rsi) if direction == 1 else (rsi - rsi_hi)

    if isinstance(ts, pd.Timestamp):
        hour = ts.hour
        dow = ts.dayofweek
    else:
        hour = 0
        dow = 0

    trend_up = int(row.get("_trend_up", 0))
    trend_down = int(row.get("_trend_down", 0))
    trend_regime = 1 if trend_up else (-1 if trend_down else 0)

    def cve(span):
        ema = _safe(row.get(f"_ema{span}", close), close)
        return (close - ema) / (close + 1e-9)

    return {
        "rsi_value":      rsi,
        "rsi_dist":       _safe(rsi_dist),
        "atr_pct":        _safe(row.get("_atr_pct", 0.0)),
        "atr_percentile": _safe(row.get("_atr_percentile", 0.0)),
        "ema20_slope":    _safe(row.get("_ema20_slope", 0.0)),
        "macd_hist":      _safe(row.get("_macd_hist", 0.0)),
        "adx":            _safe(row.get("_adx", 0.0)),
        "bb_width":       _safe(row.get("_bb_width", 0.0)),
        "bb_pos":         _safe(row.get("_bb_pos", 0.5)),
        "vol_ratio":      _safe(row.get("_vol_ratio", 1.0), 1.0),
        "obv_slope":      _safe(row.get("_obv_slope", 0.0)),
        "close_vs_ema9":   _safe(cve(9)),
        "close_vs_ema21":  _safe(cve(21)),
        "close_vs_ema50":  _safe(cve(50)),
        "close_vs_ema100": _safe(cve(100)),
        "hour_sin":       float(np.sin(2 * np.pi * hour / 24)),
        "hour_cos":       float(np.cos(2 * np.pi * hour / 24)),
        "day_of_week":    float(dow),
        "direction":      float(direction),
        "trend_regime":   float(trend_regime),
        "ret1":           _safe(row.get("_ret1", 0.0)),
        "ret3":           _safe(row.get("_ret3", 0.0)),
        "body_ratio":     _safe(row.get("_body_ratio", 0.0)),
        "loss_streak":    float(loss_streak),
    }


def extract_sequence(df: pd.DataFrame, idx: int, seq_len: int = 20) -> np.ndarray:
    """LSTM용 (seq_len, 7) np.float32 배열. idx 포함 그 이전 seq_len 봉. 앞 0-pad."""
    out = np.zeros((seq_len, 7), dtype=np.float32)
    start = idx - seq_len + 1
    for j in range(seq_len):
        i = start + j
        if i < 0:
            continue
        row = df.iloc[i]
        ret1 = _safe(row.get("_ret1", 0.0))
        rsi = _safe(row.get("_rsi", 50.0), 50.0) / 100.0
        atr_pct = _safe(row.get("_atr_pct", 0.0))
        vol_ratio = _safe(row.get("_vol_ratio", 1.0), 1.0)
        ema_slope = _safe(row.get("_ema20_slope", 0.0))
        macd_hist = _safe(row.get("_macd_hist", 0.0))
        bb_pos = _safe(row.get("_bb_pos", 0.5))
        out[j] = [ret1, rsi, atr_pct, vol_ratio, ema_slope, macd_hist, bb_pos]
    return out
