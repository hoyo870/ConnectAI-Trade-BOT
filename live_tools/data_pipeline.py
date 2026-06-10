"""
[Data Pipeline]
5분봉 OHLCV 데이터 전처리, 기술적 지표 생성, 레이블 생성, 스케일링 로직.

레이블 정의:
  - Long  : 향후 N봉 내 가격이 threshold% 이상 상승
  - Short : 향후 N봉 내 가격이 threshold% 이상 하락
  - Context: 변동성 상위 구간 (ATR 기반)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib


# ── 설정 상수 ─────────────────────────────────────────────────────────────────
LABEL_LOOKAHEAD = 12        # 레이블 계산용 선행 봉 수 (12 * 5분 = 1시간)
LONG_THRESHOLD   = 0.005    # Long 레이블 임계치 (+0.5%)
SHORT_THRESHOLD  = 0.005    # Short 레이블 임계치 (-0.5%)
CONTEXT_ATR_PCT  = 0.6      # ATR 상위 N% → Context=1 (변동성 높은 구간)

SCALER_PATH = "models/production/btc_scaler.pkl"


# ── 지표 계산 헬퍼 ────────────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에 기술적 지표 컬럼을 추가하여 반환."""
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── 추세 지표 ──────────────────────────────────────────────────────────────
    # EMA
    for span in [9, 21, 50, 100]:
        df[f"ema_{span}"] = _ema(close, span)

    df["ema_cross_9_21"]   = df["ema_9"]  - df["ema_21"]
    df["ema_cross_21_50"]  = df["ema_21"] - df["ema_50"]

    # MACD (12, 26, 9)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── 모멘텀 지표 ────────────────────────────────────────────────────────────
    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Stochastic RSI (14, 3, 3)
    rsi = df["rsi_14"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9)
    df["stoch_rsi_k"] = stoch_rsi.rolling(3).mean() * 100
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    # ROC (Rate of Change) - 5, 15봉
    for period in [5, 15]:
        df[f"roc_{period}"] = close.pct_change(period) * 100

    # ── 변동성 지표 ────────────────────────────────────────────────────────────
    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(com=13, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / close  # 가격 대비 ATR (정규화)

    # Bollinger Bands (20, 2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)
    df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # 가격 변화율 (1, 3, 6봉)
    for period in [1, 3, 6]:
        df[f"returns_{period}"] = close.pct_change(period)

    # ── 거래량 지표 ────────────────────────────────────────────────────────────
    df["vol_ema_20"]  = _ema(vol, 20)
    df["vol_ratio"]   = vol / (df["vol_ema_20"] + 1e-9)  # 거래량 상대 비율

    # OBV (On-Balance Volume)
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["obv"]         = obv
    df["obv_ema_20"]  = _ema(obv, 20)
    df["obv_slope"]   = df["obv_ema_20"].diff(5)

    # VWAP (세션 내 근사 — 롤링 20봉)
    typical_price = (high + low + close) / 3
    df["vwap_20"]      = (typical_price * vol).rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)
    df["price_vs_vwap"] = (close - df["vwap_20"]) / (df["vwap_20"] + 1e-9)

    # ── 시장 구조 ──────────────────────────────────────────────────────────────
    # 고가/저가 대비 현재 위치
    df["high_low_range"] = (high - low) / (close + 1e-9)
    df["close_pos"]      = (close - low) / (high - low + 1e-9)  # 캔들 내 위치

    # 추세 강도 (ADX 근사)
    pos_dm = (high.diff()).clip(lower=0)
    neg_dm = (-low.diff()).clip(lower=0)
    mask   = pos_dm < neg_dm
    pos_dm[mask] = 0
    neg_dm[~mask] = 0
    atr_s   = true_range.ewm(com=13, adjust=False).mean()
    pdi     = 100 * pos_dm.ewm(com=13, adjust=False).mean() / (atr_s + 1e-9)
    ndi     = 100 * neg_dm.ewm(com=13, adjust=False).mean() / (atr_s + 1e-9)
    dx      = (pdi - ndi).abs() / (pdi + ndi + 1e-9) * 100
    df["adx_14"] = dx.ewm(com=13, adjust=False).mean()
    df["pdi_14"] = pdi
    df["ndi_14"] = ndi

    return df


# ── 레이블 생성 ───────────────────────────────────────────────────────────────

def create_labels(
    df: pd.DataFrame,
    lookahead: int   = LABEL_LOOKAHEAD,
    long_thr: float  = LONG_THRESHOLD,
    short_thr: float = SHORT_THRESHOLD,
    context_pct: float = CONTEXT_ATR_PCT,
) -> pd.DataFrame:
    """
    Long / Short / Context 이진 레이블 생성.

    Long=1  : 향후 `lookahead`봉 내 최고 수익률 >= long_thr
    Short=1 : 향후 `lookahead`봉 내 최저 수익률 <= -short_thr
    Context=1: ATR_pct가 상위 (1-context_pct) 분위수 이상 (고변동 구간)
    """
    df = df.copy()
    close = df["close"]

    # 미래 수익률 행렬 벡터화: shift(-k) 로 lookahead 개 열 생성 후 max/min
    future_rets = pd.concat(
        [(close.shift(-k) - close) / (close + 1e-9) for k in range(1, lookahead + 1)],
        axis=1,
    )
    future_max_ret = future_rets.max(axis=1)
    future_min_ret = future_rets.min(axis=1)

    df["label_long"]  = (future_max_ret >= long_thr).astype(int)
    df["label_short"] = (future_min_ret <= -short_thr).astype(int)

    # Context: ATR 기반 고변동 구간
    atr_threshold = df["atr_pct"].quantile(context_pct)
    df["label_context"] = (df["atr_pct"] >= atr_threshold).astype(int)

    # 마지막 lookahead 봉은 레이블 계산 불가 → NaN 처리 후 제거
    df.loc[df.index[-lookahead:], ["label_long", "label_short"]] = np.nan

    return df


# ── 특성 선택 ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # 추세
    "ema_cross_9_21", "ema_cross_21_50",
    "macd", "macd_signal", "macd_hist",
    # 모멘텀
    "rsi_14", "stoch_rsi_k", "stoch_rsi_d",
    "roc_5", "roc_15",
    # 변동성
    "atr_pct", "bb_width", "bb_position",
    "returns_1", "returns_3", "returns_6",
    # 거래량
    "vol_ratio", "obv_slope", "price_vs_vwap",
    # 시장 구조
    "high_low_range", "close_pos",
    "adx_14", "pdi_14", "ndi_14",
]

LABEL_COLS = ["label_long", "label_short", "label_context"]


# ── 스케일링 ──────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, save_path: str = SCALER_PATH) -> RobustScaler:
    """학습 데이터로 스케일러를 피팅하고 저장."""
    scaler = RobustScaler()
    scaler.fit(df[FEATURE_COLS].dropna())
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"[Data] 스케일러 저장 완료: {save_path}")
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
    load_path: str = SCALER_PATH,
) -> pd.DataFrame:
    """스케일러를 적용하여 특성 컬럼을 정규화."""
    if scaler is None:
        scaler = joblib.load(load_path)
    df = df.copy()
    df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df


# ── 전체 파이프라인 ───────────────────────────────────────────────────────────

def build_dataset(
    raw_df: pd.DataFrame,
    fit_scaler_flag: bool = True,
    scaler: RobustScaler | None = None,
) -> tuple[pd.DataFrame, RobustScaler]:
    """
    raw OHLCV DataFrame → 지표 + 레이블 + 스케일된 특성 DataFrame 반환.

    Parameters
    ----------
    raw_df : OHLCV 컬럼(open, high, low, close, volume)을 가진 DataFrame.
             인덱스는 datetime.
    fit_scaler_flag : True이면 스케일러를 새로 피팅, False이면 load_path에서 로드.
    scaler : 외부에서 전달하는 스케일러 (fit_scaler_flag=False 시 사용).

    Returns
    -------
    df_scaled : 특성 + 레이블이 포함된 전처리 완료 DataFrame
    scaler    : 피팅된 RobustScaler
    """
    print(f"[Data] 원본 데이터 크기: {raw_df.shape}")

    # 1) 지표 생성
    df = add_technical_indicators(raw_df)

    # 2) 레이블 생성
    df = create_labels(df)

    # 3) 결측치 제거 (지표 계산 초반 NaN 및 마지막 lookahead 구간)
    df.dropna(subset=FEATURE_COLS + LABEL_COLS, inplace=True)
    print(f"[Data] NaN 제거 후 크기: {df.shape}")

    # 4) 클래스 불균형 확인 (로그)
    for col in LABEL_COLS:
        ratio = df[col].mean()
        print(f"[Data] {col} 양성 비율: {ratio:.3f} ({df[col].sum():.0f} / {len(df)})")

    # 5) 스케일링
    if fit_scaler_flag:
        scaler = fit_scaler(df)
    df_scaled = apply_scaler(df, scaler=scaler)

    return df_scaled, scaler


# ── 데이터 분할 ───────────────────────────────────────────────────────────────

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float   = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """시계열 순서를 유지한 train / val / test 분할 (랜덤 셔플 없음)."""
    n = len(df)
    i_train = int(n * train_ratio)
    i_val   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:i_train].copy()
    val   = df.iloc[i_train:i_val].copy()
    test  = df.iloc[i_val:].copy()

    print(f"[Data] 분할 — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


# ── CSV 로더 (편의 함수) ──────────────────────────────────────────────────────

def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    CSV 파일 로드. 컬럼명은 소문자 open/high/low/close/volume 를 기대.
    timestamp 또는 datetime 컬럼을 인덱스로 설정.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    time_col = next((c for c in df.columns if c in ("timestamp", "datetime", "time", "date")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

    df.sort_index(inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# ── 실행 예시 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/btcusdt_5m.csv"

    raw = load_ohlcv_csv(csv_path)
    df_processed, scaler = build_dataset(raw, fit_scaler_flag=True)

    train_df, val_df, test_df = train_val_test_split(df_processed)

    # 저장
    os.makedirs("data", exist_ok=True)
    train_df.to_parquet("data/train.parquet")
    val_df.to_parquet("data/val.parquet")
    test_df.to_parquet("data/test.parquet")
    print("[Data] 전처리 완료. data/ 폴더에 parquet 저장.")
