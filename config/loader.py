"""
config/loader.py — 코인 OHLCV 원본 데이터 로더 단일 소스 (지표 계산 없음)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent


COIN_CONFIG = {
    "btc": {"label": "BTC", "hist_start": "2020-01-01"},
    "eth": {"label": "ETH", "hist_start": "2021-04-01"},
    "sol": {"label": "SOL", "hist_start": "2021-06-01"},
    "xrp": {"label": "XRP", "hist_start": "2020-06-01"},
}


def load_ohlcv_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_index()


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.tz_convert(None) if df.index.tz else df.index
    return df


def load_coin_raw(coin: str) -> pd.DataFrame:
    """코인별 전체 OHLCV 로드 (지표 미포함). BTC/ETH: 전용 경로, SOL/XRP: data/raw/ 자동 탐색."""
    coin = coin.lower()
    label = coin.upper()
    print(f"{label} 데이터 로드 중...")

    if coin == "btc":
        pieces = []
        for f in sorted((ROOT / "data/raw").glob("BTCUSDT_5m_*.csv")):
            try:
                pieces.append(_normalize_index(load_ohlcv_csv(f)))
            except Exception:
                pass
        if not pieces:
            par = pd.read_parquet(ROOT / "data/signals_2026/backtest_2026_signals.parquet")
            pieces.append(_normalize_index(par[["open", "high", "low", "close", "volume"]].copy()))

    elif coin == "eth":
        pieces = [
            _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_history.parquet")),
            _normalize_index(pd.read_parquet(ROOT / "data/eth/ETHUSDT_5m_2026.parquet")),
        ]
        for f in sorted((ROOT / "data/raw").glob("ETHUSDT_5m_*.csv")):
            try:
                pieces.append(_normalize_index(load_ohlcv_csv(f)))
            except Exception:
                pass

    else:
        sym = f"{label}USDT"
        candidates = sorted((ROOT / "data/raw").glob(f"{sym}_5m_*.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"{sym} 데이터 없음. 먼저 다운로드:\n"
                f"  python src/data_fetcher.py --symbol {label}/USDT --start 2021-01-01"
            )
        pieces = [_normalize_index(load_ohlcv_csv(f)) for f in candidates]

    all_df = pd.concat(pieces).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="last")]
    all_df = all_df[all_df["close"].notna() & (all_df["close"] > 0)]
    print(f"  {all_df.index[0].date()} ~ {all_df.index[-1].date()}  ({len(all_df):,}행)")
    return all_df
