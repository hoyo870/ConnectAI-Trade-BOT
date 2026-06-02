"""
[Data Fetcher] ccxt를 사용해 Binance에서 OHLCV 데이터를 다운로드.

사용 예:
  python src/data_fetcher.py --symbol BTC/USDT --start 2022-01-01 --end 2024-01-01
"""

import os
import time
import argparse
from datetime import datetime, timezone

import ccxt
import pandas as pd


RAW_DIR      = "data/raw"
DEFAULT_SYM  = "BTC/USDT"
DEFAULT_TF   = "5m"
RATE_LIMIT_S = 0.5   # 요청 사이 대기 시간 (초)


def _parse_dt(dt_str: str) -> int:
    """'YYYY-MM-DD' 또는 'YYYY-MM-DD HH:MM:SS' → millisecond timestamp."""
    fmt = "%Y-%m-%d %H:%M:%S" if " " in dt_str else "%Y-%m-%d"
    dt  = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_ohlcv(
    symbol:     str = DEFAULT_SYM,
    timeframe:  str = DEFAULT_TF,
    start_date: str = "2022-01-01",
    end_date:   str | None = None,
    save: bool  = True,
) -> pd.DataFrame:
    """
    Binance에서 OHLCV 데이터를 페이지네이션으로 전체 다운로드.

    Parameters
    ----------
    symbol     : 거래 쌍 (ex. 'BTC/USDT')
    timeframe  : 봉 단위 (ex. '5m', '1h', '1d')
    start_date : 시작일 (YYYY-MM-DD)
    end_date   : 종료일 (None이면 현재 시각)
    save       : True면 data/raw/ 에 CSV 저장

    Returns
    -------
    DataFrame with columns: timestamp, open, high, low, close, volume
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    since_ms = _parse_dt(start_date)
    until_ms  = _parse_dt(end_date) if end_date else int(time.time() * 1000)
    limit     = 1000   # Binance 최대 요청 수

    sym_label  = symbol.replace("/", "")
    print(f"[Fetcher] {symbol} {timeframe} | {start_date} ~ {end_date or 'now'}")

    all_candles: list = []
    cursor = since_ms

    while cursor < until_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except ccxt.NetworkError as e:
            print(f"[Fetcher] 네트워크 오류: {e} — 5초 후 재시도")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"[Fetcher] 거래소 오류: {e}")
            break

        if not candles:
            break

        all_candles.extend(candles)
        cursor = candles[-1][0] + 1   # 마지막 타임스탬프 다음 ms
        fetched = len(all_candles)

        if fetched % 10_000 < limit:
            print(f"  누적 {fetched:,} 봉 수집 중...")

        time.sleep(RATE_LIMIT_S)

        if candles[-1][0] >= until_ms:
            break

    if not all_candles:
        print("[Fetcher] 데이터 없음.")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[df["timestamp"] <= pd.Timestamp(until_ms, unit="ms", tz="UTC")]
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[Fetcher] 완료: {len(df):,} 봉 | {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

    if save:
        os.makedirs(RAW_DIR, exist_ok=True)
        fname = f"{RAW_DIR}/{sym_label}_{timeframe}_{start_date[:10].replace('-','')}_{(end_date or 'now')[:10].replace('-','')}.csv"
        df.to_csv(fname, index=False)
        print(f"[Fetcher] 저장: {fname}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance OHLCV 데이터 수집")
    parser.add_argument("--symbol",    default=DEFAULT_SYM,  help="거래 쌍 (기본: BTC/USDT)")
    parser.add_argument("--timeframe", default=DEFAULT_TF,   help="봉 단위 (기본: 5m)")
    parser.add_argument("--start",     default="2022-01-01", help="시작일 YYYY-MM-DD")
    parser.add_argument("--end",       default=None,         help="종료일 YYYY-MM-DD (기본: 오늘)")
    args = parser.parse_args()

    fetch_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
    )
