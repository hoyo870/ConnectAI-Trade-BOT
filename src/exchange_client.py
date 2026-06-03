"""
Bybit 거래소 클라이언트 (ccxt 기반)
TRADE_MODE=paper   → 공개 시세만 사용 / 주문 없음
TRADE_MODE=sandbox → 테스트넷
TRADE_MODE=real    → 실계좌
"""
import os
import time
from pathlib import Path
from typing import Optional

import ccxt

ENV_PATH = Path(__file__).parent.parent / ".env"

# Bybit USDT 무기한 선물 심볼 — COIN 환경변수로 제어
# 지원 코인: BTC, ETH, SOL, XRP (기본: BTC)
_COIN_SYMBOL_MAP = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
}
_MIN_QTY_MAP = {
    "BTC": 0.001,
    "ETH": 0.01,
    "SOL": 0.1,
    "XRP": 1.0,
}

def get_symbol() -> str:
    coin = os.environ.get("COIN", "BTC").upper()
    return _COIN_SYMBOL_MAP.get(coin, f"{coin}/USDT:USDT")

def get_min_qty() -> float:
    coin = os.environ.get("COIN", "BTC").upper()
    return _MIN_QTY_MAP.get(coin, 0.001)

# 하위 호환: 모듈 로드 시점의 심볼 (단일 코인 실행 시 사용)
SYMBOL = get_symbol()


def _load_env() -> dict:
    env = dict(os.environ)
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip())
    return env


def build_exchange(mode: Optional[str] = None) -> tuple[ccxt.bybit, str]:
    """거래소 인스턴스와 정규화된 모드 문자열을 반환."""
    env  = _load_env()
    mode = (mode or env.get("TRADE_MODE", "sandbox")).lower()

    if mode not in {"real", "sandbox", "paper"}:
        mode = "sandbox"

    if mode == "real":
        api_key    = env.get("BYBIT_REAL_API_KEY", "")
        api_secret = env.get("BYBIT_REAL_API_SECRET", "")
    elif mode == "sandbox":
        api_key    = env.get("BYBIT_SANDBOX_API_KEY", "")
        api_secret = env.get("BYBIT_SANDBOX_API_SECRET", "")
    else:
        api_key    = ""
        api_secret = ""

    exchange = ccxt.bybit({
        "apiKey":  api_key,
        "secret":  api_secret,
        "options": {"defaultType": "linear"},
    })
    if mode == "sandbox":
        exchange.set_sandbox_mode(True)

    return exchange, mode


def get_usdt_balance(exchange: ccxt.bybit) -> float:
    """사용 가능한 USDT 잔고 반환."""
    balance = exchange.fetch_balance()
    return float(balance.get("USDT", {}).get("free", 0.0))


def get_position(exchange: ccxt.bybit) -> dict:
    """현재 포지션 반환 (없으면 side=None). COIN 환경변수 기준."""
    positions = exchange.fetch_positions([get_symbol()])
    for pos in positions:
        contracts = float(pos.get("contracts") or 0)
        if contracts > 0:
            return {
                "side":         pos["side"],          # "long" | "short"
                "size":         contracts,
                "entry_price":  float(pos.get("entryPrice") or 0),
                "mark_price":   float(pos.get("markPrice") or 0),
                "leverage":     float(pos.get("leverage") or 1),
                "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
            }
    return {
        "side": None,
        "size": 0.0,
        "entry_price": 0.0,
        "mark_price": 0.0,
        "leverage": 1.0,
        "unrealized_pnl": 0.0,
    }


def set_leverage(exchange: ccxt.bybit, leverage: int):
    """레버리지 설정 (에러 무시 — 이미 설정된 경우)."""
    try:
        exchange.set_leverage(leverage, get_symbol(), params={"category": "linear"})
    except Exception:
        pass


def place_market_order(
    exchange:  ccxt.bybit,
    side:      str,          # "buy" | "sell"
    qty:       float,        # BTC 수량
    reduce_only: bool = False,
) -> dict:
    """시장가 주문 실행."""
    params = {"reduceOnly": reduce_only, "category": "linear"}
    return exchange.create_order(get_symbol(), "market", side, qty, params=params)


def close_position(exchange: ccxt.bybit, pos: dict):
    """포지션 전체 청산."""
    if pos["side"] is None or pos["size"] == 0:
        return
    close_side = "sell" if pos["side"] == "long" else "buy"
    place_market_order(exchange, close_side, pos["size"], reduce_only=True)


def fetch_ohlcv_df(exchange: ccxt.bybit, limit: int = 500):
    """5분봉 OHLCV DataFrame 반환 (index = DatetimeIndex UTC). Rate limit 시 최대 3회 재시도."""
    import pandas as pd
    for attempt in range(3):
        try:
            raw = exchange.fetch_ohlcv(get_symbol(), "5m", limit=limit)
            df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df.astype(float)
        except Exception as e:
            if "10006" in str(e) and attempt < 2:   # Rate limit
                time.sleep(10 * (attempt + 1))       # 10s, 20s
                continue
            raise


def calc_qty(capital: float, entry_rr: float, leverage: float, price: float) -> float:
    """포지션 수량 계산. COIN별 최소 단위 자동 적용 (BTC 0.001, ETH 0.01, SOL 0.1, XRP 1.0)."""
    coin = os.environ.get("COIN", "BTC").upper()
    decimals = {"BTC": 3, "ETH": 2, "SOL": 1, "XRP": 0}.get(coin, 3)
    margin = capital * entry_rr
    qty    = (margin * leverage) / price
    return round(qty, decimals)
