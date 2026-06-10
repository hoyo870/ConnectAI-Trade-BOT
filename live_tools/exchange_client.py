"""
거래소 클라이언트 (ccxt 기반) — Bybit / BingX 지원
EXCHANGE    : bybit(기본) | bingx
TRADE_MODE  : paper | sandbox | real
  paper    → 공개 시세만 사용, 주문 없음
  sandbox  → 테스트넷 (Bybit만 지원, BingX는 paper로 동작)
  real     → 실계좌
"""
import os
import time
from pathlib import Path
from typing import Optional

import ccxt

ENV_PATH = Path(__file__).parent.parent / ".env"

# USDT 무기한 선물 심볼 — BingX / Bybit 동일 형식
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


def _order_params(exchange, reduce_only: bool = False) -> dict:
    """거래소별 주문 파라미터. Bybit은 category=linear 필요, BingX는 불필요."""
    if exchange.id == "bybit":
        return {"reduceOnly": reduce_only, "category": "linear"}
    return {"reduceOnly": reduce_only}


def build_exchange(mode: Optional[str] = None):
    """거래소 인스턴스와 정규화된 모드 문자열을 반환."""
    env      = _load_env()
    mode     = (mode or env.get("TRADE_MODE", "sandbox")).lower()
    exchange_id = env.get("EXCHANGE", "bybit").lower()

    if mode not in {"real", "sandbox", "paper"}:
        mode = "sandbox"

    # BingX는 sandbox 미지원 → paper로 대체
    if exchange_id == "bingx" and mode == "sandbox":
        mode = "paper"

    if mode == "real":
        if exchange_id == "bingx":
            api_key    = env.get("BINGX_REAL_API_KEY", "")
            api_secret = env.get("BINGX_REAL_API_SECRET", "")
        else:
            api_key    = env.get("BYBIT_REAL_API_KEY", "")
            api_secret = env.get("BYBIT_REAL_API_SECRET", "")
    elif mode == "sandbox":
        api_key    = env.get("BYBIT_SANDBOX_API_KEY", "")
        api_secret = env.get("BYBIT_SANDBOX_API_SECRET", "")
    else:
        api_key, api_secret = "", ""

    if exchange_id == "bingx":
        exchange = ccxt.bingx({
            "apiKey": api_key,
            "secret": api_secret,
            "options": {"defaultType": "swap"},
        })
    else:
        exchange = ccxt.bybit({
            "apiKey":  api_key,
            "secret":  api_secret,
            "options": {"defaultType": "linear"},
        })
        if mode == "sandbox":
            exchange.set_sandbox_mode(True)

    return exchange, mode


def get_usdt_balance(exchange) -> float:
    """사용 가능한 USDT 잔고 반환."""
    balance = exchange.fetch_balance()
    return float(balance.get("USDT", {}).get("free", 0.0))


def get_position(exchange) -> dict:
    """현재 포지션 반환 (없으면 side=None). COIN 환경변수 기준."""
    positions = exchange.fetch_positions([get_symbol()])
    for pos in positions:
        contracts = float(pos.get("contracts") or 0)
        if contracts > 0:
            return {
                "side":           pos["side"],
                "size":           contracts,
                "entry_price":    float(pos.get("entryPrice") or 0),
                "mark_price":     float(pos.get("markPrice") or 0),
                "leverage":       float(pos.get("leverage") or 1),
                "unrealized_pnl": float(pos.get("unrealizedPnl") or 0),
            }
    return {"side": None, "size": 0.0, "entry_price": 0.0,
            "mark_price": 0.0, "leverage": 1.0, "unrealized_pnl": 0.0}


def set_leverage(exchange, leverage: int):
    """레버리지 설정 (에러 무시 — 이미 설정된 경우)."""
    try:
        params = {"category": "linear"} if exchange.id == "bybit" else {}
        exchange.set_leverage(leverage, get_symbol(), params=params)
    except Exception:
        pass


def place_market_order(exchange, side: str, qty: float, reduce_only: bool = False) -> dict:
    """시장가 주문 실행."""
    return exchange.create_order(
        get_symbol(), "market", side, qty, params=_order_params(exchange, reduce_only)
    )


def close_position(exchange, pos: dict):
    """포지션 전체 청산."""
    if pos["side"] is None or pos["size"] == 0:
        return
    close_side = "sell" if pos["side"] == "long" else "buy"
    place_market_order(exchange, close_side, pos["size"], reduce_only=True)


def fetch_ohlcv_df(exchange, limit: int = 500):
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
            if attempt < 2 and any(c in str(e) for c in ["10006", "429", "rate"]):
                time.sleep(10 * (attempt + 1))
                continue
            raise


def calc_qty(capital: float, entry_rr: float, leverage: float, price: float) -> float:
    """포지션 수량 계산. COIN별 최소 단위 자동 적용."""
    coin = os.environ.get("COIN", "BTC").upper()
    decimals = {"BTC": 3, "ETH": 2, "SOL": 1, "XRP": 0}.get(coin, 3)
    qty = (capital * entry_rr * leverage) / price
    return round(qty, decimals)
