"""
temp/scripts/40_bingx_paper_test.py
BingX paper 모드 연결 테스트

확인 항목:
  1. 거래소 빌드 (sandbox→paper 자동 전환 포함)
  2. 4종목 OHLCV 수신 (공개 API, 키 불필요)
  3. 심볼 형식 확인
  4. 마지막 봉 가격/시간 출력
"""
import sys
from datetime import datetime, timezone
sys.path.insert(0, "src")

from exchange_client import build_exchange, fetch_ohlcv_df, get_symbol
import os

COINS = ["BTC", "ETH", "SOL", "XRP"]

def run():
    print("=" * 60)
    print("  BingX Paper 모드 연결 테스트")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # ── 거래소 빌드 ──────────────────────────────────────────────
    print("\n[1] 거래소 빌드")
    exchange, mode = build_exchange()
    print(f"  exchange.id : {exchange.id}")
    print(f"  mode        : {mode}  ({'✅ paper' if mode == 'paper' else '⚠ ' + mode})")
    assert exchange.id == "bingx", f"거래소 ID 불일치: {exchange.id}"
    assert mode == "paper", f"모드 불일치: {mode}"

    # ── OHLCV 수신 테스트 ────────────────────────────────────────
    print("\n[2] 4종목 OHLCV 수신 (5분봉, 최근 10봉)")
    print(f"  {'코인':<5}  {'심볼':<20}  {'마지막 봉 시각(UTC)':<22}  {'종가':>10}  {'상태'}")
    print(f"  {'─'*75}")

    results = {}
    for coin in COINS:
        os.environ["COIN"] = coin
        symbol = get_symbol()
        try:
            df = fetch_ohlcv_df(exchange, limit=10)
            last = df.iloc[-1]
            ts   = df.index[-1].strftime("%Y-%m-%d %H:%M")
            close = last["close"]
            n_rows = len(df)
            status = f"✅ ({n_rows}봉)"
            results[coin] = True
        except Exception as e:
            ts, close, status = "N/A", 0, f"❌ {e}"
            results[coin] = False
        print(f"  {coin:<5}  {symbol:<20}  {ts:<22}  {close:>10,.2f}  {status}")

    # ── 최종 판정 ────────────────────────────────────────────────
    print(f"\n[3] 최종 판정")
    all_ok = all(results.values())
    for coin, ok in results.items():
        print(f"  {coin}: {'✅ 정상' if ok else '❌ 실패'}")

    print()
    if all_ok:
        print("  ✅ BingX paper 모드 정상 동작 확인")
        print("  → TRADE_MODE=paper EXCHANGE=bingx 로 live_trader.py 실행 가능")
    else:
        failed = [c for c, ok in results.items() if not ok]
        print(f"  ❌ 일부 코인 실패: {failed}")
        print("  → BingX API 상태 또는 ccxt 버전 확인 필요")

if __name__ == "__main__":
    run()
