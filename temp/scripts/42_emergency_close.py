"""
temp/scripts/42_emergency_close.py
긴급 청산 스크립트 — 봇 상태 무관 전 코인 강제 청산

봇 크래시 / 재시작 시 실제 거래소 포지션이 남아있을 때 사용.
live_trader.py 의 state.json을 무시하고 거래소 직접 조회 후 청산.

사용법:
  python temp/scripts/42_emergency_close.py           # dry-run (확인만)
  python temp/scripts/42_emergency_close.py --execute # 실제 청산

※ live_trader.py 수정 금지
"""
import sys, os, argparse
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, "src")

ROOT  = Path(__file__).parent.parent.parent
COINS = ["BTC", "ETH", "SOL", "XRP"]


def run(execute: bool = False):
    from exchange_client import build_exchange, get_position, close_position, get_symbol

    print("=" * 60)
    print(f"  긴급 청산 스크립트")
    print(f"  모드: {'🔴 실행 (실제 청산)' if execute else '🔵 DRY-RUN (확인만)'}")
    print(f"  시각: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # 실계좌 연결
    os.environ.pop("TRADE_MODE", None)
    original_env = {k: os.environ.get(k) for k in ["TRADE_MODE", "COIN"]}
    os.environ["TRADE_MODE"] = "real"

    try:
        exchange, mode = build_exchange("real")
        print(f"\n  거래소: {exchange.id} | 모드: {mode}")
        if mode != "real":
            print("  ⚠️  실계좌 모드가 아님 — .env 의 EXCHANGE/API 키 확인 필요")
            return
    except Exception as e:
        print(f"  ❌ 거래소 연결 실패: {e}")
        return

    print()
    found_any = False
    closed = []
    failed = []

    for coin in COINS:
        os.environ["COIN"] = coin
        symbol = get_symbol()
        try:
            pos = get_position(exchange)
            if pos["side"] is None or pos["size"] == 0:
                print(f"  {coin:<4} ({symbol}): 포지션 없음")
                continue

            found_any = True
            side_str  = "LONG 🟢" if pos["side"] == "long" else "SHORT 🔴"
            size      = pos["size"]
            entry     = pos["entry_price"]
            mark      = pos["mark_price"]
            upnl      = pos["unrealized_pnl"]
            print(f"  {coin:<4} ({symbol}): {side_str} | 수량={size} | "
                  f"진입={entry:,.4f} | 현재={mark:,.4f} | 미실현PnL={upnl:+.4f}")

            if execute:
                try:
                    close_position(exchange, pos)
                    print(f"       → ✅ 청산 완료")
                    closed.append(coin)
                except Exception as ce:
                    print(f"       → ❌ 청산 실패: {ce}")
                    failed.append(coin)
            else:
                print(f"       → (DRY-RUN: 실제 청산하려면 --execute 옵션 추가)")

        except Exception as e:
            print(f"  {coin:<4}: 포지션 조회 실패 — {e}")
            failed.append(coin)

    if not found_any:
        print("\n  ✅ 열린 포지션 없음 — 모든 코인 청산 완료 상태")
    elif execute:
        print(f"\n  결과: 청산 성공 {closed} | 실패 {failed}")
        if failed:
            print("  ⚠️  실패 코인은 거래소 앱에서 직접 청산하세요.")
    else:
        print(f"\n  위 포지션을 실제 청산하려면:")
        print(f"  python temp/scripts/42_emergency_close.py --execute")

    # 환경변수 복원
    for k, v in original_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="실제 청산 실행 (기본: dry-run)")
    args = parser.parse_args()
    run(execute=args.execute)
