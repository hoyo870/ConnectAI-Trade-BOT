"""
temp/scripts/41_real_state_init.py
실계좌 state 파일 초기화 — 1000 USDT 기준 코인당 250 USDT

실계좌 전환 전 반드시 실행:
  DEFAULT_STATE["capital"] = 10000 (하드코딩 버그)를
  state 파일을 사전에 생성하여 우회한다.

동작:
  1. 기존 state 파일 내 open position 체크 (있으면 중단)
  2. 올바른 capital(250 USDT) 로 state 파일 생성/업데이트
  3. 생성 결과 검증 출력

※ live_trader.py 수정 금지
"""
import sys, json, copy
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, "src")

ROOT       = Path(__file__).parent.parent.parent
LOGS_DIR   = ROOT / "logs"
REAL_SEED  = 1000.0          # 실계좌 총 시드 (USDT)
COIN_ALLOC = 0.25            # 코인당 25%
COIN_SEED  = REAL_SEED * COIN_ALLOC   # 250 USDT per coin
COINS      = ["BTC", "ETH", "SOL", "XRP"]

# 코인별 실계좌 state 파일 경로
def state_path(coin: str) -> Path:
    prefix = f"_{coin.lower()}" if coin.upper() != "BTC" else ""
    return LOGS_DIR / f"live_state{prefix}.json"

# DEFAULT_STATE 복사 (live_trader.py 와 동일 구조)
DEFAULT_STATE = {
    "position":            0,
    "entry_price":         0.0,
    "entry_time":          None,
    "entry_lev":           1.0,
    "entry_rr":            0.0,
    "entry_bar":           0,
    "entry_sig_long":      0.0,
    "entry_sig_short":     0.0,
    "current_bar":         0,
    "last_price":          0.0,
    "last_candle_ts":      None,
    "capital":             COIN_SEED,      # ← 250 USDT (핵심 수정)
    "peak_capital":        COIN_SEED,
    "daily_start_capital": COIN_SEED,
    "daily_date":          None,
    "daily_halt":          False,
    "tg_update_offset":    0,
    "cooling_left":        0,
    "cb_triggers":         0,
    "sig_long_hist":       [],
    "sig_short_hist":      [],
    "trade_log":           [],
    "af_trail_sl":         0.0,
    "af_peak_price":       0.0,
    "af_pyramid_count":    0,
    "af_current_rr":       0.0,
    "af_entry_atr":        0.0,
}


def load_existing(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def check_no_position(state: dict, coin: str) -> bool:
    if state.get("position", 0) != 0:
        print(f"  ⛔ {coin}: 열린 포지션 존재 (position={state['position']}, "
              f"entry={state.get('entry_price',0):.4f})")
        print(f"     → 거래소에서 수동 청산 후 다시 실행하세요.")
        return False
    return True


def init_state(coin: str, force: bool = False) -> bool:  # noqa
    path = state_path(coin)
    existing = load_existing(path)

    if existing is not None:
        # 기존 파일 있음
        if not check_no_position(existing, coin):
            return False

        old_cap = existing.get("capital", 0)
        if abs(old_cap - COIN_SEED) < 1.0 and not force:
            print(f"  ✅ {coin}: 이미 올바른 자본 ({old_cap:,.0f} USDT) — 변경 없음")
            return True

        # 자본만 업데이트 (trade_log 등 기존 데이터 유지)
        existing["capital"]             = COIN_SEED
        existing["peak_capital"]        = max(existing.get("peak_capital", COIN_SEED), COIN_SEED)
        existing["daily_start_capital"] = COIN_SEED
        state = existing
        action = f"자본 업데이트: {old_cap:,.0f} → {COIN_SEED:,.0f} USDT"
    else:
        # 파일 없음 → 새로 생성
        state  = copy.deepcopy(DEFAULT_STATE)
        action = f"신규 생성: {COIN_SEED:,.0f} USDT"

    # atomic write
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)

    print(f"  ✅ {coin}: {action}  → {path.name}")
    return True


def verify_all():
    print("\n[검증] state 파일 확인")
    total = 0.0
    all_ok = True
    for coin in COINS:
        path = state_path(coin)
        if not path.exists():
            print(f"  ❌ {coin}: 파일 없음!")
            all_ok = False
            continue
        s = json.loads(path.read_text())
        cap = s.get("capital", 0)
        pos = s.get("position", 0)
        total += cap
        ok = abs(cap - COIN_SEED) < 1.0 and pos == 0
        mark = "✅" if ok else "⚠️"
        print(f"  {mark} {coin}: capital={cap:,.0f} USDT | position={pos} | "
              f"trades={len(s.get('trade_log', []))}")
        if not ok:
            all_ok = False
    print(f"\n  총 트래킹 자본: {total:,.0f} USDT (목표: {REAL_SEED:,.0f} USDT)")
    return all_ok


def main():
    print("=" * 60)
    print(f"  실계좌 state 초기화")
    print(f"  총 시드: {REAL_SEED:,.0f} USDT | 코인당: {COIN_SEED:,.0f} USDT")
    print(f"  대상 파일: logs/live_state*.json")
    print("=" * 60)
    print()

    all_ok = True
    for coin in COINS:
        ok = init_state(coin)
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n❌ 초기화 실패 — 위 오류 해결 후 재실행하세요.")
        return

    ok = verify_all()
    print()
    if ok:
        print("✅ 초기화 완료. 실계좌 실행 준비됨.")
        print(f"   TRADE_MODE=real EXCHANGE=bybit python src/live_trader.py")
    else:
        print("⚠️  일부 항목 불일치 — 확인 후 진행하세요.")


if __name__ == "__main__":
    main()
