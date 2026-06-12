"""
temp/scripts/bot_manage.py
실계좌 통합 관리 스크립트

사용법:
  python temp/scripts/bot_manage.py init              # state 초기화 (1000 USDT, 코인당 250)
  python temp/scripts/bot_manage.py preflight         # 실계좌 시작 전 사전 검증
  python temp/scripts/bot_manage.py close             # 긴급 청산 dry-run
  python temp/scripts/bot_manage.py close --execute   # 긴급 청산 실행
  python temp/scripts/bot_manage.py watch             # 포트폴리오 감시 데몬
  python temp/scripts/bot_manage.py watch --kill      # 손실 초과 시 봇 자동 종료
  python temp/scripts/bot_manage.py watch --dry-run   # 1회 체크만 (테스트)

수수료 기준:
  Bybit : taker 0.044%  maker 0.020%
  BingX : taker 0.050%  maker 0.020%  (50% 페이백 → 실효 taker 0.025%)
"""
import sys, os, json, copy, time, signal, subprocess, argparse, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

sys.path.insert(0, "src")

ROOT     = Path(__file__).parent.parent.parent
LOGS_DIR = ROOT / "logs"
COINS    = ["BTC", "ETH", "SOL", "XRP"]

REAL_SEED          = 1000.0
COIN_SEED          = REAL_SEED * 0.25     # 250 USDT per coin
PORTFOLIO_DD_LIMIT = 0.05                 # 5% 포트폴리오 손실 한도
BOT_STALE_SECONDS  = 360                  # 6분 미갱신 → 봇 이상 판단
CHECK_INTERVAL     = 30                   # 감시 주기 (초)

# 수수료 상수
FEE = {
    "bybit": {"taker": 0.00044, "maker": 0.00020, "label": "Bybit (taker 0.044%)"},
    "bingx": {"taker": 0.00025, "maker": 0.00010, "label": "BingX (taker 0.025% after 50% rebate)"},
}

PASS = "✅"
WARN = "⚠️ "
FAIL = "❌"

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
    "capital":             COIN_SEED,
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


# ─── 공통 유틸 ────────────────────────────────────────────────────────────────

def state_path(coin: str) -> Path:
    prefix = f"_{coin.lower()}" if coin.upper() != "BTC" else ""
    return LOGS_DIR / f"live_state{prefix}.json"


def load_state(coin: str) -> Optional[dict]:
    p = state_path(coin)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def atomic_write(path: Path, data: dict):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ─── init ─────────────────────────────────────────────────────────────────────

def cmd_init(args):
    print("=" * 60)
    print(f"  실계좌 state 초기화")
    print(f"  총 시드: {REAL_SEED:,.0f} USDT | 코인당: {COIN_SEED:,.0f} USDT")
    print(f"  {now_str()}")
    print("=" * 60)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for coin in COINS:
        path    = state_path(coin)
        existing = load_state(coin)

        if existing is not None:
            if existing.get("position", 0) != 0:
                entry = existing.get("entry_price", 0)
                print(f"  {FAIL} {coin}: 열린 포지션 존재 (pos={existing['position']}, entry={entry:.4f})")
                print(f"       → 거래소에서 수동 청산 후 재실행")
                all_ok = False
                continue

            old_cap = existing.get("capital", 0)
            if abs(old_cap - COIN_SEED) < 1.0 and not args.force:
                print(f"  {PASS} {coin}: 이미 올바른 자본 ({old_cap:,.0f} USDT) — 변경 없음")
                continue

            existing["capital"]             = COIN_SEED
            existing["peak_capital"]        = max(existing.get("peak_capital", COIN_SEED), COIN_SEED)
            existing["daily_start_capital"] = COIN_SEED
            atomic_write(path, existing)
            print(f"  {PASS} {coin}: 자본 업데이트 {old_cap:,.0f} → {COIN_SEED:,.0f} USDT")
        else:
            state = copy.deepcopy(DEFAULT_STATE)
            atomic_write(path, state)
            print(f"  {PASS} {coin}: 신규 생성 {COIN_SEED:,.0f} USDT → {path.name}")

    if not all_ok:
        print(f"\n{FAIL} 초기화 실패 — 포지션 해소 후 재실행")
        return

    # 검증
    print(f"\n[검증]")
    total = 0.0
    for coin in COINS:
        s   = load_state(coin)
        cap = s.get("capital", 0) if s else 0
        pos = s.get("position", 0) if s else -1
        total += cap
        ok   = abs(cap - COIN_SEED) < 1.0 and pos == 0
        mark = PASS if ok else WARN
        trades = len(s.get("trade_log", [])) if s else 0
        print(f"  {mark} {coin}: {cap:,.0f} USDT | pos={pos} | trades={trades}")
    print(f"\n  총 트래킹: {total:,.0f} USDT")
    print(f"\n{PASS} 초기화 완료 — 실계좌 실행 준비됨")
    print(f"     TRADE_MODE=real EXCHANGE=bingx python src/live_trader.py")


# ─── preflight ────────────────────────────────────────────────────────────────

def cmd_preflight(args):
    from exchange_client import build_exchange, get_usdt_balance, get_position, get_symbol, set_leverage

    exchange_name = os.environ.get("EXCHANGE", "bingx").lower()
    fee_info      = FEE.get(exchange_name, FEE["bingx"])

    print("=" * 65)
    print(f"  실계좌 사전 검증 (Pre-flight Check)")
    print(f"  {now_str()}")
    print(f"  총 시드: {REAL_SEED:,.0f} USDT | 코인당: {COIN_SEED:,.0f} USDT")
    print(f"  수수료: {fee_info['label']}")
    print("=" * 65)

    issues = []

    # 1. 거래소 연결
    print("\n[1] 거래소 API 연결")
    try:
        t0 = time.time()
        os.environ["TRADE_MODE"] = "real"
        exchange, mode = build_exchange("real")
        latency = (time.time() - t0) * 1000
        if mode != "real":
            print(f"  {FAIL} 모드 불일치: {mode} (real 이어야 함)")
            issues.append("TRADE_MODE=real 확인 필요")
        else:
            print(f"  {PASS} {exchange.id} 연결 성공 | 지연: {latency:.0f}ms")
    except Exception as e:
        print(f"  {FAIL} 연결 실패: {e}")
        print("  → API 키 / .env 설정 확인")
        return

    # 2. USDT 잔고
    print("\n[2] USDT 가용 잔고")
    try:
        bal = get_usdt_balance(exchange)
        mark = PASS if bal >= REAL_SEED * 0.9 else WARN
        print(f"  {mark} 가용 잔고: {bal:,.2f} USDT (목표 ≥ {REAL_SEED:,.0f} — 이체 전이면 무시)")
    except Exception as e:
        print(f"  {FAIL} 잔고 조회 실패: {e}")
        issues.append("잔고 조회 불가")

    # 3. State 파일 자본
    print("\n[3] State 파일 자본 검증")
    total_tracked = 0.0
    for coin in COINS:
        s    = load_state(coin)
        path = state_path(coin)
        if s is None:
            print(f"  {FAIL} {coin}: 파일 없음 → python temp/scripts/bot_manage.py init")
            issues.append(f"{coin} state 파일 없음")
            continue
        cap    = s.get("capital", 0)
        pos    = s.get("position", 0)
        trades = len(s.get("trade_log", []))
        total_tracked += cap
        pos_str = "없음" if pos == 0 else (f"LONG {s.get('entry_price',0):,.4f}" if pos==1 else f"SHORT {s.get('entry_price',0):,.4f}")
        mark   = PASS if cap > 0 else FAIL
        if cap <= 0:
            issues.append(f"{coin} capital 이상: {cap}")
        print(f"  {mark} {coin}: {cap:,.0f} USDT | pos={pos_str} | trades={trades}")
    print(f"\n  총 트래킹: {total_tracked:,.0f} USDT")
    if abs(total_tracked - REAL_SEED) > 200:
        print(f"  {WARN} 총 자본 편차 큼 → bot_manage.py init 재실행")

    # 4. 포지션 일치
    print("\n[4] 거래소 포지션 vs State 불일치 검사")
    for coin in COINS:
        os.environ["COIN"] = coin
        s        = load_state(coin)
        state_pos = s.get("position", 0) if s else 0
        try:
            ex_pos    = get_position(exchange)
            ex_has    = ex_pos["side"] is not None and ex_pos["size"] > 0
            st_has    = state_pos != 0
            if ex_has != st_has:
                ex_str = f"{ex_pos['side']} {ex_pos['size']}" if ex_has else "없음"
                st_str = f"position={state_pos}" if st_has else "없음"
                print(f"  {FAIL} {coin}: 불일치! 거래소={ex_str} | state={st_str}")
                print(f"       → bot_manage.py close --execute 로 처리")
                issues.append(f"{coin} 포지션 불일치")
            else:
                status = f"거래소={ex_pos['side']} {ex_pos['size']}" if ex_has else "둘 다 없음"
                print(f"  {PASS} {coin}: 일치 ({status})")
        except Exception as e:
            print(f"  {WARN} {coin}: 포지션 조회 실패 ({e})")

    # 5. 레버리지
    print("\n[5] 레버리지 설정 확인 (목표: 3x)")
    print(f"  {PASS} 진입 시 set_leverage(3) 자동 호출 (exchange_client.py)")
    for coin in COINS:
        os.environ["COIN"] = coin
        try:
            t0 = time.time()
            set_leverage(exchange, 3)
            print(f"  {PASS} {coin}: set_leverage(3) 성공 | {(time.time()-t0)*1000:.0f}ms")
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ["position", "no position", "102100"]):
                print(f"  {PASS} {coin}: 포지션 없음 시 설정 불가 (진입 시 자동 설정)")
            else:
                print(f"  {WARN} {coin}: set_leverage 오류 — {e}")
                issues.append(f"{coin} 레버리지 설정 오류")

    # 최종 판정
    print("\n" + "=" * 65)
    if not issues:
        print(f"  {PASS} 모든 체크 통과 — 실계좌 시작 가능")
        print(f"\n  실행:")
        print(f"  TRADE_MODE=real EXCHANGE={exchange_name} python src/live_trader.py")
    else:
        print(f"  {FAIL} {len(issues)}개 문제 발견:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    print("=" * 65)


# ─── close ────────────────────────────────────────────────────────────────────

def cmd_close(args):
    from exchange_client import build_exchange, get_position, close_position, get_symbol

    execute = args.execute
    print("=" * 60)
    print(f"  긴급 청산 스크립트")
    print(f"  모드: {'🔴 실행 (실제 청산)' if execute else '🔵 DRY-RUN (확인만)'}")
    print(f"  {now_str()}")
    print("=" * 60)

    os.environ["TRADE_MODE"] = "real"
    try:
        exchange, mode = build_exchange("real")
        print(f"\n  거래소: {exchange.id} | 모드: {mode}")
        if mode != "real":
            print("  ⚠️  실계좌 모드가 아님 — API 키 확인")
            return
    except Exception as e:
        print(f"  {FAIL} 거래소 연결 실패: {e}")
        return

    print()
    found_any = False
    closed, failed = [], []

    for coin in COINS:
        os.environ["COIN"] = coin
        try:
            pos = get_position(exchange)
            if pos["side"] is None or pos["size"] == 0:
                print(f"  {coin:<4}: 포지션 없음")
                continue
            found_any = True
            side_str  = "LONG 🟢" if pos["side"] == "long" else "SHORT 🔴"
            print(f"  {coin:<4}: {side_str} | 수량={pos['size']} | "
                  f"진입={pos['entry_price']:,.4f} | 현재={pos['mark_price']:,.4f} | "
                  f"미실현PnL={pos['unrealized_pnl']:+.4f}")
            if execute:
                try:
                    close_position(exchange, pos)
                    print(f"       → {PASS} 청산 완료")
                    closed.append(coin)
                except Exception as ce:
                    print(f"       → {FAIL} 청산 실패: {ce}")
                    failed.append(coin)
            else:
                print(f"       → (DRY-RUN: --execute 옵션 추가 시 실제 청산)")
        except Exception as e:
            print(f"  {coin:<4}: 포지션 조회 실패 — {e}")
            failed.append(coin)

    if not found_any:
        print(f"\n  {PASS} 열린 포지션 없음")
    elif execute:
        print(f"\n  결과: 청산 성공 {closed} | 실패 {failed}")
        if failed:
            print("  ⚠️  실패 코인은 거래소 앱에서 직접 청산하세요.")
    else:
        print(f"\n  실제 청산하려면: python temp/scripts/bot_manage.py close --execute")


# ─── watch ────────────────────────────────────────────────────────────────────

log = logging.getLogger("watchdog")


def _load_all_states() -> dict:
    states = {}
    for coin in COINS:
        p = state_path(coin)
        if p.exists():
            try:
                s = json.loads(p.read_text())
                s["_mtime"] = p.stat().st_mtime
                states[coin] = s
            except Exception:
                pass
    return states


def _get_bot_pids() -> list:
    try:
        result = subprocess.run(["pgrep", "-f", "live_trader.py"],
                                capture_output=True, text=True)
        return [int(p) for p in result.stdout.strip().split("\n") if p.strip().isdigit()]
    except Exception:
        return []


def _send_alert(msg: str):
    try:
        from telegram_notifier import send_trade_alert
        send_trade_alert(f"🛡️ <b>[Watchdog]</b>\n{msg}")
    except Exception as e:
        log.warning(f"텔레그램 전송 실패: {e}")


def _kill_bot(pids: list):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log.warning(f"봇 프로세스 종료: PID {pid}")
        except Exception as e:
            log.error(f"PID {pid} 종료 실패: {e}")


def _check_once(auto_kill: bool) -> dict:
    states    = _load_all_states()
    now_ts    = time.time()
    now_utc   = now_str()
    issues    = []

    total_cap   = sum(states.get(c, {}).get("capital", REAL_SEED/4) for c in COINS)
    total_start = sum(states.get(c, {}).get("daily_start_capital", REAL_SEED/4) for c in COINS)
    port_dd     = (total_cap - total_start) / (total_start + 1e-9)

    if port_dd <= -PORTFOLIO_DD_LIMIT:
        msg = (f"⚠️ 포트폴리오 손실 한도 초과!\n"
               f"손실: {port_dd:.2%} (한도: -{PORTFOLIO_DD_LIMIT:.0%})\n"
               f"총 자본: {total_cap:,.0f} USDT | {now_utc}")
        log.warning(msg)
        _send_alert(msg)
        issues.append(f"포트폴리오 손실 {port_dd:.2%}")
        if auto_kill:
            pids = _get_bot_pids()
            if pids:
                _kill_bot(pids)
                _send_alert(f"🛑 봇 프로세스 종료 완료 (PID: {pids})")
            else:
                _send_alert("🔍 봇 프로세스를 찾지 못했습니다. 수동 확인 필요.")

    for coin in COINS:
        mtime = states.get(coin, {}).get("_mtime", 0)
        if mtime > 0:
            age = now_ts - mtime
            if age > BOT_STALE_SECONDS:
                msg = (f"⏱️ 봇 응답 없음 ({coin})\n"
                       f"State 파일 {age/60:.1f}분 미갱신")
                log.warning(msg)
                issues.append(f"{coin} state {age/60:.1f}분 미갱신")
                _send_alert(msg)

    return {"portfolio_dd_pct": port_dd * 100, "total_capital": total_cap,
            "now": now_utc, "issues": issues}


def cmd_watch(args):
    if args.dry_run:
        print("=== Watchdog DRY-RUN (1회 체크) ===")
        summary = _check_once(auto_kill=False)
        print(f"포트폴리오 일일 손익: {summary['portfolio_dd_pct']:+.2f}%")
        print(f"총 자본: {summary['total_capital']:,.0f} USDT")
        issues = summary.get("issues", [])
        print(f"{'⚠️  이슈: ' + str(issues) if issues else PASS + ' 이상 없음'}")
        return

    # 데몬 모드
    log_file = LOGS_DIR / "watchdog.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    auto_kill = args.kill
    interval  = args.interval

    log.info(f"Watchdog 시작 | 주기={interval}s | auto_kill={auto_kill}")
    _send_alert(f"🛡️ Portfolio Watchdog 시작\n"
                f"총 시드: {REAL_SEED:,.0f} USDT | 손실 한도: -{PORTFOLIO_DD_LIMIT:.0%}\n"
                f"자동 종료: {'ON' if auto_kill else 'OFF'}")

    last_hourly = time.time()

    while True:
        try:
            summary  = _check_once(auto_kill)
            dd       = summary["portfolio_dd_pct"]
            cap      = summary["total_capital"]
            issue_ct = len(summary["issues"])
            log.info(f"포트폴리오: {cap:,.0f} USDT | 일일 {dd:+.2f}%"
                     + (f" | ⚠️ {issue_ct}건" if issue_ct else ""))

            if time.time() - last_hourly >= 3600:
                states = _load_all_states()
                lines  = [f"🛡️ <b>Watchdog 시간 보고</b>\n{summary['now']}\n",
                          f"💰 총 자본: {cap:,.0f} USDT (일일 {dd:+.2f}%)"]
                for coin in COINS:
                    s = states.get(coin, {})
                    p = s.get("position", 0)
                    lines.append(f"  {coin}: {s.get('capital',0):,.0f} USDT | "
                                 f"{'없음' if p==0 else 'LONG' if p==1 else 'SHORT'}")
                _send_alert("\n".join(lines))
                last_hourly = time.time()

        except KeyboardInterrupt:
            log.info("Watchdog 종료 (Ctrl+C)")
            _send_alert("🛡️ Watchdog 정상 종료")
            break
        except Exception as e:
            log.error(f"체크 오류: {e}")

        time.sleep(interval)


# ─── 수수료 참고 ──────────────────────────────────────────────────────────────

def cmd_fee(args):
    print("=" * 55)
    print("  거래소 수수료 참고")
    print("=" * 55)
    print()
    headers = ["거래소", "Taker", "Maker", "실효 Taker", "비고"]
    rows = [
        ["Bybit",  "0.044%", "0.020%", "0.044%",  "페이백 없음"],
        ["BingX",  "0.050%", "0.020%", "0.025%",  "50% 페이백"],
    ]
    col_w = [10, 8, 8, 12, 16]
    fmt   = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  " + "-" * 58)
    for row in rows:
        print(fmt.format(*row))
    print()
    print("  백테스트 기준: TRADING_FEE=0.0005 + SLIPPAGE=0.0002")
    print("  (실제 수수료보다 보수적 — 양쪽 모두 백테스트가 불리)")
    print()
    print("  ※ AF 전략은 시장가(taker) 진입·청산")
    print("    BingX 실효: 진입 0.025% + 청산 0.025% = 왕복 0.05%")
    print("    Bybit 실효: 진입 0.044% + 청산 0.044% = 왕복 0.088%")


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="실계좌 통합 관리 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python temp/scripts/bot_manage.py init
  python temp/scripts/bot_manage.py preflight
  python temp/scripts/bot_manage.py close
  python temp/scripts/bot_manage.py close --execute
  python temp/scripts/bot_manage.py watch --dry-run
  python temp/scripts/bot_manage.py watch --kill
  python temp/scripts/bot_manage.py fee
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="state 파일 초기화 (1000 USDT, 코인당 250)")
    p_init.add_argument("--force", action="store_true", help="자본 동일해도 강제 덮어쓰기")

    # preflight
    sub.add_parser("preflight", help="실계좌 시작 전 사전 검증")

    # close
    p_close = sub.add_parser("close", help="긴급 청산 (기본: dry-run)")
    p_close.add_argument("--execute", action="store_true", help="실제 청산 실행")

    # watch
    p_watch = sub.add_parser("watch", help="포트폴리오 감시 데몬")
    p_watch.add_argument("--kill",      action="store_true", help="손실 한도 초과 시 봇 자동 종료")
    p_watch.add_argument("--interval",  type=int, default=CHECK_INTERVAL, help=f"감시 주기 (초, 기본: {CHECK_INTERVAL})")
    p_watch.add_argument("--dry-run",   action="store_true", dest="dry_run", help="1회만 체크 (테스트용)")

    # fee
    sub.add_parser("fee", help="수수료 참고 출력")

    args = parser.parse_args()

    dispatch = {
        "init":      cmd_init,
        "preflight": cmd_preflight,
        "close":     cmd_close,
        "watch":     cmd_watch,
        "fee":       cmd_fee,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
