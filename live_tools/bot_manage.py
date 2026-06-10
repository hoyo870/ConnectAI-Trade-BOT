"""
live_tools/bot_manage.py
실계좌 통합 관리 스크립트

사용법:
  python live_tools/bot_manage.py init              # state 초기화 (거래소 잔고 자동 조회)
  python live_tools/bot_manage.py preflight         # 실계좌 시작 전 사전 검증
  python live_tools/bot_manage.py close             # 긴급 청산 dry-run
  python live_tools/bot_manage.py close --execute   # 긴급 청산 실행
  python live_tools/bot_manage.py watch             # 포트폴리오 감시 데몬
  python live_tools/bot_manage.py watch --kill      # 손실 초과 시 봇 자동 종료
  python live_tools/bot_manage.py watch --dry-run   # 1회 체크만 (테스트)
  python live_tools/bot_manage.py fee               # 수수료 참고 출력

수수료 기준:
  Bybit : taker 0.044%  maker 0.020%
  BingX : taker 0.050%  maker 0.020%  (50% 페이백 → 실효 taker 0.025%)
"""
import sys, os, json, copy, time, signal, subprocess, argparse, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

ROOT     = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"
sys.path.insert(0, str(Path(__file__).parent))

# .env 로드 (live_trader.py와 동일 패턴)
def _load_env_file():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
_load_env_file()

COINS    = ["BTC", "ETH", "SOL", "XRP"]

# BOT_SEED: .env 또는 환경변수로 지정 (없으면 1000 fallback, cmd_init은 exchange 자동 조회)
REAL_SEED          = float(os.environ.get("BOT_SEED", "1000"))
COIN_SEED          = REAL_SEED * 0.25
AF_LEVERAGE        = int(os.environ.get("LEVERAGE", "5"))
PORTFOLIO_DD_LIMIT = 0.05                 # 5% 포트폴리오 손실 한도
BOT_STALE_SECONDS  = 360                  # 6분 미갱신 → 봇 이상 판단
CHECK_INTERVAL     = 30                   # 감시 주기 (초)

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
    "initial_capital":     COIN_SEED,
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
    # 시드 결정: --seed 인자 > 거래소 잔고 자동 조회 > BOT_SEED env > fallback 1000
    seed = getattr(args, "seed", None)
    if seed is None:
        try:
            from exchange_client import build_exchange, get_usdt_balance
            os.environ["TRADE_MODE"] = "real"
            _ex, _ = build_exchange("real")
            bal = get_usdt_balance(_ex)
            if bal > 0:
                seed = bal
                print(f"  거래소 잔고 자동 조회: {bal:.2f} USDT")
            else:
                print(f"  {WARN} 거래소 잔고 0 — BOT_SEED/기본값 사용")
                seed = REAL_SEED
        except Exception as _e:
            print(f"  {WARN} 거래소 조회 실패 → BOT_SEED/기본값 사용: {_e}")
            seed = REAL_SEED

    coin_seed = round(seed * 0.25, 4)

    print("=" * 60)
    print(f"  실계좌 state 초기화")
    print(f"  총 시드: {seed:,.2f} USDT | 코인당: {coin_seed:,.2f} USDT")
    print(f"  {now_str()}")
    print("=" * 60)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for coin in COINS:
        path     = state_path(coin)
        existing = load_state(coin)

        if existing is not None:
            if existing.get("position", 0) != 0:
                entry = existing.get("entry_price", 0)
                print(f"  {FAIL} {coin}: 열린 포지션 존재 (pos={existing['position']}, entry={entry:.4f})")
                print(f"       → 거래소에서 수동 청산 후 재실행")
                all_ok = False
                continue

            old_cap = existing.get("capital", 0)
            if abs(old_cap - coin_seed) < 0.5 and not args.force:
                print(f"  {PASS} {coin}: 이미 올바른 자본 ({old_cap:,.2f} USDT) — 변경 없음")
                continue

            existing["capital"]          = coin_seed
            existing["initial_capital"]  = coin_seed
            existing["peak_capital"]     = max(existing.get("peak_capital", coin_seed), coin_seed)
            existing["daily_start_capital"] = coin_seed
            atomic_write(path, existing)
            print(f"  {PASS} {coin}: 자본 업데이트 {old_cap:,.2f} → {coin_seed:,.2f} USDT")
        else:
            state = copy.deepcopy(DEFAULT_STATE)
            state["capital"]          = coin_seed
            state["initial_capital"]  = coin_seed
            state["peak_capital"]     = coin_seed
            state["daily_start_capital"] = coin_seed
            atomic_write(path, state)
            print(f"  {PASS} {coin}: 신규 생성 {coin_seed:,.2f} USDT → {path.name}")

    if not all_ok:
        print(f"\n{FAIL} 초기화 실패 — 포지션 해소 후 재실행")
        return

    print(f"\n[검증]")
    total = 0.0
    for coin in COINS:
        s      = load_state(coin)
        cap    = s.get("capital", 0) if s else 0
        pos    = s.get("position", 0) if s else -1
        trades = len(s.get("trade_log", [])) if s else 0
        total += cap
        mark   = PASS if cap > 0 and pos == 0 else WARN
        print(f"  {mark} {coin}: {cap:,.2f} USDT | pos={pos} | trades={trades}")
    print(f"\n  총 트래킹: {total:,.2f} USDT")
    print(f"\n{PASS} 초기화 완료 — 실계좌 실행 준비됨")
    exchange_name = os.environ.get("EXCHANGE", "bybit")
    print(f"     python live_tools/run.py")


# ─── preflight ────────────────────────────────────────────────────────────────

def cmd_preflight(args):
    from exchange_client import build_exchange, get_usdt_balance, get_position, set_leverage

    exchange_name = os.environ.get("EXCHANGE", "bybit").lower()
    fee_info      = FEE.get(exchange_name, FEE["bybit"])

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
        bal  = get_usdt_balance(exchange)
        mark = PASS if bal > 0 else FAIL
        print(f"  {mark} 가용 잔고: {bal:,.2f} USDT")
        if bal <= 0:
            issues.append("잔고 0 — 거래소 입금 확인")
    except Exception as e:
        print(f"  {FAIL} 잔고 조회 실패: {e}")
        issues.append("잔고 조회 불가")

    # 3. State 파일 자본
    print("\n[3] State 파일 자본 검증")
    total_tracked = 0.0
    for coin in COINS:
        s   = load_state(coin)
        if s is None:
            print(f"  {FAIL} {coin}: 파일 없음 → python live_tools/bot_manage.py init")
            issues.append(f"{coin} state 파일 없음")
            continue
        cap    = s.get("capital", 0)
        pos    = s.get("position", 0)
        trades = len(s.get("trade_log", []))
        total_tracked += cap
        pos_str = "없음" if pos == 0 else (f"LONG {s.get('entry_price',0):,.4f}" if pos == 1 else f"SHORT {s.get('entry_price',0):,.4f}")
        mark   = PASS if cap > 0 else FAIL
        if cap <= 0:
            issues.append(f"{coin} capital 이상: {cap}")
        print(f"  {mark} {coin}: {cap:,.0f} USDT | pos={pos_str} | trades={trades}")
    print(f"\n  총 트래킹: {total_tracked:,.2f} USDT")
    if total_tracked <= 0:
        print(f"  {WARN} 총 자본 0 → bot_manage.py init 재실행")

    # 4. 포지션 일치
    print("\n[4] 거래소 포지션 vs State 불일치 검사")
    for coin in COINS:
        os.environ["COIN"] = coin
        s         = load_state(coin)
        state_pos = s.get("position", 0) if s else 0
        try:
            ex_pos = get_position(exchange)
            ex_has = ex_pos["side"] is not None and ex_pos["size"] > 0
            st_has = state_pos != 0
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
    print(f"\n[5] 레버리지 설정 확인 (목표: {AF_LEVERAGE}x)")
    print(f"  {PASS} 진입 시 set_leverage({AF_LEVERAGE}) 자동 호출 (exchange_client.py)")
    for coin in COINS:
        os.environ["COIN"] = coin
        try:
            t0 = time.time()
            set_leverage(exchange, AF_LEVERAGE)
            print(f"  {PASS} {coin}: set_leverage({AF_LEVERAGE}) 성공 | {(time.time()-t0)*1000:.0f}ms")
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
        print(f"  TRADE_MODE=real EXCHANGE={exchange_name} python live_tools/live_trader.py")
    else:
        print(f"  {FAIL} {len(issues)}개 문제 발견:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    print("=" * 65)


# ─── close ────────────────────────────────────────────────────────────────────

def cmd_close(args):
    from exchange_client import build_exchange, get_position, close_position

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
        print(f"\n  실제 청산하려면: python live_tools/bot_manage.py close --execute")


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
    log.warning("긴급 청산 시도 (SIGTERM 전)...")
    try:
        import sys as _sys
        from pathlib import Path as _Path
        root = _Path(__file__).parent.parent
        import subprocess as _sp
        _sp.run(
            [_sys.executable, "live_tools/bot_manage.py", "close", "--execute"],
            cwd=str(root), timeout=30, capture_output=True
        )
        log.info("긴급 청산 완료")
    except Exception as _e:
        log.warning(f"긴급 청산 실패: {_e}")

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log.warning(f"봇 프로세스 종료: PID {pid}")
        except Exception as e:
            log.error(f"PID {pid} 종료 실패: {e}")


def _check_once(auto_kill: bool) -> dict:
    states      = _load_all_states()
    now_ts      = time.time()
    now_utc     = now_str()
    issues      = []

    total_cap   = sum(states.get(c, {}).get("capital", 0) for c in COINS)
    total_start = sum(states.get(c, {}).get("daily_start_capital", states.get(c, {}).get("initial_capital", 0)) for c in COINS)

    # 미실현 PnL 추정 (state의 last_price 기반)
    for coin in COINS:
        s = states.get(coin, {})
        pos = s.get("position", 0)
        if pos == 0:
            continue
        last_px  = s.get("last_price", 0)
        entry_p  = s.get("entry_price", 0)
        lev      = s.get("entry_lev", 1)
        rr       = s.get("entry_rr", 0.1)
        cap      = s.get("capital", 0)
        if last_px > 0 and entry_p > 0 and cap > 0:
            pnl_raw = pos * (last_px - entry_p) / (entry_p + 1e-9)
            unrealized = cap * pnl_raw * lev * rr
            total_cap += unrealized

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
                f"손실 한도: -{PORTFOLIO_DD_LIMIT:.0%} | 자동 종료: {'ON' if auto_kill else 'OFF'}")

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


# ─── resume ──────────────────────────────────────────────────────────────────

def cmd_resume(args):
    print("=" * 55)
    print(f"  일일 Halt 해제 (Resume)")
    print(f"  {now_str()}")
    print("=" * 55)
    for coin in COINS:
        s = load_state(coin)
        if s is None:
            print(f"  {WARN} {coin}: state 파일 없음")
            continue
        halted = s.get("daily_halt", False)
        if halted or args.force:
            s["daily_halt"]          = False
            s["daily_start_capital"] = s.get("capital", s.get("initial_capital", 0))
            s["cb_triggers"]         = 0
            atomic_write(state_path(coin), s)
            print(f"  {PASS} {coin}: halt 해제 | 일일 기준자본 = {s['daily_start_capital']:,.0f} USDT")
        else:
            print(f"  {PASS} {coin}: halt 없음 — 변경 불필요")
    print("\n  재시작: python live_tools/run.py")


# ─── backup ───────────────────────────────────────────────────────────────────

def cmd_backup(args):
    from datetime import datetime, timezone
    import shutil
    ts    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dst   = LOGS_DIR / "backup" / ts
    dst.mkdir(parents=True, exist_ok=True)
    backed = 0
    for coin in COINS:
        p = state_path(coin)
        if p.exists():
            shutil.copy2(p, dst / p.name)
            backed += 1
    print(f"{PASS} state 파일 {backed}개 백업 완료 → {dst}")


# ─── fee ─────────────────────────────────────────────────────────────────────

def cmd_fee(args):
    print("=" * 55)
    print("  거래소 수수료 참고")
    print("=" * 55)
    rows = [
        ["Bybit",  "0.044%", "0.020%", "0.044%",  "페이백 없음"],
        ["BingX",  "0.050%", "0.020%", "0.025%",  "50% 페이백"],
    ]
    col_w = [10, 8, 8, 12, 16]
    fmt   = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format("거래소", "Taker", "Maker", "실효 Taker", "비고"))
    print("  " + "-" * 58)
    for row in rows:
        print(fmt.format(*row))
    print()
    print("  백테스트 기준: TRADING_FEE=0.0005 + SLIPPAGE=0.0002 (보수적)")
    print("  BingX 왕복: 진입 0.025% + 청산 0.025% = 0.05%")
    print("  Bybit 왕복: 진입 0.044% + 청산 0.044% = 0.088%")


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="실계좌 통합 관리 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
커맨드:
  init        state 파일 초기화 (거래소 잔고 자동 조회, --seed 수동 지정 가능)
  preflight   실계좌 시작 전 전체 검증
  close       긴급 청산 dry-run (--execute 로 실행)
  watch       포트폴리오 감시 데몬 (--kill, --dry-run)
  fee         수수료 참고 출력
        """
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="state 파일 초기화 (거래소 잔고 자동 조회)")
    p_init.add_argument("--force", action="store_true", help="강제 덮어쓰기")
    p_init.add_argument("--seed",  type=float, default=None, help="총 시드 USDT 수동 지정 (미입력 시 거래소 잔고 자동 조회)")

    sub.add_parser("preflight", help="실계좌 시작 전 사전 검증")

    p_close = sub.add_parser("close", help="긴급 청산 (기본: dry-run)")
    p_close.add_argument("--execute", action="store_true", help="실제 청산 실행")

    p_watch = sub.add_parser("watch", help="포트폴리오 감시 데몬")
    p_watch.add_argument("--kill",     action="store_true", help="손실 한도 초과 시 봇 자동 종료")
    p_watch.add_argument("--interval", type=int, default=CHECK_INTERVAL, help=f"감시 주기 초 (기본: {CHECK_INTERVAL})")
    p_watch.add_argument("--dry-run",  action="store_true", dest="dry_run", help="1회만 체크")

    sub.add_parser("fee", help="수수료 참고 출력")

    p_resume = sub.add_parser("resume", help="일일 halt 해제")
    p_resume.add_argument("--force", action="store_true", help="halt 상태와 무관하게 강제 적용")

    sub.add_parser("backup", help="state 파일 백업")

    args = parser.parse_args()
    {"init": cmd_init, "preflight": cmd_preflight,
     "close": cmd_close, "watch": cmd_watch, "fee": cmd_fee,
     "resume": cmd_resume, "backup": cmd_backup}[args.command](args)


if __name__ == "__main__":
    main()
