"""
temp/scripts/44_portfolio_watchdog.py
포트폴리오 수준 외부 서킷브레이커 / 감시 데몬

live_trader.py 와 독립적으로 실행. 30초마다 체크.

기능:
  1. 총 포트폴리오 손실 감시 (기준: state 파일 daily_start_capital 합산)
  2. 총 손실 -5% 초과 시 텔레그램 경고 + (--kill 옵션 시) 봇 프로세스 종료
  3. 봇 프로세스 생존 감시 (5분 이상 state 파일 미갱신 시 경고)
  4. 매 시간 잔고 스냅샷 로그

실행:
  # 기본 (감시만, 자동 종료 안 함)
  nohup python temp/scripts/44_portfolio_watchdog.py > logs/watchdog.log 2>&1 &

  # 강제 종료 모드 (손실 임계 초과 시 봇 프로세스 kill)
  python temp/scripts/44_portfolio_watchdog.py --kill

※ live_trader.py 수정 금지
"""
import sys, os, json, time, signal, subprocess, argparse, logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, "src")
ROOT     = Path(__file__).parent.parent.parent
LOGS_DIR = ROOT / "logs"
COINS    = ["BTC", "ETH", "SOL", "XRP"]

# 설정값
REAL_SEED         = 1000.0     # 총 시드
PORTFOLIO_DD_LIMIT = 0.05      # 5% 총 손실 → 경고/종료
BOT_STALE_SECONDS  = 360       # 6분 이상 state 미갱신 시 봇 이상으로 판단
CHECK_INTERVAL     = 30        # 감시 주기 (초)

log = logging.getLogger("watchdog")


def state_path(coin: str) -> Path:
    prefix = f"_{coin.lower()}" if coin.upper() != "BTC" else ""
    return LOGS_DIR / f"live_state{prefix}.json"


def load_all_states() -> dict:
    states = {}
    for coin in COINS:
        p = state_path(coin)
        if p.exists():
            try:
                states[coin] = json.loads(p.read_text())
                states[coin]["_mtime"] = p.stat().st_mtime
            except Exception:
                pass
    return states


def get_bot_pid() -> list[int]:
    """live_trader.py 실행 중인 PID 목록."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "live_trader.py"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip().isdigit()]
        return pids
    except Exception:
        return []


def send_alert(msg: str):
    try:
        from telegram_notifier import send_trade_alert
        send_trade_alert(f"🛡️ <b>[Watchdog]</b>\n{msg}")
    except Exception as e:
        log.warning(f"텔레그램 전송 실패: {e}")


def kill_bot(pids: list[int]):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log.warning(f"봇 프로세스 종료: PID {pid}")
        except Exception as e:
            log.error(f"PID {pid} 종료 실패: {e}")


def check_once(auto_kill: bool) -> dict:
    states   = load_all_states()
    now_ts   = time.time()
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    issues   = []
    summary  = {}

    # ── 포트폴리오 손실 계산 ─────────────────────────────────────────
    total_cap    = 0.0
    total_start  = 0.0
    for coin in COINS:
        s = states.get(coin, {})
        cap   = s.get("capital", REAL_SEED / 4)
        start = s.get("daily_start_capital", REAL_SEED / 4)
        total_cap   += cap
        total_start += start

    port_dd = (total_cap - total_start) / (total_start + 1e-9)
    summary["portfolio_dd_pct"] = port_dd * 100
    summary["total_capital"]    = total_cap
    summary["now"]              = now_str

    if port_dd <= -PORTFOLIO_DD_LIMIT:
        msg = (
            f"⚠️ 포트폴리오 손실 한도 초과!\n"
            f"현재 손실: {port_dd:.2%} (한도: -{PORTFOLIO_DD_LIMIT:.0%})\n"
            f"총 자본: {total_cap:,.0f} USDT\n"
            f"시각: {now_str}"
        )
        log.warning(msg)
        send_alert(msg)
        issues.append(f"포트폴리오 손실 {port_dd:.2%}")

        if auto_kill:
            pids = get_bot_pid()
            if pids:
                kill_bot(pids)
                send_alert(f"🛑 봇 프로세스 종료 완료 (PID: {pids})")
            else:
                send_alert("🔍 봇 프로세스를 찾지 못했습니다. 수동 확인 필요.")

    # ── 봇 생존 감시 ────────────────────────────────────────────────
    for coin in COINS:
        s     = states.get(coin, {})
        mtime = s.get("_mtime", 0)
        if mtime > 0:
            age_sec = now_ts - mtime
            if age_sec > BOT_STALE_SECONDS:
                msg = (
                    f"⏱️ 봇 응답 없음 ({coin})\n"
                    f"State 파일 마지막 갱신: {age_sec/60:.1f}분 전\n"
                    f"봇이 멈췄을 수 있습니다."
                )
                log.warning(msg)
                issues.append(f"{coin} state 파일 {age_sec/60:.1f}분 미갱신")
                send_alert(msg)

    summary["issues"] = issues
    return summary


def run(auto_kill: bool, interval: int):
    # 로깅 설정
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

    log.info(f"Watchdog 시작 | 주기={interval}s | auto_kill={auto_kill}")
    log.info(f"포트폴리오 손실 한도: -{PORTFOLIO_DD_LIMIT:.0%} | 봇 응답 한도: {BOT_STALE_SECONDS}s")

    send_alert(
        f"🛡️ Portfolio Watchdog 시작\n"
        f"총 시드: {REAL_SEED:,.0f} USDT | 손실 한도: -{PORTFOLIO_DD_LIMIT:.0%}\n"
        f"자동 종료: {'ON' if auto_kill else 'OFF'}"
    )

    last_hourly = time.time()

    while True:
        try:
            summary = check_once(auto_kill)
            dd      = summary.get("portfolio_dd_pct", 0)
            cap     = summary.get("total_capital", 0)

            log.info(f"포트폴리오: {cap:,.0f} USDT | 일일 손익: {dd:+.2f}%"
                     + (f" | ⚠️ {len(summary['issues'])}건" if summary["issues"] else ""))

            # 매 시간 상태 요약 텔레그램
            if time.time() - last_hourly >= 3600:
                states = load_all_states()
                lines = [f"🛡️ <b>Watchdog 시간 보고</b>\n{summary['now']}\n"]
                lines.append(f"💰 총 자본: {cap:,.0f} USDT (일일 {dd:+.2f}%)")
                for coin in COINS:
                    s = states.get(coin, {})
                    c = s.get("capital", 0)
                    p = s.get("position", 0)
                    pos_str = "없음" if p == 0 else ("LONG" if p == 1 else "SHORT")
                    lines.append(f"  {coin}: {c:,.0f} USDT | {pos_str}")
                send_alert("\n".join(lines))
                last_hourly = time.time()

        except KeyboardInterrupt:
            log.info("Watchdog 종료 (Ctrl+C)")
            send_alert("🛡️ Watchdog 정상 종료")
            break
        except Exception as e:
            log.error(f"체크 오류: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill", action="store_true",
                        help="손실 한도 초과 시 봇 프로세스 자동 종료")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL,
                        help=f"감시 주기 (초, 기본: {CHECK_INTERVAL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="1회만 체크하고 종료 (테스트용)")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Watchdog DRY-RUN (1회 체크) ===")
        from datetime import datetime, timezone
        summary = check_once(auto_kill=False)
        print(f"포트폴리오 일일 손익: {summary.get('portfolio_dd_pct', 0):+.2f}%")
        print(f"총 자본: {summary.get('total_capital', 0):,.0f} USDT")
        issues = summary.get("issues", [])
        if issues:
            print(f"⚠️ 이슈: {issues}")
        else:
            print("✅ 이상 없음")
    else:
        run(auto_kill=args.kill, interval=args.interval)
