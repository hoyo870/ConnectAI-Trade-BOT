"""
live_tools/run.py
실계좌 통합 실행 + 로컬 웹 대시보드 (supervisor + Flask)

실행:
  python live_tools/run.py                    # real 모드 (기본)
  python live_tools/run.py --paper            # paper 모드 (봇만, watchdog 없음)
  python live_tools/run.py --port 8765        # 포트 변경
  python live_tools/run.py --no-auto-restart  # 자동 재시작 비활성화

브라우저: http://127.0.0.1:8765

프로세스 관리:
  - live_trader.py + bot_manage.py watch --kill 자동 시작
  - 비정상 종료 감지 시 자동 재시작 (crash-loop 보호)
  - 포트폴리오 손실 -5% 초과 시 트레이더 재시작 차단 (Halted)
  - 중복 실행 방지 (logs/run.pid 락 파일)
"""

import sys, os, json, time, signal, subprocess, threading, argparse, logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

ROOT     = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"

from flask import Flask, jsonify, request, Response, render_template


def _load_env() -> None:
    """프로젝트 루트 .env를 os.environ에 로드 (이미 설정된 값은 덮어쓰지 않음)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

# ─── 설정 ─────────────────────────────────────────────────────────────────────

COINS            = ["BTC", "ETH", "SOL", "XRP"]
PORTFOLIO_DD_LIMIT = 0.05        # 5% 손실 → 트레이더 재시작 차단
RESTART_MAX      = 3             # 크래시루프 임계값
RESTART_WINDOW   = 600           # 10분 내 재시작 횟수 집계
BACKOFF          = [5, 15, 30, 60]
POLL_INTERVAL    = 2             # 프로세스 감시 주기 (초)
PID_FILE         = LOGS_DIR / "run.pid"

log = logging.getLogger("run")

# ─── 상태 공유 ────────────────────────────────────────────────────────────────

_lock = threading.Lock()

class ManagedProcess:
    def __init__(self, name, cmd, restart=True):
        self.name          = name
        self.cmd           = cmd
        self.should_restart = restart
        self.proc: Optional[subprocess.Popen] = None
        self.started_at    = 0.0
        self.restart_times = []      # 최근 재시작 타임스탬프
        self.status        = "stopped"   # running / stopped / halted / crash_loop
        self.last_exit_code: Optional[int] = None
        self.restart_count = 0

    def uptime(self) -> str:
        if self.status != "running" or self.started_at == 0:
            return "-"
        secs = int(time.time() - self.started_at)
        h, m = divmod(secs // 60, 60)
        return f"{h}h {m}m" if h else f"{m}m {secs%60}s"

    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "status":        self.status,
            "pid":           self.pid(),
            "uptime":        self.uptime(),
            "restart_count": self.restart_count,
            "last_exit":     self.last_exit_code,
        }

# ─── 프로세스 정의 ────────────────────────────────────────────────────────────

_paper_mode = False

def _make_processes(paper: bool) -> list:
    global _paper_mode
    _paper_mode = paper
    # .env에서 읽은 값을 명시적으로 하위 프로세스에 전달
    trader_env = {
        "TRADE_MODE": os.environ.get("TRADE_MODE", "paper"),
        "EXCHANGE":   os.environ.get("EXCHANGE", "bybit"),
        "LEVERAGE":   os.environ.get("LEVERAGE", "5"),
        "STRATEGY":   os.environ.get("STRATEGY", "antifragile"),
    }
    trader_cmd = [sys.executable, "live_tools/live_trader.py"]
    _wk_enabled = os.environ.get("WATCHDOG_KILL_ENABLED", "false").lower() in ("1", "true", "yes")
    watch_cmd   = [sys.executable, "live_tools/bot_manage.py", "watch"] + (["--kill"] if _wk_enabled else [])
    procs = [ManagedProcess("trader",   trader_cmd, restart=True)]
    if not paper:
        procs.append(ManagedProcess("watchdog", watch_cmd,  restart=True))
    _EXTRA_ENV.update(trader_env)
    return procs

_EXTRA_ENV: dict = {}
_processes: list = []
_auto_restart = True
_shutting_down = False

# ─── 리스크 체크 ──────────────────────────────────────────────────────────────

def _portfolio_drawdown() -> float:
    prefix_map = {"BTC": "", "ETH": "_eth", "SOL": "_sol", "XRP": "_xrp"}
    prefix_key  = "paper" if _paper_mode else "live"
    total_cap, total_start = 0.0, 0.0
    for coin, sfx in prefix_map.items():
        p = LOGS_DIR / f"{prefix_key}_state{sfx}.json"
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text())
            total_cap   += s.get("capital", 0)
            total_start += s.get("daily_start_capital", s.get("capital", 0))
        except Exception:
            pass
    if total_start < 1:
        return 0.0
    return (total_cap - total_start) / total_start


def _too_many_restarts(mp: ManagedProcess) -> bool:
    now = time.time()
    mp.restart_times = [t for t in mp.restart_times if now - t < RESTART_WINDOW]
    return len(mp.restart_times) >= RESTART_MAX

# ─── 프로세스 시작/종료 ───────────────────────────────────────────────────────

def _start(mp: ManagedProcess, extra_env: dict):
    env = os.environ.copy()
    env.update(extra_env)
    env["PYTHONUNBUFFERED"] = "1"

    log_name = f"supervisor_{mp.name}.log"
    log_path = LOGS_DIR / log_name
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8")

    mp.proc = subprocess.Popen(
        mp.cmd,
        cwd=str(ROOT),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    mp.started_at = time.time()
    mp.status     = "running"
    mp.restart_times.append(mp.started_at)
    log.info(f"[{mp.name}] 시작 PID={mp.proc.pid}")


def _stop(mp: ManagedProcess, timeout: int = 10):
    if mp.proc is None:
        return
    try:
        mp.proc.send_signal(signal.SIGTERM)
        mp.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        mp.proc.kill()
    except Exception:
        pass
    mp.proc   = None
    mp.status = "stopped"
    log.info(f"[{mp.name}] 종료")

# ─── 슈퍼바이저 루프 ──────────────────────────────────────────────────────────

def _supervisor_loop():
    global _shutting_down
    for mp in _processes:
        _start(mp, _EXTRA_ENV)
    time.sleep(3)

    backoff_idx = {mp.name: 0 for mp in _processes}

    while not _shutting_down:
        pending_restarts = []  # list of (mp, delay)
        with _lock:
            for mp in _processes:
                if mp.proc is None:
                    continue
                code = mp.proc.poll()
                if code is None:
                    mp.status = "running"
                    continue

                mp.last_exit_code = code
                mp.proc = None
                log.warning(f"[{mp.name}] 종료 감지 (exit={code})")

                if not _auto_restart or not mp.should_restart:
                    mp.status = "stopped"
                    continue

                # 트레이더 리스크 halt 체크
                if mp.name == "trader":
                    dd = _portfolio_drawdown()
                    if dd <= -PORTFOLIO_DD_LIMIT:
                        mp.status = "halted"
                        log.warning(f"[trader] 손실 한도 초과 ({dd:.1%}) — 재시작 차단")
                        _telegram_alert(f"🛑 [{mp.name}] HALTED — 손실 한도 {dd:.1%} 초과. 수동 확인 필요.")
                        continue

                if _too_many_restarts(mp):
                    mp.status = "crash_loop"
                    log.error(f"[{mp.name}] 크래시루프 감지 — 재시작 중단")
                    _telegram_alert(f"💥 [{mp.name}] CRASH LOOP — {RESTART_MAX}회 재시작 실패. 수동 확인 필요.")
                    continue

                delay = BACKOFF[min(backoff_idx[mp.name], len(BACKOFF)-1)]
                backoff_idx[mp.name] = min(backoff_idx[mp.name]+1, len(BACKOFF)-1)
                log.info(f"[{mp.name}] {delay}초 후 재시작...")
                pending_restarts.append((mp, delay))

        # Release lock during sleep to keep dashboard responsive
        for mp, delay in pending_restarts:
            time.sleep(delay)
            with _lock:
                _start(mp, _EXTRA_ENV)
                mp.restart_count += 1
                backoff_idx[mp.name] = 0

        time.sleep(POLL_INTERVAL)

# ─── 상태 읽기 ────────────────────────────────────────────────────────────────

def _read_states() -> dict:
    prefix_key = "paper" if _paper_mode else "live"
    sfx_map = {"BTC": "", "ETH": "_eth", "SOL": "_sol", "XRP": "_xrp"}
    result = {}
    for coin, sfx in sfx_map.items():
        p = LOGS_DIR / f"{prefix_key}_state{sfx}.json"
        if not p.exists():
            continue
        try:
            result[coin] = json.loads(p.read_text())
        except Exception:
            pass
    return result


def _portfolio_summary(states: dict) -> dict:
    total_cap   = sum(s.get("capital", 0) for s in states.values())
    total_start = sum(s.get("daily_start_capital", s.get("capital", 0)) for s in states.values())
    dd = (total_cap - total_start) / (total_start + 1e-9)
    return {"total_capital": round(total_cap, 2), "daily_pnl_pct": round(dd * 100, 2)}


def _tail_log(path: Path, lines: int = 100) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 40_000))
            content = f.read().decode("utf-8", errors="replace")
        return content.splitlines()[-lines:]
    except Exception:
        return []


def _capital_series(states: dict) -> dict:
    series = {}
    for coin, s in states.items():
        pts = [{"t": t["time"], "v": round(t["capital"], 2)}
               for t in s.get("trade_log", []) if "capital" in t]
        # 현재 자본을 마지막 포인트로 추가
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        pts.append({"t": s.get("last_candle_ts") or now, "v": round(s.get("capital", 0), 2)})
        series[coin] = pts
    return series

# ─── 텔레그램 알림 헬퍼 ───────────────────────────────────────────────────────

def _telegram_alert(msg: str):
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from telegram_notifier import send_trade_alert
        send_trade_alert(f"🤖 <b>[run.py Supervisor]</b>\n{msg}")
    except Exception as e:
        log.warning(f"텔레그램 알림 실패: {e}")


# ─── Flask 앱 ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(_HERE / "templates"),
    static_folder=str(_HERE / "static"),
)
app.logger.disabled = True
log_flask = logging.getLogger("werkzeug")
log_flask.setLevel(logging.WARNING)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    with _lock:
        procs = [mp.to_dict() for mp in _processes]
    states  = _read_states()
    summary = _portfolio_summary(states)
    coins   = {
        coin: {
            "capital":     round(s.get("capital", 0), 2),
            "position":    s.get("position", 0),
            "entry_price": round(s.get("avg_entry_price") or s.get("entry_price", 0), 4),
            "entry_rr":    round(s.get("entry_rr") or s.get("af_current_rr", 0.1), 3),
            "entry_lev":   int(s.get("entry_lev", 0)),
            "trail_sl":    round(s.get("af_trail_sl", 0), 4),
            "last_price":  round(s.get("last_price", 0), 4),
            "pyramid_cnt": int(s.get("af_pyramid_count", 0)),
        }
        for coin, s in states.items()
    }
    return jsonify({
        "processes": procs,
        "portfolio": summary,
        "coins":     coins,
        "mode":      "paper" if _paper_mode else "real",
        "exchange":  os.environ.get("EXCHANGE", "bingx").lower(),
        "leverage":  int(os.environ.get("LEVERAGE", "5")),
        "ts":        datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
    })


@app.route("/api/logs")
def api_logs():
    log_file = LOGS_DIR / ("paper_multi.log" if _paper_mode else "live_multi.log")
    n = int(request.args.get("n", 100))
    return jsonify({"lines": _tail_log(log_file, n)})


@app.route("/api/capital")
def api_capital():
    states = _read_states()
    return jsonify(_capital_series(states))


@app.route("/api/trades")
def api_trades():
    prefix_key = "paper" if _paper_mode else "live"
    sfx_map = {"BTC": "", "ETH": "_eth", "SOL": "_sol", "XRP": "_xrp"}
    all_trades = []
    for coin, sfx in sfx_map.items():
        p = LOGS_DIR / f"{prefix_key}_state{sfx}.json"
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text())
            for t in s.get("trade_log", []):
                t["coin"] = coin
                all_trades.append(t)
        except Exception:
            pass
    all_trades.sort(key=lambda t: t.get("time", ""), reverse=True)
    return jsonify({"trades": all_trades[:20]})


@app.route("/api/emergency_close", methods=["POST"])
def api_emergency_close():
    try:
        result = subprocess.run(
            [sys.executable, "live_tools/bot_manage.py", "close", "--execute"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30
        )
        return jsonify({"ok": True, "message": "청산 요청 완료", "output": result.stdout[-500:]})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.route("/api/action/<action>/<name>", methods=["POST"])
def api_action(action: str, name: str):
    with _lock:
        mp = next((p for p in _processes if p.name == name), None)
        if mp is None:
            return jsonify({"error": "unknown process"}), 404
        if action == "start":
            if mp.status in ("stopped", "halted", "crash_loop"):
                mp.restart_times.clear()
                mp.status = "starting"
                threading.Thread(target=lambda: _start(mp, _EXTRA_ENV), daemon=True).start()
            return jsonify({"ok": True})
        elif action == "stop":
            threading.Thread(target=lambda: _stop(mp), daemon=True).start()
            return jsonify({"ok": True})
        return jsonify({"error": "unknown action"}), 400



# ─── 락 파일 ─────────────────────────────────────────────────────────────────

def _acquire_lock():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            os.kill(old_pid, 0)
            print(f"[run.py] 이미 실행 중입니다 (PID {old_pid}). 종료 후 재실행하세요.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass
    PID_FILE.write_text(str(os.getpid()))


def _release_lock():
    PID_FILE.unlink(missing_ok=True)

# ─── 종료 처리 ────────────────────────────────────────────────────────────────

def _shutdown(sig=None, frame=None):
    global _shutting_down
    _shutting_down = True
    log.info("종료 신호 수신 — 하위 프로세스 정리 중...")
    with _lock:
        for mp in _processes:
            _stop(mp)
    _release_lock()
    log.info("종료 완료")
    sys.exit(0)


# ─── 실계좌 시작 검증 ─────────────────────────────────────────────────────────

def _auto_init_if_needed():
    """state 파일 없거나 자본 0이면 bot_manage.py init 자동 실행."""
    import json, subprocess
    sfx_map = {"BTC": "", "ETH": "_eth", "SOL": "_sol", "XRP": "_xrp"}
    total = 0.0
    missing = []
    for coin, sfx in sfx_map.items():
        p = LOGS_DIR / f"live_state{sfx}.json"
        if not p.exists():
            missing.append(coin)
            continue
        try:
            s = json.loads(p.read_text())
            total += s.get("capital", 0)
        except Exception:
            missing.append(coin)

    needs_init = bool(missing) or total <= 0
    if needs_init:
        if missing:
            print(f"[run.py] state 파일 없음 {missing} → bot_manage.py init 자동 실행")
        else:
            print(f"[run.py] 총 자본 {total:.2f} USDT → bot_manage.py init 자동 실행")
        result = subprocess.run(
            [sys.executable, "live_tools/bot_manage.py", "init"],
            cwd=str(ROOT)
        )
        if result.returncode != 0:
            print("[run.py] ❌ init 실패 — 수동 확인 필요")
            sys.exit(1)
    else:
        print(f"[run.py] ✅ 실계좌 검증 통과 | 총 트래킹 자본: {total:,.2f} USDT")


def _validate_real_mode_startup():
    _auto_init_if_needed()


def _set_leverage_all_coins():
    """시작 시 전 종목 레버리지 거래소 등록. 성공/실패 supervisor.log에 기록."""
    from exchange_client import build_exchange, set_leverage as _set_lev
    leverage = int(os.environ.get("LEVERAGE", "5"))
    log.info(f"[startup] 레버리지 일괄 설정 시작: {leverage}x")
    try:
        exchange, _ = build_exchange("real")
    except Exception as e:
        log.warning(f"[startup] 거래소 연결 실패 — 레버리지 설정 스킵: {e}")
        return
    for coin in ["BTC", "ETH", "SOL", "XRP"]:
        os.environ["COIN"] = coin
        _set_lev(exchange, leverage)
    log.info(f"[startup] 레버리지 일괄 설정 완료")


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main():
    global _processes, _auto_restart

    # .env 우선 로드 (os.environ에 없는 값만 설정)
    _load_env()

    parser = argparse.ArgumentParser(description="TradeBot 통합 실행 + 대시보드")
    parser.add_argument("--port",            type=int, default=8765, help="대시보드 포트 (기본: 8765)")
    parser.add_argument("--no-auto-restart", action="store_true",   dest="no_restart",
                        help="자동 재시작 비활성화 (기본: 활성화)")
    args = parser.parse_args()

    trade_mode = os.environ.get("TRADE_MODE", "paper").lower()
    paper_mode = (trade_mode != "real")   # paper / sandbox 모두 paper_mode=True

    _auto_restart = not args.no_restart
    _processes    = _make_processes(paper_mode)

    # 로깅 (startup 이전에 초기화해야 exchange API 로그가 기록됨)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "supervisor.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    if not paper_mode:
        # state 파일 검증 + 없으면 자동 init
        _validate_real_mode_startup()
        # 전 종목 레버리지 거래소 등록
        _set_leverage_all_coins()

    _acquire_lock()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # 슈퍼바이저 스레드 시작
    t = threading.Thread(target=_supervisor_loop, daemon=True, name="supervisor")
    t.start()

    mode_str  = "PAPER" if paper_mode else "REAL"
    proc_list = " + ".join(p.name for p in _processes)
    print(f"\n{'='*55}")
    print(f"  TradeBot Dashboard — {mode_str} MODE")
    print(f"  프로세스: {proc_list}")
    print(f"  대시보드: http://127.0.0.1:{args.port}")
    print(f"  종료:     Ctrl+C")
    print(f"{'='*55}\n")

    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
