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

from flask import Flask, jsonify, request, Response


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
    watch_cmd  = [sys.executable, "live_tools/bot_manage.py", "watch", "--kill"]
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

app = Flask(__name__)
app.logger.disabled = True
log_flask = logging.getLogger("werkzeug")
log_flask.setLevel(logging.WARNING)


@app.route("/")
def index():
    return Response(_HTML, mimetype="text/html")


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
            "daily_halt":  s.get("daily_halt", False),
            "entry_price": round(s.get("entry_price", 0), 4),
            "trail_sl":    round(s.get("af_trail_sl", 0), 4),
            "last_price":  round(s.get("last_price", 0), 4),
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


# ─── 내장 HTML 대시보드 ───────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TradeBot Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:      #020617;
    --surface: #0d1526;
    --card:    #0d1526;
    --border:  #1e293b;
    --border2: #334155;
    --text:    #f1f5f9;
    --muted:   #94a3b8;
    --dim:     #475569;
    --accent:  #38bdf8;
    --up:      #10b981;
    --down:    #f43f5e;
    --warn:    #f59e0b;
  }
  *, *::before, *::after { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', 'Courier New', sans-serif; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.25rem; }
  .label { font-size: 0.68rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  .kpi-val { font-size: 1.4rem; font-weight: 700; line-height: 1.2; }
  .log-box { background: #000d1a; color: #22d3ee; font-size: 0.68rem; font-family: 'Courier New', monospace;
             height: 200px; overflow-y: auto; padding: 0.75rem;
             border-radius: 8px; white-space: pre-wrap; word-break: break-all; }
  .badge-up   { background: rgba(16,185,129,.15); color: #10b981; border: 1px solid rgba(16,185,129,.35); }
  .badge-down { background: rgba(244,63,94,.15);  color: #f43f5e; border: 1px solid rgba(244,63,94,.35); }
  .badge-halt { background: rgba(245,158,11,.15); color: #f59e0b; border: 1px solid rgba(245,158,11,.35); }
  .badge-loop { background: rgba(139,92,246,.15); color: #a78bfa; border: 1px solid rgba(139,92,246,.35); }
  .badge { padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 700; }
  .modal-wrap {
    position: fixed; inset: 0; z-index: 50;
    display: flex; align-items: center; justify-content: center; padding: 1rem;
    background: rgba(2,6,23,.9); backdrop-filter: blur(6px);
  }
  .modal-box {
    background: #0d1526; border: 1px solid var(--border2); border-radius: 18px; padding: 1.5rem;
    width: 100%; max-height: calc(100vh - 2rem); overflow-y: auto;
    box-shadow: 0 30px 80px rgba(0,0,0,.7);
  }
  #capitalChart { width: 100% !important; height: 420px !important; }
</style>
</head>
<body class="min-h-screen p-4 md:p-6">

<!-- 헤더 -->
<div class="flex flex-wrap justify-between items-start gap-4 mb-6 pb-5 border-b border-slate-800">
  <div>
    <h1 class="text-lg font-bold" style="color:var(--accent)">ConnectAI TradeBot</h1>
    <div id="mode-badge" class="text-xs mt-1" style="color:var(--muted)">로딩 중...</div>
  </div>
  <div class="flex items-start gap-6">
    <div class="text-right">
      <div class="label mb-1">총 자본</div>
      <div id="total-capital" class="text-2xl font-bold" style="color:var(--up)">-</div>
      <div id="daily-pnl" class="text-sm mt-0.5">-</div>
      <div id="last-update" class="text-xs mt-1" style="color:var(--dim)">-</div>
    </div>
    <div class="flex flex-col gap-2 pt-1">
      <button onclick="openModal('capital-modal')"
        class="px-4 py-2 rounded-lg text-xs font-semibold transition-colors"
        style="background:rgba(56,189,248,.12);color:var(--accent);border:1px solid rgba(56,189,248,.3)">
        📈 자본 추이
      </button>
      <button onclick="openModal('trades-modal')"
        class="px-4 py-2 rounded-lg text-xs font-semibold transition-colors"
        style="background:rgba(255,255,255,.05);color:var(--text);border:1px solid var(--border2)">
        📋 최근 거래
      </button>
    </div>
  </div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-12 gap-4">

  <!-- 프로세스 -->
  <div class="lg:col-span-4 card">
    <div class="label mb-3">프로세스</div>
    <div id="process-cards" class="space-y-2"><!-- JS --></div>
    <button onclick="emergencyClose()"
      class="w-full mt-3 py-2 rounded-lg font-bold text-sm transition-colors"
      style="background:rgba(244,63,94,.1);color:#f43f5e;border:1px solid rgba(244,63,94,.3)">
      🚨 Emergency Close All
    </button>
  </div>

  <!-- 코인 현황 -->
  <div class="lg:col-span-4 card">
    <div class="label mb-3">코인 현황</div>
    <div id="coin-cards" class="space-y-1"><!-- JS --></div>
  </div>

  <!-- KPI + 빠른 보기 -->
  <div class="lg:col-span-4 flex flex-col gap-4">
    <div class="card flex-1">
      <div class="label mb-4">포트폴리오 요약</div>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <div class="label mb-1">활성 포지션</div>
          <div id="kpi-positions" class="kpi-val" style="color:var(--accent)">-</div>
        </div>
        <div>
          <div class="label mb-1">레버리지</div>
          <div id="kpi-leverage" class="kpi-val" style="color:var(--warn)">-</div>
        </div>
      </div>
      <div class="mt-4 pt-3" style="border-top:1px solid var(--border)">
        <button onclick="openModal('capital-modal')"
          class="w-full py-2.5 rounded-xl text-sm font-semibold mb-2 transition-colors"
          style="background:rgba(56,189,248,.1);color:var(--accent);border:1px solid rgba(56,189,248,.25)">
          📈 자본 추이
        </button>
        <button onclick="openModal('trades-modal')"
          class="w-full py-2.5 rounded-xl text-sm font-semibold transition-colors"
          style="background:rgba(255,255,255,.05);color:var(--text);border:1px solid var(--border2)">
          📋 최근 거래
        </button>
      </div>
    </div>
  </div>

  <!-- 로그 (전체 너비) -->
  <div class="lg:col-span-12 card">
    <div class="flex justify-between items-center mb-3">
      <div class="label">로그</div>
      <div class="flex gap-3 items-center">
        <label style="font-size:.72rem;color:var(--muted);cursor:pointer">
          <input type="checkbox" id="auto-scroll" checked class="mr-1">자동 스크롤
        </label>
        <span style="font-size:.7rem;color:var(--dim)" id="log-update">-</span>
      </div>
    </div>
    <div class="log-box" id="log-box">로딩 중...</div>
  </div>

</div>

<!-- 자본 추이 모달 -->
<div id="capital-modal" class="modal-wrap hidden" role="dialog" aria-modal="true"
     onclick="if(event.target===this)closeModal('capital-modal')">
  <div class="modal-box" style="max-width:900px" onclick="event.stopPropagation()">
    <div class="flex items-center justify-between mb-5">
      <div class="font-semibold text-sm" style="color:var(--accent)">📈 자본 추이</div>
      <button onclick="closeModal('capital-modal')"
        class="text-xs px-3 py-1.5 rounded-lg transition-colors"
        style="background:var(--border);color:var(--muted)">✕ 닫기</button>
    </div>
    <div style="height:420px">
      <canvas id="capitalChart"></canvas>
    </div>
  </div>
</div>

<!-- 최근 거래 모달 -->
<div id="trades-modal" class="modal-wrap hidden" role="dialog" aria-modal="true"
     onclick="if(event.target===this)closeModal('trades-modal')">
  <div class="modal-box" style="max-width:1100px" onclick="event.stopPropagation()">
    <div class="flex items-center justify-between mb-5">
      <div class="font-semibold text-sm" style="color:var(--text)">📋 최근 거래</div>
      <button onclick="closeModal('trades-modal')"
        class="text-xs px-3 py-1.5 rounded-lg transition-colors"
        style="background:var(--border);color:var(--muted)">✕ 닫기</button>
    </div>
    <div id="trades-table" class="overflow-x-auto">로딩 중...</div>
  </div>
</div>

<script>
const COIN_COLORS = {BTC:'#f59e0b', ETH:'#6366f1', SOL:'#10b981', XRP:'#3b82f6'};
let chart = null;
let lastCapitalSeries = null;

function toKST(utcStr) {
  if (!utcStr || utcStr === '-') return '-';
  try {
    const d = new Date(utcStr.replace(' UTC','').replace(' ','T') + 'Z');
    return d.toLocaleString('ko-KR', {
      timeZone:'Asia/Seoul', month:'2-digit', day:'2-digit',
      hour:'2-digit', minute:'2-digit', hour12:false
    }) + ' KST';
  } catch(e) { return utcStr; }
}

// ─── 모달 열기/닫기 ─────────────────────────────────────────────────────
function openModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  if (id === 'capital-modal') {
    requestAnimationFrame(() => {
      if (lastCapitalSeries) renderChart(lastCapitalSeries);
      else updateChart();
    });
  }
  if (id === 'trades-modal') updateTrades();
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.classList.add('hidden');
  if (!document.querySelector('.modal-wrap:not(.hidden)'))
    document.body.style.overflow = '';
}

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  closeModal('capital-modal');
  closeModal('trades-modal');
});

// ─── Chart 초기화 ──────────────────────────────────────────────────────
function renderChart(series) {
  const ctx = document.getElementById('capitalChart').getContext('2d');
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: Object.entries(series)
        .filter(([,pts]) => pts.length > 1)
        .map(([coin, pts]) => ({
          label: coin,
          data: pts.map(p => ({ x: new Date(p.t.replace(' UTC','').replace(' ','T')+'Z'), y: p.v })),
          borderColor: COIN_COLORS[coin] || '#94a3b8',
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.3,
        }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color:'#94a3b8', font:{size:11} } } },
      scales: {
        x: { type:'time', time:{unit:'hour', displayFormats:{hour:'MM-dd HH:mm'}},
             ticks:{color:'#475569'}, grid:{color:'#1e293b'} },
        y: { ticks:{color:'#475569', callback:v=>'$'+v.toFixed(0)},
             grid:{color:'#1e293b'} }
      }
    }
  });
  chart.resize();
}

// ─── 상태 업데이트 ─────────────────────────────────────────────────────
function badgeClass(status) {
  if (status === 'running')    return 'badge badge-up';
  if (status === 'halted')     return 'badge badge-halt';
  if (status === 'crash_loop') return 'badge badge-loop';
  return 'badge badge-down';
}
function badgeText(status) {
  const m = {running:'UP', stopped:'DOWN', halted:'HALTED', crash_loop:'LOOP', starting:'...'};
  return m[status] || status.toUpperCase();
}

function renderProcesses(procs) {
  const el = document.getElementById('process-cards');
  el.innerHTML = procs.map(p => `
    <div class="flex items-center justify-between py-2 border-b border-slate-700 last:border-0">
      <div>
        <div class="flex items-center gap-2">
          <span class="${badgeClass(p.status)}">${badgeText(p.status)}</span>
          <span class="text-sm font-medium">${p.name}</span>
        </div>
        <div class="text-xs text-slate-500 mt-1">
          PID ${p.pid||'-'} · 가동 ${p.uptime} · 재시작 ${p.restart_count}회
        </div>
      </div>
      <div class="flex gap-1">
        <button onclick="action('start','${p.name}')"
          class="text-xs px-2 py-1 rounded bg-green-900 text-green-300 hover:bg-green-800">▶ 시작</button>
        <button onclick="action('stop','${p.name}')"
          class="text-xs px-2 py-1 rounded bg-red-900 text-red-300 hover:bg-red-800">■ 정지</button>
      </div>
    </div>`).join('');
}

function renderCoins(coins) {
  const el = document.getElementById('coin-cards');
  el.innerHTML = Object.entries(coins).map(([c,s]) => {
    const isLong  = s.position === 1;
    const isShort = s.position === -1;
    const posLabel = s.position === 0 ? '<span style="color:var(--dim)">대기</span>'
                   : isLong  ? '<span style="color:var(--up);font-weight:600">LONG ▲</span>'
                              : '<span style="color:var(--down);font-weight:600">SHORT ▼</span>';
    const halt = s.daily_halt ? '<span style="color:var(--down)" title="일일 한도 초과">⬤</span> ' : '';
    const detail = s.position !== 0 && s.entry_price > 0
      ? `<div style="font-size:.68rem;color:var(--dim);margin-top:3px;padding-left:2px">
           진입 <span style="color:var(--muted)">${s.entry_price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:4})}</span>
           &nbsp;·&nbsp;
           Trail <span style="color:var(--warn)">${s.trail_sl > 0 ? s.trail_sl.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:4}) : '-'}</span>
         </div>`
      : '';
    return `<div style="padding:.55rem 0;border-bottom:1px solid var(--border);last:border-0">
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:.85rem">
        <span style="color:${COIN_COLORS[c]||'#94a3b8'};font-weight:600">${halt}${c}</span>
        <span style="font-family:monospace;font-size:.8rem">$${s.capital.toFixed(2)}</span>
        <span style="font-size:.75rem">${posLabel}</span>
      </div>${detail}
    </div>`;
  }).join('');
}

async function updateStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const pnl = d.portfolio.daily_pnl_pct;
    const pnlColor = pnl >= 0 ? 'color:var(--up)' : 'color:var(--down)';
    document.getElementById('mode-badge').textContent =
      (d.mode === 'paper' ? '📝 PAPER' : '🔴 REAL') + ' · ' + (d.exchange || 'bybit').toUpperCase();
    document.getElementById('total-capital').textContent = '$' + d.portfolio.total_capital.toLocaleString();
    document.getElementById('daily-pnl').innerHTML =
      `<span style="${pnlColor};font-size:.85rem">일일 ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%</span>`;
    document.getElementById('last-update').textContent =
      new Date().toLocaleTimeString('ko-KR', {timeZone:'Asia/Seoul', hour12:false}) + ' KST';
    // KPI
    const activeCnt = Object.values(d.coins).filter(s => s.position !== 0).length;
    const kpiPos = document.getElementById('kpi-positions');
    if (kpiPos) kpiPos.textContent = activeCnt + ' / ' + Object.keys(d.coins).length;
    const kpiLev = document.getElementById('kpi-leverage');
    if (kpiLev) kpiLev.textContent = (d.leverage || '-') + (d.leverage ? 'x' : '');
    renderProcesses(d.processes);
    renderCoins(d.coins);
  } catch(e) { console.warn('status err', e); }
}

// ─── 자본 차트 업데이트 ─────────────────────────────────────────────────
async function updateChart() {
  try {
    const r = await fetch('/api/capital');
    const series = await r.json();
    lastCapitalSeries = series;
    if (!document.getElementById('capital-modal').classList.contains('hidden')) {
      renderChart(series);
    }
  } catch(e) { console.warn('chart err', e); }
}

// ─── 로그 업데이트 ─────────────────────────────────────────────────────
async function updateLogs() {
  try {
    const r  = await fetch('/api/logs?n=120');
    const d  = await r.json();
    const el = document.getElementById('log-box');
    el.textContent = d.lines.join('\\n');
    if (document.getElementById('auto-scroll').checked)
      el.scrollTop = el.scrollHeight;
    document.getElementById('log-update').textContent =
      new Date().toLocaleTimeString('ko-KR');
  } catch(e) { console.warn('log err', e); }
}

// ─── 프로세스 액션 ─────────────────────────────────────────────────────
async function action(act, name) {
  await fetch(`/api/action/${act}/${name}`, { method:'POST' });
  setTimeout(updateStatus, 1000);
}

// ─── 긴급 청산 ─────────────────────────────────────────────────────────
async function emergencyClose() {
  if (!confirm('모든 포지션을 즉시 청산하시겠습니까?')) return;
  const r = await fetch('/api/emergency_close', {method:'POST'});
  const d = await r.json();
  alert(d.message || '청산 요청 완료');
  setTimeout(updateStatus, 2000);
}

// ─── 최근 거래 업데이트 ─────────────────────────────────────────────────
async function updateTrades() {
  try {
    const r = await fetch('/api/trades');
    const d = await r.json();
    const el = document.getElementById('trades-table');
    if (!d.trades || d.trades.length === 0) {
      el.innerHTML = '<p class="text-slate-500 text-sm">거래 이력 없음</p>';
      return;
    }
    const rows = d.trades.map(t => {
      const pnl = t.pnl || 0;
      const pc = pnl >= 0 ? 'text-green-400' : 'text-red-400';
      const dir = t.direction === 1 ? '🟢 LONG' : '🔴 SHORT';
      const slip = t.slippage_exit_pct != null ? (t.slippage_exit_pct >= 0 ? '+' : '') + t.slippage_exit_pct.toFixed(3) + '%' : '-';
      const capAfter = t.capital || 0;
      const capBefore = (1 + pnl) !== 0 ? capAfter / (1 + pnl) : capAfter;
      const pnlUsdt = capAfter - capBefore;
      const pnlSign = pnl >= 0 ? '+' : '';
      return `<tr class="border-b border-slate-800">
        <td class="py-1 px-2 text-xs text-slate-400">${toKST(t.time||'-')}</td>
        <td class="py-1 px-2 text-xs" style="color:${COIN_COLORS[t.coin]||'#94a3b8'}">${t.coin}</td>
        <td class="py-1 px-2 text-xs">${dir}</td>
        <td class="py-1 px-2 text-xs font-mono">${(t.entry||0).toLocaleString()}</td>
        <td class="py-1 px-2 text-xs font-mono">${(t.exit||0).toLocaleString()}</td>
        <td class="py-1 px-2 text-xs font-mono ${pc}">${pnlSign}$${pnlUsdt.toFixed(2)}<span class="text-slate-500 ml-1">(${pnlSign}${(pnl*100).toFixed(2)}%)</span></td>
        <td class="py-1 px-2 text-xs text-slate-400">${slip}</td>
        <td class="py-1 px-2 text-xs text-slate-400">${t.reason||'-'}</td>
      </tr>`;
    }).join('');
    el.innerHTML = `<table class="w-full text-left"><thead><tr class="text-slate-500 text-xs uppercase">
      <th class="py-1 px-2">시각(KST)</th><th class="py-1 px-2">코인</th><th class="py-1 px-2">방향</th>
      <th class="py-1 px-2">진입가</th><th class="py-1 px-2">청산가</th>
      <th class="py-1 px-2">PnL(USDT/%)</th><th class="py-1 px-2">슬리피지</th><th class="py-1 px-2">이유</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
  } catch(e) { console.warn('trades err', e); }
}

// ─── 초기화 + 자동 갱신 ─────────────────────────────────────────────────
updateStatus();
updateLogs();
updateChart();
updateTrades();
setInterval(updateStatus, 5000);
setInterval(updateLogs,   5000);
setInterval(updateChart, 60000);
setInterval(updateTrades, 30000);
</script>
</body>
</html>"""

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
