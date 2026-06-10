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
        "EXCHANGE":   os.environ.get("EXCHANGE", "bingx"),
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
            "capital":  round(s.get("capital", 0), 2),
            "position": s.get("position", 0),
            "daily_halt": s.get("daily_halt", False),
        }
        for coin, s in states.items()
    }
    return jsonify({
        "processes": procs,
        "portfolio": summary,
        "coins":     coins,
        "mode":      "paper" if _paper_mode else "real",
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
<style>
  body { background: #0f172a; color: #e2e8f0; font-family: 'Courier New', monospace; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; }
  .log-box { background: #020617; color: #4ade80; font-size: 0.72rem;
             height: 260px; overflow-y: auto; padding: 0.75rem;
             border-radius: 8px; white-space: pre-wrap; word-break: break-all; }
  .badge-up   { background:#166534; color:#4ade80; }
  .badge-down { background:#7f1d1d; color:#f87171; }
  .badge-halt { background:#78350f; color:#fbbf24; }
  .badge-loop { background:#581c87; color:#d8b4fe; }
  .badge { padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; }
  canvas { max-height: 260px; }
</style>
</head>
<body class="min-h-screen p-4">

<!-- 헤더 -->
<div class="flex justify-between items-center mb-5 pb-4 border-b border-slate-700">
  <div>
    <h1 class="text-xl font-bold text-blue-400">ConnectAI TradeBot</h1>
    <div id="mode-badge" class="text-xs text-slate-400 mt-1">로딩 중...</div>
  </div>
  <div class="text-right">
    <div class="text-slate-400 text-xs mb-1">총 자본</div>
    <div id="total-capital" class="text-2xl font-bold text-green-400">-</div>
    <div id="daily-pnl" class="text-sm mt-0.5">-</div>
    <div id="last-update" class="text-slate-500 text-xs mt-1">-</div>
  </div>
</div>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-4">

  <!-- 왼쪽: 프로세스 + 코인 상태 -->
  <div class="space-y-4">
    <div class="card">
      <div class="text-slate-400 text-xs font-bold mb-3 uppercase tracking-wide">프로세스</div>
      <div id="process-cards" class="space-y-3"><!-- JS --></div>
      <button onclick="emergencyClose()"
        class="w-full mt-3 py-2 rounded-lg bg-red-800 hover:bg-red-700 text-red-200 font-bold text-sm border border-red-600">
        🚨 Emergency Close All
      </button>
    </div>
    <div class="card">
      <div class="text-slate-400 text-xs font-bold mb-3 uppercase tracking-wide">코인 현황</div>
      <div id="coin-cards" class="space-y-2"><!-- JS --></div>
    </div>
  </div>

  <!-- 오른쪽: 자본 차트 -->
  <div class="lg:col-span-2 card">
    <div class="text-slate-400 text-xs font-bold mb-3 uppercase tracking-wide">자본 추이</div>
    <canvas id="capitalChart"></canvas>
  </div>

  <!-- 로그 (전체 너비) -->
  <div class="lg:col-span-3 card">
    <div class="flex justify-between items-center mb-3">
      <div class="text-slate-400 text-xs font-bold uppercase tracking-wide">로그</div>
      <div class="flex gap-2 items-center">
        <label class="text-slate-500 text-xs">
          <input type="checkbox" id="auto-scroll" checked class="mr-1">자동 스크롤
        </label>
        <span class="text-slate-600 text-xs" id="log-update">-</span>
      </div>
    </div>
    <div class="log-box" id="log-box">로딩 중...</div>
  </div>

  <!-- 최근 거래 (전체 너비) -->
  <div class="lg:col-span-3 card">
    <div class="text-slate-400 text-xs font-bold mb-3 uppercase tracking-wide">최근 거래</div>
    <div id="trades-table" class="overflow-x-auto">로딩 중...</div>
  </div>

</div>

<script>
const COIN_COLORS = {BTC:'#f59e0b', ETH:'#6366f1', SOL:'#10b981', XRP:'#3b82f6'};
let chart = null;

// ─── Chart 초기화 ──────────────────────────────────────────────────────
function initChart() {
  const ctx = document.getElementById('capitalChart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: { datasets: [] },
    options: {
      responsive: true,
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
    const posStr = s.position === 0 ? '없음' : (s.position === 1 ? 'LONG 🟢' : 'SHORT 🔴');
    const dot = s.daily_halt ? '🔴' : '';
    return `<div class="flex justify-between text-sm py-1 border-b border-slate-800 last:border-0">
      <span style="color:${COIN_COLORS[c]||'#94a3b8'}">${c} ${dot}</span>
      <span>$${s.capital.toFixed(0)}</span>
      <span class="text-slate-400 text-xs">${posStr}</span>
    </div>`;
  }).join('');
}

async function updateStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    const pnl = d.portfolio.daily_pnl_pct;
    const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
    document.getElementById('mode-badge').textContent =
      (d.mode === 'paper' ? '📝 PAPER MODE' : '🔴 REAL MODE') + '  |  BingX';
    document.getElementById('total-capital').textContent = '$' + d.portfolio.total_capital.toLocaleString();
    document.getElementById('daily-pnl').innerHTML =
      `<span class="${pnlColor}">일일 ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%</span>`;
    document.getElementById('last-update').textContent = d.ts;
    renderProcesses(d.processes);
    renderCoins(d.coins);
  } catch(e) { console.warn('status err', e); }
}

// ─── 자본 차트 업데이트 ─────────────────────────────────────────────────
async function updateChart() {
  try {
    const r = await fetch('/api/capital');
    const series = await r.json();
    if (!chart) return;
    chart.data.datasets = Object.entries(series)
      .filter(([,pts]) => pts.length > 1)
      .map(([coin, pts]) => ({
        label: coin,
        data: pts.map(p => ({ x: new Date(p.t.replace(' UTC','')), y: p.v })),
        borderColor: COIN_COLORS[coin] || '#94a3b8',
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.3,
      }));
    chart.update('none');
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
      return `<tr class="border-b border-slate-800">
        <td class="py-1 px-2 text-xs text-slate-400">${t.time||'-'}</td>
        <td class="py-1 px-2 text-xs" style="color:${COIN_COLORS[t.coin]||'#94a3b8'}">${t.coin}</td>
        <td class="py-1 px-2 text-xs">${dir}</td>
        <td class="py-1 px-2 text-xs font-mono">${(t.entry||0).toLocaleString()}</td>
        <td class="py-1 px-2 text-xs font-mono">${(t.exit||0).toLocaleString()}</td>
        <td class="py-1 px-2 text-xs font-mono ${pc}">${(pnl*100).toFixed(2)}%</td>
        <td class="py-1 px-2 text-xs text-slate-400">${slip}</td>
        <td class="py-1 px-2 text-xs text-slate-400">${t.reason||'-'}</td>
      </tr>`;
    }).join('');
    el.innerHTML = `<table class="w-full text-left"><thead><tr class="text-slate-500 text-xs uppercase">
      <th class="py-1 px-2">시각</th><th class="py-1 px-2">코인</th><th class="py-1 px-2">방향</th>
      <th class="py-1 px-2">진입가</th><th class="py-1 px-2">청산가</th>
      <th class="py-1 px-2">PnL</th><th class="py-1 px-2">슬리피지</th><th class="py-1 px-2">이유</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
  } catch(e) { console.warn('trades err', e); }
}

// ─── 초기화 + 자동 갱신 ─────────────────────────────────────────────────
initChart();
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

def _validate_real_mode_startup():
    import json
    from pathlib import Path
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
    if missing:
        print(f"\n[run.py] ❌ state 파일 없음: {missing}")
        print("  먼저 실행: python live_tools/bot_manage.py init")
        sys.exit(1)
    if total < 100:
        print(f"\n[run.py] ❌ 총 자본 이상: {total:.0f} USDT (최소 100 필요)")
        print("  먼저 실행: python live_tools/bot_manage.py init")
        sys.exit(1)
    print(f"[run.py] ✅ 실계좌 검증 통과 | 총 트래킹 자본: {total:,.0f} USDT")


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

    if not paper_mode:
        # state 파일 검증
        _validate_real_mode_startup()

    # 로깅
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "supervisor.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )

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
