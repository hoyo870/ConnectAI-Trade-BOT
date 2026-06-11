const COIN_COLORS = { BTC: '#f59e0b', ETH: '#6366f1', SOL: '#10b981', XRP: '#3b82f6' };
let chart = null;
let lastCapitalSeries = null;

// ─── 유틸 ──────────────────────────────────────────────────────────────
function toKST(utcStr) {
  if (!utcStr || utcStr === '-') return '-';
  try {
    const d = new Date(utcStr.replace(' UTC', '').replace(' ', 'T') + 'Z');
    return d.toLocaleString('ko-KR', {
      timeZone: 'Asia/Seoul', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hour12: false
    }) + ' KST';
  } catch (e) { return utcStr; }
}

// ─── 모달 열기/닫기 ────────────────────────────────────────────────────
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

// ─── 자본 차트 ─────────────────────────────────────────────────────────
function renderChart(series) {
  const ctx = document.getElementById('capitalChart').getContext('2d');
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: Object.entries(series)
        .filter(([, pts]) => pts.length > 1)
        .map(([coin, pts]) => ({
          label: coin,
          data: pts.map(p => ({ x: new Date(p.t.replace(' UTC', '').replace(' ', 'T') + 'Z'), y: p.v })),
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
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
      scales: {
        x: {
          type: 'time', time: { unit: 'hour', displayFormats: { hour: 'MM-dd HH:mm' } },
          ticks: { color: '#475569' }, grid: { color: '#1e293b' }
        },
        y: {
          ticks: { color: '#475569', callback: v => '$' + v.toFixed(0) },
          grid: { color: '#1e293b' }
        }
      }
    }
  });
  chart.resize();
}

// ─── 상태 렌더링 ───────────────────────────────────────────────────────
function badgeClass(status) {
  if (status === 'running')    return 'badge badge-up';
  if (status === 'halted')     return 'badge badge-halt';
  if (status === 'crash_loop') return 'badge badge-loop';
  return 'badge badge-down';
}
function badgeText(status) {
  const m = { running: 'UP', stopped: 'DOWN', halted: 'HALTED', crash_loop: 'LOOP', starting: '...' };
  return m[status] || status.toUpperCase();
}

function renderProcesses(procs) {
  const el = document.getElementById('process-cards');
  el.innerHTML = procs.map(p => `
    <div class="flex items-center justify-between py-2 border-b last:border-0" style="border-color:var(--border)">
      <div>
        <div class="flex items-center gap-2">
          <span class="${badgeClass(p.status)}">${badgeText(p.status)}</span>
          <span style="font-size:.85rem;font-weight:500">${p.name}</span>
        </div>
        <div style="font-size:.7rem;color:var(--dim);margin-top:3px">
          PID ${p.pid || '-'} · 가동 ${p.uptime} · 재시작 ${p.restart_count}회
        </div>
      </div>
      <div class="flex gap-1">
        <button onclick="action('start','${p.name}')"
          style="font-size:.7rem;padding:3px 8px;border-radius:6px;background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3)">▶ 시작</button>
        <button onclick="action('stop','${p.name}')"
          style="font-size:.7rem;padding:3px 8px;border-radius:6px;background:rgba(244,63,94,.1);color:#f43f5e;border:1px solid rgba(244,63,94,.3)">■ 정지</button>
      </div>
    </div>`).join('');
}

function renderCoins(coins) {
  const el = document.getElementById('coin-cards');
  el.innerHTML = Object.entries(coins).map(([c, s]) => {
    const posLabel = s.position === 0
      ? `<span style="color:var(--dim)">대기</span>`
      : s.position === 1
        ? `<span style="color:var(--up);font-weight:600">LONG ▲</span>`
        : `<span style="color:var(--down);font-weight:600">SHORT ▼</span>`;
    const halt = s.daily_halt ? '<span style="color:var(--down)" title="일일 한도 초과">⬤</span> ' : '';
    const detail = s.position !== 0 && s.entry_price > 0
      ? `<div style="font-size:.68rem;color:var(--dim);margin-top:3px;padding-left:2px">
           진입 <span style="color:var(--muted)">${s.entry_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</span>
           &nbsp;·&nbsp;
           Trail <span style="color:var(--warn)">${s.trail_sl > 0 ? s.trail_sl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }) : '-'}</span>
         </div>`
      : '';
    return `<div style="padding:.55rem 0;border-bottom:1px solid var(--border)">
      <div style="display:flex;justify-content:space-between;align-items:center;font-size:.85rem">
        <span style="color:${COIN_COLORS[c] || '#94a3b8'};font-weight:600">${halt}${c}</span>
        <span style="font-family:monospace;font-size:.8rem">$${s.capital.toFixed(2)}</span>
        <span style="font-size:.75rem">${posLabel}</span>
      </div>${detail}
    </div>`;
  }).join('');
}

// ─── API 업데이트 ──────────────────────────────────────────────────────
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
      new Date().toLocaleTimeString('ko-KR', { timeZone: 'Asia/Seoul', hour12: false }) + ' KST';
    const activeCnt = Object.values(d.coins).filter(s => s.position !== 0).length;
    const kpiPos = document.getElementById('kpi-positions');
    if (kpiPos) kpiPos.textContent = activeCnt + ' / ' + Object.keys(d.coins).length;
    const kpiLev = document.getElementById('kpi-leverage');
    if (kpiLev) kpiLev.textContent = (d.leverage || '-') + (d.leverage ? 'x' : '');
    renderProcesses(d.processes);
    renderCoins(d.coins);
  } catch (e) { console.warn('status err', e); }
}

async function updateChart() {
  try {
    const r = await fetch('/api/capital');
    const series = await r.json();
    lastCapitalSeries = series;
    if (!document.getElementById('capital-modal').classList.contains('hidden'))
      renderChart(series);
  } catch (e) { console.warn('chart err', e); }
}

async function updateLogs() {
  try {
    const r = await fetch('/api/logs?n=120');
    const d = await r.json();
    const el = document.getElementById('log-box');
    el.textContent = d.lines.join('\n');
    if (document.getElementById('auto-scroll').checked)
      el.scrollTop = el.scrollHeight;
    document.getElementById('log-update').textContent =
      new Date().toLocaleTimeString('ko-KR');
  } catch (e) { console.warn('log err', e); }
}

async function action(act, name) {
  await fetch(`/api/action/${act}/${name}`, { method: 'POST' });
  setTimeout(updateStatus, 1000);
}

async function emergencyClose() {
  if (!confirm('모든 포지션을 즉시 청산하시겠습니까?')) return;
  const r = await fetch('/api/emergency_close', { method: 'POST' });
  const d = await r.json();
  alert(d.message || '청산 요청 완료');
  setTimeout(updateStatus, 2000);
}

async function updateTrades() {
  try {
    const r = await fetch('/api/trades');
    const d = await r.json();
    const el = document.getElementById('trades-table');
    if (!d.trades || d.trades.length === 0) {
      el.innerHTML = '<p style="color:var(--muted);font-size:.85rem;padding:.5rem 0">거래 이력 없음</p>';
      return;
    }
    const rows = d.trades.map(t => {
      const pnl = t.pnl || 0;
      const pnlColor = pnl >= 0 ? 'color:var(--up)' : 'color:var(--down)';
      const dir = t.direction === 1 ? '🟢 LONG' : '🔴 SHORT';
      const slip = t.slippage_exit_pct != null
        ? (t.slippage_exit_pct >= 0 ? '+' : '') + t.slippage_exit_pct.toFixed(3) + '%'
        : '-';
      const capAfter = t.capital || 0;
      const capBefore = (1 + pnl) !== 0 ? capAfter / (1 + pnl) : capAfter;
      const pnlUsdt = capAfter - capBefore;
      const pnlSign = pnl >= 0 ? '+' : '';
      return `<tr style="border-bottom:1px solid var(--border)">
        <td style="padding:6px 8px;font-size:.72rem;color:var(--muted)">${toKST(t.time || '-')}</td>
        <td style="padding:6px 8px;font-size:.72rem;color:${COIN_COLORS[t.coin] || '#94a3b8'};font-weight:600">${t.coin}</td>
        <td style="padding:6px 8px;font-size:.72rem">${dir}</td>
        <td style="padding:6px 8px;font-size:.72rem;font-family:monospace">${(t.entry || 0).toLocaleString()}</td>
        <td style="padding:6px 8px;font-size:.72rem;font-family:monospace">${(t.exit || 0).toLocaleString()}</td>
        <td style="padding:6px 8px;font-size:.72rem;font-family:monospace;${pnlColor}">${pnlSign}$${pnlUsdt.toFixed(2)}<span style="color:var(--dim);margin-left:4px">(${pnlSign}${(pnl * 100).toFixed(2)}%)</span></td>
        <td style="padding:6px 8px;font-size:.72rem;color:var(--dim)">${slip}</td>
        <td style="padding:6px 8px;font-size:.72rem;color:var(--dim)">${t.reason || '-'}</td>
      </tr>`;
    }).join('');
    el.innerHTML = `<table style="width:100%;border-collapse:collapse;text-align:left">
      <thead><tr style="color:var(--dim);font-size:.65rem;text-transform:uppercase;letter-spacing:.06em">
        <th style="padding:6px 8px">시각(KST)</th><th style="padding:6px 8px">코인</th>
        <th style="padding:6px 8px">방향</th><th style="padding:6px 8px">진입가</th>
        <th style="padding:6px 8px">청산가</th><th style="padding:6px 8px">PnL</th>
        <th style="padding:6px 8px">슬리피지</th><th style="padding:6px 8px">이유</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  } catch (e) { console.warn('trades err', e); }
}

// ─── 초기화 ────────────────────────────────────────────────────────────
updateStatus();
updateLogs();
updateChart();
updateTrades();
setInterval(updateStatus,  5000);
setInterval(updateLogs,    5000);
setInterval(updateChart,  60000);
setInterval(updateTrades, 30000);
