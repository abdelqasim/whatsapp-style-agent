"""
Monitoring Dashboard
────────────────────
Real-time HTML dashboard served at /dashboard.
Shows service health, recent logs, conversation history, and pipeline stats.
"""

import json
import logging
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from src.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

LOG_DIR = Path("./data/conversation_logs")
MEMORY_DIR = Path("./data/conversation_logs/memory")

_log_buffer: deque[str] = deque(maxlen=100)


class _DashboardLogHandler(logging.Handler):
    def emit(self, record):
        try:
            _log_buffer.append(self.format(record))
        except Exception:
            pass


_handler = _DashboardLogHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s", datefmt="%H:%M:%S"))
logging.getLogger().addHandler(_handler)
logging.getLogger("uvicorn.access").addHandler(_handler)


def _load_todays_log() -> list[dict]:
    log_file = LOG_DIR / f"{date.today().isoformat()}.json"
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            return json.load(f)
    return []


def _get_all_senders() -> dict[str, list[dict]]:
    result = {}
    if MEMORY_DIR.exists():
        for f in sorted(MEMORY_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                result[f.stem] = data
            except (json.JSONDecodeError, OSError):
                pass
    return result


@router.get("/api/dashboard-data")
async def dashboard_data():
    services = {}
    for name, url in [
        ("backend", "http://localhost:8000/health"),
        ("n8n", "http://n8n:5678/healthz"),
        ("qdrant", "http://qdrant:6333/healthz"),
    ]:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(url)
                services[name] = "up" if r.status_code == 200 else "down"
        except Exception:
            services[name] = "down"

    log = _load_todays_log()

    intent_counts = defaultdict(int)
    latencies = []
    senders = set()
    ungrounded = 0
    for entry in log:
        intent_counts[entry.get("intent", "unknown")] += 1
        if entry.get("latency_ms"):
            latencies.append(entry["latency_ms"])
        senders.add(entry.get("sender", "unknown"))
        if not entry.get("grounded", True):
            ungrounded += 1

    memories = _get_all_senders()

    from src.followups import get_pending_count
    pending_followups = get_pending_count()

    log_lines = list(_log_buffer)

    return {
        "services": services,
        "stats": {
            "total_messages": len(log),
            "unique_senders": len(senders),
            "intent_counts": dict(intent_counts),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "ungrounded": ungrounded,
            "pending_followups": pending_followups,
        },
        "recent_interactions": log[-20:],
        "memories": {k: v[-6:] for k, v in memories.items()},
        "docker_logs": log_lines,
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhatsApp Agent — Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e1e4e8; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a1f36 0%, #0d1025 100%); padding: 20px 30px; border-bottom: 1px solid #2d3148; display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 1.4em; font-weight: 600; }
  .header h1 span { color: #25D366; }
  .header .status { font-size: 0.85em; color: #8b8fa3; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; padding: 20px; max-width: 1400px; margin: 0 auto; }
  .card { background: #161b22; border: 1px solid #2d3148; border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 0.95em; color: #8b8fa3; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 14px; }
  .full-width { grid-column: 1 / -1; }
  .services { display: flex; gap: 12px; flex-wrap: wrap; }
  .svc { padding: 10px 18px; border-radius: 8px; font-size: 0.9em; font-weight: 500; display: flex; align-items: center; gap: 8px; }
  .svc .dot { width: 10px; height: 10px; border-radius: 50%; }
  .svc.up { background: #0d2818; border: 1px solid #1a4d2e; }
  .svc.up .dot { background: #25D366; box-shadow: 0 0 6px #25D366; }
  .svc.down { background: #2d1216; border: 1px solid #5c2127; }
  .svc.down .dot { background: #f85149; box-shadow: 0 0 6px #f85149; }
  .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .stat { background: #1c2130; border-radius: 8px; padding: 14px; }
  .stat .val { font-size: 1.8em; font-weight: 700; color: #58a6ff; }
  .stat .label { font-size: 0.78em; color: #8b8fa3; margin-top: 2px; }
  .intents { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .intent-tag { padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 500; }
  .intent-tag.casual_chat { background: #1a3a5c; color: #79c0ff; }
  .intent-tag.knowledge_query { background: #1a3c2a; color: #56d364; }
  .intent-tag.scheduling { background: #3d2b1a; color: #f0b95c; }
  .intent-tag.default { background: #2d2d3d; color: #b3b8d0; }
  .log-box { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px; max-height: 400px; overflow-y: auto; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.78em; line-height: 1.7; }
  .log-line { padding: 2px 0; border-bottom: 1px solid #161b22; word-break: break-all; }
  .log-line.error { color: #f85149; }
  .log-line.info { color: #8b949e; }
  .log-line.intent { color: #58a6ff; }
  .log-line.style { color: #d2a8ff; }
  .log-line.rag { color: #56d364; }
  .msg-list { max-height: 400px; overflow-y: auto; }
  .msg { padding: 10px; margin-bottom: 8px; border-radius: 8px; background: #1c2130; }
  .msg .meta { font-size: 0.75em; color: #8b8fa3; margin-bottom: 4px; display: flex; gap: 10px; }
  .msg .meta .intent-badge { padding: 1px 8px; border-radius: 10px; font-size: 0.85em; }
  .msg .incoming { color: #c9d1d9; margin-bottom: 4px; }
  .msg .reply { color: #56d364; font-size: 0.9em; }
  .memory-section { margin-bottom: 14px; }
  .memory-section h3 { font-size: 0.85em; color: #58a6ff; margin-bottom: 6px; }
  .memory-msg { padding: 4px 8px; font-size: 0.82em; border-left: 2px solid #2d3148; margin-bottom: 3px; }
  .memory-msg.user { border-left-color: #58a6ff; color: #c9d1d9; }
  .memory-msg.assistant { border-left-color: #25D366; color: #8b949e; }
  .refresh-bar { text-align: center; padding: 8px; font-size: 0.8em; color: #484f58; }
  .auto-tag { background: #1a3a5c; color: #58a6ff; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; cursor: pointer; }
</style>
</head>
<body>

<div class="header">
  <h1><span>WhatsApp</span> Agent Dashboard</h1>
  <div class="status">
    <span class="auto-tag" onclick="toggleAuto()">Auto-refresh: <span id="autoLabel">ON</span></span>
    &nbsp; Last update: <span id="lastUpdate">—</span>
  </div>
</div>

<div class="grid" id="content">
  <div class="card" id="servicesCard"><h2>Services</h2><div class="services" id="services">Loading...</div></div>
  <div class="card" id="statsCard"><h2>Today's Stats</h2><div id="stats">Loading...</div></div>
  <div class="card" id="interactionsCard"><h2>Recent Interactions</h2><div class="msg-list" id="interactions">Loading...</div></div>
  <div class="card" id="memoryCard"><h2>Conversation Memory</h2><div id="memories">Loading...</div></div>
  <div class="card full-width" id="logsCard"><h2>Backend Logs</h2><div class="log-box" id="logs">Loading...</div></div>
</div>

<div class="refresh-bar">Refreshes every 5 seconds</div>

<script>
let autoRefresh = true;
let timer;

function toggleAuto() {
  autoRefresh = !autoRefresh;
  document.getElementById('autoLabel').textContent = autoRefresh ? 'ON' : 'OFF';
  if (autoRefresh) startTimer(); else clearInterval(timer);
}

function startTimer() { clearInterval(timer); timer = setInterval(refresh, 5000); }

function classifyLog(line) {
  if (line.includes('ERROR')) return 'error';
  if (line.includes('[intent]')) return 'intent';
  if (line.includes('[style]')) return 'style';
  if (line.includes('[rag]')) return 'rag';
  return 'info';
}

function intentColor(intent) {
  if (intent === 'casual_chat') return 'casual_chat';
  if (intent === 'knowledge_query') return 'knowledge_query';
  if (intent === 'scheduling') return 'scheduling';
  return 'default';
}

async function refresh() {
  try {
    const r = await fetch('/api/dashboard-data');
    const d = await r.json();

    // Services
    let svcs = '';
    for (const [name, status] of Object.entries(d.services)) {
      svcs += `<div class="svc ${status}"><div class="dot"></div>${name}</div>`;
    }
    document.getElementById('services').innerHTML = svcs;

    // Stats
    const s = d.stats;
    let intents = '';
    for (const [k, v] of Object.entries(s.intent_counts)) {
      intents += `<span class="intent-tag ${intentColor(k)}">${k}: ${v}</span>`;
    }
    document.getElementById('stats').innerHTML = `
      <div class="stat-grid">
        <div class="stat"><div class="val">${s.total_messages}</div><div class="label">Messages Today</div></div>
        <div class="stat"><div class="val">${s.unique_senders}</div><div class="label">Unique Senders</div></div>
        <div class="stat"><div class="val">${s.avg_latency_ms}ms</div><div class="label">Avg Latency</div></div>
        <div class="stat"><div class="val">${s.pending_followups || 0}</div><div class="label">Pending Follow-ups</div></div>
      </div>
      <div class="intents">${intents || '<span style="color:#484f58">No intents yet</span>'}</div>
    `;

    // Interactions
    let msgs = '';
    for (const m of (d.recent_interactions || []).reverse()) {
      msgs += `<div class="msg">
        <div class="meta">
          <span>${(m.timestamp||'').substring(11,16)}</span>
          <span>From: ${m.sender}</span>
          <span class="intent-badge intent-tag ${intentColor(m.intent)}">${m.intent}</span>
          ${m.grounded ? '' : '<span style="color:#f85149">ungrounded</span>'}
        </div>
        <div class="incoming">${escHtml(m.incoming || '')}</div>
        <div class="reply">${escHtml(m.reply || '')}</div>
      </div>`;
    }
    document.getElementById('interactions').innerHTML = msgs || '<div style="color:#484f58">No interactions yet today</div>';

    // Memory
    let mem = '';
    for (const [sender, turns] of Object.entries(d.memories || {})) {
      mem += `<div class="memory-section"><h3>${sender}</h3>`;
      for (const t of turns) {
        mem += `<div class="memory-msg ${t.role}">${t.role === 'user' ? 'User' : 'Bot'}: ${escHtml(t.content || '')}</div>`;
      }
      mem += '</div>';
    }
    document.getElementById('memories').innerHTML = mem || '<div style="color:#484f58">No memory data</div>';

    // Logs
    let logs = '';
    for (const line of d.docker_logs || []) {
      logs += `<div class="log-line ${classifyLog(line)}">${escHtml(line)}</div>`;
    }
    document.getElementById('logs').innerHTML = logs;
    const logBox = document.getElementById('logs');
    logBox.scrollTop = logBox.scrollHeight;

    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
  } catch(e) {
    console.error('Dashboard refresh failed:', e);
  }
}

function escHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

refresh();
startTimer();
</script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML
