// Prefilter AI Platform — Dashboard JavaScript

const API = '';  // relative to current host

// ── State ────────────────────────────────────────────────────────────
let currentResult = null;
let sessionId = generateSessionId();
let activeTab = 'ir';
const SESSION_CONV = [
  'gaming laptops',
  'Only Lenovo',
  'Actually 32GB RAM',
  'Under $600',
];

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('sessionIdDisplay').textContent = sessionId;
  checkHealth();
  setInterval(checkHealth, 15000);

  document.getElementById('queryInput').addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runPipeline();
  });
});

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(API + '/health');
    const ok = r.ok;
    document.getElementById('healthDot').className = 'health-indicator ' + (ok ? 'ok' : 'error');
    document.getElementById('healthText').textContent = ok ? 'API Online' : 'API Error';
  } catch {
    document.getElementById('healthDot').className = 'health-indicator error';
    document.getElementById('healthText').textContent = 'Offline';
  }
}

// ── View switching ────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + name).classList.remove('hidden');
  document.getElementById('nav-' + name).classList.add('active');
}

// ── Tab switching ─────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tc => { tc.classList.remove('active'); tc.classList.add('hidden'); });
  document.getElementById('tab-' + name).classList.remove('hidden');
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  activeTab = name;
}

// ── Pipeline ──────────────────────────────────────────────────────────
async function runPipeline() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) return;

  const parser = document.getElementById('parserSelect').value;
  const btn = document.getElementById('runBtn');
  btn.classList.add('loading');
  btn.textContent = '⏳ Processing...';

  // Reset stage pills
  animateStages();

  try {
    const res = await fetch(API + '/v1/parse', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query, parser }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Request failed');
    }

    currentResult = await res.json();
    renderOutput(currentResult);
  } catch (e) {
    alert('Pipeline error: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = '<span class="run-btn-icon">▶</span> Run Pipeline';
  }
}

function animateStages() {
  const pills = ['sp-parse','sp-ontology','sp-validate','sp-relax','sp-translate'];
  pills.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active','done');
  });

  let i = 0;
  const interval = setInterval(() => {
    if (i > 0) document.getElementById(pills[i-1]).classList.replace('active','done');
    if (i < pills.length) {
      document.getElementById(pills[i]).classList.add('active');
      i++;
    } else {
      clearInterval(interval);
    }
  }, 200);
}

function renderOutput(data) {
  document.getElementById('emptyState').classList.add('hidden');
  document.getElementById('outputContent').classList.remove('hidden');

  // Conflicts
  const conflictBanner = document.getElementById('conflictBanner');
  if (data.conflicts && data.conflicts.length > 0) {
    conflictBanner.classList.remove('hidden');
    document.getElementById('conflictMessages').innerHTML =
      data.conflicts.map(c => `<div>• ${escapeHtml(c)}</div>`).join('') +
      (data.warnings || []).slice(0, 2).map(w => `<div style="margin-top:4px;color:#aaa">↳ ${escapeHtml(w)}</div>`).join('');
  } else {
    conflictBanner.classList.add('hidden');
  }

  // Meta chips
  document.getElementById('domainChip').textContent = '📂 ' + (data.domain || 'general');
  document.getElementById('filterCount').textContent = `${(data.filters||[]).length} filters`;
  document.getElementById('prefCount').textContent = `${(data.preferences||[]).length} prefs`;
  document.getElementById('latencyChip').textContent = `⚡ ${(data.total_latency_ms||0).toFixed(1)}ms`;

  // Stage pills — mark all done
  setTimeout(() => {
    ['sp-parse','sp-ontology','sp-validate','sp-relax','sp-translate'].forEach(id => {
      const el = document.getElementById(id);
      el.classList.remove('active');
      el.classList.add('done');
    });
  }, 1200);

  // IR tab
  const irDisplay = {
    domain: data.domain,
    intent: data.intent,
    filters: data.filters,
    preferences: data.preferences,
  };
  document.getElementById('irOutput').textContent = JSON.stringify(irDisplay, null, 2);

  // SQL
  document.getElementById('sqlOutput').textContent = data.sql
    ? `-- SQL WHERE clause\nWHERE ${data.sql}`
    : '-- No SQL output';

  // MongoDB
  document.getElementById('mongoOutput').textContent =
    JSON.stringify(data.mongodb || {}, null, 2);

  // Elasticsearch
  document.getElementById('esOutput').textContent =
    JSON.stringify(data.elasticsearch || {}, null, 2);

  // ChromaDB
  document.getElementById('chromaOutput').textContent =
    JSON.stringify(data.chromadb || {}, null, 2);

  // Explanation
  const explainEl = document.getElementById('explainOutput');
  explainEl.innerHTML = '';
  const exp = data.explanation || {};
  if (Object.keys(exp).length === 0) {
    explainEl.innerHTML = '<p style="color:var(--text-dim);font-size:12px;">No explanation data.</p>';
  } else {
    Object.entries(exp).forEach(([field, text]) => {
      const div = document.createElement('div');
      div.className = 'explain-item';
      div.innerHTML = `<div class="explain-field">${escapeHtml(field)}</div><div class="explain-text">${escapeHtml(text)}</div>`;
      explainEl.appendChild(div);
    });
  }

  // Relaxed IR
  if (data.relaxed) {
    document.getElementById('relaxedOutput').textContent =
      JSON.stringify(data.relaxed, null, 2);
  } else {
    document.getElementById('relaxedOutput').textContent = '// No conflicts detected — relaxation not triggered.';
  }
}

// ── Session ───────────────────────────────────────────────────────────
async function sendSession() {
  const input = document.getElementById('sessionInput');
  const query = input.value.trim();
  if (!query) return;

  const parser = document.getElementById('parserSelect').value;
  input.value = '';

  try {
    const res = await fetch(API + '/v1/session/' + sessionId, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query, parser }),
    });
    const data = await res.json();
    appendTurnCard(data);
  } catch (e) {
    alert('Session error: ' + e.message);
  }
}

function sessionConv(idx) {
  document.getElementById('sessionInput').value = SESSION_CONV[idx] || '';
  sendSession();
}

function appendTurnCard(data) {
  const history = document.getElementById('sessionHistory');
  const turn = data.turn || history.children.length + 1;
  const card = document.createElement('div');
  card.className = 'turn-card';

  const filters = (data.filters || []).map(f => {
    const op = f.operator === 'eq' ? '' : `${f.operator}:`;
    return `<span class="filter-tag">${f.field}: ${op}${f.value}</span>`;
  }).join('');

  const conflictHtml = (data.conflicts || []).length > 0
    ? `<div class="turn-conflict">⚠ ${escapeHtml(data.conflicts[0])}</div>`
    : '';

  card.innerHTML = `
    <div class="turn-header">
      <span class="turn-num">Turn ${turn}</span>
      <span class="turn-query">"${escapeHtml(data.query)}"</span>
      <span class="turn-domain">${data.domain || 'general'}</span>
    </div>
    <div class="turn-filters">${filters || '<span style="color:var(--text-dim);font-size:11px;">No filters extracted</span>'}</div>
    ${conflictHtml}
  `;
  history.appendChild(card);
  history.scrollTop = history.scrollHeight;
}

async function resetSession() {
  try {
    await fetch(API + '/v1/session/' + sessionId, { method: 'DELETE' });
  } catch {}
  sessionId = generateSessionId();
  document.getElementById('sessionIdDisplay').textContent = sessionId;
  document.getElementById('sessionHistory').innerHTML = '';
}

// ── Analytics ─────────────────────────────────────────────────────────
async function loadAnalytics() {
  try {
    const res = await fetch(API + '/v1/analytics');
    const data = await res.json();

    document.getElementById('stat-total').textContent = data.total_queries;
    document.getElementById('stat-conflict').textContent = (data.conflict_rate * 100).toFixed(0) + '%';
    document.getElementById('stat-relax').textContent = (data.relaxation_rate * 100).toFixed(0) + '%';
    document.getElementById('stat-latency').textContent = data.avg_latency_ms.toFixed(1) + 'ms';
    document.getElementById('stat-sessions').textContent = data.active_sessions;

    // Domain distribution
    const dist = document.getElementById('domainDistribution');
    dist.innerHTML = '';
    const entries = Object.entries(data.domain_distribution || {}).sort((a,b) => b[1]-a[1]);
    const max = Math.max(...entries.map(e => e[1]), 1);
    entries.forEach(([domain, count]) => {
      const pct = Math.round((count / max) * 100);
      dist.innerHTML += `
        <div class="domain-bar-row">
          <span class="domain-bar-label">${domain}</span>
          <div class="domain-bar-track"><div class="domain-bar-fill" style="width:${pct}%"></div></div>
          <span class="domain-bar-count">${count}</span>
        </div>`;
    });
    if (entries.length === 0) {
      dist.innerHTML = '<p style="color:var(--text-dim);font-size:12px;padding:8px 0;">No queries yet. Run some queries first.</p>';
    }
  } catch (e) {
    console.error('Analytics error:', e);
  }
}

// ── Domains ───────────────────────────────────────────────────────────
async function loadDomains() {
  try {
    const res = await fetch(API + '/v1/domains');
    const data = await res.json();
    const grid = document.getElementById('domainsGrid');
    grid.innerHTML = '';

    Object.entries(data.domains || {}).forEach(([name, schema]) => {
      const fieldsHtml = Object.entries(schema.fields || {}).map(([fname, fdef]) => {
        const impClass = { HIGH: 'imp-high', MEDIUM: 'imp-medium', LOW: 'imp-low' }[fdef.importance] || 'imp-low';
        return `
          <div class="field-row">
            <span class="field-name">${fname}</span>
            <span class="field-type">${fdef.type}</span>
            <span class="importance-badge ${impClass}">${fdef.importance}</span>
          </div>`;
      }).join('');

      grid.innerHTML += `
        <div class="domain-card">
          <div class="domain-card-name">📂 ${name.replace('_',' ')}</div>
          <div class="domain-card-desc">${schema.description || ''}</div>
          <div class="field-list">${fieldsHtml}</div>
        </div>`;
    });
  } catch (e) {
    console.error('Domains error:', e);
  }
}

// ── Utils ─────────────────────────────────────────────────────────────
function setQuery(q) {
  document.getElementById('queryInput').value = q;
  showView('pipeline');
  document.getElementById('queryInput').focus();
}

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).slice(2, 10);
}

function escapeHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
