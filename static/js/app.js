/* ─── FairSight App.js ───────────────────────────────────────────────────── */

// ─── State ──────────────────────────────────────────────────────────────────
let state = {
  fileId: null,
  filename: null,
  columns: [],
  dtypes: {},
  numericCols: [],
  categoricalCols: [],
  analysisResult: null,
  barChart: null,
  pieChart: null
};

// ─── Utility ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = id => $( id )?.classList.remove('hidden');
const hide = id => $( id )?.classList.add('hidden');

function showToast(msg, type = 'success') {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast ${type}`;
  t.style.opacity = '1';
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.classList.add('hidden'), 300); }, 3500);
}

function scoreColor(s) {
  if (s >= 85) return '#10b981';
  if (s >= 70) return '#f59e0b';
  return '#ef4444';
}

function scoreLabel(s) {
  if (s >= 85) return 'Low Bias Detected';
  if (s >= 70) return 'Moderate Bias Detected';
  return 'High Bias Detected';
}

function scoreDesc(s, flags) {
  const critical = flags.filter(f => f.level === 'critical').length;
  const warning = flags.filter(f => f.level === 'warning').length;
  if (s >= 85) return `Dataset appears relatively fair. ${warning} advisory warning${warning !== 1 ? 's' : ''} noted.`;
  if (s >= 70) return `Some bias patterns detected. ${warning} warning${warning !== 1 ? 's' : ''} — review recommended.`;
  return `Significant bias detected. ${critical} critical issue${critical !== 1 ? 's' : ''} require immediate attention.`;
}

// ─── Upload & Drag-Drop ──────────────────────────────────────────────────────
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
uploadArea.addEventListener('click', e => {
  if (e.target === uploadArea || e.target.classList.contains('upload-icon') || e.target.classList.contains('upload-text')) {
    fileInput.click();
  }
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  if (!file.name.match(/\.(csv|json)$/i)) {
    showToast('Please upload a CSV or JSON file', 'error'); return;
  }
  const formData = new FormData();
  formData.append('file', file);

  showLoading('Uploading file…', 'Parsing columns and data types');
  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Upload failed');
    hideLoading();
    applyFileData(data);
  } catch (err) {
    hideLoading();
    showToast(err.message, 'error');
  }
}

function applyFileData(data) {
  state.fileId = data.file_id;
  state.filename = data.filename;
  state.columns = data.columns;
  state.dtypes = data.dtypes;
  state.numericCols = data.numeric_cols;
  state.categoricalCols = data.categorical_cols;

  // File info card
  $('fileName').textContent = data.filename;
  $('fileShape').textContent = `${data.shape.rows.toLocaleString()} rows × ${data.shape.cols} columns`;

  const statsHtml = [
    `<div class="fstat"><strong>${data.shape.rows.toLocaleString()}</strong>Rows</div>`,
    `<div class="fstat"><strong>${data.shape.cols}</strong>Columns</div>`,
    `<div class="fstat"><strong>${data.numeric_cols.length}</strong>Numeric</div>`,
    `<div class="fstat"><strong>${data.categorical_cols.length}</strong>Categorical</div>`,
    Object.keys(data.missing).length > 0
      ? `<div class="fstat"><strong style="color:var(--yellow)">${Object.values(data.missing).reduce((a,b)=>a+b,0)}</strong>Missing Values</div>`
      : `<div class="fstat"><strong style="color:var(--green)">0</strong>Missing Values</div>`
  ].join('');
  $('fileStatsRow').innerHTML = statsHtml;

  show('fileInfoCard');

  // Populate selects
  populateSelects(data.columns, data.dtypes, data.categorical_cols, data.numeric_cols, data.unique_counts);

  // Preview table
  renderPreviewTable(data.preview, data.columns);
  $('previewRowCount').textContent = `(first ${Math.min(10, data.shape.rows)} of ${data.shape.rows.toLocaleString()} rows)`;

  show('configSection');
  $('configSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function populateSelects(columns, dtypes, catCols, numCols, uniqueCounts) {
  const sensitiveEl = $('sensitiveCol');
  const targetEl = $('targetCol');
  const predictedEl = $('predictedCol');
  const extraEl = $('extraSensitive');

  sensitiveEl.innerHTML = '<option value="">Select a column…</option>';
  targetEl.innerHTML = '<option value="">Select a column…</option>';
  predictedEl.innerHTML = '<option value="">None (skip equalized odds)</option>';
  extraEl.innerHTML = '';

  columns.forEach(col => {
    const dtype = dtypes[col] || 'object';
    const isNumeric = numCols.includes(col);
    const uniq = uniqueCounts[col] || 0;
    const label = `${col} (${isNumeric ? 'numeric' : 'text'}, ${uniq} unique)`;

    // Sensitive: prefer categorical with few unique values
    const opt1 = new Option(label, col);
    sensitiveEl.appendChild(opt1);

    // Target: prefer binary numeric
    const opt2 = new Option(label, col);
    targetEl.appendChild(opt2);

    // Predicted
    if (isNumeric) {
      const opt3 = new Option(label, col);
      predictedEl.appendChild(opt3);
    }

    // Extra sensitive
    const opt4 = new Option(label, col);
    extraEl.appendChild(opt4);
  });

  // Auto-select smart defaults
  autoSelectColumns(columns, dtypes, catCols, numCols, uniqueCounts);
}

function autoSelectColumns(columns, dtypes, catCols, numCols, uniqueCounts) {
  const sensitiveKeywords = ['gender','sex','race','ethnicity','age','nationality','religion','disability'];
  const targetKeywords = ['approved','hired','admitted','outcome','decision','label','target','result','granted'];

  let bestSensitive = null, bestTarget = null;

  for (const kw of sensitiveKeywords) {
    const found = columns.find(c => c.toLowerCase().includes(kw));
    if (found) { bestSensitive = found; break; }
  }

  for (const kw of targetKeywords) {
    const found = columns.find(c => c.toLowerCase().includes(kw));
    if (found) { bestTarget = found; break; }
  }

  // Fallback: first categorical for sensitive, last binary numeric for target
  if (!bestSensitive) {
    bestSensitive = catCols[0] || columns[0];
  }
  if (!bestTarget) {
    bestTarget = numCols.find(c => uniqueCounts[c] <= 2) || numCols[numCols.length - 1] || columns[columns.length - 1];
  }

  if (bestSensitive) $('sensitiveCol').value = bestSensitive;
  if (bestTarget) $('targetCol').value = bestTarget;
}

function renderPreviewTable(rows, columns) {
  if (!rows || rows.length === 0) { $('previewTable').innerHTML = '<p style="padding:16px;color:var(--text-muted)">No preview available</p>'; return; }
  const headers = columns.map(c => `<th>${c}</th>`).join('');
  const bodyRows = rows.map(row =>
    `<tr>${columns.map(c => `<td>${row[c] ?? ''}</td>`).join('')}</tr>`
  ).join('');
  $('previewTable').innerHTML = `<table class="data-table"><thead><tr>${headers}</tr></thead><tbody>${bodyRows}</tbody></table>`;
}

// ─── Sample Data ─────────────────────────────────────────────────────────────
async function loadSample(type) {
  showLoading('Loading sample data…', 'Preparing demo dataset');
  try {
    const res = await fetch('/sample-data');
    const datasets = await res.json();
    const dataset = datasets[type];
    if (!dataset) throw new Error('Sample not found');

    // Convert to CSV and upload as blob
    const keys = Object.keys(dataset.data[0]);
    const csvLines = [keys.join(','), ...dataset.data.map(row => keys.map(k => row[k]).join(','))];
    const csvBlob = new Blob([csvLines.join('\n')], { type: 'text/csv' });
    const file = new File([csvBlob], `${type}_sample.csv`, { type: 'text/csv' });

    const formData = new FormData();
    formData.append('file', file);
    const uploadRes = await fetch('/upload', { method: 'POST', body: formData });
    const data = await uploadRes.json();
    if (!uploadRes.ok) throw new Error(data.error);

    hideLoading();
    applyFileData(data);
    showToast(`Loaded: ${dataset.name}`, 'success');
  } catch (err) {
    hideLoading();
    showToast(err.message, 'error');
  }
}

// ─── Analysis ────────────────────────────────────────────────────────────────
async function runAnalysis() {
  const sensitiveCol = $('sensitiveCol').value;
  const targetCol = $('targetCol').value;
  const predictedCol = $('predictedCol').value;
  const extraSensitive = Array.from($('extraSensitive').selectedOptions).map(o => o.value);

  if (!sensitiveCol) { showToast('Please select a sensitive attribute column', 'error'); return; }
  if (!targetCol) { showToast('Please select a target/outcome column', 'error'); return; }
  if (sensitiveCol === targetCol) { showToast('Sensitive and target columns must be different', 'error'); return; }

  showLoading('Running Fairness Analysis…', 'Computing metrics across all groups');
  animateLoadingSteps();

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_id: state.fileId,
        sensitive_col: sensitiveCol,
        target_col: targetCol,
        predicted_col: predictedCol || null,
        extra_sensitive: extraSensitive
      })
    });
    const data = await res.json();
    document.getElementById("ai-advice-box").innerText =
    data.ai_advice;
    if (!res.ok) throw new Error(data.error || 'Analysis failed');

    state.analysisResult = data;
    hideLoading();

    renderResults(data);
    show('resultsSection');
    $('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    $('resultsDatasetDesc').textContent = `Sensitive: "${sensitiveCol}" · Target: "${targetCol}" · ${data.dataset_info.total_rows.toLocaleString()} rows analyzed`;

  } catch (err) {
    hideLoading();
    showToast(err.message, 'error');
  }
}

let loadingStepTimer;
function animateLoadingSteps() {
  const steps = ['ls1','ls2','ls3','ls4','ls5'];
  let i = 0;
  loadingStepTimer = setInterval(() => {
    if (i > 0) { const el = $(steps[i-1]); if(el) { el.classList.remove('active'); el.classList.add('done'); el.textContent = '✅ ' + el.textContent.replace(/^[^a-zA-Z]*/, ''); } }
    if (i < steps.length) { const el = $(steps[i]); if(el) el.classList.add('active'); }
    i++;
    if (i >= steps.length + 1) clearInterval(loadingStepTimer);
  }, 600);
}

// ─── Render Results ───────────────────────────────────────────────────────────
function renderResults(data) {
  renderScoreBanner(data);
  renderFlags(data.flags);
  renderCharts(data);
  renderMetricsGrid(data);
  renderEqOdds(data.equalized_odds);
  renderIntersectional(data.intersectional);
  renderCorrelations(data.correlations);
  renderMitigations(data.mitigations);
}

// Score Banner
function renderScoreBanner(data) {
  const score = data.overall_score;
  const color = scoreColor(score);

  // Draw ring on canvas
  const canvas = $('scoreCanvas');
  const ctx = canvas.getContext('2d');
  const cx = 70, cy = 70, r = 56;
  ctx.clearRect(0, 0, 140, 140);

  // Track
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#1e1e35'; ctx.lineWidth = 10; ctx.stroke();

  // Fill
  const angle = (score / 100) * Math.PI * 2 - Math.PI / 2;
  const grad = ctx.createLinearGradient(0, 0, 140, 140);
  grad.addColorStop(0, color); grad.addColorStop(1, color + 'aa');
  ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, angle);
  ctx.strokeStyle = grad; ctx.lineWidth = 10; ctx.lineCap = 'round'; ctx.stroke();

  $('scoreValue').textContent = score;
  $('scoreValue').style.color = color;

  const labelEl = document.querySelector('#scoreBanner .score-info h3');
  const descEl = document.querySelector('#scoreBanner .score-info p');
  if (labelEl) labelEl.textContent = scoreLabel(score);
  if (descEl) descEl.textContent = scoreDesc(score, data.flags);

  // Metric pills
  const di = data.disparate_impact;
  const spd = data.statistical_parity_diff;
  const groups = Object.keys(data.selection_rates).length;

  $('scoreMetrics').innerHTML = [
    metricPill('Disparate Impact', di !== null ? di + '%' : 'N/A', di !== null ? (di >= 80 ? 'green' : 'red') : 'text'),
    metricPill('Stat. Parity Diff', (spd * 100).toFixed(1) + '%', spd > 0.2 ? 'red' : spd > 0.1 ? 'yellow' : 'green'),
    metricPill('Groups Analyzed', groups, 'text'),
    metricPill('Total Rows', data.dataset_info.total_rows.toLocaleString(), 'text'),
  ].join('');
}

function metricPill(label, val, colorClass) {
  const cls = colorClass === 'green' ? 'var(--green)' : colorClass === 'red' ? 'var(--red)' : colorClass === 'yellow' ? 'var(--yellow)' : 'var(--text)';
  return `<div class="score-metric"><div class="sm-value" style="color:${cls}">${val}</div><div class="sm-label">${label}</div></div>`;
}

// Flags
function renderFlags(flags) {
  $('flagsSection').innerHTML = flags.map(f => `
    <div class="flag-card ${f.level}">
      <div class="flag-icon">${f.icon}</div>
      <div class="flag-content">
        <div class="flag-title">${f.title}</div>
        <div class="flag-detail">${f.detail}</div>
        ${f.fix ? `<div class="flag-fix">💡 Fix: ${f.fix}</div>` : ''}
      </div>
    </div>
  `).join('');
}

// Charts
function renderCharts(data) {
  const groups = Object.keys(data.selection_rates);
  const rates = Object.values(data.selection_rates);
  const colors = ['#6366f1','#8b5cf6','#06b6d4','#10b981','#f59e0b','#ef4444','#f97316','#ec4899'];

  // Bar Chart
  if (state.barChart) state.barChart.destroy();
  const barCtx = $('barChart').getContext('2d');
  state.barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: groups,
      datasets: [{
        label: 'Positive Outcome Rate',
        data: rates.map(r => (r * 100).toFixed(2)),
        backgroundColor: colors.slice(0, groups.length).map(c => c + 'cc'),
        borderColor: colors.slice(0, groups.length),
        borderWidth: 2, borderRadius: 6
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `${ctx.parsed.y}%` } } },
      scales: {
        y: {
          beginAtZero: true, max: 100,
          grid: { color: '#1e1e35' },
          ticks: { color: '#8888aa', callback: v => v + '%' }
        },
        x: { grid: { display: false }, ticks: { color: '#8888aa' } }
      }
    }
  });

  // Pie Chart
  if (state.pieChart) state.pieChart.destroy();
  const props = Object.values(data.group_proportions);
  const pieCtx = $('pieChart').getContext('2d');
  state.pieChart = new Chart(pieCtx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(data.group_proportions),
      datasets: [{
        data: props.map(p => (p * 100).toFixed(1)),
        backgroundColor: colors.slice(0, groups.length).map(c => c + 'cc'),
        borderColor: colors.slice(0, groups.length),
        borderWidth: 2
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { color: '#8888aa', padding: 14, font: { size: 12 } } },
        tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.parsed}%` } }
      },
      cutout: '62%'
    }
  });
}

// Metrics Grid
function renderMetricsGrid(data) {
  const di = data.disparate_impact;
  const spd = data.statistical_parity_diff;

  const blocks = [
    {
      label: 'Disparate Impact Score',
      value: di !== null ? di + '%' : 'N/A',
      sub: di !== null ? (di >= 80 ? '✅ Above 80% threshold' : '🚨 Below legal 80% threshold') : 'Insufficient data',
      color: di !== null ? (di >= 80 ? 'var(--green)' : 'var(--red)') : 'var(--text-muted)'
    },
    {
      label: 'Statistical Parity Diff',
      value: (spd * 100).toFixed(1) + '%',
      sub: spd <= 0.1 ? '✅ Within acceptable range' : spd <= 0.2 ? '⚠️ Moderate gap' : '🚨 Large outcome gap',
      color: spd <= 0.1 ? 'var(--green)' : spd <= 0.2 ? 'var(--yellow)' : 'var(--red)'
    },
    {
      label: 'Groups Analyzed',
      value: Object.keys(data.selection_rates).length,
      sub: Object.keys(data.selection_rates).join(', '),
      color: 'var(--primary-light)'
    }
  ];

  // Per-group outcome rates
  Object.entries(data.group_distributions).forEach(([group, stats]) => {
    blocks.push({
      label: `Group: ${group}`,
      value: (stats.mean * 100).toFixed(1) + '%',
      sub: `${stats.positive} pos / ${stats.negative} neg out of ${stats.count}`,
      color: 'var(--accent)'
    });
  });

  $('metricsGrid').innerHTML = blocks.map(b => `
    <div class="metric-block">
      <div class="metric-block-label">${b.label}</div>
      <div class="metric-block-value" style="color:${b.color}">${b.value}</div>
      <div class="metric-block-sub">${b.sub}</div>
    </div>
  `).join('');
}

// Equalized Odds
function renderEqOdds(eqOdds) {
  if (!eqOdds) { hide('eqOddsSection'); return; }
  show('eqOddsSection');

  const cards = Object.entries(eqOdds.groups).map(([group, stats]) => `
    <div class="eq-group-card">
      <div class="eq-group-name">Group: ${group}</div>
      <div class="eq-stat"><span>True Positive Rate</span><span style="color:var(--green)">${(stats.tpr * 100).toFixed(1)}%</span></div>
      <div class="eq-stat"><span>False Positive Rate</span><span style="color:var(--red)">${(stats.fpr * 100).toFixed(1)}%</span></div>
      <div class="eq-stat"><span>Precision</span><span style="color:var(--primary-light)">${(stats.precision * 100).toFixed(1)}%</span></div>
      <div class="cm-grid">
        <div class="cm-cell tp"><strong>${stats.tp}</strong>True Pos</div>
        <div class="cm-cell fp"><strong>${stats.fp}</strong>False Pos</div>
        <div class="cm-cell fn"><strong>${stats.fn}</strong>False Neg</div>
        <div class="cm-cell tn"><strong>${stats.tn}</strong>True Neg</div>
      </div>
    </div>
  `).join('');

  const diffCard = `
    <div class="eq-group-card" style="border-color:var(--primary);background:var(--primary-glow)">
      <div class="eq-group-name" style="color:var(--primary-light)">⚖️ Parity Gaps</div>
      <div class="eq-stat"><span>TPR Difference</span><span style="color:${eqOdds.tpr_diff > 0.1 ? 'var(--red)' : 'var(--green)'}">${(eqOdds.tpr_diff * 100).toFixed(1)}%</span></div>
      <div class="eq-stat"><span>FPR Difference</span><span style="color:${eqOdds.fpr_diff > 0.1 ? 'var(--red)' : 'var(--green)'}">${(eqOdds.fpr_diff * 100).toFixed(1)}%</span></div>
      <p style="font-size:11px;color:var(--text-muted);margin-top:10px">Target: both gaps < 10% for equalized odds fairness</p>
    </div>
  `;

  $('eqOddsGrid').innerHTML = diffCard + cards;
}

// Intersectional
function renderIntersectional(data) {
  if (!data || data.length === 0) { hide('intersectSection'); return; }
  show('intersectSection');

  const maxRate = Math.max(...data.map(d => d.rate));
  $('intersectGrid').innerHTML = data.map(d => `
    <div class="intersect-row">
      <div class="intersect-label">${d.group}</div>
      <div class="intersect-bar-wrap">
        <div class="intersect-bar" style="width:${maxRate > 0 ? (d.rate / maxRate * 100).toFixed(1) : 0}%"></div>
      </div>
      <div class="intersect-pct">${(d.rate * 100).toFixed(1)}%</div>
      <div class="intersect-count">n=${d.count}</div>
    </div>
  `).join('');
}

// Correlations
function renderCorrelations(correlations) {
  if (!correlations || Object.keys(correlations).length === 0) { hide('corrSection'); return; }
  show('corrSection');

  const sorted = Object.entries(correlations).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...sorted.map(e => Math.abs(e[1])));

  $('corrBars').innerHTML = sorted.map(([col, val]) => {
    const pct = maxAbs > 0 ? (Math.abs(val) / maxAbs * 100).toFixed(1) : 0;
    const isPos = val >= 0;
    const absVal = Math.abs(val);
    const riskLevel = absVal > 0.5 ? `<span class="tag complexity-high">High Proxy Risk</span>` :
                      absVal > 0.3 ? `<span class="tag complexity-medium">Moderate Risk</span>` :
                      `<span class="tag">Low Risk</span>`;
    return `
      <div class="corr-row">
        <div class="corr-label">${col}</div>
        <div class="corr-bar-container">
          <div class="corr-bar-track"><div class="corr-bar-fill ${isPos ? 'corr-positive' : 'corr-negative'}" style="width:${pct}%"></div></div>
          ${riskLevel}
        </div>
        <div class="corr-val" style="color:${absVal > 0.4 ? 'var(--red)' : 'var(--text-muted)'}">${val > 0 ? '+' : ''}${val.toFixed(3)}</div>
      </div>
    `;
  }).join('');
}

// Mitigations
function renderMitigations(mitigations) {
  $('mitPhases').innerHTML = mitigations.map(phase => `
    <div class="mit-phase">
      <div class="mit-phase-header">
        <div class="mit-phase-icon">${phase.icon}</div>
        <div>
          <div class="mit-phase-title">${phase.phase}</div>
        </div>
      </div>
      ${phase.strategies.map(s => `
        <div class="mit-strategy">
          <div class="mit-strategy-name">${s.name}</div>
          <div class="mit-strategy-desc">${s.desc}</div>
          <div class="mit-strategy-tags">
            <span class="tag">When: ${s.when}</span>
            <span class="tag complexity-${s.complexity.toLowerCase()}">Complexity: ${s.complexity}</span>
          </div>
        </div>
      `).join('')}
    </div>
  `).join('');
}

// ─── Loading ──────────────────────────────────────────────────────────────────
function showLoading(title, step) {
  clearInterval(loadingStepTimer);
  // Reset steps
  ['ls1','ls2','ls3','ls4','ls5'].forEach((id, i) => {
    const el = $(id);
    if (el) { el.classList.remove('active','done'); }
  });
  $('loadingTitle').textContent = title;
  $('loadingStep').textContent = step;
  show('loadingOverlay');
}

function hideLoading() {
  clearInterval(loadingStepTimer);
  hide('loadingOverlay');
}

// ─── Reset ────────────────────────────────────────────────────────────────────
function resetUpload() {
  state.fileId = null; state.filename = null;
  state.columns = []; state.analysisResult = null;
  hide('fileInfoCard');
  hide('configSection');
  hide('resultsSection');
  fileInput.value = '';
}

function resetAll() {
  resetUpload();
  if (state.barChart) { state.barChart.destroy(); state.barChart = null; }
  if (state.pieChart) { state.pieChart.destroy(); state.pieChart = null; }
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ─── Export ───────────────────────────────────────────────────────────────────
function exportReport() {
  if (!state.analysisResult) { showToast('No analysis results to export', 'error'); return; }
  const report = {
    generated_at: new Date().toISOString(),
    tool: 'FairSight Bias Detection Platform',
    dataset: state.filename,
    analysis: state.analysisResult
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `fairsight_report_${Date.now()}.json`;
  a.click(); URL.revokeObjectURL(url);
  showToast('Report exported successfully', 'success');
}

// ─── Nav active state ─────────────────────────────────────────────────────────
const navLinks = document.querySelectorAll('.nav-link');
window.addEventListener('scroll', () => {
  const scrollY = window.scrollY;
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (!href.startsWith('#')) return;
    const target = document.querySelector(href);
    if (!target) return;
    const top = target.offsetTop - 80;
    const bottom = top + target.offsetHeight;
    link.classList.toggle('active', scrollY >= top && scrollY < bottom);
  });
});
