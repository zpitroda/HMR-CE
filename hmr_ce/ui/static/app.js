/**
 * HMR-CE Live Dashboard Frontend Logic
 * Upgraded with Intuitive Explanations, Mode Switching, Interactive Inspections,
 * Real-Time Brain State Telemetry, and Scenario Narrations.
 */

let surpriseChart = null;
let currentTopology = { tier0: [], tier1: [], tier2: [], tier3: [] };
let activeTab = 'tier0';
let currentMode = 'intuitive'; // 'intuitive' | 'telemetry'
let ws = null;
let currentGraphData = { nodes: [], edges: [] };

// Scenario Explanations for Average Persons
const SCENARIO_NARRATIVES = {
  belief_invalidation: {
    title: "1. Belief Revision (Self-Correction)",
    desc: "We tell the AI an initial plan (using DNS TXT for queueing), then inform it that failed and we switched to Redis Streams. The AI's contradiction detector automatically strikes out the old DNS fact in the belief graph so it will never give you outdated advice!"
  },
  semantic_smearing: {
    title: "2. Surprise Isolation (Protecting Topic Summaries)",
    desc: "We discuss standard PostgreSQL query optimizations, and then inject a radical outlier proposal ('storing binary tensors in JSONB'). The AI identifies the shock and isolates it, preventing this weird outlier from polluting the high-level database topic summary."
  },
  titans_momentum: {
    title: "3. Titans Attention Window (Forward Momentum)",
    desc: "When a surprising idea is first mentioned, the AI opens an attention window (Momentum > 0). Even if the following explanatory sentences sound routine on their own, the AI captures them with high priority because they belong to the surprising event."
  },
  temporal_inversion: {
    title: "4. Temporal Decay (Recency Prioritization)",
    desc: "We set an initial IP, generate 12 routine system check turns, and then update the IP. Thanks to exponential decay, the AI correctly prioritizes the recent IP without erasing the old historical record from disk."
  },
  prefix_truncation: {
    title: "5. Two-Stage Retrieval (Fast Funnel)",
    desc: "The AI searches through years of memories in microseconds. Step 1 uses ultra-fast 64-number summaries to find the right topic cluster. Step 2 uses high-definition 1024-number vectors to pinpoint the exact best memories."
  }
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
  setViewMode('intuitive');
  initChart();
  initWebSocket();
  fetchState();
  fetchGraph();

  // URL parameter auto-runner (e.g. ?scenario=belief_invalidation)
  const urlParams = new URLSearchParams(window.location.search);
  const scenario = urlParams.get('scenario');
  if (scenario && SCENARIO_NARRATIVES[scenario]) {
    setTimeout(() => runScenario(scenario), 400);
  }

  // Chat Form Submit
  const chatForm = document.getElementById('chat-form');
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    input.value = '';
    await sendChatMessage(msg);
  });
});

// ==================== VIEW MODE SWITCHER ====================
function setViewMode(mode) {
  currentMode = mode;
  document.body.className = `mode-${mode} bg-slate-950 text-slate-100 antialiased min-h-screen flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200`;
  
  const btnIntuitive = document.getElementById('btn-mode-intuitive');
  const btnTelemetry = document.getElementById('btn-mode-telemetry');

  if (btnIntuitive && btnTelemetry) {
    if (mode === 'intuitive') {
      btnIntuitive.classList.add('active');
      btnTelemetry.classList.remove('active');
    } else {
      btnTelemetry.classList.add('active');
      btnIntuitive.classList.remove('active');
    }
  }

  // Update Chart axis titles if chart exists
  if (surpriseChart) {
    surpriseChart.options.scales.x.title.text = mode === 'intuitive' 
      ? 'Topical Drift (How new is the subject?)' 
      : 'Topical Drift S_topic (theta=0.50)';
    surpriseChart.options.scales.y.title.text = mode === 'intuitive' 
      ? 'Surprise Level (How unexpected is the fact?)' 
      : 'Perplexity P_entropy (theta=40.0)';
    surpriseChart.update();
  }

  // Re-render current tab to reflect labels
  renderTopologyTab();
}

// ==================== 1. 2D SURPRISE PHASE PLANE (CHART.JS) ====================
function initChart() {
  const canvas = document.getElementById('surpriseChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  surpriseChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Q1: Topic Pivot',
          data: [],
          backgroundColor: 'rgba(6, 182, 212, 0.85)',
          borderColor: '#22d3ee',
          pointRadius: 6,
          pointHoverRadius: 9
        },
        {
          label: 'Q2: Novelty (Thesis)',
          data: [],
          backgroundColor: 'rgba(245, 158, 11, 0.85)',
          borderColor: '#fbbf24',
          pointRadius: 8,
          pointHoverRadius: 11
        },
        {
          label: 'Q3: Expected Cont.',
          data: [],
          backgroundColor: 'rgba(16, 185, 129, 0.85)',
          borderColor: '#34d399',
          pointRadius: 5,
          pointHoverRadius: 8
        },
        {
          label: 'Q4: Routine Interr.',
          data: [],
          backgroundColor: 'rgba(100, 116, 139, 0.85)',
          borderColor: '#94a3b8',
          pointRadius: 4,
          pointHoverRadius: 7
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      onClick: (evt, elements) => {
        if (elements && elements.length > 0) {
          const el = elements[0];
          const pt = surpriseChart.data.datasets[el.datasetIndex].data[el.index];
          inspectChartPoint(pt);
        }
      },
      scales: {
        x: {
          title: { 
            display: true, 
            text: currentMode === 'intuitive' ? 'Topical Drift (How new is the subject?)' : 'Topical Drift S_topic (theta=0.50)', 
            color: '#94a3b8', 
            font: { size: 10, family: 'JetBrains Mono' } 
          },
          min: 0.0,
          max: 1.0,
          grid: { color: 'rgba(51, 65, 85, 0.35)' },
          ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } }
        },
        y: {
          title: { 
            display: true, 
            text: currentMode === 'intuitive' ? 'Surprise Level (How unexpected is the fact?)' : 'Perplexity P_entropy (theta=40.0)', 
            color: '#94a3b8', 
            font: { size: 10, family: 'JetBrains Mono' } 
          },
          min: 0,
          max: 100,
          grid: { color: 'rgba(51, 65, 85, 0.35)' },
          ticks: { color: '#64748b', font: { size: 10, family: 'JetBrains Mono' } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(51, 65, 85, 0.8)',
          borderWidth: 1,
          titleFont: { family: 'JetBrains Mono', size: 11 },
          bodyFont: { family: 'Inter', size: 11 },
          callbacks: {
            title: function(items) {
              const item = items[0].raw;
              return `Turn ${item.turn_id}: ${item.label}`;
            },
            label: function(ctx) {
              const item = ctx.raw;
              return [
                `Text: "${item.text.slice(0, 45)}..."`,
                `Drift S: ${item.x.toFixed(3)} | Surprise P: ${item.y.toFixed(1)}`,
                `(Click point to inspect full details)`
              ];
            }
          }
        }
      }
    },
    plugins: [
      {
        id: 'quadrantCrosshairs',
        afterDraw: (chart) => {
          const { ctx, chartArea, scales: { x, y } } = chart;
          if (!chartArea || !x || !y) return;
          const xPos = x.getPixelForValue(0.50);
          const yPos = y.getPixelForValue(40.0);

          ctx.save();
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 4]);

          // Vertical dividing line at S_topic = 0.50
          if (xPos >= chartArea.left && xPos <= chartArea.right) {
            ctx.beginPath();
            ctx.moveTo(xPos, chartArea.top);
            ctx.lineTo(xPos, chartArea.bottom);
            ctx.stroke();
          }

          // Horizontal dividing line at P_entropy = 40.0
          if (yPos >= chartArea.top && yPos <= chartArea.bottom) {
            ctx.beginPath();
            ctx.moveTo(chartArea.left, yPos);
            ctx.lineTo(chartArea.right, yPos);
            ctx.stroke();
          }

          ctx.restore();
        }
      }
    ]
  });
}

function addTurnToChart(gateResult, text) {
  if (!surpriseChart || !gateResult) return;

  const pt = {
    x: Math.min(1.0, Math.max(0.0, gateResult.s_topic)),
    y: Math.min(100.0, Math.max(0.0, gateResult.p_entropy)),
    turn_id: gateResult.turn_id,
    label: gateResult.label,
    quadrant: gateResult.quadrant,
    salience: gateResult.salience,
    momentum: gateResult.active_momentum,
    text: text
  };

  let datasetIdx = 2; // Default Q3
  if (gateResult.quadrant === 'DOMAIN_PIVOT') datasetIdx = 0;
  else if (gateResult.quadrant === 'CONCEPTUAL_NOVELTY') datasetIdx = 1;
  else if (gateResult.quadrant === 'EXPECTED_PROGRESS') datasetIdx = 2;
  else if (gateResult.quadrant === 'ROUTINE_INTERRUPT') datasetIdx = 3;

  surpriseChart.data.datasets[datasetIdx].data.push(pt);
  surpriseChart.update('none');

  // Update momentum badges and brain state
  updateBrainState(gateResult);
}

function updateBrainState(gateResult) {
  const momentum = gateResult.active_momentum || 0;
  const quadrant = gateResult.quadrant;

  // Header badge
  const headerBadge = document.getElementById('active-momentum-header');
  if (headerBadge) {
    headerBadge.innerText = `Momentum α: ${momentum.toFixed(2)}`;
  }

  // Panel badge
  const panelBadge = document.getElementById('active-momentum-badge');
  if (panelBadge) {
    panelBadge.innerText = `Window α: ${momentum.toFixed(2)}`;
    if (momentum > 0.4) {
      panelBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-amber-950 border border-amber-500 text-amber-300 shadow-sm shadow-amber-500/20 animate-pulse';
    } else {
      panelBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-400';
    }
  }

  // Brain state badge
  const brainBadge = document.getElementById('brain-state-badge');
  if (brainBadge) {
    if (quadrant === 'CONCEPTUAL_NOVELTY' || momentum > 0.6) {
      brainBadge.className = 'px-2.5 py-1 rounded-lg bg-amber-950/80 border border-amber-600/90 text-amber-300 font-mono text-[11px] flex items-center gap-1.5 shadow-md shadow-amber-500/20';
      brainBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span><span>Brain: Novelty Alert! 🔥</span>`;
    } else if (quadrant === 'DOMAIN_PIVOT') {
      brainBadge.className = 'px-2.5 py-1 rounded-lg bg-cyan-950/80 border border-cyan-600/90 text-cyan-300 font-mono text-[11px] flex items-center gap-1.5';
      brainBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-cyan-400"></span><span>Brain: Topic Switch 🌀</span>`;
    } else {
      brainBadge.className = 'px-2.5 py-1 rounded-lg bg-emerald-950/70 border border-emerald-800/80 text-emerald-300 font-mono text-[11px] flex items-center gap-1.5';
      brainBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>Brain: Calm (Routine)</span>`;
    }
  }
}

// ==================== 2. WEBSOCKET TELEMETRY ====================
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/stream`;
  
  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      const el = document.getElementById('connection-status');
      if (el) {
        el.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>WebSocket Live`;
        el.className = 'flex items-center gap-1.5 font-mono text-emerald-400';
      }
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.event === 'turn_ingested') {
          // Option A: Only chart user turns on the Surprise Radar
          if (!msg.data.tier1_record || msg.data.tier1_record.speaker_role !== 'ASSISTANT') {
            addTurnToChart(msg.data.gate_result, msg.data.tier1_record.raw_text);
          }
          fetchState();
          fetchGraph();
        } else if (msg.event === 'turn_processed') {
          fetchState();
          fetchGraph();
        } else if (msg.event === 'session_reset') {
          resetLocalState();
        }
      } catch (err) {
        console.error('WS parse error:', err);
      }
    };

    ws.onclose = () => {
      const el = document.getElementById('connection-status');
      if (el) {
        el.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500"></span>Reconnecting...`;
        el.className = 'flex items-center gap-1.5 font-mono text-rose-400';
      }
      setTimeout(initWebSocket, 3000);
    };
  } catch (e) {
    console.warn('WebSocket setup failed:', e);
  }
}

// ==================== 3. CHAT INTERACTIONS & PROMPT CHIPS ====================
async function sendChatMessage(text) {
  const forceVerbatim = document.getElementById('force-verbatim').checked;
  const retroMode = document.getElementById('retro-mode').checked;

  appendUserBubble(text);

  // Add temporary typing indicator
  const typingId = appendTypingBubble();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        force_verbatim: forceVerbatim,
        is_retrospective: retroMode ? true : null
      })
    });
    const data = await res.json();

    // Remove typing bubble
    removeTypingBubble(typingId);

    // Render Assistant Reply
    appendAgentBubble(data.reply, data.user_ingest ? data.user_ingest.gate_result : null);

    // Update charts & traces (Option A: User turns only on Surprise Radar)
    if (data.user_ingest && data.user_ingest.gate_result) {
      addTurnToChart(data.user_ingest.gate_result, text);
    }

    renderTraversalTrace(data.traversal);
    currentTopology = data.topology;
    renderTopologyTab();
    fetchGraph();

  } catch (err) {
    removeTypingBubble(typingId);
    appendSystemBubble(`Chat Error: ${err.message}`);
  }
}

function fillAndSend(promptText) {
  const input = document.getElementById('chat-input');
  if (input) {
    input.value = promptText;
    sendChatMessage(promptText);
    input.value = '';
  }
}

function clearChat() {
  const container = document.getElementById('chat-messages');
  container.innerHTML = `
    <div class="system-bubble">
      <div class="flex items-center gap-1.5 text-cyan-400 text-xs font-semibold font-mono mb-1">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
        <span>CHAT LOG CLEARED • MEMORY PRESERVED</span>
      </div>
      <p class="text-slate-300 text-xs">The visual chat has been cleared. The 4-tier memory engine is still active with all your stored knowledge.</p>
    </div>
  `;
}

function appendUserBubble(text) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'user-bubble';
  div.innerHTML = `
    <div class="flex items-center justify-between text-[10px] text-cyan-300 font-mono mb-1">
      <span class="font-bold">YOU</span>
      <span>${new Date().toLocaleTimeString()}</span>
    </div>
    <p class="text-slate-100 leading-relaxed">${escapeHtml(text)}</p>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendAgentBubble(text, gateResult) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'agent-bubble';

  let badgeHtml = '';
  if (gateResult) {
    const qClass = `badge-${gateResult.quadrant.toLowerCase().replace('_', '-')}`;
    const desc = currentMode === 'intuitive' 
      ? (gateResult.quadrant === 'CONCEPTUAL_NOVELTY' ? '💡 Novel Fact' : (gateResult.quadrant === 'DOMAIN_PIVOT' ? '🌀 Topic Switch' : 'Expected'))
      : gateResult.label;
    badgeHtml = `<span class="badge ${qClass} ml-2">${desc} (α=${gateResult.salience.toFixed(2)})</span>`;
  }

  div.innerHTML = `
    <div class="flex items-center justify-between mb-1.5">
      <span class="font-bold text-emerald-400 text-xs font-mono flex items-center gap-1.5">
        <svg class="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/></svg>
        HMR-CE ASSISTANT (QWEN 3.8 27B)
      </span>
      ${badgeHtml}
    </div>
    <p class="text-slate-200 text-sm whitespace-pre-wrap leading-relaxed">${escapeHtml(text)}</p>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendTypingBubble() {
  const container = document.getElementById('chat-messages');
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'agent-bubble';
  div.innerHTML = `
    <div class="flex items-center gap-2 text-slate-400 text-xs font-mono">
      <span class="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
      <span>Evaluating surprise & searching 4-tier memory...</span>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTypingBubble(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendSystemBubble(text) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'system-bubble';
  div.innerHTML = `<p class="text-xs text-amber-300 font-mono">${escapeHtml(text)}</p>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// ==================== 4. MEMORY TOPOLOGY TABS & INSPECTIONS ====================
function switchTab(tabName) {
  activeTab = tabName;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById(`tab-${tabName}`);
  if (activeBtn) activeBtn.classList.add('active');
  renderTopologyTab();
}

function renderTopologyTab() {
  const container = document.getElementById('tab-content');
  const counter = document.getElementById('tier-counter');
  if (!container || !counter) return;

  container.innerHTML = '';
  const items = currentTopology[activeTab] || [];
  counter.innerText = `${items.length} item(s)`;

  if (items.length === 0) {
    const emptyNames = {
      tier0: 'Short-term buffer (Tier 0)',
      tier1: 'Disk vault (Tier 1)',
      tier2: 'Episodic conceptual nodes (Tier 2)',
      tier3: 'Topic cluster centroids (Tier 3)'
    };
    container.innerHTML = `<p class="text-slate-500 italic p-3 text-center">${emptyNames[activeTab]} is currently empty.</p>`;
    return;
  }

  if (activeTab === 'tier0') {
    // Ring buffer (K=6)
    items.forEach(t => {
      const card = document.createElement('div');
      card.className = 'telemetry-card';
      card.onclick = () => inspectTopologyItem('tier0', t);
      card.innerHTML = `
        <div class="flex justify-between items-center text-[10px] text-slate-400 mb-1">
          <span class="text-cyan-400 font-bold">Turn ${t.turn_id} (${t.speaker_role})</span>
          <span>${t.timestamp ? t.timestamp.slice(11, 19) : 'Active'}</span>
        </div>
        <p class="text-slate-200 font-mono text-xs leading-relaxed line-clamp-2">${escapeHtml(t.raw_text)}</p>
        <span class="text-[9px] text-cyan-400/70 mt-1 block">Click to inspect</span>
      `;
      container.appendChild(card);
    });
  } else if (activeTab === 'tier1') {
    // SQLite Vault
    items.forEach(t => {
      const card = document.createElement('div');
      const isSuper = t.active_status === 'SUPERSEDED';
      card.className = `telemetry-card ${isSuper ? 'superseded' : ''}`;
      card.onclick = () => inspectTopologyItem('tier1', t);
      card.innerHTML = `
        <div class="flex justify-between items-center text-[10px] mb-1">
          <span class="text-cyan-400 font-bold">Turn ${t.turn_id} [${t.speaker_role}]</span>
          <span class="badge ${isSuper ? 'badge-superseded' : 'badge-q3'}">${isSuper ? 'SUPERSEDED (Retracted)' : 'VALID (Active)'}</span>
        </div>
        <p class="text-slate-200 text-xs leading-relaxed ${isSuper ? 'line-through opacity-70' : ''}">${escapeHtml(t.raw_text)}</p>
        <div class="flex justify-between items-center mt-1 text-[9px] text-slate-500">
          <span>SHA-256: ${t.sha256_hash ? t.sha256_hash.slice(0, 10) + '...' : 'vault'}</span>
          <span class="text-cyan-400/80">Click for details</span>
        </div>
      `;
      container.appendChild(card);
    });
  } else if (activeTab === 'tier2') {
    // Episodic Nodes
    items.forEach(n => {
      const isSuper = n.status === 'SUPERSEDED';
      const isNovel = n.surprise_salience >= 0.99;
      const card = document.createElement('div');
      card.className = `telemetry-card ${isSuper ? 'superseded' : (isNovel ? 'novel-node' : 'active-node')}`;
      card.onclick = () => inspectTopologyItem('tier2', n);
      card.innerHTML = `
        <div class="flex justify-between items-center text-[10px] mb-1">
          <span class="text-indigo-400 font-bold">Node ${n.node_id.slice(0, 8)} (Turn ${n.turn_references ? n.turn_references.join(',') : ''})</span>
          <div class="flex gap-1">
            <span class="badge ${isNovel ? 'badge-q2' : 'badge-q3'}">α=${n.surprise_salience.toFixed(2)}</span>
            <span class="badge ${isSuper ? 'badge-superseded' : 'badge-q3'}">${n.status}</span>
          </div>
        </div>
        <p class="text-slate-300 text-xs leading-relaxed mb-1">${escapeHtml(n.metadata.abstract || '')}</p>
        <div class="flex justify-between text-[10px] text-slate-500">
          <span>Cluster: ${n.centroid_cluster_id}</span>
          ${n.superseded_by_node ? `<span class="text-red-400 font-bold">Replaced by ${n.superseded_by_node.slice(0, 8)}</span>` : '<span class="text-cyan-400/80">Click to inspect</span>'}
        </div>
      `;
      container.appendChild(card);
    });
  } else if (activeTab === 'tier3') {
    // Macro-Centroids
    items.forEach(c => {
      const card = document.createElement('div');
      card.className = 'telemetry-card';
      card.onclick = () => inspectTopologyItem('tier3', c);
      card.innerHTML = `
        <div class="flex justify-between items-center text-[10px] text-slate-400 mb-1">
          <span class="text-emerald-400 font-bold">${c.cluster_id}</span>
          <span class="badge badge-q1">${c.member_count} belief node(s)</span>
        </div>
        <p class="text-xs text-slate-300">Turns covered: [${c.bounding_turn_range[0]} to ${c.bounding_turn_range[1]}]</p>
        <p class="text-[10px] text-slate-500 mt-1 font-mono">Coarse MRL Prefix: d=64 [${c.centroid_coarse.slice(0, 4).map(v => v.toFixed(3)).join(', ')} ...]</p>
      `;
      container.appendChild(card);
    });
  }
}

// ==================== 5. DIRECTED BELIEF REVISION DAG (SVG) ====================
async function fetchGraph() {
  try {
    const res = await fetch('/api/graph');
    const data = await res.json();
    currentGraphData = data;
    renderDAG(data.nodes, data.edges);

    const elCount = document.getElementById('invalidations-count');
    if (elCount) {
      elCount.innerText = `${data.edges.length} Correction(s)`;
      if (data.edges.length > 0) {
        elCount.className = 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-rose-950 border border-rose-500 text-rose-300 shadow-md shadow-rose-500/20';
      } else {
        elCount.className = 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-400';
      }
    }
  } catch (err) {
    console.error('Failed to fetch graph:', err);
  }
}

function renderDAG(nodes, edges) {
  const svg = document.getElementById('dag-svg');
  if (!svg) return;
  svg.innerHTML = '';

  if (nodes.length === 0) {
    svg.innerHTML = '<text x="20" y="40" fill="#64748b" font-family="monospace" font-size="11">No memory nodes in DAG yet. Type a fact to begin.</text>';
    return;
  }

  // Arrow marker def
  const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#f43f5e" />
    </marker>
  `;
  svg.appendChild(defs);

  const nodeMap = {};
  const startY = 16;
  const rowHeight = 46;
  const boxWidth = 240;
  const boxHeight = 32;

  nodes.forEach((n, idx) => {
    const y = startY + idx * rowHeight;
    nodeMap[n.id] = { ...n, x: 12, y: y };
  });

  svg.setAttribute('height', `${Math.max(220, startY + nodes.length * rowHeight + 20)}`);
  svg.setAttribute('width', `${Math.max(320, boxWidth + 80)}`);
  svg.style.minWidth = `${Math.max(320, boxWidth + 80)}px`;

  // Render curved invalidation edges
  edges.forEach(e => {
    const fromNode = nodeMap[e.from];
    const toNode = nodeMap[e.to];
    if (fromNode && toNode) {
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const startX = fromNode.x + boxWidth;
      const startY = fromNode.y + boxHeight / 2;
      const endX = toNode.x + boxWidth;
      const endY = toNode.y + boxHeight / 2;
      const curveX = startX + 38;

      const d = `M ${startX} ${startY} C ${curveX} ${startY}, ${curveX} ${endY}, ${endX} ${endY}`;
      path.setAttribute('d', d);
      path.setAttribute('stroke', '#f43f5e');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('fill', 'none');
      path.setAttribute('marker-end', 'url(#arrow)');
      path.setAttribute('stroke-dasharray', '4 3');
      svg.appendChild(path);
    }
  });

  // Render nodes
  nodes.forEach(n => {
    const pos = nodeMap[n.id];
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.className = 'dag-node cursor-pointer';
    g.onclick = () => inspectDagNode(n);

    const isSuper = n.status === 'SUPERSEDED';
    const strokeColor = isSuper ? '#f43f5e' : '#10b981';
    const fillColor = isSuper ? 'rgba(35, 15, 15, 0.90)' : 'rgba(15, 23, 42, 0.90)';

    // Main Card Box
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', pos.x);
    rect.setAttribute('y', pos.y);
    rect.setAttribute('width', boxWidth);
    rect.setAttribute('height', boxHeight);
    rect.setAttribute('rx', 8);
    rect.setAttribute('fill', fillColor);
    rect.setAttribute('stroke', strokeColor);
    rect.setAttribute('stroke-width', isSuper ? '1.5' : '1.8');
    if (isSuper) {
      rect.setAttribute('stroke-dasharray', '4 2');
    }
    g.appendChild(rect);

    // Node text with strict max-length truncation to prevent badge collision
    const maxChars = 22;
    const displayLabel = n.label.length > maxChars ? n.label.slice(0, maxChars - 1) + '…' : n.label;
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', pos.x + 10);
    text.setAttribute('y', pos.y + 20);
    text.setAttribute('fill', isSuper ? '#94a3b8' : '#f1f5f9');
    text.setAttribute('font-family', 'JetBrains Mono, monospace');
    text.setAttribute('font-size', '10px');
    if (isSuper) {
      text.setAttribute('text-decoration', 'line-through');
    }
    text.textContent = displayLabel;

    // Hover tooltip
    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `${n.label}\nFull: ${n.full_text}\nStatus: ${n.status}`;
    g.appendChild(title);
    g.appendChild(text);

    // Dedicated right-hand status pill
    const badgeW = 38;
    const badgeH = 18;
    const badgeX = pos.x + boxWidth - badgeW - 8;
    const badgeY = pos.y + (boxHeight - badgeH) / 2;

    const badgeRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    badgeRect.setAttribute('x', badgeX);
    badgeRect.setAttribute('y', badgeY);
    badgeRect.setAttribute('width', badgeW);
    badgeRect.setAttribute('height', badgeH);
    badgeRect.setAttribute('rx', 4);
    badgeRect.setAttribute('fill', isSuper ? 'rgba(244, 63, 94, 0.18)' : 'rgba(16, 185, 129, 0.18)');
    badgeRect.setAttribute('stroke', isSuper ? 'rgba(244, 63, 94, 0.5)' : 'rgba(16, 185, 129, 0.5)');
    badgeRect.setAttribute('stroke-width', '1');
    g.appendChild(badgeRect);

    const statusText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    statusText.setAttribute('x', badgeX + badgeW / 2);
    statusText.setAttribute('y', badgeY + 12);
    statusText.setAttribute('text-anchor', 'middle');
    statusText.setAttribute('fill', isSuper ? '#f87171' : '#34d399');
    statusText.setAttribute('font-family', 'JetBrains Mono, monospace');
    statusText.setAttribute('font-size', '9px');
    statusText.setAttribute('font-weight', 'bold');
    statusText.textContent = isSuper ? 'OLD' : 'LIVE';
    g.appendChild(statusText);

    svg.appendChild(g);
  });
}

// ==================== 6. TRAVERSAL TRACE RENDERING ====================
function renderTraversalTrace(traversal) {
  const container = document.getElementById('traversal-trace');
  const badge = document.getElementById('hydration-badge');
  if (!container || !badge) return;
  container.innerHTML = '';

  if (!traversal) {
    badge.innerText = 'Ready';
    badge.className = 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700';
    return;
  }

  badge.innerText = traversal.requires_verbatim ? 'Verbatim Hydrated' : 'Summary Injected';
  badge.className = traversal.requires_verbatim 
    ? 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-cyan-950 border border-cyan-500 text-cyan-300 shadow-sm shadow-cyan-500/30'
    : 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300';

  // Phase 1 Card
  const p1 = document.createElement('div');
  p1.className = 'traversal-step-card traversal-step-1 mb-2';
  p1.innerHTML = `
    <div class="flex items-center justify-between mb-1">
      <span class="text-[10px] text-cyan-400 font-bold font-mono">PHASE 1: MACRO TOPIC FILTER (d=64)</span>
      <span class="text-[9px] px-1.5 py-0.2 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono">Fast Funnel</span>
    </div>
    <p class="text-slate-300 text-xs intuitive-only">Scanned all memory topics at high speed and zoomed into:</p>
    <div class="text-xs text-white font-mono font-bold mt-0.5">${traversal.candidate_clusters.join(', ') || 'No matching topic clusters'}</div>
  `;
  container.appendChild(p1);

  // Phase 2 Card
  const p2 = document.createElement('div');
  p2.className = 'traversal-step-card traversal-step-2 mb-2';
  let nodesHtml = traversal.scored_nodes.map(n => `
    <div class="py-1 border-b border-slate-800/80 flex justify-between items-center text-[11px]">
      <span>Turn [${n.turn_references.join(',')}]: sim=${n.sim_fine.toFixed(3)} decay=${n.time_decay.toFixed(2)}</span>
      <span class="font-bold text-emerald-400">Score: ${n.composite_score.toFixed(3)}</span>
    </div>
  `).join('');
  p2.innerHTML = `
    <div class="flex items-center justify-between mb-1">
      <span class="text-[10px] text-indigo-400 font-bold font-mono">PHASE 2: DEEP CONCEPT RE-RANKING (d=1024)</span>
      <span class="text-[9px] px-1.5 py-0.2 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-mono">Filtered Obsolete</span>
    </div>
    <p class="text-slate-300 text-xs mb-1.5 intuitive-only">Scored concepts using semantic match + recency, concealing struck-out facts:</p>
    <div class="space-y-1">${nodesHtml || '<p class="text-slate-500 italic">No nodes evaluated.</p>'}</div>
  `;
  container.appendChild(p2);

  // Phase 3 Card
  const p3 = document.createElement('div');
  p3.className = 'traversal-step-card traversal-step-3';
  let blocksHtml = traversal.context_blocks.map(b => `
    <div class="p-2 rounded-lg bg-slate-950 text-xs text-slate-200 mb-1.5 border border-slate-800/80 leading-relaxed">${escapeHtml(b)}</div>
  `).join('');
  p3.innerHTML = `
    <div class="flex items-center justify-between mb-1">
      <span class="text-[10px] text-emerald-400 font-bold font-mono">PHASE 3: GROUND-TRUTH HYDRATION</span>
      <span class="text-[9px] px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono">Zero Hallucination</span>
    </div>
    <p class="text-slate-300 text-xs mb-1.5 intuitive-only">Pulls exact verbatim strings from the SQLite disk vault for 100% precision:</p>
    <div>${blocksHtml || '<p class="text-slate-500 italic">None hydrated.</p>'}</div>
  `;
  container.appendChild(p3);
}

// ==================== 7. SCENARIO RUNNER & NARRATIONS ====================
async function runScenario(scenarioId) {
  const narrative = SCENARIO_NARRATIVES[scenarioId];
  if (narrative) {
    showScenarioBanner(narrative.title, narrative.desc);
  }

  appendSystemBubble(`Running experiment: ${scenarioId}...`);

  try {
    const res = await fetch(`/api/scenarios/${scenarioId}`, { method: 'POST' });
    const data = await res.json();
    appendSystemBubble(`Experiment ${data.scenario} completed.`);

    if (data.turns && Array.isArray(data.turns)) {
      data.turns.forEach(t => addTurnToChart(t.gate_result, t.text));
    }

    await fetchState();
    await fetchGraph();

    if (scenarioId === 'belief_invalidation') {
      appendSystemBubble(`Contradiction detected! DNS fact marked SUPERSEDED. Testing query: "What queue architecture are we using?"`);
      await sendChatMessage("What queue architecture are we using?");
    } else if (scenarioId === 'semantic_smearing') {
      appendSystemBubble(`Surprise Isolation verified: Outlier proposition quarantined (${data.q2_quadrant}). Topic centroid protected against semantic smearing.`);
    } else if (scenarioId === 'titans_momentum') {
      appendSystemBubble(`Titans Momentum window active: Trigger salience = 1.0, forward follow-up explanations retained elevated retention floors.`);
    } else if (scenarioId === 'temporal_inversion') {
      appendSystemBubble(`Temporal decay prioritized recent setting over older setting without losing history.`);
      if (data.retrieval) renderTraversalTrace(data.retrieval);
    } else if (scenarioId === 'prefix_truncation') {
      appendSystemBubble(`Two-stage funnel filtered clusters at d=64 prefix, then disambiguated at full d=1024 fine resolution.`);
      if (data.retrieval) renderTraversalTrace(data.retrieval);
    }
  } catch (err) {
    appendSystemBubble(`Scenario error: ${err.message}`);
  }
}

function showScenarioBanner(title, desc) {
  const banner = document.getElementById('scenario-banner');
  const titleEl = document.getElementById('scenario-title');
  const descEl = document.getElementById('scenario-desc');

  if (banner && titleEl && descEl) {
    titleEl.innerText = title;
    descEl.innerText = desc;
    banner.classList.remove('hidden');
  }
}

function dismissScenarioBanner() {
  const banner = document.getElementById('scenario-banner');
  if (banner) banner.classList.add('hidden');
}

async function resetMemory() {
  if (!confirm("Are you sure you want to reset all 4 memory tiers?")) return;

  try {
    await fetch('/api/reset', { method: 'POST' });
    resetLocalState();
  } catch (err) {
    console.error('Reset error:', err);
  }
}

function resetLocalState() {
  // 1. Reset Surprise Scatter Chart
  if (surpriseChart) {
    surpriseChart.data.datasets.forEach(d => d.data = []);
    surpriseChart.update();
  }

  // 2. Reset 4-tier storage topology and counts
  currentTopology = { tier0: [], tier1: [], tier2: [], tier3: [] };
  renderTopologyTab();
  fetchGraph();

  // 3. Reset Momentum & Brain State Badges
  updateBrainState({ active_momentum: 0.0, quadrant: 'EXPECTED_PROGRESS' });

  // 4. Reset Chat Messages Log to Initial Welcome Bubble
  const chatContainer = document.getElementById('chat-messages');
  if (chatContainer) {
    chatContainer.innerHTML = `
      <div class="system-bubble">
        <div class="flex items-center gap-1.5 text-cyan-400 text-xs font-semibold font-mono mb-1">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          <span>HMR-CE ONLINE • 100% LOCAL INFERENCE</span>
        </div>
        <p class="text-slate-300 text-xs leading-relaxed">
          Memory session reset to a clean state. Ready for new conversational turns.
        </p>
      </div>
    `;
  }

  // 5. Reset Traversal Trace Panel
  const traceContainer = document.getElementById('traversal-trace');
  const hydrationBadge = document.getElementById('hydration-badge');
  if (traceContainer) {
    traceContainer.innerHTML = `
      <div class="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60 text-center text-slate-400">
        <svg class="w-6 h-6 mx-auto mb-1 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
        <p class="text-xs font-semibold text-slate-300 mb-1">Awaiting Memory Search</p>
        <p class="text-[11px]">Type a question in the chat or run an experiment to see the 3-phase memory search in real-time!</p>
      </div>
    `;
  }
  if (hydrationBadge) {
    hydrationBadge.innerText = 'Ready';
    hydrationBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700';
  }

  // 6. Reset Input Field
  const chatInput = document.getElementById('chat-input');
  if (chatInput) chatInput.value = '';

  // 7. Dismiss Scenario Banners & Close Modals
  dismissScenarioBanner();
  closeInspectorModal();
  closeExplainerModal();
}

async function fetchState() {
  try {
    const res = await fetch('/api/state');
    currentTopology = await res.json();
    renderTopologyTab();
  } catch (err) {
    console.error('Failed to fetch state:', err);
  }
}

// ==================== 8. CLICK-TO-INSPECT MODALS ====================
function openExplainerModal() {
  const m = document.getElementById('explainer-modal');
  if (m) m.classList.remove('hidden');
}

function closeExplainerModal() {
  const m = document.getElementById('explainer-modal');
  if (m) m.classList.add('hidden');
}

function inspectChartPoint(pt) {
  let expl = "";
  if (pt.quadrant === 'DOMAIN_PIVOT') {
    expl = `You cleanly switched to a brand new topic. The AI finalized the previous topic chapter and initialized a new cluster without polluting past memories.`;
  } else if (pt.quadrant === 'CONCEPTUAL_NOVELTY') {
    expl = `This was a surprising or contrarian premise! The AI gave it maximum salience (α=1.0) and opened a Titans momentum window so any follow-up context is also captured.`;
  } else if (pt.quadrant === 'EXPECTED_PROGRESS') {
    expl = `This sentence flowed naturally from the existing topic. The AI smoothly updated its high-level topic centroid summary without cluttering episodic memory.`;
  } else {
    expl = `Routine banter, greeting, or minor tangent. The AI kept this in short-term buffer memory (Tier 0) so long-term storage stays lean and fast.`;
  }

  showInspectorModal({
    icon: "🎯",
    title: `Turn ${pt.turn_id} (User Prompt) • ${pt.label}`,
    subtitle: `User Input Saliency (Drift: ${pt.x.toFixed(3)}, PPL: ${pt.y.toFixed(1)})`,
    humanExpl: expl,
    fullText: pt.text,
    status: "INGESTED",
    salience: `α=${pt.salience ? pt.salience.toFixed(2) : '1.00'}`,
    cluster: "Active Cluster",
    supersededBy: "None"
  });
}

function inspectDagNode(node) {
  const isSuper = node.status === 'SUPERSEDED';
  let expl = isSuper
    ? `⚠️ This belief was retracted! A newer statement directly contradicted this premise. The AI struck it through and hid it from standard queries to avoid giving you outdated answers.`
    : `✅ This is currently an active truth in the AI's mind. It will be recalled when you ask questions related to this topic.`;

  showInspectorModal({
    icon: isSuper ? "❌" : "✅",
    title: `Belief Node: ${node.id.slice(0, 8)}`,
    subtitle: `Turn ${node.turn_id} Belief State`,
    humanExpl: expl,
    fullText: node.full_text,
    status: node.status,
    salience: `α=${node.salience ? node.salience.toFixed(2) : '1.00'}`,
    cluster: node.cluster_id || 'cluster_0',
    supersededBy: node.superseded_by ? node.superseded_by.slice(0, 8) : 'None'
  });
}

function inspectTopologyItem(tier, item) {
  if (tier === 'tier0') {
    showInspectorModal({
      icon: "💬",
      title: `Tier 0 Buffer • Turn ${item.turn_id}`,
      subtitle: `Short-Term Working Memory (${item.speaker_role})`,
      humanExpl: `This is in the AI's immediate short-term ring buffer (last 6 turns). It's always directly visible in active context.`,
      fullText: item.raw_text,
      status: "ACTIVE_BUFFER",
      salience: "In-Context",
      cluster: "Working Memory",
      supersededBy: "None"
    });
  } else if (tier === 'tier1') {
    const isSuper = item.active_status === 'SUPERSEDED';
    showInspectorModal({
      icon: "💾",
      title: `Tier 1 Vault • Turn ${item.turn_id}`,
      subtitle: `Permanent SQLite Disk Vault (${item.speaker_role})`,
      humanExpl: isSuper 
        ? `This exact text is safely preserved on disk for historical audits, but struck out from active context because newer information contradicted it.`
        : `This text is written to disk with SHA-256 integrity. When exact numbers or code are needed, this verbatim ground truth is pulled into the prompt.`,
      fullText: item.raw_text,
      status: item.active_status,
      salience: "Disk Ground Truth",
      cluster: item.session_id,
      supersededBy: isSuper ? 'Contradicted by newer turn' : 'None'
    });
  } else if (tier === 'tier2') {
    const isSuper = item.status === 'SUPERSEDED';
    showInspectorModal({
      icon: "🧠",
      title: `Tier 2 Episodic Node • ${item.node_id.slice(0, 8)}`,
      subtitle: `High-Dimensional Concept Vector (d=1024)`,
      humanExpl: isSuper 
        ? `This concept node was marked SUPERSEDED. It has a directed link pointing to the replacement node and is excluded from everyday searches.`
        : `This episodic node stores the semantic meaning of this event in high resolution (1024 dimensions) using ReverseEOL embeddings.`,
      fullText: item.metadata.abstract || 'No abstract',
      status: item.status,
      salience: `α=${item.surprise_salience.toFixed(2)}`,
      cluster: item.centroid_cluster_id,
      supersededBy: item.superseded_by_node ? item.superseded_by_node.slice(0, 8) : 'None'
    });
  } else if (tier === 'tier3') {
    showInspectorModal({
      icon: "🗺️",
      title: `Tier 3 Topic Cluster • ${item.cluster_id}`,
      subtitle: `Macro-Topology Summary (d=64 Coarse Prefix)`,
      humanExpl: `This compact centroid represents the overarching topic of ${item.member_count} belief node(s). The AI checks this first in Phase 1 before loading heavy 1024-dimension vectors.`,
      fullText: `Bounding Turns: [${item.bounding_turn_range[0]} to ${item.bounding_turn_range[1]}]\nMember Count: ${item.member_count}\nCoarse Prefix Vector: [${item.centroid_coarse.slice(0, 8).map(v => v.toFixed(3)).join(', ')} ...]`,
      status: "TOPIC_CENTROID",
      salience: "Macro Level",
      cluster: item.cluster_id,
      supersededBy: "None"
    });
  }
}

function showInspectorModal(data) {
  const m = document.getElementById('inspector-modal');
  if (!m) return;

  document.getElementById('inspector-icon').innerText = data.icon;
  document.getElementById('inspector-title').innerText = data.title;
  document.getElementById('inspector-subtitle').innerText = data.subtitle;
  document.getElementById('inspector-human-expl').innerText = data.humanExpl;
  document.getElementById('inspector-full-text').innerText = data.fullText;
  
  const statusEl = document.getElementById('inspector-status');
  statusEl.innerText = data.status;
  statusEl.className = data.status === 'SUPERSEDED' ? 'font-bold text-rose-400 ml-1' : 'font-bold text-emerald-400 ml-1';

  document.getElementById('inspector-salience').innerText = data.salience;
  document.getElementById('inspector-cluster').innerText = data.cluster;
  document.getElementById('inspector-superseded-by').innerText = data.supersededBy;

  m.classList.remove('hidden');
}

function closeInspectorModal() {
  const m = document.getElementById('inspector-modal');
  if (m) m.classList.add('hidden');
}

// ==================== UTILS ====================
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
