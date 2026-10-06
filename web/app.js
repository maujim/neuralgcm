import { createGlobe } from './globe.js';

const $ = (selector) => document.querySelector(selector);
const IDS = ['a', 'b'];
const cities = [
  ['London', 51.51, -0.13], ['New York', 40.71, -74.01], ['Tokyo', 35.68, 139.69],
  ['Nairobi', -1.29, 36.82], ['Sydney', -33.87, 151.21], ['São Paulo', -23.55, -46.63],
  ['Reykjavík', 64.15, -21.94], ['Singapore', 1.29, 103.85], ['Mumbai', 19.08, 72.88],
  ['Mexico City', 19.43, -99.13],
];
const state = {
  config: null, jobs: {}, frames: { a: {}, b: {} }, provenance: {}, tokens: { a: 0, b: 0 },
  inFlight: { a: new Map(), b: new Map() }, field: 'temperature', index: 0, city: null,
  playing: false, playTimer: null, bEnabled: true,
};
const globes = {};

async function api(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || `Request failed (${response.status})`);
  return payload;
}
function setMessage(message) { $('#run-status').textContent = message || ''; }
function activeIds() { return IDS.filter((id) => id === 'a' || state.bEnabled); }
function clearGlobe(id) { globes[id]?.setFrame({ grid: null, frame: null, field: state.field, range: { min: 0, max: 1 } }); }
function clearScenario(id) { state.jobs[id] = null; state.frames[id] = {}; state.provenance[id] = null; state.inFlight[id].clear(); clearGlobe(id); }
function fillConfig(config) {
  state.config = config;
  IDS.forEach((id) => {
    const duration = $(`#duration-${id}`);
    duration.replaceChildren(...config.duration_hours.map((hours) => new Option(`${hours} hours`, hours)));
    duration.value = config.defaults.duration_hours;
    const model = $(`#model-${id}`);
    model.replaceChildren(...config.models.map((item) => {
      const option = new Option(`${item.label}${item.available ? '' : ` — unavailable: ${item.unavailable_reason || 'not installed'}`}`, item.id);
      option.disabled = !item.available;
      return option;
    }));
    model.value = config.defaults.model_id;
    updateSeed(id);
  });
  $('#field-select').replaceChildren(...config.fields.map((field) => new Option(`${field.label} (${field.unit})`, field.id)));
  $('#field-select').value = state.field;
  renderProvenance();
}
function updateSeed(id) {
  const model = state.config.models.find((item) => item.id === $(`#model-${id}`).value);
  const input = $(`#seed-${id}`);
  input.disabled = Boolean(model && !model.stochastic);
  $(`#seed-help-${id}`).textContent = input.disabled ? 'Deterministic model: seed is retained for provenance but has no stochastic effect.' : '';
}
function requestBody(id) { return { model_id: $(`#model-${id}`).value, duration_hours: Number($(`#duration-${id}`).value), seed: Number($(`#seed-${id}`).value) }; }
function stopPlayback() { state.playing = false; clearTimeout(state.playTimer); state.playTimer = null; $('#play').textContent = '▶ Play'; }
async function run(id) {
  stopPlayback();
  const token = ++state.tokens[id];
  clearScenario(id); state.index = 0; renderJobs(); render();
  try {
    const job = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(requestBody(id)) });
    if (token !== state.tokens[id]) return;
    state.jobs[id] = job; state.provenance[id] = job.provenance || null; renderJobs(); await poll(id, token);
  } catch (error) { if (token === state.tokens[id]) setMessage(error.message); }
}
async function poll(id, token) {
  const job = state.jobs[id];
  if (!job || token !== state.tokens[id]) return;
  try {
    const updated = await api(`/api/jobs/${job.id}`);
    if (token !== state.tokens[id] || state.jobs[id]?.id !== job.id) return;
    state.jobs[id] = updated; state.provenance[id] = updated.provenance || state.provenance[id]; renderJobs(); renderProvenance();
    if (updated.status === 'cancelled') {
      state.frames[id] = {};
      clearGlobe(id);
      renderJobs();
      updateTimeline();
      render();
      return;
    }
    await loadAllFrames(id, token);
    if (updated.status !== 'completed' && updated.status !== 'failed') setTimeout(() => poll(id, token), 700);
  } catch (error) { if (token === state.tokens[id]) setMessage(error.message); }
}
async function clearQueued() {
  const button = $('#clear-queued');
  const status = $('#clear-status');
  button.disabled = true;
  status.textContent = '';
  try {
    const { cleared_count: clearedCount } = await api('/api/jobs/clear', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    status.textContent = `Cleared ${clearedCount} queued job${clearedCount === 1 ? '' : 's'}.`;
    await Promise.all(IDS.map((id) => {
      const job = state.jobs[id];
      if (!job) return Promise.resolve();
      return poll(id, state.tokens[id]);
    }));
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}
async function loadFrame(id, index, token = state.tokens[id]) {
  const job = state.jobs[id];
  if (!job || token !== state.tokens[id] || index >= (job.available_frames || 0) || state.frames[id][index]) return;
  if (state.inFlight[id].has(index)) return state.inFlight[id].get(index);
  const request = api(`/api/jobs/${job.id}/frames/${index}`).then((frame) => {
    if (token === state.tokens[id] && state.jobs[id]?.id === job.id) state.frames[id][index] = frame;
  }).catch((error) => { if (token === state.tokens[id]) setMessage(error.message); }).finally(() => { if (state.inFlight[id].get(index) === request) state.inFlight[id].delete(index); });
  state.inFlight[id].set(index, request);
  return request;
}
async function loadAllFrames(id, token) {
  const job = state.jobs[id];
  if (!job || token !== state.tokens[id]) return;
  await Promise.all(Array.from({ length: job.available_frames || 0 }, (_, index) => loadFrame(id, index, token)));
  if (token === state.tokens[id]) { updateTimeline(); render(); }
}
function renderJobs() {
  const list = $('#job-list'); list.replaceChildren();
  IDS.forEach((id) => { const job = state.jobs[id]; if (!job) return; const item = document.createElement('div'); item.className = 'job'; const progress = job.progress || {}; item.textContent = job.status === 'cancelled' ? `Scenario ${id.toUpperCase()}: cleared from queue — forecast not run (${progress.completed_steps || 0}/${progress.total_steps || 0} steps)` : `Scenario ${id.toUpperCase()}: ${job.phase} — ${progress.completed_steps || 0}/${progress.total_steps || 0} steps`; if (job.error) { const error = document.createElement('span'); error.className = 'error'; error.textContent = ` ${job.error.message}`; item.append(document.createElement('br'), error); } list.append(item); });
  if (!list.children.length) list.innerHTML = '<p class="small">No forecast submitted.</p>';
  updateTimeline();
}
function renderProvenance() {
  const dl = $('#provenance'); dl.replaceChildren();
  const entries = Object.entries(state.provenance).flatMap(([id, provenance]) => provenance ? [[`scenario_${id}`, ''], ...Object.entries(provenance)] : []);
  if (!entries.length && state.config?.provenance) entries.push(...Object.entries(state.config.provenance));
  entries.forEach(([key, value]) => { const dt = document.createElement('dt'); dt.textContent = key.replaceAll('_', ' '); const dd = document.createElement('dd'); dd.textContent = Array.isArray(value) ? value.join('; ') : String(value ?? '—'); dl.append(dt, dd); });
}
function commonRange(field) {
  const values = activeIds().flatMap((id) => state.frames[id][state.index]?.fields[field] || []);
  return values.length ? { min: Math.min(...values), max: Math.max(...values) } : { min: 0, max: 1 };
}
function render() {
  const field = state.config?.fields.find((item) => item.id === state.field); if (!field) return;
  const range = commonRange(field.id);
  IDS.forEach((id) => { const grid = state.jobs[id]?.grid; if (globes[id]) globes[id].setFrame({ grid: grid || null, frame: state.frames[id][state.index] || null, field: state.field, range }); });
  const frame = activeIds().map((id) => state.frames[id][state.index]).find(Boolean);
  $('#timeline-label').textContent = frame ? `${frame.kind === 'initial_reanalysis' ? 'Initial reanalysis' : 'Prediction'} · ${frame.valid_time}` : 'Waiting for selected frame';
  if (state.city) renderCity();
}
function nearest(grid, city) {
  if (!grid || !city) return null; let best = null; let distance = Infinity;
  grid.latitudes.forEach((latitude, row) => grid.longitudes.forEach((longitude, column) => { const dLat = latitude - city[1]; const deltaLon = Math.abs(longitude - city[2]); const dLon = Math.min(deltaLon, 360 - deltaLon) * Math.cos(city[1] * Math.PI / 180); const d = dLat * dLat + dLon * dLon; if (d < distance) { distance = d; best = row * grid.longitudes.length + column; } })); return best;
}
function chartSvg(points, unit) {
  const validPoints = points.filter((point) => Number.isFinite(point.value) && Number.isFinite(point.lead_hours));
  if (!validPoints.length) return '<p class="unavailable">No returned values yet.</p>';
  const width = 560; const height = 190; const pad = 28; const values = validPoints.map((point) => point.value);
  const min = Math.min(...values); const max = Math.max(...values); const span = max - min || 1;
  const minLead = Math.min(...validPoints.map((point) => point.lead_hours)); const maxLead = Math.max(...validPoints.map((point) => point.lead_hours)); const leadSpan = maxLead - minLead || 1;
  const x = (point) => pad + (point.lead_hours - minLead) * (width - pad * 2) / leadSpan;
  const y = (point) => height - pad - ((point.value - min) / span) * (height - pad * 2);
  const path = validPoints.map((point) => `${x(point)},${y(point)}`).join(' ');
  const initial = validPoints.find((point) => point.kind === 'initial_reanalysis');
  const marker = initial ? `<circle class="era5-marker" cx="${x(initial)}" cy="${y(initial)}" r="5"/>` : '';
  const firstLabel = validPoints[0].label; const lastLabel = validPoints.at(-1).label;
  return `<svg class="city-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${unit} hourly values"><line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"/><polyline points="${path}"/>${marker}<g class="chart-labels"><text x="${pad}" y="${height - 7}">${firstLabel}</text><text x="${width - pad}" y="${height - 7}" text-anchor="end">${lastLabel}</text><text x="${pad}" y="14">${max.toPrecision(4)} ${unit}</text></g></svg>`;
}
function renderCity() {
  const cards = activeIds().filter((id) => state.jobs[id]); $('#city-chart').replaceChildren();
  cards.forEach((id) => { const grid = state.jobs[id].grid; const index = nearest(grid, state.city); const card = document.createElement('div'); card.className = 'chart-card'; const title = document.createElement('h3'); title.textContent = `Scenario ${id.toUpperCase()} · ${state.city[0]} (${state.city[1]}°, ${state.city[2]}°)`; card.append(title); const note = document.createElement('p'); note.className = 'small'; note.textContent = 'Nearest returned 850 hPa model-grid cell; initial ERA5 is distinct from forecast values.'; card.append(note);
    [['temperature', '°C'], ['wind_speed', 'm/s'], ['specific_humidity', 'g/kg']].forEach(([field, unit]) => { const heading = document.createElement('h4'); heading.textContent = `${field === 'wind_speed' ? 'Wind speed' : field === 'specific_humidity' ? 'Specific humidity' : 'Temperature'} (${unit})`; card.append(heading); const points = Object.keys(state.frames[id]).map(Number).sort((a, b) => a - b).map((frameIndex) => { const frame = state.frames[id][frameIndex]; return { value: frame?.fields?.[field]?.[index], lead_hours: frame?.lead_hours, kind: frame?.kind, label: frame?.kind === 'initial_reanalysis' ? 'ERA5' : `Prediction +${frame?.lead_hours ?? frameIndex}h` }; }); card.insertAdjacentHTML('beforeend', chartSvg(points, unit)); }); $('#city-chart').append(card); });
  if (!cards.length) $('#city-chart').innerHTML = '<p class="empty-state">Select a city after running a scenario.</p>';
}
function renderCities() { const query = $('#city-search').value.toLowerCase(); $('#city-results').replaceChildren(...cities.filter((city) => city[0].toLowerCase().includes(query)).map((city) => { const button = document.createElement('button'); button.type = 'button'; button.textContent = city[0]; button.onclick = () => { state.city = city; renderCities(); renderCity(); }; if (state.city?.[0] === city[0]) button.classList.add('selected'); return button; })); }
 
async function sync() {
  const selected = state.index;
  const tokenById = Object.fromEntries(activeIds().map((id) => [id, state.tokens[id]]));
  await Promise.all(activeIds().map((id) => loadFrame(id, selected, tokenById[id])));
  if (selected === state.index && activeIds().every((id) => tokenById[id] === state.tokens[id])) render();
}


function play() { if (!state.playing) return; const max = Number($('#timeline').max); if (state.index >= max) { stopPlayback(); return; } state.index += 1; $('#timeline').value = state.index; sync().finally(() => { state.playTimer = setTimeout(play, 450); }); }
function updateTimeline() { const max = Math.max(0, ...activeIds().map((id) => state.jobs[id]?.available_frames || 0)) - 1; const bounded = Math.max(0, max); state.index = Math.min(state.index, bounded); $('#timeline').max = bounded; $('#timeline').value = state.index; }
async function init() { try { fillConfig(await api('/api/config')); IDS.forEach((id) => { globes[id] = createGlobe($(`#globe-${id}`)); $(`#model-${id}`).onchange = () => updateSeed(id); $(`#run-${id}`).onclick = () => run(id); }); renderCities(); $('#city-search').oninput = renderCities; $('#field-select').onchange = (event) => { state.field = event.target.value; render(); }; $('#toggle-b').onclick = () => { state.bEnabled = !state.bEnabled; if (!state.bEnabled) { stopPlayback(); clearGlobe('b'); } $('#scenario-b').classList.toggle('disabled', !state.bEnabled); $('#scenario-b').hidden = !state.bEnabled; $('#globe-card-b').hidden = !state.bEnabled; $('#toggle-b').textContent = state.bEnabled ? 'Disable scenario B' : 'Enable scenario B'; updateTimeline(); render(); }; $('#timeline').oninput = async (event) => { state.index = Number(event.target.value); await sync(); }; $('#play').onclick = () => { state.playing ? stopPlayback() : (state.playing = true, $('#play').textContent = '⏸ Pause', play()); }; window.onresize = () => Object.values(globes).forEach((globe) => globe.resize()); updateTimeline(); } catch (error) { setMessage(error.message); } }
new MutationObserver(updateTimeline).observe($('#job-list'), { childList: true });
$('#clear-queued').onclick = clearQueued;
init();
