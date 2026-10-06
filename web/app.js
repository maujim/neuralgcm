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
  config: null, initial: null, archived: null, jobs: {}, frames: { a: {}, b: {} }, provenance: {}, tokens: { a: 0, b: 0 },
  inFlight: { a: new Map(), b: new Map() }, field: 'temperature', index: 0, city: cities[0],
  view: 'starting', playing: false, playTimer: null, bEnabled: false, submitting: { a: false, b: false }, requestedModels: {},
};
let globe;
const advancedGlobes = {};

async function api(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error?.message || `Request failed (${response.status})`);
  return payload;
}
function setMessage(message, isError = false) {
  const element = $('#run-status'); element.textContent = message || ''; element.classList.toggle('error', isError);
}
function activeIds() { return IDS.filter((id) => id === 'a' || state.bEnabled); }
function clearScenario(id) {
  state.jobs[id] = null; state.frames[id] = {}; state.provenance[id] = null; state.inFlight[id].clear();
}
function fillConfig(config) {
  state.config = config;
  const preferredModel = config.models.find((item) => item.id === 'deterministic_2_8_deg' && item.available)
    || config.models.find((item) => item.id === 'toy_tl63' && item.available)
    || config.models.find((item) => item.id === config.defaults.model_id && item.available);
  IDS.forEach((id) => {
    const duration = $(`#duration-${id}`);
    duration.replaceChildren(...config.duration_hours.map((hours) => new Option(`${hours} hours`, hours)));
    duration.value = String(Math.min(1, ...config.duration_hours));
    const model = $(`#model-${id}`);
    model.replaceChildren(...config.models.map((item) => {
      const label = item.id === 'toy_tl63' ? `${item.label} — toy model` : item.label;
      const option = new Option(`${label}${item.available ? '' : ` — unavailable: ${item.unavailable_reason || 'not installed'}`}`, item.id);
      option.disabled = !item.available;
      return option;
    }));
    if (preferredModel) model.value = preferredModel.id;
    updateSeed(id);
  });
  const fields = config.fields.map((field) => new Option(`${field.label} (${field.unit})`, field.id));
  $('#field-select').replaceChildren(...fields.map((option) => option.cloneNode(true)));
  $('#advanced-field-select').replaceChildren(...fields);
  $('#field-select').value = state.field;
  $('#advanced-field-select').value = state.field;
  renderProvenance();
}
function updateSeed(id) {
  const model = state.config.models.find((item) => item.id === $(`#model-${id}`).value);
  const input = $(`#seed-${id}`);
  input.disabled = Boolean(model && !model.stochastic);
  $(`#seed-help-${id}`).textContent = input.disabled ? 'Deterministic model: seed is retained for provenance but has no stochastic effect.' : '';
}
function requestBody(id, durationOverride = null) {
  return { model_id: $(`#model-${id}`).value, duration_hours: durationOverride ?? Number($(`#duration-${id}`).value), seed: Number($(`#seed-${id}`).value) };
}
function stopPlayback() { state.playing = false; clearTimeout(state.playTimer); state.playTimer = null; $('#play').textContent = '▶ Play timeline'; }
function jobBusy(job) { return job && !['completed', 'failed', 'cancelled'].includes(job.status); }
async function run(id, durationOverride = null) {
  stopPlayback();
  const token = ++state.tokens[id];
  clearScenario(id); state.index = 0; state.view = 'starting';
  if (id === 'a') { state.initial = state.archived; $('#show-future').disabled = true; }
  state.submitting[id] = true;
  const body = requestBody(id, durationOverride);
  state.requestedModels[id] = body.model_id;
  setMessage('');
  renderJobs(); render();
  try {
    const job = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (token !== state.tokens[id]) return;
    state.submitting[id] = false;
    state.jobs[id] = job; state.provenance[id] = job.provenance || null; renderJobs(); renderProvenance(); await poll(id, token);
  } catch (error) {
    if (token === state.tokens[id]) { state.submitting[id] = false; setMessage(error.message, true); renderJobs(); }
  }
}
async function poll(id, token) {
  const job = state.jobs[id];
  if (!job || token !== state.tokens[id]) return;
  try {
    const updated = await api(`/api/jobs/${job.id}`);
    if (token !== state.tokens[id] || state.jobs[id]?.id !== job.id) return;
    state.jobs[id] = updated; state.provenance[id] = updated.provenance || state.provenance[id];
    renderJobs(); renderProvenance();
    await loadAllFrames(id, token);
    if (updated.status === 'failed') setMessage(updated.error?.message || 'Forecast failed. No prediction is available.', true);
    else if (updated.status === 'cancelled') setMessage('Queued forecast cancelled. You can submit another forecast.', false);
    else if (updated.status === 'completed') setMessage('Forecast complete. The returned frames are shown; predictions are not measurements.');
    else if (id === 'a') setMessage('');
    if (updated.status !== 'completed' && updated.status !== 'failed' && updated.status !== 'cancelled') setTimeout(() => poll(id, token), 700);
  } catch (error) {
    if (token === state.tokens[id]) { setMessage(error.message, true); if (jobBusy(state.jobs[id])) setTimeout(() => poll(id, token), 1500); }
  }
}
async function clearQueued() {
  const button = $('#clear-queued'); const status = $('#clear-status');
  button.disabled = true; status.textContent = '';
  try {
    const { cleared_count: clearedCount } = await api('/api/jobs/clear', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
    });
    status.textContent = `Cleared ${clearedCount} queued job${clearedCount === 1 ? '' : 's'}. Running forecasts continue.`;
    await Promise.all(IDS.map((id) => state.jobs[id] ? poll(id, state.tokens[id]) : Promise.resolve()));
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
}
async function loadFrame(id, index, token = state.tokens[id]) {
  const job = state.jobs[id];
  if (!job || token !== state.tokens[id] || index >= (job.available_frames || 0) || state.frames[id][index]) return;
  if (state.inFlight[id].has(index)) return state.inFlight[id].get(index);
  const request = api(`/api/jobs/${job.id}/frames/${index}`).then((frame) => {
    if (token === state.tokens[id] && state.jobs[id]?.id === job.id) {
      state.frames[id][index] = frame;
      if (id === 'a' && index === 0) state.initial = { grid: job.grid, frame };
    }
  }).catch((error) => { if (token === state.tokens[id]) setMessage(error.message, true); }).finally(() => { if (state.inFlight[id].get(index) === request) state.inFlight[id].delete(index); });
  state.inFlight[id].set(index, request);
  return request;
}
async function loadAllFrames(id, token) {
  const job = state.jobs[id];
  if (!job || token !== state.tokens[id]) return;
  await Promise.all(Array.from({ length: job.available_frames || 0 }, (_, index) => loadFrame(id, index, token)));
  if (token === state.tokens[id]) { renderJobs(); updateTimeline(); render(); }
}
function phaseLabel(phase) {
  const value = String(phase || '').toLowerCase();
  if (value.includes('load')) return 'Loading NeuralGCM with MLX on this Mac';
  if (value.includes('encod') || value.includes('prepar')) return 'Preparing global atmosphere inputs';
  if (value.includes('forecast') || value.includes('advance') || value.includes('step') || value.includes('integrat')) return 'Predicting global atmosphere';
  if (value.includes('decod')) return 'Reading returned prediction';
  return phase ? String(phase).replaceAll('_', ' ') : 'Forecast in progress';
}
function stage(text, status) {
  const item = document.createElement('div'); item.className = `lesson-stage ${status}`; item.textContent = text; return item;
}
function renderJobs() {
  const list = $('#job-list'); list.replaceChildren();
  const job = state.jobs.a; const startingReady = Boolean(state.initial?.frame);
  list.append(
    stage(startingReady ? '1 · Starting weather read' : '1 · Reading starting weather', startingReady ? 'complete' : 'active'),
    stage(state.submitting.a ? '2 · Submitting to forecast queue' : job?.status === 'queued' ? '2 · Waiting in local queue' : jobBusy(job) ? `2 · ${phaseLabel(job.phase)}` : job?.status === 'completed' ? '2 · Model run complete' : job?.status === 'failed' ? '2 · Model run failed' : job?.status === 'cancelled' ? '2 · Queue cancelled' : '2 · Run the model when ready', state.submitting.a || jobBusy(job) ? 'active' : job?.status === 'completed' ? 'complete' : job?.status === 'failed' ? 'failed' : 'waiting'),
    stage(state.frames.a[targetFrameIndex(job)] ? `3 · Prediction +${state.frames.a[targetFrameIndex(job)].lead_hours}h returned` : job?.status === 'failed' ? '3 · Prediction unavailable' : job?.status === 'cancelled' ? '3 · No prediction returned' : `3 · Waiting for +${targetFrameIndex(job)}h prediction`, state.frames.a[targetFrameIndex(job)] ? 'complete' : 'waiting'),
  );
  $('#predict-one-hour').disabled = !startingReady || state.submitting.a || jobBusy(job);
  $('#predict-three-hours').disabled = !startingReady || state.submitting.a || jobBusy(job);
  $('#run-a').disabled = !startingReady || state.submitting.a || jobBusy(job);
  $('#run-b').disabled = !startingReady || state.submitting.b || jobBusy(state.jobs.b);
  updateTimeline();
}
function renderProvenance() {
  const dl = $('#provenance'); if (!dl) return;
  dl.replaceChildren();
  const entries = Object.entries(state.provenance).flatMap(([id, provenance]) => provenance ? [[`scenario_${id}`, ''], ...Object.entries(provenance)] : []);
  if (!entries.length && state.initial?.provenance) entries.push(...Object.entries(state.initial.provenance));
  if (!entries.length && state.config?.provenance) entries.push(...Object.entries(state.config.provenance));
  entries.forEach(([key, value]) => { const dt = document.createElement('dt'); dt.textContent = key.replaceAll('_', ' '); const dd = document.createElement('dd'); dd.textContent = Array.isArray(value) ? value.join('; ') : String(value ?? '—'); dl.append(dt, dd); });
}
function nearest(grid, city) {
  if (!grid || !city) return null; let best = null; let distance = Infinity;
  grid.latitudes.forEach((latitude, row) => grid.longitudes.forEach((longitude, column) => {
    const dLat = latitude - city[1]; const deltaLon = Math.abs(longitude - city[2]);
    const dLon = Math.min(deltaLon, 360 - deltaLon) * Math.cos(city[1] * Math.PI / 180); const d = dLat * dLat + dLon * dLon;
    if (d < distance) { distance = d; best = row * grid.longitudes.length + column; }
  }));
  return best;
}
function selectedFrame() {
  if (state.view === 'timeline') return state.frames.a[state.index] ?? (state.index === 0 ? state.initial?.frame || null : null);
  if (state.view === 'future') return currentHorizonFrame();
  return state.initial?.frame || null;
}
function requestedDuration(job) { return Number(job?.request?.duration_hours ?? job?.duration_hours ?? 1); }
function targetFrameIndex(job) { return requestedDuration(job); }
function currentHorizonFrame() {
  const job = state.jobs.a;
  if (!job) return null;
  return state.frames.a[targetFrameIndex(job)] || null;
}
function formatValue(value, unit) { return Number.isFinite(value) ? `${value.toFixed(2)} ${unit}` : 'Not available in returned frame'; }
function fieldInfo() { return state.config?.fields.find((item) => item.id === state.field) || { id: 'temperature', label: 'Temperature', unit: '°C' }; }
function renderValue() {
  const field = fieldInfo(); const frame = selectedFrame(); const grid = state.initial?.grid;
  const index = nearest(grid, state.city); const value = index == null ? NaN : frame?.fields?.[field.id]?.[index];
  const lead = frame?.lead_hours || 0; const displayName = frame ? frame.kind === 'initial_reanalysis' ? 'Starting weather' : `${lead} hour${lead === 1 ? '' : 's'} later` : `No returned frame at +${state.index}h`;
  $('#comparison-value-label').textContent = `${state.city[0]} · ${displayName.toLowerCase()} · ${field.label.toLowerCase()}`;
  $('#starting-value').textContent = formatValue(value, field.unit);
  $('#starting-time').textContent = frame?.kind === 'initial_reanalysis'
    ? `ERA5 · ${frame.valid_time} · 850 hPa upper-air value.`
    : frame ? `Prediction +${lead}h · ${frame.valid_time} · 850 hPa; not a measurement.` : 'A real returned frame is not available yet.';
  const prediction = state.view === 'timeline' ? (frame?.kind === 'prediction' ? frame : null) : currentHorizonFrame();
  const startGrid = prediction ? state.jobs.a.grid : grid;
  const cityIndex = nearest(startGrid, state.city);
  const start = prediction ? state.frames.a[0]?.fields?.[field.id]?.[cityIndex] : value;
  const future = prediction?.fields?.[field.id]?.[cityIndex];
  $('#change-value').textContent = Number.isFinite(start) && Number.isFinite(future) ? `Returned ${field.label.toLowerCase()} change over ${prediction.lead_hours} hour${prediction.lead_hours === 1 ? '' : 's'}: ${future - start >= 0 ? '+' : ''}${(future - start).toFixed(2)} ${field.unit} (same model-grid cell).` : '';
}
function renderLegend(field, values, frameAvailable) {
  const legend = $('#legend-main'); legend.replaceChildren();
  const title = document.createElement('h2'); title.textContent = `${field.label} · ${field.unit}`;
  const range = document.createElement('p');
  const finite = values.filter(Number.isFinite);
  const scale = finite.length ? `${Math.min(...finite).toFixed(1)} to ${Math.max(...finite).toFixed(1)} ${field.unit}` : '';
  range.textContent = frameAvailable ? scale ? `Map scale: ${scale}` : 'Waiting for returned field values.' : `No returned frame at this time. ${scale ? `Scale for returned comparison frames: ${scale}` : ''}`;
  legend.append(title, range);
}
function render() {
  if (!state.initial || !globe) return;
  const field = fieldInfo();
  const frame = selectedFrame();
  const grid = frame?.kind === 'initial_reanalysis' ? state.initial.grid : state.jobs.a?.grid || state.initial.grid;
  const values = frame?.fields?.[field.id] || [];
  const comparisonFrames = [state.frames.a[0], currentHorizonFrame()].filter(Boolean);
  const scaleValues = [values, ...comparisonFrames.map((item) => item.fields?.[field.id] || [])].flat().filter(Number.isFinite);
  const range = scaleValues.length ? { min: Math.min(...scaleValues), max: Math.max(...scaleValues) } : { min: 0, max: 1 };
  globe.setFrame({ grid, frame, field: field.id, range });
  globe.setLocation({ name: state.city[0], latitude: state.city[1], longitude: state.city[2] }, { focus: false });
  $('#map-heading').textContent = `${frame?.kind === 'initial_reanalysis' ? 'Starting weather' : frame ? `Predicted weather · +${frame.lead_hours}h` : 'No forecast frame returned'} · ${state.city[0]} highlighted`;
  $('#map-caption').textContent = `${frame?.kind === 'initial_reanalysis' ? 'Archived global ERA5 starting conditions' : frame ? `NeuralGCM prediction · ${frame.valid_time}` : 'No matching forecast frame returned'} · 850 hPa`;
  renderLegend(field, scaleValues, Boolean(frame));
  const future = currentHorizonFrame();
  $('#show-future').disabled = !future;
  $('#show-future').textContent = future ? `${future.lead_hours} hour${future.lead_hours === 1 ? '' : 's'} later` : 'One hour later';
  $('#show-starting').classList.toggle('selected', state.view === 'starting');
  $('#show-future').classList.toggle('selected', state.view === 'future');
  $('#show-starting').setAttribute('aria-pressed', String(state.view === 'starting'));
  $('#show-future').setAttribute('aria-pressed', String(state.view === 'future'));
  const completed = state.jobs.a?.status === 'completed' && Boolean(state.frames.a[1]);
  $('#exploration').hidden = !completed;
  $('#field-control').hidden = !completed;
  $('#timeline-panel').hidden = !completed;
  renderValue();
  renderProvenance();
  renderCity();
  updateTimelineLabel();
  renderAdvanced();
  const job = state.jobs.a;
  const provenance = job?.provenance || state.provenance.a || {};
  const modelIdValue = modelId('a');
  const model = state.config?.models.find((item) => item.id === modelIdValue);
  const modelNameValue = model?.label || provenance.model_name || modelIdValue || 'local MLX model';
  $('#model-note').textContent = job ? `NeuralGCM · ${modelNameValue}${modelIdValue === 'toy_tl63' ? ' · toy model' : ''} · run with MLX on this Mac.` : '';
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
  const city = state.city; const chart = $('#city-chart'); chart.replaceChildren();
  activeIds().forEach((id) => {
    const job = state.jobs[id];
    if (!job || !state.frames[id][0]) return;
    const grid = job.grid; const index = nearest(grid, city);
    const section = document.createElement('section'); section.className = 'scenario-city';
    const heading = document.createElement('h3'); heading.textContent = `Scenario ${id.toUpperCase()} · ${modelName(id)}`; section.append(heading);
    const cards = document.createElement('div'); cards.className = 'scenario-city-grid';
    [['temperature', '°C', 'Temperature'], ['wind_speed', 'm/s', 'Wind speed'], ['specific_humidity', 'g/kg', 'Specific humidity']].forEach(([key, unit, label]) => {
      const card = document.createElement('div'); card.className = 'chart-card';
      const title = document.createElement('h4'); title.textContent = `${city[0]} · ${label}`; card.append(title);
      const note = document.createElement('p'); note.className = 'small'; note.textContent = 'Nearest point on this job’s returned grid; not a station measurement.'; card.append(note);
      const points = Object.keys(state.frames[id]).map(Number).sort((a, b) => a - b).map((frameIndex) => {
        const frame = state.frames[id][frameIndex];
        return { value: frame?.fields?.[key]?.[index], lead_hours: frame?.lead_hours, kind: frame?.kind, label: frame?.kind === 'initial_reanalysis' ? 'ERA5' : `+${frame?.lead_hours ?? frameIndex}h` };
      });
      card.insertAdjacentHTML('beforeend', chartSvg(points, unit)); cards.append(card);
    });
    section.append(cards); chart.append(section);
  });
}
function modelId(id) {
  const job = state.jobs[id]; const provenance = job?.provenance || state.provenance[id] || {};
  return job?.request?.model_id || provenance.model_id || job?.model_id || state.requestedModels[id] || '';
}
function modelName(id) {
  const idValue = modelId(id); const model = state.config?.models.find((item) => item.id === idValue);
  const provenance = state.jobs[id]?.provenance || state.provenance[id] || {};
  return model?.label || provenance.model_name || idValue || 'Local model';
}
function renderAdvanced() {
  if (!advancedGlobes.a || !advancedGlobes.b) return;
  $('#advanced-results').hidden = !activeIds().some((id) => state.jobs[id] || state.submitting[id]);
  $('#globe-card-a').hidden = !state.jobs.a;
  $('#globe-card-b').hidden = !state.bEnabled || (!state.jobs.b && !state.submitting.b);
  const mapValues = activeIds().flatMap((id) => [state.frames[id][0], state.frames[id][state.index]].filter(Boolean).map((frame) => frame.fields?.[state.field] || []));
  const values = mapValues.flat().filter(Number.isFinite);
  const range = values.length ? { min: Math.min(...values), max: Math.max(...values) } : { min: 0, max: 1 };
  const field = fieldInfo();
  $('#advanced-results h2').textContent = range.min === range.max ? `A/B results · ${field.label} · shared scale ${range.min.toFixed(1)} ${field.unit}` : `A/B results · ${field.label} · shared scale ${range.min.toFixed(1)} to ${range.max.toFixed(1)} ${field.unit}`;
  IDS.forEach((id) => {
    const job = state.jobs[id]; const frame = state.frames[id][state.index] ?? (state.index === 0 ? state.frames[id][0] || null : null);
    advancedGlobes[id].setFrame({ grid: job?.grid || null, frame, field: state.field, range });
    advancedGlobes[id].setLocation({ name: state.city[0], latitude: state.city[1], longitude: state.city[2] }, { focus: false });
    $(`#advanced-label-${id}`).textContent = frame ? `${frame.valid_time} · ${modelName(id)}` : job ? `${job.phase || job.status} · ${modelName(id)}` : 'Waiting for forecast';
    const card = $(`#globe-card-${id}`);
    let caption = card.querySelector('.advanced-map-note');
    if (!caption) { caption = document.createElement('p'); caption.className = 'advanced-map-note'; card.append(caption); }
    caption.textContent = frame ? `${frame.kind === 'initial_reanalysis' ? 'ERA5 starting weather' : `Prediction +${frame.lead_hours}h`} · shared field scale` : 'No returned frame at this selected time.';
  });
  const statuses = $('#advanced-job-list'); statuses.replaceChildren();
  activeIds().forEach((id) => {
    if (state.jobs[id] || state.submitting[id]) {
      const status = document.createElement('p'); status.textContent = state.submitting[id] ? `Scenario ${id.toUpperCase()}: submitting…` : `Scenario ${id.toUpperCase()}: ${state.jobs[id].status === 'queued' ? 'waiting in local queue' : jobBusy(state.jobs[id]) ? phaseLabel(state.jobs[id].phase) : state.jobs[id].status}`;
      statuses.append(status);
    }
  });
}
async function sync() {
  const selected = state.index; const ids = activeIds();
  const tokens = Object.fromEntries(ids.map((id) => [id, state.tokens[id]]));
  await Promise.all(ids.map((id) => loadFrame(id, selected, tokens[id])));
  if (selected === state.index && ids.every((id) => tokens[id] === state.tokens[id])) { renderCity(); render(); }
}
function play() {
  if (!state.playing) return;
  const max = Number($('#timeline').max);
  if (state.index >= max) { stopPlayback(); return; }
  state.index += 1; $('#timeline').value = state.index;
  sync().finally(() => { state.playTimer = setTimeout(play, 450); });
}
function updateTimeline() {
  const max = Math.max(0, ...activeIds().map((id) => state.jobs[id]?.available_frames || 0)) - 1;
  const bounded = Math.max(0, max); state.index = Math.min(state.index, bounded);
  $('#timeline').max = bounded; $('#timeline').value = state.index;
  updateTimelineLabel();
}
function updateTimelineLabel() {
  const frame = state.view === 'timeline' ? state.frames.a[state.index] ?? (state.index === 0 ? state.initial?.frame : null) : selectedFrame();
  const label = frame ? `${frame.kind === 'initial_reanalysis' ? 'ERA5 starting conditions' : `Prediction +${frame.lead_hours}h`} · ${frame.valid_time}` : state.index ? `No Scenario A prediction frame returned at +${state.index}h` : 'ERA5 starting conditions';
  $('#timeline-label').textContent = label;
}
function setView(view) {
  state.view = view;
  if (view === 'starting') state.index = 0;
  else if (view === 'future') state.index = targetFrameIndex(state.jobs.a);
  $('#timeline').value = state.index;
  render();
}
async function init() {
  try {
    fillConfig(await api('/api/config'));
    $('#run-a').disabled = true; $('#run-b').disabled = true;
    globe = createGlobe($('#globe-main'));
    globe.setLocation({ name: 'London', latitude: 51.51, longitude: -0.13 }, { focus: true });
    advancedGlobes.a = createGlobe($('#globe-a'));
    advancedGlobes.b = createGlobe($('#globe-b'));
    $('#model-a').onchange = () => updateSeed('a'); $('#model-b').onchange = () => updateSeed('b');
    $('#run-a').onclick = () => run('a'); $('#run-b').onclick = () => run('b');
    $('#predict-one-hour').onclick = () => run('a', 1);
    $('#predict-three-hours').onclick = () => run('a', 3);
    $('#field-select').onchange = (event) => { state.field = event.target.value; $('#advanced-field-select').value = state.field; render(); };
    $('#advanced-field-select').onchange = (event) => { state.field = event.target.value; $('#field-select').value = state.field; render(); };
    $('#show-starting').onclick = () => setView('starting'); $('#show-future').onclick = () => { if (!$('#show-future').disabled) setView('future'); };
    $('#city-select').replaceChildren(...cities.map(([name]) => new Option(name, name)));
    $('#city-select').onchange = () => { state.city = cities.find(([name]) => name === $('#city-select').value); globe.setLocation({ name: state.city[0], latitude: state.city[1], longitude: state.city[2] }, { focus: true }); render(); };
    $('#toggle-b').onclick = () => {
      state.bEnabled = !state.bEnabled; $('#scenario-b').hidden = !state.bEnabled;
      $('#toggle-b').textContent = state.bEnabled ? 'Disable scenario B' : 'Enable scenario B';
      renderJobs(); render();
      if (state.bEnabled) requestAnimationFrame(() => advancedGlobes.b.resize());
    };
    $('#clear-queued').onclick = clearQueued;
    $('#timeline').oninput = async (event) => { state.index = Number(event.target.value); state.view = 'timeline'; await sync(); };
    $('#play').onclick = () => {
      if (state.playing) stopPlayback();
      else { state.view = 'timeline'; state.playing = true; $('#play').textContent = '⏸ Pause'; play(); }
    };
    window.addEventListener('resize', () => { globe.resize(); Object.values(advancedGlobes).forEach((item) => item.resize()); });
    document.querySelector('.advanced').addEventListener('toggle', (event) => { if (event.target.open) requestAnimationFrame(() => Object.values(advancedGlobes).forEach((item) => item.resize())); });
    $('#starting-value').textContent = 'Loading archived ERA5…';
    const initial = await api('/api/initial');
    if (!initial?.grid || !initial?.frame) throw new Error('The archived historical starting frame is unavailable.');
    state.initial = initial; state.archived = initial; $('#predict-one-hour').disabled = false; renderProvenance(); renderJobs(); render();
  } catch (error) {
    setMessage(error.message, true);
    $('#starting-value').textContent = 'Historical starting frame unavailable';
    $('#starting-time').textContent = error.message;
  }
}
init();
