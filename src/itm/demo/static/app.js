const chains = [[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]];
const colors = { ground_truth: '#247ba0', generated: '#168f79' };
const state = {
  result: null, frame: 0, playing: false, lastTime: 0, panels: [], samples: [], split: 'test',
  selectedSamples: { a: null, b: null },
};
const el = id => document.getElementById(id);
const sampleKey = id => id === 'sample-a' ? 'a' : 'b';

function selectedSampleId(id) {
  return state.selectedSamples[sampleKey(id)];
}

function setSelectedSample(id, value) {
  state.selectedSamples[sampleKey(id)] = value || null;
}

function setCandidatesVisible(id, visible) {
  const wrapper = el(`${id}-options-wrap`);
  const toggle = el(`${id}-toggle`);
  wrapper.hidden = !visible;
  toggle.textContent = visible ? 'Hide list' : 'Show list';
  toggle.setAttribute('aria-expanded', String(visible));
}

function renderSamplePicker(id) {
  const filter = el(`${id}-filter`).value.trim().toLowerCase();
  const selected = selectedSampleId(id);
  const matches = state.samples.filter(sample =>
    !filter || sample.motion_id.toLowerCase().includes(filter) || sample.caption.toLowerCase().includes(filter)
  );
  const visible = matches.slice(0, 100);
  const selectedSample = state.samples.find(sample => sample.motion_id === selected);
  if (selectedSample && !visible.some(sample => sample.motion_id === selected)) visible.unshift(selectedSample);
  el(id).innerHTML = visible.map(sample =>
    `<option value="${escapeHtml(sample.motion_id)}">${escapeHtml(sample.motion_id)} · ${escapeHtml(sample.caption)}</option>`
  ).join('');
  el(id).value = selected || '';
}

async function loadSamples() {
  state.split = el('split').value;
  setStatus('Loading samples…');
  const response = await fetch(`/api/samples?split=${state.split}&limit=2000`);
  state.samples = await response.json();
  const chooseRandom = !selectedSampleId('sample-a');
  for (const id of ['sample-a', 'sample-b']) {
    const selected = selectedSampleId(id);
    if (!state.samples.some(sample => sample.motion_id === selected)) setSelectedSample(id, null);
  }
  if (chooseRandom) randomizeSamples();
  if (selectedSampleId('sample-b') === selectedSampleId('sample-a') && state.samples.length > 1) {
    setSelectedSample('sample-b', state.samples.find(sample => sample.motion_id !== selectedSampleId('sample-a')).motion_id);
  }
  renderSamplePicker('sample-a'); renderSamplePicker('sample-b');
  updateDetails();
  setStatus(`${state.samples.length} samples available`);
}

function randomizeSamples() {
  if (!state.samples.length) return;
  setSelectedSample('sample-a', state.samples[Math.floor(Math.random() * state.samples.length)].motion_id);
  do { setSelectedSample('sample-b', state.samples[Math.floor(Math.random() * state.samples.length)].motion_id); }
  while (state.samples.length > 1 && selectedSampleId('sample-b') === selectedSampleId('sample-a'));
  renderSamplePicker('sample-a'); renderSamplePicker('sample-b');
  updateDetails();
}

async function loadRunList() {
  const source = el('load-source').value;
  const root = el('experiment-root').value;
  el('experiment-root-row').hidden = source !== 'experiments';
  setStatus(`Loading ${source === 'experiments' ? 'experiment results' : 'runs'}…`);
  const endpoint = source === 'experiments' ? `/api/experiments?root=${encodeURIComponent(root)}&limit=200` : '/api/runs?limit=100';
  const response = await fetch(endpoint), runs = await response.json();
  el('recent-runs').innerHTML = runs.map(run => {
    const pair = run.sample_b ? `${run.sample_a} + ${run.sample_b}` : (run.sample_a || run.sensor_config || '');
    const metrics = run.active_sensor_error == null ? '' : ` · err ${Number(run.active_sensor_error).toFixed(3)} · jerk ${Number(run.jerk_ratio).toFixed(2)}`;
    const rootLabel = run.root_key ? `${run.root_key} · ` : '';
    return `<option value="${escapeHtml(run.run_id)}">${escapeHtml(rootLabel)}${escapeHtml(run.run_id)} · ${escapeHtml(run.mode)} · ${escapeHtml(pair)}${metrics}</option>`;
  }).join('');
  if (runs.length) el('load-run-id').value = runs[0].run_id;
  setStatus(`${runs.length} cached ${source === 'experiments' ? 'experiment results' : 'runs'} available`);
}

function setSidebarMode(mode) {
  const loading = mode === 'load';
  el('generate-pane').hidden = loading; el('load-pane').hidden = !loading;
  el('generate-tab').classList.toggle('active', !loading); el('load-tab').classList.toggle('active', loading);
  el('generate-tab').setAttribute('aria-selected', String(!loading)); el('load-tab').setAttribute('aria-selected', String(loading));
  if (loading) loadRunList().catch(error => setStatus(error.message, true));
}

function updateDetails() {
  const ids = el('mode').value === 'matrix' ? ['sample-a', 'sample-b'] : ['sample-a'];
  el('sample-details').innerHTML = ids.map((id, index) => {
    const sample = state.samples.find(value => value.motion_id === selectedSampleId(id));
    return sample ? `<div class="sample-detail"><strong>${index ? 'B' : 'A'} · ${sample.motion_id}</strong>${escapeHtml(sample.caption)}<br>${sample.frames} IMU frames</div>` : '';
  }).join('');
}

el('mode').addEventListener('change', () => { el('sample-b-row').hidden = el('mode').value !== 'matrix'; updateDetails(); });
el('split').addEventListener('change', loadSamples);
for (const id of ['sample-a', 'sample-b']) {
  el(`${id}-filter`).addEventListener('input', () => renderSamplePicker(id));
  el(id).addEventListener('change', () => { setSelectedSample(id, el(id).value); updateDetails(); });
  el(`${id}-toggle`).addEventListener('click', () => {
    setCandidatesVisible(id, el(`${id}-options-wrap`).hidden);
  });
}
el('randomize').addEventListener('click', randomizeSamples);
el('generate-tab').addEventListener('click', () => setSidebarMode('generate'));
el('load-tab').addEventListener('click', () => setSidebarMode('load'));
el('load-source').addEventListener('change', loadRunList);
el('experiment-root').addEventListener('change', loadRunList);
el('recent-runs').addEventListener('change', () => el('load-run-id').value = el('recent-runs').value);
el('load-run').addEventListener('click', async () => {
  const runId = el('load-run-id').value.trim();
  if (!runId) return;
  const source = el('load-source').value;
  setStatus(`Loading cached ${source === 'experiments' ? 'experiment' : 'run'}…`);
  try {
    const encoded = source === 'experiments' ? runId.split('/').map(encodeURIComponent).join('/') : encodeURIComponent(runId);
    const query = source === 'experiments' ? `?root=${encodeURIComponent(el('experiment-root').value)}` : '';
    const response = await fetch(`/api/${source}/${encoded}${query}`), data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Result not found');
    setResult(data); setStatus(`Run ${data.run_id} loaded`);
  } catch (error) { setStatus(error.message, true); }
});

el('generate-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = el('generate');
  button.disabled = true;
  setStatus('Generating on GPU…');
  const payload = {
    mode: el('mode').value, split: el('split').value, sample_a: selectedSampleId('sample-a'),
    sample_b: el('mode').value === 'matrix' ? selectedSampleId('sample-b') : null,
    sensor_config: el('sensor').value, seed: Number(el('seed').value),
    text_scale: Number(el('text-scale').value), imu_scale: Number(el('imu-scale').value), device: el('device').value
  };
  try {
    const response = await fetch('/api/generate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Generation failed');
    setResult(data);
    setStatus(`Run ${data.run_id} ready`);
  } catch (error) { setStatus(error.message, true); }
  finally { button.disabled = false; }
});

function setResult(result) {
  state.result = result; state.frame = 0; state.playing = false; state.panels = [];
  el('run-id').textContent = result.run_id; el('empty').hidden = true;
  el('timeline').max = result.frame_count - 1; el('timeline').value = 0; el('play').textContent = '▶';
  const isMatrix = result.request.mode === 'matrix' || result.request.experiment === 'matrix';
  if (isMatrix) {
    const groundTruth = result.panels.slice(0, 2).map((panel, index) => panelMarkup(panel, index, result)).join('');
    const combinations = result.panels.slice(2).map((panel, offset) => panelMarkup(panel, offset + 2, result)).join('');
    const info = `<article class="matrix-empty"><strong>None + None</strong>${runInfoList(result)}</article>`;
    el('panels').innerHTML = `<section class="gt-row">${groundTruth}</section><div class="matrix-scroll"><section class="matrix-grid">${info}${combinations}</section></div>`;
  } else {
    const gridClass = result.panels.length > 4 ? 'experiment-grid' : 'four-way-grid';
    el('panels').innerHTML = `${summaryMarkup(result)}<section class="${gridClass}">${result.panels.map((panel, index) => panelMarkup(panel, index, result)).join('')}</section>`;
  }
  result.panels.forEach((panel, index) => setupMotionCanvas(panel, el(`motion-${index}`)));
  renderIMU(result); renderFrame();
}

function summaryMarkup(result) {
  const sampleKeys = Object.keys(result.samples || {});
  const sampleText = sampleKeys.length <= 2 ? sampleKeys.map(key => result.samples[key].motion_id || key).join(' + ') : `${sampleKeys.length} samples`;
  const sensor = result.request.sensor_config || 'mixed';
  const experiment = result.request.experiment || result.request.mode || 'result';
  const source = result.source || {};
  const sourcePath = source.root_path ? `${source.root_path}/${source.relative_path || ''}` : '';
  return `<section class="run-summary"><span>Experiment<strong>${escapeHtml(experiment)}</strong></span><span>Sample<strong>${escapeHtml(sampleText || 'n/a')}</strong></span><span>Split<strong>${escapeHtml(result.request.split || 'n/a')}</strong></span><span>Sensors<strong>${escapeHtml(sensor)}</strong></span><span>Guidance<strong>Text ${result.request.text_scale} · IMU ${result.request.imu_scale}</strong></span><span>Seed<strong>${result.request.seed}</strong></span>${sourcePath ? `<span class="source-path">Source<strong>${escapeHtml(sourcePath)}</strong></span>` : ''}</section>`;
}

function runInfoList(result) {
  const source = result.source || {};
  const sourcePath = source.root_path ? `${source.root_path}/${source.relative_path || ''}` : '';
  return `<dl><dt>Run</dt><dd>${escapeHtml(result.run_id)}</dd>${sourcePath ? `<dt>Source</dt><dd>${escapeHtml(sourcePath)}</dd>` : ''}<dt>Seed</dt><dd>${result.request.seed}</dd><dt>Text scale</dt><dd>${result.request.text_scale}</dd><dt>IMU scale</dt><dd>${result.request.imu_scale}</dd><dt>Sensors</dt><dd>${escapeHtml(result.request.sensor_config || 'mixed')}</dd></dl>`;
}

function panelCaption(panel, result) {
  if (panel.caption) return panel.caption;
  const samples = result.samples || {};
  const sourceSample = panel.text_source && samples[panel.text_source];
  if (sourceSample && sourceSample.caption) return sourceSample.caption;
  const motionSample = panel.motion_id && samples[panel.motion_id];
  if (motionSample && motionSample.caption) return motionSample.caption;
  return panel.kind.replace('_', ' ');
}

function panelMarkup(panel, index, result) {
  const caption = panelCaption(panel, result);
  return `<article class="motion-panel"><header><h3>${escapeHtml(panel.label)}</h3><div class="tags"><span class="tag ${panel.text_source ? '' : 'none'}">Text ${panel.text_source || 'None'}</span><span class="tag ${panel.imu_source ? '' : 'none'}">IMU ${panel.imu_source || 'None'}</span></div><p title="${escapeHtml(caption)}">${escapeHtml(caption)}</p></header><canvas id="motion-${index}" width="480" height="355"></canvas></article>`;
}

function setupMotionCanvas(panel, canvas) {
  const view = { panel, canvas, yaw: -0.7, pitch: 0.12, dragging: false, x: 0, y: 0 };
  canvas.addEventListener('pointerdown', e => { view.dragging = true; view.x = e.clientX; view.y = e.clientY; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => { if (!view.dragging) return; view.yaw += (e.clientX - view.x) * .01; view.pitch = Math.max(-.6, Math.min(.6, view.pitch + (e.clientY - view.y) * .006)); view.x = e.clientX; view.y = e.clientY; drawMotion(view); });
  canvas.addEventListener('pointerup', () => view.dragging = false);
  state.panels.push(view);
}

function drawMotion(view) {
  const { canvas, panel, yaw, pitch } = view, ctx = canvas.getContext('2d');
  const joints = panel.motion[Math.min(state.frame, panel.motion.length - 1)], root = joints[0];
  ctx.clearRect(0,0,canvas.width,canvas.height); ctx.strokeStyle = '#e4e9eb'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(40, canvas.height-42); ctx.lineTo(canvas.width-40, canvas.height-42); ctx.stroke();
  const projected = joints.map(joint => {
    const x = joint[0]-root[0], y=joint[1], z=joint[2]-root[2];
    const rx = x*Math.cos(yaw)-z*Math.sin(yaw), rz=x*Math.sin(yaw)+z*Math.cos(yaw);
    const ry = y*Math.cos(pitch)-rz*Math.sin(pitch);
    return [canvas.width/2 + rx*105, canvas.height-40-ry*105];
  });
  ctx.strokeStyle = colors[panel.kind] || colors.generated; ctx.lineWidth = 5; ctx.lineCap='round'; ctx.lineJoin='round';
  for (const chain of chains) { ctx.beginPath(); chain.forEach((joint,index) => index ? ctx.lineTo(...projected[joint]) : ctx.moveTo(...projected[joint])); ctx.stroke(); }
  ctx.fillStyle='#17202a'; ctx.font='12px system-ui';
  const travel = Math.hypot(joints[0][0]-panel.motion[0][0][0], joints[0][2]-panel.motion[0][0][2]);
  ctx.fillText(`root travel ${travel.toFixed(2)} m`, 12, 20);
}

function renderIMU(result) {
  el('imu-section').hidden = false;
  const entries = Object.entries(result.imu).slice(0, 12);
  el('imu-panels').innerHTML = entries.map(([key, value], index) => `<article class="imu-panel"><h3>IMU ${escapeHtml(key)} · ${escapeHtml(result.request.sensor_config || value.sensor_config || '')}</h3><canvas id="imu-${index}" width="800" height="170"></canvas></article>`).join('');
  entries.forEach(([,value], index) => drawSignal(el(`imu-${index}`), value.acceleration));
}

function drawSignal(canvas, signal) {
  const ctx=canvas.getContext('2d'), w=canvas.width, h=canvas.height, colors=['#d1495b','#2e8b57','#247ba0'];
  const max=Math.max(1,...signal.flat().map(Math.abs)); ctx.clearRect(0,0,w,h); ctx.strokeStyle='#e2e7e9'; ctx.beginPath(); ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  for(let c=0;c<3;c++){ctx.strokeStyle=colors[c];ctx.lineWidth=1.5;ctx.beginPath();signal.forEach((v,i)=>{const x=i/(signal.length-1)*w,y=h/2-v[c]/max*(h*.42);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();}
}

function renderFrame() {
  if (!state.result) return; state.panels.forEach(drawMotion);
  el('timeline').value=state.frame; el('time').value=`${(state.frame/state.result.fps).toFixed(2)} s`;
}
function tick(time) { if(state.playing&&state.result){const step=(time-state.lastTime)/1000*state.result.fps*Number(el('speed').value);if(step>=1){state.frame=(state.frame+Math.floor(step))%state.result.frame_count;state.lastTime=time;renderFrame();}}requestAnimationFrame(tick); }
el('play').addEventListener('click',()=>{state.playing=!state.playing;state.lastTime=performance.now();el('play').textContent=state.playing?'❚❚':'▶';});
el('timeline').addEventListener('input',()=>{state.frame=Number(el('timeline').value);renderFrame();});
function setPanelSize(value) {
  const size = Math.max(220, Math.min(620, Number(value)));
  document.documentElement.style.setProperty('--panel-size', `${size}px`);
  el('panel-size').value = size;
}
el('panel-size').addEventListener('input', () => setPanelSize(el('panel-size').value));
document.querySelector('main').addEventListener('wheel', event => {
  if (!event.ctrlKey || !state.result) return;
  event.preventDefault();
  setPanelSize(Number(el('panel-size').value) + (event.deltaY < 0 ? 20 : -20));
}, { passive: false });
function setStatus(message,error=false){el('status').textContent=message;el('status').style.borderColor=error?'#d1495b':'#1abc9c';}
function escapeHtml(value){return String(value).replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));}
loadSamples().catch(error=>setStatus(error.message,true)); requestAnimationFrame(tick);
