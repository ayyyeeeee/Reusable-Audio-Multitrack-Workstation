'use strict';

const $ = (id) => document.getElementById(id);
const colors = ['#9ed7b9', '#d6b28a', '#a4b9ef', '#d5a7c6', '#91c9d2', '#c2c98c', '#aaa2cc', '#91a8bb'];
const state = {
  session: null, busy: false, localBusy: false, selected: null, solos: new Set(),
  mode: 'mix', context: null, master: null, sources: [], gains: new Map(),
  buffers: new Map(), bufferPromises: new Map(), peaks: new Map(),
  playing: false, loading: false, startTime: 0, offset: 0, playGeneration: 0,
  draft: 0, saved: 0, saveTimer: null, savePromise: null, pollPromise: null,
  projectSignature: '', lastError: '', seekDrag: false, renderGeneration: 0,
};
let toastTimer;

function formatTime(value) {
  const safe = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(safe / 60);
  const seconds = (safe % 60).toFixed(2).padStart(5, '0');
  return `${minutes}:${seconds}`;
}

function showError(error) {
  $('errorText').textContent = error instanceof Error ? error.message : String(error);
  $('errorBanner').hidden = false;
}

function toast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  let result;
  try { result = await response.json(); }
  catch { throw new Error(`服务返回了无效响应（${response.status}）。`); }
  if (!response.ok) throw new Error(result.error || `请求失败（${response.status}）。`);
  return result;
}

function post(path, data = {}) {
  return request(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
}

function controls(forAudition = false) {
  return state.session.tracks.map((track) => ({
    id: track.id, name: track.name, gain: track.gain,
    muted: forAudition && state.solos.size > 0 ? !state.solos.has(track.id) : track.muted,
  }));
}

function markDirty() {
  state.draft += 1;
  $('saveStatus').textContent = '有未保存更改';
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => { saveChanges().catch(showError); }, 500);
}

function saveChanges() {
  clearTimeout(state.saveTimer);
  if (state.savePromise) return state.savePromise;
  if (!state.session || !state.session.tracks.length || state.saved >= state.draft) return Promise.resolve();
  // One drain loop serves autosave and action callers, including edits made while saving.
  state.savePromise = (async () => {
    try {
      while (state.session?.tracks.length && state.saved < state.draft) {
        const identifier = state.session.id;
        const version = state.draft;
        const payload = { tracks: controls(), name: state.session.name };
        $('saveStatus').textContent = '正在保存…';
        await post('/api/session', payload);
        if (state.session?.id !== identifier) return;
        state.saved = Math.max(state.saved, version);
      }
      $('saveStatus').textContent = '更改已保存';
    } catch (error) {
      $('saveStatus').textContent = '保存失败，请重试';
      throw error;
    }
  })().finally(() => { state.savePromise = null; });
  return state.savePromise;
}

async function runAction(action) {
  if (state.localBusy) return;
  state.localBusy = true;
  updateControls();
  try {
    await saveChanges();
    await action();
    await poll();
  } catch (error) { showError(error); }
  finally { state.localBusy = false; updateControls(); }
}

function getContext() {
  if (!state.context) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('此浏览器不支持音频工作台，请使用新版 Edge 或 Chrome。');
    state.context = new AudioContextClass();
    state.master = state.context.createGain();
    state.master.gain.value = Number($('monitorVolume').value);
    state.master.connect(state.context.destination);
  }
  return state.context;
}

async function getBuffer(url) {
  if (state.buffers.has(url)) return state.buffers.get(url);
  if (state.bufferPromises.has(url)) return state.bufferPromises.get(url);
  const generation = state.renderGeneration;
  const promise = (async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error('音轨加载失败，请确认本地服务仍在运行。');
    const raw = await response.arrayBuffer();
    let buffer;
    try { buffer = await getContext().decodeAudioData(raw); }
    catch { throw new Error('浏览器无法解码这条音轨，请使用新版 Edge 或 Chrome。'); }
    if (generation === state.renderGeneration) state.buffers.set(url, buffer);
    return buffer;
  })();
  state.bufferPromises.set(url, promise);
  try { return await promise; }
  finally { if (generation === state.renderGeneration) state.bufferPromises.delete(url); }
}

function position() {
  if (!state.playing || !state.context) return state.offset;
  return Math.min(state.session?.duration || 0, state.offset + Math.max(0, state.context.currentTime - state.startTime));
}

function stopSources() {
  for (const source of state.sources) {
    try { source.stop(); } catch { /* A naturally completed source is already stopped. */ }
    source.disconnect();
  }
  for (const gain of state.gains.values()) gain.disconnect();
  state.sources = [];
  state.gains.clear();
}

function pause(reset = false) {
  state.offset = reset ? 0 : position();
  state.playGeneration += 1;
  state.playing = false;
  state.loading = false;
  stopSources();
  updateTransport();
}

function effectiveGain(track) {
  // Audition a removed sound without changing its saved mute state.
  if (state.solos.size) return state.solos.has(track.id) ? track.gain : 0;
  if (track.muted) return 0;
  return track.gain;
}

function applyGains() {
  if (state.context && state.mode === 'mix') {
    const now = state.context.currentTime;
    for (const track of state.session?.tracks || []) {
      const node = state.gains.get(track.id);
      if (node) {
        node.gain.cancelScheduledValues(now);
        node.gain.setTargetAtTime(effectiveGain(track), now, 0.008);
      }
    }
  }
  updateTransport();
}

async function play() {
  if (!state.session || state.loading) return;
  const generation = ++state.playGeneration;
  const sessionId = state.session.id;
  state.loading = true;
  updateTransport();
  try {
    const context = getContext();
    await context.resume();
    const list = state.mode === 'original' || !state.session.tracks.length
      ? [{ id: 'original', url: state.session.original_url, gain: 1, muted: false }]
      : state.session.tracks;
    // Load all sources before starting any: all tracks share one sample clock.
    const buffers = [];
    for (const track of list) {
      buffers.push(await getBuffer(track.url));
      if (generation !== state.playGeneration || state.session?.id !== sessionId) return;
    }
    if (generation !== state.playGeneration) return;
    if (state.offset >= state.session.duration - 0.01) state.offset = 0;
    stopSources();
    state.startTime = context.currentTime + 0.045;
    list.forEach((track, index) => {
      const source = context.createBufferSource();
      const gain = context.createGain();
      source.buffer = buffers[index];
      gain.gain.value = track.id === 'original' ? 1 : effectiveGain(track);
      source.connect(gain);
      gain.connect(state.master);
      source.start(state.startTime, Math.min(state.offset, Math.max(0, source.buffer.duration - 0.001)));
      state.sources.push(source);
      state.gains.set(track.id, gain);
    });
    state.playing = true;
  } catch (error) { pause(); showError(error); }
  finally {
    if (generation === state.playGeneration) state.loading = false;
    updateTransport();
  }
}

function seek(seconds) {
  const wasPlaying = state.playing;
  pause();
  state.offset = Math.max(0, Math.min(state.session?.duration || 0, seconds));
  updateTransport();
  if (wasPlaying) play();
}

function changeMode(mode) {
  if (state.mode === mode) return;
  const resume = state.playing;
  pause();
  state.mode = mode;
  updateTransport();
  if (resume) play();
}

function updateTransport() {
  const hasTracks = Boolean(state.session?.tracks.length);
  $('playIcon').textContent = state.loading ? '· ·' : state.playing ? 'Ⅱ' : '▶';
  $('playButton').setAttribute('aria-label', state.playing ? '暂停' : state.loading ? '加载音轨中' : '播放');
  document.querySelector('.transport-panel').classList.toggle('loading', state.loading);
  $('mixMode').classList.toggle('active', state.mode === 'mix');
  $('originalMode').classList.toggle('active', state.mode === 'original');
  $('mixMode').setAttribute('aria-pressed', String(state.mode === 'mix'));
  $('originalMode').setAttribute('aria-pressed', String(state.mode === 'original'));
  let text = '准备就绪';
  if (state.loading) text = '正在载入音轨，准备同步播放…';
  else if (state.session) {
    if (state.mode === 'original' || !hasTracks) text = '试听：原始录音';
    else if (state.solos.size) text = `试听：独听 ${state.solos.size} 条轨 · 导出混音也仅包含独听轨`;
    else {
      const audible = state.session.tracks.filter((t) => !t.muted && t.gain > 0).length;
      text = `试听：当前混音 · ${audible} / ${state.session.tracks.length} 条轨开启`;
    }
  }
  $('listeningStatus').textContent = text;
  $('currentTime').textContent = formatTime(position());
  if (!state.seekDrag) $('seekBar').value = position();
  const ratio = (state.session?.duration || 0) > 0 ? position() / state.session.duration : 0;
  document.querySelectorAll('.playhead').forEach((line) => { line.style.left = `${Math.min(100, ratio * 100)}%`; });
}

function animate() {
  if (state.playing && position() >= (state.session?.duration || 0)) pause(true);
  if (state.playing) updateTransport();
  requestAnimationFrame(animate);
}

function drawWave(canvas, peaks, color) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const width = rect.width, height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#2b3742';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
  if (!peaks?.length) return;
  // A common, compressed display scale preserves comparative loudness between tracks.
  ctx.fillStyle = color;
  const bars = Math.ceil(width / 2);
  for (let x = 0; x < bars; x++) {
    const start = Math.floor(x * peaks.length / bars);
    const end = Math.max(start + 1, Math.floor((x + 1) * peaks.length / bars));
    let peak = 0;
    for (let i = start; i < Math.min(end, peaks.length); i++) peak = Math.max(peak, peaks[i]);
    const amplitude = Math.min(1, Math.sqrt(Math.max(0, peak))) * (height - 5);
    if (amplitude > .1) ctx.fillRect(x * 2, (height - amplitude) / 2, 1.3, Math.max(1, amplitude));
  }
}

function redrawWaves() {
  drawWave($('overviewCanvas'), state.peaks.get('original'), '#88b9a0');
  state.session?.tracks.forEach((track, index) => {
    const canvas = document.querySelector(`.track-row[data-id="${track.id}"] canvas`);
    if (canvas) drawWave(canvas, state.peaks.get(track.id), colors[index % colors.length]);
  });
}

async function loadPeaks() {
  if (!state.session) return;
  const generation = state.renderGeneration;
  const entries = [
    { id: 'original', peak_url: state.session.original_url.replace(/\.wav$/, '.peaks.json'), url: state.session.original_url },
    ...state.session.tracks,
  ];
  await Promise.allSettled(entries.map(async (track) => {
    try {
      const response = await fetch(track.peak_url);
      if (!response.ok) throw new Error('No peak data');
      const data = await response.json();
      if (!Array.isArray(data.peaks)) throw new Error('Invalid peak data');
      if (generation !== state.renderGeneration) return;
      state.peaks.set(track.id, data.peaks);
    } catch {
      // Waveform failure does not block listening; derive it from decoded audio when needed.
      try {
        const buffer = await getBuffer(track.url);
        if (generation !== state.renderGeneration) return;
        const samples = buffer.getChannelData(0);
        const chunk = Math.max(1, Math.ceil(samples.length / 2048));
        const peaks = [];
        for (let i = 0; i < samples.length; i += chunk) {
          let peak = 0;
          for (let j = i; j < Math.min(samples.length, i + chunk); j++) peak = Math.max(peak, Math.abs(samples[j]));
          peaks.push(peak);
        }
        state.peaks.set(track.id, peaks);
      } catch { return; }
    }
    if (generation === state.renderGeneration) redrawWaves();
  }));
}

function selectTrack(identifier) {
  state.selected = identifier;
  document.querySelectorAll('.track-row').forEach((row) => row.classList.toggle('selected', row.dataset.id === identifier));
  const track = state.session?.tracks.find((item) => item.id === identifier);
  $('splitPanel').hidden = !track;
  $('selectedTrackName').textContent = track?.name || '';
  updateControls();
}

function updateTrackAppearance(track) {
  const row = document.querySelector(`.track-row[data-id="${track.id}"]`);
  if (!row) return;
  row.classList.toggle('muted', effectiveGain(track) === 0);
  row.classList.toggle('solo', state.solos.has(track.id));
  const mute = row.querySelector('.mute');
  mute.classList.toggle('active', track.muted);
  mute.setAttribute('aria-pressed', String(track.muted));
  mute.textContent = track.muted ? '已静音' : '静音';
  const solo = row.querySelector('.solo');
  solo.classList.toggle('active', state.solos.has(track.id));
  solo.setAttribute('aria-pressed', String(state.solos.has(track.id)));
}

function redrawMixState() {
  state.session?.tracks.forEach(updateTrackAppearance);
  $('clearSoloButton').disabled = !state.solos.size;
  applyGains();
}

function renderTracks() {
  const container = $('tracks');
  const tracks = state.session?.tracks || [];
  $('trackCount').textContent = tracks.length;
  if (!tracks.length) {
    container.replaceChildren();
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.innerHTML = '<div class="empty-wave" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><h3></h3><p></p>';
    empty.querySelector('h3').textContent = state.session ? '录音已就绪，开始分离声音' : '把混在一起的声音，交给你逐轨控制';
    empty.querySelector('p').textContent = state.session ? '先听原始录音，再在左侧选择分离方式。分轨完成后即可独听、命名、静音和导出。' : '导入录音并开始分轨。独听每条音轨，命名为你听到的声音，然后静音不需要的部分。';
    container.append(empty);
    $('splitPanel').hidden = true;
    return;
  }
  container.replaceChildren();
  tracks.forEach((track, index) => {
    const row = document.createElement('article');
    row.className = 'track-row';
    row.dataset.id = track.id;
    row.innerHTML = '<div class="track-label"><span class="track-index"></span><div class="track-label-main"><input class="track-name" type="text" maxlength="100"><div class="track-meta"></div></div></div><div class="track-wave waveform-container" title="点击波形跳转"><canvas aria-hidden="true"></canvas><div class="playhead"></div></div><div class="track-controls"><div class="track-buttons"><button class="solo" type="button">独听</button><button class="mute" type="button">静音</button><button class="download-track" type="button" title="导出这条音轨" aria-label="导出这条音轨">↓</button></div><label class="gain-control"><input type="range" min="0" max="2" step="0.01" aria-label="音轨音量"><output></output></label></div>';
    row.querySelector('.track-index').textContent = String(index + 1).padStart(2, '0');
    row.querySelector('.track-index').style.borderLeft = `2px solid ${colors[index % colors.length]}`;
    const name = row.querySelector('.track-name');
    name.value = track.name;
    name.setAttribute('aria-label', `音轨 ${index + 1} 名称`);
    name.title = '输入你听到的声音名称，修改自动保存';
    const methodName = { demucs: '神经网络候选', practice: '节拍与神经网络分轨', features: '频谱特征分组', repeating: '重复节拍拆分' }[track.method] || '分离音轨';
    row.querySelector('.track-meta').textContent = methodName + (track.parent_id ? ' · 再次拆分' : '');
    row.addEventListener('click', () => selectTrack(track.id));
    name.addEventListener('input', () => {
      track.name = name.value.trim() || `音轨 ${index + 1}`;
      if (state.selected === track.id) $('selectedTrackName').textContent = track.name;
      markDirty();
    });
    name.addEventListener('change', () => { name.value = track.name; });
    row.querySelector('.mute').addEventListener('click', (event) => {
      event.stopPropagation();
      track.muted = !track.muted;
      updateTrackAppearance(track);
      applyGains(); markDirty();
    });
    row.querySelector('.solo').addEventListener('click', (event) => {
      event.stopPropagation();
      if (state.solos.has(track.id)) state.solos.delete(track.id); else state.solos.add(track.id);
      changeMode('mix');
      redrawMixState();
    });
    row.querySelector('.solo').title = '临时独听此轨（包括已静音的轨），不修改保存的静音状态';
    const gain = row.querySelector('.gain-control input');
    const output = row.querySelector('output');
    gain.value = track.gain;
    output.textContent = `${Math.round(track.gain * 100)}%`;
    gain.addEventListener('input', () => {
      track.gain = Number(gain.value);
      output.textContent = `${Math.round(track.gain * 100)}%`;
      applyGains(); markDirty();
    });
    gain.addEventListener('dblclick', () => {
      gain.value = 1; track.gain = 1; output.textContent = '100%';
      applyGains(); markDirty();
    });
    row.querySelector('.track-wave').addEventListener('click', (event) => {
      const bounds = event.currentTarget.getBoundingClientRect();
      seek((event.clientX - bounds.left) / bounds.width * state.session.duration);
    });
    row.querySelector('.download-track').addEventListener('click', (event) => {
      event.stopPropagation();
      exportAudio('mix', [{ id: track.id, gain: track.gain, muted: false }]);
    });
    container.append(row);
    updateTrackAppearance(track);
  });
  if (!tracks.some((track) => track.id === state.selected)) state.selected = tracks[0].id;
  selectTrack(state.selected);
  redrawWaves();
}

function acceptSession(session) {
  if (!session) return;
  const changed = !state.session || state.session.id !== session.id || state.session.revision !== session.revision;
  if (!changed) {
    state.session.can_undo = session.can_undo;
    return;
  }
  pause(true);
  clearTimeout(state.saveTimer);
  state.renderGeneration += 1;
  state.session = session;
  state.draft = 0; state.saved = 0;
  state.buffers.clear(); state.bufferPromises.clear(); state.peaks.clear(); state.solos.clear();
  state.selected = null;
  state.mode = session.tracks.length ? 'mix' : 'original';
  $('projectName').value = session.name;
  $('audioInfo').textContent = `${formatTime(session.duration)}  /  ${(session.sample_rate / 1000).toFixed(1)} kHz  /  ${session.channels === 1 ? '单声道' : '立体声'}`;
  $('durationTime').textContent = formatTime(session.duration);
  $('seekBar').max = session.duration;
  $('seekBar').value = 0;
  $('overviewPlaceholder').hidden = true;
  $('saveStatus').textContent = '工程已保存';
  $('modelNotes').textContent = (session.notes || []).join('\n');
  $('exportResult').hidden = true;
  renderTracks(); updateTransport(); loadPeaks();
}

function renderProjects(projects) {
  const signature = JSON.stringify(projects) + (state.session?.id || '');
  if (signature === state.projectSignature) return;
  state.projectSignature = signature;
  $('projectCount').textContent = projects.length;
  const list = $('projectList');
  list.replaceChildren();
  if (!projects.length) {
    const empty = document.createElement('p'); empty.className = 'micro';
    empty.textContent = '导入后自动保存在本机。'; list.append(empty); return;
  }
  projects.forEach((project) => {
    const button = document.createElement('button');
    button.type = 'button'; button.className = 'project-item';
    button.classList.toggle('active', project.id === state.session?.id);
    button.dataset.id = project.id;
    button.title = project.name;
    button.innerHTML = '<span class="project-icon" aria-hidden="true">▤</span><span class="project-title"></span><small></small>';
    button.querySelector('.project-title').textContent = project.name;
    button.querySelector('small').textContent = Number.isInteger(project.tracks) ? `${project.tracks} 轨` : '';
    button.addEventListener('click', () => runAction(async () => { pause(); acceptSession(await post('/api/open', { session_id: project.id })); }));
    list.append(button);
  });
}

function updateControls() {
  const blocked = state.busy || state.localBusy;
  const hasSession = Boolean(state.session);
  const hasTracks = Boolean(state.session?.tracks.length);
  $('importButton').disabled = blocked;
  $('fileInput').disabled = blocked;
  $('separateButton').disabled = blocked || !hasSession;
  $('separateButton').firstChild.textContent = hasTracks ? '重新分轨 ' : '开始分轨 ';
  $('methodSelect').disabled = blocked;
  $('componentCount').disabled = blocked;
  $('cancelButton').hidden = !state.busy;
  $('cancelButton').disabled = !state.busy;
  $('exportMixButton').disabled = blocked || !hasTracks;
  $('exportStemsButton').disabled = blocked || !hasTracks;
  $('exportFormat').disabled = blocked;
  $('undoButton').disabled = blocked || !state.session?.can_undo;
  $('enableAllButton').disabled = blocked || !hasTracks;
  $('clearSoloButton').disabled = !state.solos.size;
  $('playButton').disabled = !hasSession;
  $('rewindButton').disabled = !hasSession;
  $('seekBar').disabled = !hasSession;
  $('mixMode').disabled = !hasTracks;
  $('originalMode').disabled = !hasSession;
  $('splitButton').disabled = blocked || !state.selected;
  $('splitMethod').disabled = blocked;
  $('splitComponents').disabled = blocked;
  $('projectName').readOnly = blocked || !hasTracks;
  document.querySelectorAll('.track-name,.gain-control input,.track-buttons .mute,.download-track,.project-item').forEach((element) => { element.disabled = blocked; });
}

async function poll() {
  if (state.pollPromise) return state.pollPromise;
  state.pollPromise = (async () => {
    try {
      const result = await request('/api/state');
      state.busy = result.busy;
      acceptSession(result.session);
      renderProjects(result.sessions || []);
      $('jobStatus').textContent = result.message || '准备就绪';
      $('progressNumber').textContent = result.busy ? `${Math.round(result.progress || 0)}%` : '';
      $('progressBar').style.width = `${Math.max(0, Math.min(100, result.progress || 0))}%`;
      if (result.error && result.error !== state.lastError) { showError(result.error); state.lastError = result.error; }
      if (!result.error) state.lastError = '';
      updateControls();
    } catch (error) {
      $('jobStatus').textContent = '连接中断，请确认工作台启动窗口仍在运行。';
      $('progressNumber').textContent = '';
    }
  })();
  try { await state.pollPromise; }
  finally { state.pollPromise = null; }
}

function importFile(file) {
  if (!file || state.busy || state.localBusy) return;
  if (file.size > 256 * 1024 * 1024) { showError('文件超过 256 MiB，请先分段。'); return; }
  runAction(async () => {
    pause(true);
    $('jobStatus').textContent = '正在导入与解码音频…';
    const session = await request('/api/import', {
      method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(file.name) }, body: file,
    });
    acceptSession(session); $('errorBanner').hidden = true;
  });
}

function exportAudio(mode, subset = null) {
  if (!state.session?.tracks.length) return;
  runAction(async () => {
    const result = await post('/api/export', {
      format: $('exportFormat').value, mode,
      tracks: subset || controls(mode === 'mix'),
    });
    const link = $('downloadLink');
    link.href = result.url; link.download = result.filename;
    $('exportMessage').textContent = result.message;
    $('exportResult').hidden = false;
    link.click();
    toast('导出完成，文件也保存在本地工程的 exports 文件夹。');
  });
}

$('importButton').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', (event) => { importFile(event.target.files[0]); event.target.value = ''; });
for (const type of ['dragenter', 'dragover']) $('importButton').addEventListener(type, (event) => { event.preventDefault(); if (!state.busy) $('importButton').classList.add('drag-over'); });
for (const type of ['dragleave', 'drop']) $('importButton').addEventListener(type, (event) => { event.preventDefault(); $('importButton').classList.remove('drag-over'); });
$('importButton').addEventListener('drop', (event) => importFile(event.dataTransfer.files[0]));
document.addEventListener('dragover', (event) => event.preventDefault());
document.addEventListener('drop', (event) => { event.preventDefault(); if (!$('importButton').contains(event.target)) importFile(event.dataTransfer.files[0]); });
$('methodSelect').addEventListener('change', () => {
  const features = $('methodSelect').value === 'features';
  $('componentField').hidden = !features;
  $('methodHint').textContent = features ? '从录音学习频谱、瞬态与时间变化，生成可由你试听辨识的候选轨。' : $('methodSelect').value === 'practice' ? '适合带节拍器的录音：先提取重复声，再分离其余乐器和人声，共八条候选轨。' : '预训练模型提取人声、鼓、贝斯、吉他、钢琴与其他候选轨。';
});
$('separateButton').addEventListener('click', () => runAction(async () => {
  pause(); $('errorBanner').hidden = true;
  await post('/api/separate', { method: $('methodSelect').value, components: Number($('componentCount').value) });
}));
$('cancelButton').addEventListener('click', () => { post('/api/cancel').then(poll).catch(showError); });
$('undoButton').addEventListener('click', () => runAction(async () => { pause(); acceptSession(await post('/api/undo')); }));
$('splitMethod').addEventListener('change', () => { $('splitComponents').hidden = $('splitMethod').value === 'repeating'; });
$('splitButton').addEventListener('click', () => runAction(async () => {
  pause(); $('errorBanner').hidden = true;
  await post('/api/split', { track_id: state.selected, method: $('splitMethod').value, components: Number($('splitComponents').value) });
}));
$('projectName').addEventListener('input', () => {
  if (!state.session) return;
  state.session.name = $('projectName').value.trim() || '未命名工程'; markDirty();
});
$('projectName').addEventListener('change', () => { if (state.session) $('projectName').value = state.session.name; });
$('playButton').addEventListener('click', () => { if (state.playing || state.loading) pause(); else play(); });
$('rewindButton').addEventListener('click', () => seek(0));
$('mixMode').addEventListener('click', () => changeMode('mix'));
$('originalMode').addEventListener('click', () => changeMode('original'));
$('monitorVolume').addEventListener('input', () => {
  if (state.master) state.master.gain.setTargetAtTime(Number($('monitorVolume').value), state.context.currentTime, .01);
});
$('seekBar').addEventListener('input', () => { state.seekDrag = true; $('currentTime').textContent = formatTime(Number($('seekBar').value)); });
$('seekBar').addEventListener('change', () => { state.seekDrag = false; seek(Number($('seekBar').value)); });
$('overview').addEventListener('click', (event) => {
  if (!state.session) return;
  const bounds = $('overview').getBoundingClientRect(); seek((event.clientX - bounds.left) / bounds.width * state.session.duration);
});
$('clearSoloButton').addEventListener('click', () => { state.solos.clear(); redrawMixState(); });
$('enableAllButton').addEventListener('click', () => {
  if (!state.session) return;
  state.session.tracks.forEach((track) => { track.muted = false; });
  state.solos.clear(); redrawMixState(); markDirty();
});
$('exportMixButton').addEventListener('click', () => exportAudio('mix'));
$('exportStemsButton').addEventListener('click', () => exportAudio('stems'));
$('dismissError').addEventListener('click', () => { $('errorBanner').hidden = true; });
document.addEventListener('keydown', (event) => {
  if (event.code !== 'Space' || event.repeat || /INPUT|TEXTAREA|SELECT|BUTTON/.test(document.activeElement.tagName)) return;
  if (!state.session) return;
  event.preventDefault(); if (state.playing || state.loading) pause(); else play();
});
window.addEventListener('beforeunload', (event) => {
  if (state.saved < state.draft) { event.preventDefault(); event.returnValue = ''; }
});
new ResizeObserver(redrawWaves).observe($('tracks'));
new ResizeObserver(redrawWaves).observe($('overview'));
poll();
setInterval(poll, 900);
requestAnimationFrame(animate);
