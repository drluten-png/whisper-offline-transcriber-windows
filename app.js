'use strict';

const state = {
  info: null,
  jobs: [],
  local: new Map(),   // загрузки на этом устройстве
  hidden: new Set(),  // убранные из списка (файлы остаются в output)
  offline: false,
  total: 0,
  truncated: false,
  models: null,       // каталог моделей
  modelChoice: null,  // выбранная в списке модель
  dlPoll: null,       // опрос статуса скачивания
  renaming: null,     // id задачи, которую сейчас переименовываем
};

const $ = (id) => document.getElementById(id);
const drop = $('drop');
const fileInput = $('fileInput');

/* Как называть файловый менеджер и файл запуска на разных системах.
   Пилот на Windows: «Показать в Finder» — это macOS-термин, нужен проводник. */
function revealWord() {
  const p = String((state.info && state.info.platform) || '');
  if (p === 'darwin') return 'Finder';
  if (p.indexOf('win') === 0) return 'проводнике';
  return 'файловом менеджере';
}
function launchName() {
  const p = String((state.info && state.info.platform) || '');
  if (p === 'darwin') return '«Запустить Whisper.command»';
  if (p.indexOf('win') === 0) return 'start.bat';
  return 'скрипт запуска';
}
function revealBtnText() {
  const p = String((state.info && state.info.platform) || '');
  return (p === 'darwin') ? 'Показать в Finder' : 'Показать в папке';
}

// --------------------------------------------------------------------------
// Утилиты
// --------------------------------------------------------------------------
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmtBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return n + ' Б';
  const u = ['КБ', 'МБ', 'ГБ', 'ТБ'];
  let i = -1;
  do { n /= 1024; i++; } while (n >= 1024 && i < u.length - 1);
  return n.toFixed(n >= 10 ? 0 : 1) + ' ' + u[i];
}

function fmtClock(sec) {
  sec = Math.max(0, Math.floor(Number(sec) || 0));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  const pad = (x) => String(x).padStart(2, '0');
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

function fmtEta(sec) {
  if (!sec || sec < 5) return '';
  if (sec < 90) return 'осталось ~' + Math.round(sec / 10) * 10 + ' с';
  return 'осталось ~' + Math.round(sec / 60) + ' мин';
}

function elapsed(from, to) {
  if (!from) return '';
  return fmtClock((to || Date.now() / 1000) - from);
}

// --------------------------------------------------------------------------
// Состояние приложения
// --------------------------------------------------------------------------
async function loadInfo() {
  try {
    const r = await fetch('/api/info');
    const i = await r.json();
    state.info = i;
    $('badgeModel').textContent = i.model.replace('mlx-community/', '');
    $('badgeLang').textContent = 'язык: ' + i.language;

    const mem = $('badgeMem');
    if (i.mem_available) {
      const gb = i.mem_available / 1024 / 1024 / 1024;
      mem.textContent = 'память: ' + gb.toFixed(1) + ' ГБ';
      mem.className = 'badge' + (gb < 2.5 ? ' badge-warn' : '');
      mem.title = gb < 2.5
        ? 'Мало свободной памяти: закройте лишние вкладки и программы, иначе распознавание может тормозить'
        : 'Свободная память';
    } else {
      mem.hidden = true;
    }

    let foot = `Модель: ${i.model} · язык: ${i.language} · офлайн`;
    if (!i.ready) foot = '⚠︎ ' + (i.import_error || 'Окружение не готово');
    $('footInfo').textContent = foot;
  } catch (e) {
    $('footInfo').textContent = 'Нет связи с приложением';
  }
}

// --------------------------------------------------------------------------
// Модель распознавания
// --------------------------------------------------------------------------
async function loadModels() {
  try {
    const r = await fetch('/api/models');
    const d = await r.json();
    state.models = d;
    if (state.modelChoice == null) state.modelChoice = d.current;
    const downloading = (d.catalog || []).some(
      (m) => m.download && m.download.status === 'downloading');
    if (!downloading && state.dlPoll) {
      clearInterval(state.dlPoll);
      state.dlPoll = null;
    } else if (downloading && !state.dlPoll) {
      state.dlPoll = setInterval(loadModels, 2000);
    }
    renderModels();
  } catch (e) { /* нет связи — покажем это в баннере */ }
}

function renderModels() {
  const d = state.models;
  if (!d || !d.catalog || !d.catalog.length) return;
  const sel = $('modelSelect');
  const btn = $('modelBtn');
  const hint = $('modelHint');

  const sig = d.catalog.map((m) =>
    `${m.id}:${m.installed}:${m.current}:${m.download ? m.download.status : ''}`).join('|');
  if (sel.dataset.sig !== sig) {
    sel.innerHTML = d.catalog.map((m) => {
      let mark;
      if (m.current) mark = '✓ работает';
      else if (m.installed) mark = 'установлена';
      else if (m.download && m.download.status === 'downloading') mark = 'скачивается…';
      else mark = `скачать ${m.size_mb} МБ`;
      /* скорость относительно tiny: замеры на CPU (int8, русский) — tiny 4.6с,
         base 8.8с, small 22.8с на файле 2:55. Turbo ≈ в 9 раз медленнее tiny. */
      let sp = '';
      if (m.speed != null) {
        sp = (m.speed <= 1.05) ? ' · самая быстрая' : ` · ×${m.speed} медленнее tiny`;
      }
      return `<option value="${esc(m.id)}">${esc(m.label)} — ${mark}${sp}</option>`;
    }).join('');
    sel.dataset.sig = sig;
  }

  const chosen = d.catalog.find((m) => m.id === state.modelChoice) || d.catalog[0];
  sel.value = chosen.id;
  const dl = chosen.download || {};
  /* подпись скорости: замеры на CPU — насколько модель медленнее самой быстрой (tiny) */
  const spText = (chosen.speed != null)
    ? (chosen.speed <= 1.05
        ? 'Это самая быстрая модель.'
        : `Скорость: примерно ×${chosen.speed} от самой быстрой (tiny).`)
    : '';

  if (chosen.current) {
    btn.hidden = true;
    sel.disabled = false;
    hint.textContent = 'Эта модель используется сейчас. ' + spText + ' ' + (chosen.note || '');
  } else if (dl.status === 'downloading') {
    btn.hidden = false;
    btn.disabled = true;
    btn.textContent = `Скачиваю ${dl.progress || 0}%`;
    sel.disabled = true;
    hint.textContent = `Идёт скачивание (нужен интернет): ${dl.have_mb || 0} из `
      + `${chosen.size_mb} МБ, прошло ${dl.elapsed || 0} с.`;
  } else if (dl.status === 'error') {
    btn.hidden = false;
    btn.disabled = false;
    btn.textContent = 'Повторить скачивание';
    sel.disabled = false;
    hint.textContent = 'Не удалось скачать: ' + (dl.error || 'неизвестная ошибка');
  } else if (chosen.installed) {
    btn.hidden = false;
    btn.disabled = false;
    btn.textContent = 'Включить';
    sel.disabled = false;
    hint.textContent = (chosen.note || '') + ' Модель уже скачана — интернет не нужен.';
  } else {
    btn.hidden = false;
    btn.disabled = false;
    btn.textContent = `Скачать ${chosen.size_mb} МБ`;
    sel.disabled = false;
    hint.textContent = (chosen.note || '')
      + ' Скачивание разовое и требует интернета, потом всё работает офлайн.';
  }
}

// --------------------------------------------------------------------------
// Словарь терминов
// --------------------------------------------------------------------------
async function loadGlossary() {
  try {
    const r = await fetch('/api/glossary');
    const d = await r.json();
    const ta = $('glossaryText');
    if (ta && document.activeElement !== ta) ta.value = d.text || '';
    $('glossaryCount').textContent = `${d.terms} слов · ${d.replacements} замен`;
  } catch (e) { /* нет связи — покажет баннер */ }
}

$('glossarySave').addEventListener('click', async () => {
  const hint = $('glossaryHint');
  hint.textContent = 'Сохраняю…';
  try {
    const r = await fetch('/api/glossary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: $('glossaryText').value }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || 'ошибка');
    hint.textContent = `Сохранено: ${d.terms} слов, ${d.replacements} замен`;
    loadGlossary();
  } catch (e) {
    hint.textContent = 'Не удалось сохранить';
  }
});

function setOffline(flag) {
  if (state.offline === flag) return;
  state.offline = flag;
  renderBanner();
}

function renderBanner() {
  const b = $('banner');
  if (state.offline) {
    b.hidden = false;
    b.className = 'banner banner-error';
    b.innerHTML = '<b>Нет связи с приложением.</b> Похоже, окно Terminal закрыто или '
      + 'приложение остановлено. Запустите ' + launchName() + ' заново — '
      + 'уже готовые транскрипции останутся в папке output.';
  } else {
    b.hidden = true;
    b.innerHTML = '';
  }
}

// --------------------------------------------------------------------------
// Загрузка файлов
// --------------------------------------------------------------------------
const uploadQueue = [];
let activeUpload = null;

function handleFiles(files, dirsSkipped) {
  if (dirsSkipped) {
    addNotice(`Папки не поддерживаются — перетащите файлы (пропущено папок: ${dirsSkipped}).`);
  }
  for (const f of files) {
    if (!f) continue;
    const key = 'u_' + Math.random().toString(36).slice(2);
    state.local.set(key, {
      key, file: f, name: f.name || 'audio', size: f.size,
      progress: 0, status: 'pending', message: 'Ждёт загрузки', error: '', xhr: null,
    });
    uploadQueue.push(key);
  }
  render();
  pumpUploads();
}

const notices = [];
function addNotice(text) {
  notices.push({ text, ts: Date.now() });
  render();
}

function pumpUploads() {
  if (activeUpload) return;
  while (uploadQueue.length) {
    const key = uploadQueue.shift();
    const entry = state.local.get(key);
    if (!entry || entry.status !== 'pending') continue;
    startUpload(entry);
    return;
  }
}

function startUpload(entry) {
  entry.status = 'uploading';
  entry.message = 'Загрузка на компьютер…';
  entry.progress = 0;
  render();

  const xhr = new XMLHttpRequest();
  entry.xhr = xhr;
  activeUpload = entry;
  xhr.open('POST', '/api/upload', true);
  xhr.timeout = 6 * 60 * 60 * 1000; // 6 часов на очень большие файлы
  xhr.setRequestHeader('X-Filename', encodeURIComponent(entry.name));
  xhr.setRequestHeader('Content-Type', 'application/octet-stream');
  xhr.setRequestHeader('X-Timestamps', $('timestamps').checked ? '1' : '0');
  xhr.setRequestHeader('X-Prompt', encodeURIComponent(($('prompt').value || '').trim().slice(0, 500)));

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      entry.progress = Math.round((e.loaded / e.total) * 100);
      render();
    }
  };
  const finish = () => {
    activeUpload = null;
    entry.xhr = null;
    pumpUploads();
    render();
  };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      state.local.delete(entry.key);
      refresh();
    } else {
      let msg = 'Ошибка загрузки (HTTP ' + xhr.status + ')';
      try { const j = JSON.parse(xhr.responseText); if (j.error) msg = j.error; } catch (e) {}
      entry.status = 'error';
      entry.error = msg;
      entry.message = 'Ошибка загрузки';
    }
    finish();
  };
  xhr.onerror = () => {
    entry.status = 'error';
    entry.error = 'Связь с приложением потерялась. Проверьте окно Terminal и попробуйте снова.';
    entry.message = 'Ошибка загрузки';
    setOffline(true);
    finish();
  };
  xhr.ontimeout = () => {
    entry.status = 'error';
    entry.error = 'Загрузка длится слишком долго и была остановлена.';
    entry.message = 'Ошибка загрузки';
    finish();
  };
  xhr.onabort = () => {
    entry.status = 'canceled';
    entry.message = 'Отменено';
    finish();
  };
  xhr.send(entry.file);
}

function cancelUpload(key) {
  const e = state.local.get(key);
  if (e && e.xhr) e.xhr.abort();
}

function removeEntry(key) {
  const e = state.local.get(key);
  if (e && e.xhr) return; // сначала отмена
  state.local.delete(key);
  render();
}

function retryUpload(key) {
  const e = state.local.get(key);
  if (!e) return;
  e.status = 'pending';
  e.error = '';
  e.progress = 0;
  e.message = 'Ждёт загрузки';
  uploadQueue.push(key);
  render();
  pumpUploads();
}

// --------------------------------------------------------------------------
// Скачивание и переименование результатов
// --------------------------------------------------------------------------
async function downloadJob(id) {
  const job = state.jobs.find((j) => j.id === id);
  if (!job || !job.download) return;
  try {
    const r = await fetch(job.download);
    if (!r.ok) { addNotice('Не удалось скачать файл (код ' + r.status + ')'); return; }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (job.file_stem || job.display_name || 'транскрипт').replace(/\.txt$/i, '') + '.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 15000);
    setOffline(false);
  } catch (e) {
    setOffline(true);
    addNotice('Приложение не отвечает — скачивание невозможно. Файлы лежат в папке '
      + 'output, их можно открыть в ' + revealWord() + '.');
  }
}

function focusRenameInput() {
  const el = $('renameInput');
  if (el) { el.focus(); el.select(); }
}

async function saveRename(id) {
  const el = $('renameInput');
  if (!el) return;
  const name = (el.value || '').trim();
  if (!name) { addNotice('Имя файла не может быть пустым'); return; }
  try {
    const r = await fetch('/api/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, name }),
    });
    if (!r.ok) {
      let msg = 'Не удалось переименовать';
      try { const j = await r.json(); if (j.error) msg = j.error; } catch (e) {}
      addNotice(msg);
      return;
    }
    state.renaming = null;
    await refresh();
  } catch (e) {
    addNotice('Нет связи с приложением — переименовать нельзя');
  }
}

// --------------------------------------------------------------------------
// Список задач
// --------------------------------------------------------------------------
async function refresh() {
  try {
    const r = await fetch('/api/jobs');
    const d = await r.json();
    state.jobs = d.jobs || [];
    state.total = d.total || state.jobs.length;
    state.truncated = !!d.truncated;
    setOffline(false);
    render();
  } catch (e) {
    setOffline(true);
    render();
  }
}

function statusInfo(job) {
  switch (job.status) {
    case 'queued': return ['В очереди', 'st-queued', false];
    case 'probe': return ['Анализ', 'st-run', true];
    case 'extract': return ['Подготовка звука', 'st-run', true];
    case 'transcribe': return ['Распознавание', 'st-run', true];
    case 'save': return ['Сохранение', 'st-run', true];
    case 'done': return ['Готово', 'st-done', false];
    case 'error': return ['Ошибка', 'st-error', false];
    default: return [job.status, 'st-queued', false];
  }
}

function focusAttrs(key, action) {
  return `data-focus-key="${esc(key)}" data-focus-action="${action}"`;
}

function captureFocus() {
  const el = document.activeElement;
  if (!el || !el.dataset || !el.dataset.focusKey) return null;
  return { key: el.dataset.focusKey, action: el.dataset.focusAction };
}

function restoreFocus(f) {
  if (!f) return;
  const el = document.querySelector(
    `[data-focus-key="${f.key}"]` + (f.action ? `[data-focus-action="${f.action}"]` : ''));
  if (el) el.focus();
}

function renderJob(job) {
  const [label, cls, running] = statusInfo(job);
  const pct = Math.max(0, Math.min(100, job.progress || 0));
  const meta = [];
  if (job.size) meta.push(fmtBytes(job.size));
  if (job.duration) meta.push('длительность ' + fmtClock(job.duration));
  if (running && job.started) meta.push('прошло ' + elapsed(job.started));
  if (running && job.eta) { const s = fmtEta(job.eta); if (s) meta.push(s); }
  if (job.status === 'done' && job.started && job.finished) {
    meta.push('обработано за ' + elapsed(job.started, job.finished));
  }
  if (job.status === 'done' && job.name) meta.push('файл: ' + job.name.replace(/\.txt$/i, ''));

  const isRenaming = state.renaming === job.id;
  const title = isRenaming
    ? `<input class="rename-input" id="renameInput" maxlength="120"
             value="${esc(job.file_stem || job.display_name || job.name)}">`
    : esc(job.display_name || job.name);

  let actions = '';
  if (job.status === 'done' && job.download) {
    if (isRenaming) {
      actions = `<button class="btn" data-rename-save="${esc(job.id)}">Сохранить</button>
        <button class="btn btn-ghost" data-rename-cancel="1">Отмена</button>`;
    } else {
      actions = `<button class="btn" data-download="${esc(job.id)}" ${focusAttrs(job.id, 'download')}>Скачать TXT</button>
        <button class="btn btn-ghost" data-apply-glossary="${esc(job.id)}" ${focusAttrs(job.id, 'glossary')}>Словарь</button>
        <button class="btn btn-ghost" data-rename="${esc(job.id)}" ${focusAttrs(job.id, 'rename')}>Переименовать</button>
        <button class="btn btn-ghost" data-reveal="${esc(job.id)}" ${focusAttrs(job.id, 'reveal')}>${esc(revealBtnText())}</button>
        <button class="btn btn-ghost" data-hide="${esc(job.id)}" ${focusAttrs(job.id, 'hide')}>Убрать</button>`;
    }
  } else if (job.status === 'error') {
    if (job.can_retry) {
      actions += `<button class="btn" data-retry="${esc(job.id)}" ${focusAttrs(job.id, 'retry')}>Повторить</button>`;
    }
    actions += `<button class="btn btn-ghost" data-hide="${esc(job.id)}" ${focusAttrs(job.id, 'hide')}>Убрать</button>`;
  }

  const details = job.error_detail
    ? `<details class="job-details"><summary>Подробности</summary><pre>${esc(job.error_detail)}</pre></details>`
    : '';

  return `<div class="job">
    <div class="job-top">
      <div>
        <div class="job-name">${title}</div>
        <div class="job-meta">${esc(meta.join(' · '))}</div>
      </div>
      <div class="job-actions">${actions}</div>
    </div>
    <div class="bar${running ? ' seg' : ''}"><i style="width:${pct}%"></i></div>
    <div class="job-bottom">
      <span class="status ${cls}">● ${esc(label)}${job.message && job.status !== 'done' ? ' — ' + esc(job.message) : ''}</span>
      <span>${job.status === 'error' ? '' : pct + '%'}</span>
    </div>
    ${job.error ? `<div class="job-error">${esc(job.error)}</div>` : ''}
    ${details}
  </div>`;
}

function renderUpload(e) {
  const map = {
    pending: ['st-queued', 'Ждёт загрузки'],
    uploading: ['st-run', 'Загрузка на компьютер…'],
    error: ['st-error', 'Ошибка загрузки'],
    canceled: ['st-queued', 'Отменено'],
  };
  const [cls, label] = map[e.status] || ['st-queued', e.status];
  const pct = e.status === 'uploading' ? e.progress : (e.status === 'canceled' ? 0 : e.progress);
  let actions = '';
  if (e.status === 'uploading') {
    actions = `<button class="btn btn-ghost" data-cancel="${esc(e.key)}" ${focusAttrs(e.key, 'cancel')}>Отменить</button>`;
  } else if (e.status === 'error' || e.status === 'canceled') {
    actions = `<button class="btn" data-retry-upload="${esc(e.key)}" ${focusAttrs(e.key, 'retry-upload')}>Повторить</button>
      <button class="btn btn-ghost" data-remove="${esc(e.key)}" ${focusAttrs(e.key, 'remove')}>Убрать</button>`;
  }
  return `<div class="job">
    <div class="job-top">
      <div>
        <div class="job-name">${esc(e.name)}</div>
        <div class="job-meta">${fmtBytes(e.size)}</div>
      </div>
      <div class="job-actions">${actions}</div>
    </div>
    <div class="bar${e.status === 'uploading' ? ' seg' : ''}"><i style="width:${pct}%"></i></div>
    <div class="job-bottom">
      <span class="status ${cls}">● ${esc(label)}</span>
      <span>${e.status === 'uploading' ? pct + '%' : ''}</span>
    </div>
    ${e.error ? `<div class="job-error">${esc(e.error)}</div>` : ''}
  </div>`;
}

function render() {
  const list = $('jobsList');
  // пока пользователь переименовывает — не трогаем список, чтобы не сбивать ввод
  if (state.renaming && document.activeElement && document.activeElement.id === 'renameInput') {
    return;
  }
  const focus = captureFocus();

  const locals = Array.from(state.local.values());
  const jobs = state.jobs.filter((j) => !state.hidden.has(j.id));
  const total = locals.length + jobs.length;

  $('jobsCount').textContent = total ? `всего: ${total}` : '';
  const active = jobs.filter((j) => ['queued', 'probe', 'extract', 'transcribe', 'save'].includes(j.status)).length;
  if (total) {
    $('jobsCount').textContent = `всего: ${total}` + (active ? ` · в работе: ${active}` : '');
  }

  if (!total && !notices.length) {
    list.innerHTML = '<div class="empty">Пока пусто. Добавьте файл — транскрипция появится здесь.</div>';
    restoreFocus(focus);
    return;
  }

  let html = '';
  if (notices.length) {
    html += notices.map((n, i) =>
      `<div class="notice">${esc(n.text)} <button class="btn btn-ghost btn-mini" data-notice="${i}">Ок</button></div>`).join('');
  }
  if (state.truncated) {
    html += `<div class="notice">Показаны последние ${jobs.length} из ${state.total} записей. Все файлы лежат в папке output.</div>`;
  }
  html += locals.map(renderUpload).join('') + jobs.map(renderJob).join('');
  list.innerHTML = html;
  restoreFocus(focus);
}

// --------------------------------------------------------------------------
// События
// --------------------------------------------------------------------------
function pickFiles() { fileInput.click(); }

drop.addEventListener('click', pickFiles);
drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickFiles(); }
});
drop.setAttribute('tabindex', '0');

['dragenter', 'dragover'].forEach((ev) =>
  window.addEventListener(ev, () => drop.classList.add('over')));
['dragleave', 'drop'].forEach((ev) =>
  window.addEventListener(ev, (e) => {
    if (ev === 'dragleave' && e.relatedTarget) return;
    drop.classList.remove('over');
  }));

function filesFromTransfer(dt) {
  const files = [];
  let dirs = 0;
  if (dt.items && dt.items.length) {
    for (const item of dt.items) {
      if (item.kind !== 'file') continue;
      let entry = null;
      try { entry = item.webkitGetAsEntry && item.webkitGetAsEntry(); } catch (e) {}
      if (entry && entry.isDirectory) { dirs++; continue; }
      const f = item.getAsFile();
      if (f) files.push(f);
    }
  } else if (dt.files) {
    for (const f of dt.files) files.push(f);
  }
  return { files, dirs };
}

// Ловим файл в любом месте окна, а не только в пунктирной зоне
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  if (!e.dataTransfer) return;
  const { files, dirs } = filesFromTransfer(e.dataTransfer);
  if (files.length || dirs) handleFiles(files, dirs);
});

fileInput.addEventListener('change', () => {
  handleFiles(Array.from(fileInput.files || []), 0);
  fileInput.value = '';
});

$('openOutput').addEventListener('click', async () => {
  try { await fetch('/api/open-output', { method: 'POST' }); } catch (e) {}
});

$('modelSelect').addEventListener('change', (e) => {
  state.modelChoice = e.target.value;
  renderModels();
});

$('modelBtn').addEventListener('click', async () => {
  const d = state.models;
  if (!d) return;
  const chosen = d.catalog.find((m) => m.id === state.modelChoice);
  if (!chosen) return;
  const url = chosen.installed ? '/api/models/select' : '/api/models/download';
  const btn = $('modelBtn');
  btn.disabled = true;
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: chosen.id }),
    });
    if (!r.ok) {
      let msg = 'Не удалось переключить модель';
      try { const j = await r.json(); if (j.error) msg = j.error; } catch (e) {}
      addNotice(msg);
    }
  } catch (e) {
    addNotice('Нет связи с приложением');
  }
  await loadModels();
  await loadInfo();
});

$('jobsList').addEventListener('click', async (e) => {
  const t = e.target;
  if (!t || !t.closest) return;

  const notice = t.closest('[data-notice]');
  if (notice) {
    notices.splice(Number(notice.getAttribute('data-notice')), 1);
    render();
    return;
  }
  const dl = t.closest('[data-download]');
  if (dl) { downloadJob(dl.getAttribute('data-download')); return; }

  const gb = t.closest('[data-apply-glossary]');
  if (gb) {
    gb.disabled = true;
    gb.textContent = 'Применяю…';
    try {
      const r = await fetch('/api/apply-glossary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: gb.getAttribute('data-apply-glossary') }),
      });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        addNotice(d.error || 'Не удалось применить словарь');
      } else if (d.count) {
        addNotice(`Словарь исправил слов: ${d.count}. Оригинал сохранён в data/backup.`);
      } else {
        addNotice('Словарь: замен не потребовалось');
      }
    } catch (e) {
      addNotice('Нет связи с приложением');
    }
    gb.disabled = false;
    gb.textContent = 'Словарь';
    refresh();
    return;
  }

  const ren = t.closest('[data-rename]');
  if (ren) {
    state.renaming = ren.getAttribute('data-rename');
    render();
    focusRenameInput();
    return;
  }

  const renSave = t.closest('[data-rename-save]');
  if (renSave) { saveRename(renSave.getAttribute('data-rename-save')); return; }

  const renCancel = t.closest('[data-rename-cancel]');
  if (renCancel) { state.renaming = null; render(); return; }

  const cancel = t.closest('[data-cancel]');
  if (cancel) { cancelUpload(cancel.getAttribute('data-cancel')); return; }

  const remove = t.closest('[data-remove]');
  if (remove) { removeEntry(remove.getAttribute('data-remove')); return; }

  const retryUp = t.closest('[data-retry-upload]');
  if (retryUp) { retryUpload(retryUp.getAttribute('data-retry-upload')); return; }

  const hide = t.closest('[data-hide]');
  if (hide) { state.hidden.add(hide.getAttribute('data-hide')); render(); return; }

  const reveal = t.closest('[data-reveal]');
  if (reveal) {
    try {
      await fetch('/api/reveal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: reveal.getAttribute('data-reveal') }),
      });
    } catch (err) {}
    return;
  }
  const retry = t.closest('[data-retry]');
  if (retry) {
    retry.disabled = true;
    retry.textContent = 'Ставлю в очередь…';
    try {
      await fetch('/api/retry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: retry.getAttribute('data-retry') }),
      });
      refresh();
    } catch (err) {
      retry.disabled = false;
      retry.textContent = 'Повторить';
    }
  }
});

$('jobsList').addEventListener('keydown', (e) => {
  if (!e.target || e.target.id !== 'renameInput') return;
  if (e.key === 'Enter') {
    e.preventDefault();
    const btn = document.querySelector('[data-rename-save]');
    if (btn) saveRename(btn.getAttribute('data-rename-save'));
  } else if (e.key === 'Escape') {
    e.preventDefault();
    state.renaming = null;
    render();
  }
});

// Не даём случайно потерять долгую загрузку
window.addEventListener('beforeunload', (e) => {
  const busy = Array.from(state.local.values())
    .some((x) => x.status === 'uploading' || x.status === 'pending');
  if (!busy) return;
  e.preventDefault();
  e.returnValue = 'Файлы ещё загружаются. Точно закрыть страницу?';
  return e.returnValue;
});

// --------------------------------------------------------------------------
// Старт
// --------------------------------------------------------------------------
loadInfo();
loadModels();
loadGlossary();
refresh();
setInterval(refresh, 2000);
setInterval(loadInfo, 30000);
setInterval(loadModels, 30000);
