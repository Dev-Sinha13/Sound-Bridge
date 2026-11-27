let state = null, activeJob = null, polling = false, reportsKey = '', collectionsKey = '', optionsKey = '', previousBackend = null;
let backendInitialized = false, loadingSearch = false, busyNow = false;
const $ = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
const number = value => new Intl.NumberFormat().format(value);
const initialEmpty = $('results').firstElementChild.cloneNode(true);
const modelNames = {clap: 'CLAP', finelap: 'FineLAP', aligned: 'JEPA bridge', jepa: 'Audio-JEPA', demo: 'Tone fixture'};
function status(message, error = false) {
  if ($('status').textContent !== message) $('status').textContent = message;
  $('job-bar').classList.toggle('error', error);
  $('job-bar').classList.toggle('idle', !error && !busyNow);
}
async function request(url, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Sonic-Token': state.token}, body: JSON.stringify(body)};
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}
function time(value) { return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`; }
function showView(name, updateHash = true) {
  if (!['search', 'library', 'research'].includes(name)) name = 'search';
  document.querySelectorAll('.tab').forEach(button => {
    const selected = button.dataset.view === name;
    button.classList.toggle('active', selected); button.setAttribute('aria-selected', String(selected)); button.tabIndex = selected ? 0 : -1;
  });
  document.querySelectorAll('.view').forEach(view => view.hidden = view.id !== `${name}-view`);
  $('page-name').textContent = {search: 'Search sounds', library: 'Collections', research: 'Alignment lab'}[name];
  document.title = `Sound Bridge · ${$('page-name').textContent}`;
  if (updateHash) window.scrollTo({top:0, behavior:'instant'});
  if (updateHash && location.hash !== `#${name}`) history.pushState(null, '', `#${name}`);
}
function resetResults(message) {
  $('results').replaceChildren(message ? el('div', message, 'empty') : initialEmpty.cloneNode(true));
  $('results').setAttribute('aria-busy', 'false'); $('search-timing').textContent = '';
  $('result-count').textContent = 'Your next discovery starts with a few words.';
  loadingSearch = false;
}
function showSearchLoading() {
  if (loadingSearch) return;
  loadingSearch = true; $('results').setAttribute('aria-busy', 'true');
  const row = el('div', undefined, 'loading-state');
  const spinner = el('span', undefined, 'loading-spinner'); spinner.setAttribute('aria-hidden', 'true');
  row.append(spinner, el('span', 'Listening for a match…')); $('results').replaceChildren(row);
  $('result-count').textContent = 'Searching your collection'; $('search-timing').textContent = '';
}
function collectionChanged(clear = true) {
  const index = state?.indexes.find(i => i.name === $('collection').value);
  $('collection-info').textContent = index ? `${number(index.files)} recordings · ${number(index.segments)} moments` : 'Add or choose a compatible collection';
  $('watch').textContent = state?.watches?.[index?.name] ? 'Stop watching this folder' : 'Watch this folder';
  if (clear) resetResults();
}
function renderCollections() {
  const indexes = state.indexes.filter(i => i.settings).sort((a, b) => Number(b.compatible) - Number(a.compatible) || a.name.localeCompare(b.name));
  $('nav-count').textContent = number(indexes.length);
  $('library-count').textContent = `${number(indexes.length)} ${indexes.length === 1 ? 'collection' : 'collections'}`;
  const key = JSON.stringify(indexes);
  if (key === collectionsKey) return;
  collectionsKey = key; $('collections-list').replaceChildren();
  if (!indexes.length) $('collections-list').append(el('div', 'Your first collection starts with a folder. Add one to begin.', 'empty compact'));
  for (const index of indexes) {
    const card = el('article', undefined, 'collection-card'), content = el('div', undefined, 'collection-content');
    const icon = el('span', '≋', 'result-file-icon'); icon.setAttribute('aria-hidden', 'true');
    content.append(el('h3', index.name), el('span', modelNames[index.settings.encoder.backend] || index.settings.encoder.backend, 'backend-label'));
    content.append(el('p', index.settings.root, 'collection-path'));
    const counts = el('div', undefined, 'collection-metrics'); counts.append(el('span', `${number(index.files)} recordings`), el('span', `${number(index.segments)} moments`)); content.append(counts);
    const open = el('button', index.compatible ? 'Search collection' : 'Use a compatible model', 'secondary'); open.type = 'button';
    open.onclick = () => {
      if (index.compatible) { showView('search'); $('collection').value = index.name; collectionChanged(); $('query').focus(); }
      else {
        const backend = index.settings.encoder.backend;
        if (['clap', 'finelap'].includes(backend)) { $('search-backend').value = backend; $('search-config').value = backend === 'finelap' ? 'configs/finelap.json' : ''; }
        if (backend === 'aligned') { showView('research'); status('Export the matching alignment from its experiment and choose “Use this alignment for search”.'); }
        else if (backend === 'demo') status('Start the app with --backend demo to search this synthetic fixture.');
        else $('models-dialog').showModal();
      }
    };
    content.append(open); card.append(icon, content); $('collections-list').append(card);
  }
}
function renderResults(result) {
  loadingSearch = false; $('results').setAttribute('aria-busy', 'false'); $('results').replaceChildren();
  $('result-count').textContent = `${result.matches.length} ${result.matches.length === 1 ? 'moment' : 'moments'} for “${result.query}”`;
  if (!result.matches.length) {
    const empty = el('div', undefined, 'empty compact'); empty.append(el('h3', 'No moments with these settings.'), el('p', 'Try a broader description, clear the file filter, or lower your similarity cutoff.')); $('results').append(empty);
  }
  const feedback = async (button, label, segment) => {
    button.disabled = true;
    try { await request('/api/feedback', {index: result.index, query: result.query, label, segment}); button.textContent = 'Judgment saved'; status('Saved on this device. Judgments do not automatically change the model.'); }
    catch (error) { button.disabled = false; status(error.message, true); }
  };
  for (const [rank, match] of result.matches.entries()) {
    const card = el('article', undefined, 'result'), top = el('div', undefined, 'result-top');
    const icon = el('span', '≋', 'result-file-icon'); icon.setAttribute('aria-hidden', 'true');
    const title = el('div', undefined, 'result-title'); title.append(el('h3', match.name));
    const meta = el('div', undefined, 'result-meta'); meta.append(el('span', `${time(match.start)} – ${time(match.end)}`), el('span', `${(match.end - match.start).toFixed(1)} sec`));
    if (!rank) meta.append(el('span', 'TOP MATCH', 'result-badge')); title.append(meta);
    const score = el('div', match.score.toFixed(3), 'score'); score.append(el('small', 'similarity')); top.append(icon, title, score);
    const wave = el('canvas'); wave.width = 1000; wave.height = 60; wave.className = 'waveform'; wave.setAttribute('role', 'img'); wave.setAttribute('aria-label', `Amplitude waveform for ${match.name}`);
    const audio = el('audio'); audio.controls = true; audio.preload = 'metadata'; audio.setAttribute('aria-label', `Play ${match.name}, ${time(match.start)} to ${time(match.end)}`);
    audio.src = `/api/audio?index=${encodeURIComponent(result.index)}&id=${match.id}`;
    audio.onplay = () => document.querySelectorAll('audio').forEach(other => { if (other !== audio) other.pause(); });
    const actions = el('div', undefined, 'result-actions'), similar = el('button', 'Find similar sounds', 'text-button'); similar.type = 'button';
    similar.onclick = () => submit('/api/search', {index: result.index, query: `Sounds like ${match.name}`, similar_to: match.id, unique_files: true});
    const judgments = el('details', undefined, 'feedback-controls'); judgments.append(el('summary', 'How was this match?'));
    const buttons = el('div', undefined, 'feedback-buttons');
    for (const [label, text] of [['correct', 'Correct sound'], ['wrong_sound', 'Wrong sound'], ['wrong_timestamp', 'Wrong timestamp']]) {
      const button = el('button', text); button.type = 'button'; button.onclick = () => feedback(button, label, match.id); buttons.append(button);
    }
    judgments.append(buttons); actions.append(similar, judgments); card.append(top, wave, audio, actions); $('results').append(card);
    request(`/api/waveform?index=${encodeURIComponent(result.index)}&id=${match.id}`).then(data => {
      const ctx = wave.getContext('2d'), peak = Math.max(...data.peaks, .01), width = wave.width / data.peaks.length;
      ctx.fillStyle = '#86a871'; data.peaks.forEach((v, i) => { const h = Math.max(2, v / peak * 48); ctx.fillRect(i * width, (60 - h) / 2, Math.max(1, width - 2), h); });
    }).catch(() => { wave.hidden = true; });
  }
  const absent = el('button', 'This sound is absent from the collection', 'text-button'); absent.type = 'button'; absent.onclick = () => feedback(absent, 'absent');
  const absence = el('div', undefined, 'absent-action'); absence.append(absent); $('results').append(absence);
}
function busy(value, kind) {
  busyNow = value;
  document.querySelectorAll('form button, #demo, #open-models, #mobile-models, .collection-card button, .result-actions > .text-button').forEach(b => b.disabled = value);
  if (state && !state.can_search) document.querySelectorAll('#index-form button, #search-form button').forEach(b => b.disabled = true);
  $('job-bar').classList.toggle('running', value); $('job-bar').classList.toggle('idle', !value && !$('job-bar').classList.contains('error'));
  $('collection').disabled = value;
  $('search-submit').querySelector('.button-spinner').hidden = !(value && kind === 'search');
}
async function refresh() {
  if (polling) return; polling = true;
  try {
    state = await request('/api/state');
    $('connection-dot').className = 'connected'; $('connection-label').textContent = state.offline ? 'Offline ready' : 'Local workspace';
    $('model-name').textContent = modelNames[state.backend] || state.backend;
    $('model-detail').textContent = state.offline ? 'Cached models · on this device' : 'Inference stays on this device';
    $('mode').hidden = !state.synthetic && !state.synthetic_alignment;
    $('mode').textContent = state.synthetic ? 'Synthetic demo: only low, mid, and high tone queries are supported. These fixtures test the pipeline, not real-world retrieval.' : 'This bridge was calibrated on synthetic fixtures. It is not a validated model for real sounds.';
    $('demo').hidden = !state.synthetic; $('use-clap').hidden = state.backend !== 'aligned';
    const changedBackend = previousBackend !== null && state.backend !== previousBackend;
    if (state.backend !== previousBackend) { $('suggestions').replaceChildren(); previousBackend = state.backend; }
    if (changedBackend) { optionsKey = ''; resetResults('Model ready. Choose a compatible collection to search.'); }
    if (!backendInitialized) {
      if (['clap', 'finelap'].includes(state.backend)) $('search-backend').value = state.backend;
      if (state.backend === 'finelap') $('search-config').value = state.default_config;
      $('experiment-backend').value = state.synthetic ? 'demo' : 'jepa';
      $('experiment-config').value = 'configs/jepa.json';
      experimentBackendChanged();
      backendInitialized = true;
    }
    const compatible = state.indexes.filter(i => i.settings && i.compatible);
    const selected = $('collection').value, key = JSON.stringify(compatible.map(i => [i.name, i.files, i.segments]));
    if (key !== optionsKey) {
      optionsKey = key;
      $('collection').replaceChildren(...compatible.map(i => { const o = el('option', i.name); o.value = i.name; return o; }));
      if (compatible.some(i => i.name === selected)) $('collection').value = selected;
      if (!compatible.length) { const option = el('option', 'No compatible collections'); option.value = ''; $('collection').append(option); }
      collectionChanged(false);
    }
    if (!$('suggestions').children.length) {
      for (const text of state.synthetic ? ['low tone', 'mid tone', 'high tone'] : ['Paper crumpling', 'Footsteps on gravel', 'Rain falling']) {
        const button = el('button', text); button.type = 'button'; button.onclick = () => { $('query').value = text; $('query').focus(); }; $('suggestions').append(button);
      }
    }
    $('query').placeholder = state.synthetic ? 'A low, mid, or high tone…' : 'Paper crumpling, footsteps on gravel, distant thunder…';
    renderCollections(); renderReports(state.reports); collectionChanged(false);
    const running = state.jobs.find(j => ['running', 'queued'].includes(j.status));
    if (running && !activeJob) { activeJob = running.id; if (running.kind === 'search') showSearchLoading(); }
    $('cancel-job').hidden = !running; $('cancel-job').dataset.job = running?.id || ''; busy(Boolean(running), running?.kind);
    if (activeJob) {
      const job = state.jobs.find(j => j.id === activeJob);
      if (job) {
        status(`${job.kind}: ${job.message}${job.seconds !== undefined ? ` · ${job.seconds}s` : ''}`, job.status === 'failed');
        if (['complete', 'failed', 'cancelled'].includes(job.status)) {
          if (job.kind === 'search') {
            if (job.status === 'complete') { renderResults(job.result); $('search-timing').textContent = `${job.seconds.toFixed(2)}s`; }
            else { resetResults(job.status === 'cancelled' ? 'Search cancelled. Your collection is ready when you are.' : `Search could not finish: ${job.message}`); }
          }
          if (job.kind === 'models' && job.status === 'complete') $('models-dialog').close();
          activeJob = null;
        }
      }
    } else if ($('status').textContent.startsWith('Connecting')) status('Ready. Your audio stays on this device.');
  } catch (error) {
    $('connection-dot').className = 'disconnected'; $('connection-label').textContent = 'Disconnected';
    status(`Could not connect. Check that the local server is running. ${error.message}`, true);
  } finally { polling = false; }
}
async function submit(url, body) {
  if (busyNow) return status('A job is already running. Wait or cancel it before starting another.', true);
  busy(true, url === '/api/search' ? 'search' : undefined);
  if (url === '/api/search') showSearchLoading();
  try { const result = await request(url, body); activeJob = result.job; status('Job queued…'); await refresh(); }
  catch (error) { if (url === '/api/search') resetResults(`Search could not start: ${error.message}`); busy(false); status(error.message, true); }
}
$('index-form').onsubmit = event => { event.preventDefault(); submit('/api/index', {folder: $('folder').value, name: $('name').value, window: Number($('window').value), hop: Number($('hop').value), scales: $('scales').value.split(',').map(v => v.trim()).filter(Boolean).map(Number)}); };
$('search-form').onsubmit = event => { event.preventDefault(); if (!$('collection').value) { showView('library'); return status('Add a folder, or load the model matching an existing collection.', true); } submit('/api/search', {index: $('collection').value, query: $('query').value, min_score: $('min-score').value === '' ? null : Number($('min-score').value), name_filter: $('name-filter').value, unique_files: $('unique-files').checked}); };
$('experiment-form').onsubmit = event => { event.preventDefault(); submit('/api/experiment', {name: $('run-name').value, manifest: $('manifest').value, backend: $('experiment-backend').value, config: ['jepa', 'finelap'].includes($('experiment-backend').value) ? $('experiment-config').value : null, methods: $('methods').value.split(',').map(v => v.trim()).filter(Boolean), budgets: $('budgets').value.split(',').map(Number)}); };
$('cancel-job').onclick = async () => { try { await request('/api/cancel', {job: $('cancel-job').dataset.job}); status('Cancellation requested. Finishing the current model operation…'); } catch (error) { status(error.message, true); } };
$('use-clap').onclick = () => submit('/api/backend', {backend: 'clap'});
$('demo').onclick = async () => { try { const fixture = await request('/api/demo', {}); $('folder').value = fixture.folder; $('manifest').value = fixture.manifest; $('budgets').value = '6,12,18'; $('name').value = 'tones'; status(fixture.note); } catch (error) { status(error.message, true); } };
$('collection').onchange = () => collectionChanged();
$('watch').onclick = async () => { try { if (!$('collection').value) throw new Error('Choose a collection first.'); const enabled = !state.watches?.[$('collection').value]; await request('/api/watch', {index: $('collection').value, enabled}); status(enabled ? 'Watching this folder while the page is open. Changes are checked every 10 seconds.' : 'Folder watching stopped.'); await refresh(); } catch (error) { status(error.message, true); } };
$('model-form').onsubmit = event => { event.preventDefault(); $('models-dialog').close(); submit('/api/backend', {backend: $('search-backend').value, config: $('search-config').value || ($('search-backend').value === 'finelap' ? 'configs/finelap.json' : null)}); };
for (const id of ['open-models', 'mobile-models']) $(id).onclick = () => $('models-dialog').showModal();
$('close-models').onclick = () => $('models-dialog').close();
$('search-backend').onchange = () => { $('search-config').value = $('search-backend').value === 'finelap' ? 'configs/finelap.json' : ''; };
function experimentBackendChanged() {
  const backend = $('experiment-backend').value, native = ['clap', 'finelap'].includes(backend);
  $('experiment-config').disabled = !['jepa', 'finelap'].includes(backend);
  $('experiment-config').value = backend === 'finelap' ? 'configs/finelap.json' : backend === 'jepa' ? 'configs/jepa.json' : '';
  $('budgets').disabled = native; $('methods').closest('details').hidden = native;
}
$('experiment-backend').onchange = experimentBackendChanged;
document.querySelectorAll('.tab').forEach(button => {
  button.onclick = () => showView(button.dataset.view);
  button.onkeydown = event => {
    const tabs = [...document.querySelectorAll('.tab')], index = tabs.indexOf(button);
    if (!['ArrowDown', 'ArrowUp', 'ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault(); const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (['ArrowDown', 'ArrowRight'].includes(event.key) ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].click(); tabs[next].focus();
  };
});
document.addEventListener('click', event => { if (event.target.closest('[data-open-library]')) { showView('library'); $('folder').focus(); } });
document.addEventListener('keydown', event => { if (event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey && !$('models-dialog').open && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) && !document.activeElement?.isContentEditable) { event.preventDefault(); showView('search'); $('query').focus(); } });
window.addEventListener('hashchange', () => { const name = location.hash.slice(1); if (['search', 'library', 'research'].includes(name)) showView(name, false); });
showView(location.hash.slice(1) || 'search', false);
refresh(); setInterval(refresh, 1500);
function renderReports(reports) {
  const key = JSON.stringify(reports);
  if (key === reportsKey) return;
  reportsKey = key;
  if (!reports.length) return;
  $('reports').replaceChildren();
  for (const report of reports) {
    const block = el('div', undefined, 'report');
    block.append(el('h3', report.name), el('p', report.synthetic ? 'SYNTHETIC FIXTURE · These scores do not measure real-world retrieval.' : 'Held-out test results · higher recall is better'));
    const wrap = el('div', undefined, 'table-wrap'), table = el('table');
    const caption = el('caption', `${report.name} · held-out retrieval`); table.append(caption);
    const head = el('tr'); ['Method', 'Groups', 'Seed', 'T→A R@1', 'R@5', 'R@10', 'A→T R@1'].forEach(v => { const th = el('th', v); th.scope = 'col'; head.append(th); });
    table.append(head);
    for (const r of report.results) {
      const row = el('tr'), t = r.test.text_to_audio, a = r.test.audio_to_text;
      [r.method, r.budget || '—', r.seed ?? '—', ...[t['recall@1'], t['recall@5'], t['recall@10'], a['recall@1']].map(v => `${(v * 100).toFixed(1)}%`)].forEach(v => row.append(el('td', String(v))));
      table.append(row);
    }
    wrap.append(table);
    const selected = report.results.find(r => r.artifact === report.selected_artifact) || report.results[0];
    if (selected) {
      const stats = el('div', undefined, 'report-stats');
      for (const [label, value] of [['Text → audio R@1', selected.test.text_to_audio['recall@1']], ['Audio → text R@1', selected.test.audio_to_text['recall@1']]]) {
        const stat = el('div', undefined, 'report-stat'); stat.append(el('strong', `${(value * 100).toFixed(1)}%`), el('span', label)); stats.append(stat);
      }
      block.append(stats);
    }
    const runs = el('details', undefined, 'report-runs'); runs.append(el('summary', `View ${report.results.length} evaluated ${report.results.length === 1 ? 'run' : 'runs'}`), wrap);
    block.append(runs); $('reports').append(block);
    if (report.selected_artifact) {
      block.append(el('p', `Selected on validation: ${report.selected_artifact}`));
      if (report.encoder.backend === 'jepa') {
        const button = el('button', 'Export selected alignment', 'secondary');
        button.onclick = async () => {
          try {
            const result = await request('/api/export-alignment', {experiment: report.name});
            status(`Exported config: ${result.config}`);
            const use = el('button', 'Use this alignment for search', 'secondary');
            use.onclick = () => {
              const current = state.indexes.find(i => i.name === $('collection').value);
              if (current) { $('folder').value = current.settings.root; $('name').value = `${current.name.slice(0, 58)}-jepa`; }
              submit('/api/backend', {backend: 'aligned', config: result.config});
              document.querySelector('[data-view="search"]').click();
            };
            button.replaceWith(use);
          }
          catch (error) { status(error.message, true); }
        };
        block.append(button);
      }
    }
  }
}

const mobileNav = matchMedia('(max-width:700px)');
function updateNavOrientation() { document.querySelector('nav[role=tablist]').setAttribute('aria-orientation', mobileNav.matches ? 'horizontal' : 'vertical'); }
mobileNav.addEventListener('change', updateNavOrientation); updateNavOrientation();
