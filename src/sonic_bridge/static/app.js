let state = null;
let activeJob = null;
let polling = false;
const $ = id => document.getElementById(id);
const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
async function request(url, body) {
  const options = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Sonic-Token': state.token}, body: JSON.stringify(body)};
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}
function time(value) { return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`; }
function renderResults(result) {
  $('results').replaceChildren();
  $('result-count').textContent = `${result.matches.length} moments · ${result.query}`;
  if (!result.matches.length) $('results').append(el('p', 'No matching windows in this collection.', 'muted'));
  for (const match of result.matches) {
    const card = el('article', undefined, 'result');
    const top = el('div', undefined, 'result-top');
    const title = el('div');
    title.append(el('h3', match.name), el('p', `${time(match.start)} – ${time(match.end)}`));
    top.append(title, el('span', `${match.score.toFixed(3)} similarity`, 'score'));
    const audio = el('audio'); audio.controls = true; audio.preload = 'none';
    audio.src = `/api/audio?index=${encodeURIComponent(result.index)}&id=${match.id}`;
    card.append(top, audio); $('results').append(card);
  }
}
function renderReports(reports) {
  if (!reports.length) return;
  $('reports').replaceChildren();
  for (const report of reports) {
    const block = el('div', undefined, 'report');
    block.append(el('h3', report.name), el('p', report.synthetic ? 'SYNTHETIC FIXTURE · These scores do not measure real-world retrieval.' : 'Held-out test results · higher recall is better'));
    const wrap = el('div', undefined, 'table-wrap'), table = el('table');
    const head = el('tr'); ['Method', 'Groups', 'Seed', 'T→A R@1', 'R@5', 'R@10', 'A→T R@1'].forEach(v => head.append(el('th', v)));
    table.append(head);
    for (const r of report.results) {
      const row = el('tr'), t = r.test.text_to_audio, a = r.test.audio_to_text;
      [r.method, r.budget || '—', r.seed ?? '—', ...[t['recall@1'], t['recall@5'], t['recall@10'], a['recall@1']].map(v => `${(v * 100).toFixed(1)}%`)].forEach(v => row.append(el('td', String(v))));
      table.append(row);
    }
    wrap.append(table); block.append(wrap); $('reports').append(block);
  }
}
function busy(value) { document.querySelectorAll('form button, #demo').forEach(b => b.disabled = value); }
async function refresh() {
  if (polling) return; polling = true;
  try {
    state = await request('/api/state');
    $('mode').textContent = state.synthetic ? 'SYNTHETIC DEMO / Only low, mid, and high tone queries are supported. This verifies the pipeline; it is not a semantic audio model.' : `${state.backend.toUpperCase()} / Local inference · ${state.offline ? 'cached models only' : 'model files may download on first use'} · similarity scores are not probabilities`;
    $('demo').hidden = !state.synthetic;
    const selected = $('collection').value;
    const compatible = state.indexes.filter(i => i.settings && i.settings.encoder.backend === state.backend);
    $('collection').replaceChildren(...compatible.map(i => { const o = el('option', `${i.name} · ${i.files} recordings · ${i.segments} windows`); o.value = i.name; return o; }));
    if (compatible.some(i => i.name === selected)) $('collection').value = selected;
    if (!compatible.length) $('collection').append(el('option', 'No collections yet'));
    if (!$('suggestions').children.length) {
      for (const text of state.synthetic ? ['low tone', 'mid tone', 'high tone'] : ['Rain and thunder', 'A dog barking', 'Metal scraping']) {
        const button = el('button', text); button.type = 'button'; button.onclick = () => { $('query').value = text; }; $('suggestions').append(button);
      }
    }
    if (state.synthetic) $('query').placeholder = 'A low, mid, or high tone…';
    renderReports(state.reports);
    const running = state.jobs.find(j => ['running', 'queued'].includes(j.status));
    busy(Boolean(running));
    if (activeJob) {
      const job = state.jobs.find(j => j.id === activeJob);
      if (job) {
        status(`${job.kind}: ${job.message}${job.seconds !== undefined ? ` · ${job.seconds}s` : ''}`, job.status === 'failed');
        if (job.status === 'complete' || job.status === 'failed') {
          if (job.kind === 'search' && job.status === 'complete') renderResults(job.result);
          activeJob = null;
        }
      }
    } else if (running) status(`${running.kind}: ${running.message}`);
    else if ($('status').textContent.startsWith('Connecting')) status(`Ready · ${state.workspace}`);
  } catch (error) { status(error.message, true); }
  finally { polling = false; }
}
async function submit(url, body) {
  busy(true);
  try { const result = await request(url, body); activeJob = result.job; status('Job queued…'); await refresh(); }
  catch (error) { status(error.message, true); busy(false); }
}
$('index-form').onsubmit = event => { event.preventDefault(); submit('/api/index', {folder: $('folder').value, name: $('name').value, window: Number($('window').value), hop: Number($('hop').value)}); };
$('search-form').onsubmit = event => { event.preventDefault(); if (!state.indexes.length) return status('Index a collection first.', true); submit('/api/search', {index: $('collection').value, query: $('query').value}); };
$('experiment-form').onsubmit = event => { event.preventDefault(); submit('/api/experiment', {name: $('run-name').value, manifest: $('manifest').value, budgets: $('budgets').value.split(',').map(Number)}); };
$('demo').onclick = async () => { try { const fixture = await request('/api/demo', {}); $('folder').value = fixture.folder; $('manifest').value = fixture.manifest; $('budgets').value = '6,12,18'; $('name').value = 'tones'; status(fixture.note); } catch (error) { status(error.message, true); } };
document.querySelectorAll('.tab').forEach(button => button.onclick = () => { document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b === button)); $('search-view').hidden = button.dataset.view !== 'search'; $('research-view').hidden = button.dataset.view !== 'research'; });
refresh(); setInterval(refresh, 1500);
