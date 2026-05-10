// ─────────────────────────────────────────────────────────────────────────────
// RAG Agent – local-first personal knowledge base
// All chunks + indexes live in the browser (IndexedDB).
// The server is stateless: it only receives context + query and returns an answer.
// ─────────────────────────────────────────────────────────────────────────────

const baseUrl = window.API_BASE_URL || window.location.origin;

// ── Global state ──────────────────────────────────────────────────────────────
let collections = [];
let selectedCollection = null;
let bm25Index = null;
let bm25Chunks = [];
let chatHistory = [];

// ── IndexedDB layer ───────────────────────────────────────────────────────────
const DB_NAME = 'rag-kb';
const DB_VERSION = 2;
let _db = null;

function openDB() {
  if (_db) return Promise.resolve(_db);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const d = e.target.result;
      if (!d.objectStoreNames.contains('collections')) {
        d.createObjectStore('collections', { keyPath: 'id' });
      }
      if (!d.objectStoreNames.contains('chunks')) {
        const s = d.createObjectStore('chunks', { keyPath: 'id' });
        s.createIndex('collection', 'collection', { unique: false });
        s.createIndex('filename', 'filename', { unique: false });
      }
      if (!d.objectStoreNames.contains('chat_sessions')) {
        d.createObjectStore('chat_sessions', { keyPath: 'id' });
      }
    };
    req.onsuccess = (e) => { _db = e.target.result; resolve(_db); };
    req.onerror  = (e) => reject(e.target.error);
  });
}

async function dbPut(store, value) {
  const d = await openDB();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, 'readwrite');
    tx.objectStore(store).put(value).onsuccess = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbGetAll(store) {
  const d = await openDB();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, 'readonly');
    const req = tx.objectStore(store).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror  = () => reject(req.error);
  });
}

async function dbGetByIndex(store, indexName, value) {
  const d = await openDB();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, 'readonly');
    const req = tx.objectStore(store).index(indexName).getAll(value);
    req.onsuccess = () => resolve(req.result);
    req.onerror  = () => reject(req.error);
  });
}

async function dbDelete(store, key) {
  const d = await openDB();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, 'readwrite');
    tx.objectStore(store).delete(key).onsuccess = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function dbDeleteByIndex(store, indexName, value) {
  const d = await openDB();
  return new Promise((resolve, reject) => {
    const tx = d.transaction(store, 'readwrite');
    const s = tx.objectStore(store).index(indexName);
    const req = s.openCursor(IDBKeyRange.only(value));
    req.onsuccess = (e) => {
      const cursor = e.target.result;
      if (cursor) { cursor.delete(); cursor.continue(); }
      else resolve();
    };
    req.onerror = () => reject(req.error);
  });
}

// ── BM25 ──────────────────────────────────────────────────────────────────────
class BM25 {
  constructor(k1 = 1.5, b = 0.75) {
    this.k1 = k1; this.b = b;
    this.docs = []; this.idf = {}; this.avgdl = 0; this.N = 0;
  }

  tokenize(text) {
    return text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
  }

  build(docs) {
    this.docs = docs.map(d => ({ id: d.id, tokens: this.tokenize(d.text) }));
    this.N = this.docs.length;
    if (!this.N) return;
    this.avgdl = this.docs.reduce((s, d) => s + d.tokens.length, 0) / this.N;
    const df = {};
    for (const doc of this.docs) {
      const seen = new Set();
      for (const t of doc.tokens) {
        if (!seen.has(t)) { df[t] = (df[t] || 0) + 1; seen.add(t); }
      }
    }
    this.idf = {};
    for (const [term, freq] of Object.entries(df)) {
      this.idf[term] = Math.log((this.N - freq + 0.5) / (freq + 0.5) + 1);
    }
  }

  search(query, topK = 20) {
    if (!this.N) return [];
    const qTokens = this.tokenize(query);
    const scores = new Map();
    for (const doc of this.docs) {
      const dl = doc.tokens.length;
      const tf = {};
      for (const t of doc.tokens) tf[t] = (tf[t] || 0) + 1;
      let score = 0;
      for (const t of qTokens) {
        if (!this.idf[t]) continue;
        const f = tf[t] || 0;
        score += this.idf[t] * (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + this.b * dl / this.avgdl));
      }
      if (score > 0) scores.set(doc.id, score);
    }
    return [...scores.entries()].sort((a, b) => b[1] - a[1]).slice(0, topK).map(([id, score]) => ({ id, score }));
  }
}

// ── Text processing ───────────────────────────────────────────────────────────
function chunkText(text, maxWords = 200, overlap = 40) {
  const words = text.split(/\s+/);
  const chunks = [];
  const step = maxWords - overlap;
  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + maxWords).join(' ');
    if (chunk.trim()) chunks.push(chunk);
    if (i + maxWords >= words.length) break;
  }
  return chunks;
}

function estimateTokens(chunks) {
  return chunks.reduce((s, c) => s + Math.ceil(c.text.length / 4), 0);
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

// ── BM25 index rebuild ────────────────────────────────────────────────────────
async function rebuildBM25(collectionId) {
  const allChunks = collectionId
    ? await dbGetByIndex('chunks', 'collection', collectionId)
    : await dbGetAll('chunks');

  bm25Chunks = allChunks;
  const docs = allChunks.map(c => ({
    id: c.id,
    text: c.text + (c.questions && c.questions.length ? ' ' + c.questions.join(' ') : ''),
  }));
  bm25Index = new BM25();
  bm25Index.build(docs);
}

// ── Ingest ────────────────────────────────────────────────────────────────────
async function ingestText(text, filename, collectionId, enhancedIndexing) {
  const rawChunks = chunkText(text);
  const newChunks = rawChunks.map((t, i) => ({
    id: uid(),
    text: t,
    filename,
    source: filename,
    collection: collectionId,
    chunkIndex: i,
    questions: [],
    createdAt: Date.now(),
  }));

  for (const chunk of newChunks) await dbPut('chunks', chunk);
  await rebuildBM25();
  showStatus('Indexed ' + newChunks.length + ' chunks from "' + filename + '"', 'success');

  if (enhancedIndexing) {
    showStatus('Generating search questions for "' + filename + '"...', 'info');
    try {
      const payload = { chunks: newChunks.map(c => ({ id: c.id, text: c.text, source: c.source })) };
      const res = await fetch(baseUrl + '/ingest/questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const data = await res.json();
        for (const item of (data.results || [])) {
          const chunk = newChunks.find(c => c.id === item.id);
          if (chunk) { chunk.questions = item.questions; await dbPut('chunks', chunk); }
        }
        await rebuildBM25();
        showStatus('Enhanced indexing done for "' + filename + '"', 'success');
      }
    } catch (e) {
      showStatus('Enhanced indexing failed (BM25 still active): ' + e.message, 'info');
    }
  }

  return newChunks.length;
}

// ── Retrieval ─────────────────────────────────────────────────────────────────
async function retrieve(query, collectionId) {
  await rebuildBM25(collectionId || undefined);
  const totalTokens = estimateTokens(bm25Chunks);

  if (totalTokens < 60000 && bm25Chunks.length < 500) {
    return { mode: 'full', chunks: bm25Chunks };
  }

  const hits = bm25Index.search(query, 20);
  const chunkMap = new Map(bm25Chunks.map(c => [c.id, c]));
  const candidates = hits.map(h => chunkMap.get(h.id)).filter(Boolean);
  return { mode: 'bm25', chunks: candidates };
}

// ── Collections management ────────────────────────────────────────────────────
async function loadCollections() {
  collections = await dbGetAll('collections');
  await renderCollections();
  updateChatSelector();
}

async function createCollection(name) {
  const col = { id: uid(), name: name.trim(), createdAt: Date.now() };
  await dbPut('collections', col);
  await loadCollections();
  return col;
}

async function deleteCollection(id) {
  await dbDeleteByIndex('chunks', 'collection', id);
  await dbDelete('collections', id);
  if (selectedCollection === id) selectedCollection = null;
  await loadCollections();
  await rebuildBM25();
}

async function deleteDocument(filename, collectionId) {
  const chunks = await dbGetByIndex('chunks', 'collection', collectionId);
  const toDelete = chunks.filter(c => c.filename === filename);
  for (const c of toDelete) await dbDelete('chunks', c.id);
  await rebuildBM25();
  await renderCollections();
  showStatus('Removed "' + filename + '"', 'success');
}

// ── Render left pane ──────────────────────────────────────────────────────────
async function renderCollections() {
  const list = document.getElementById('collectionList');
  if (!list) return;
  list.innerHTML = '';

  if (!collections.length) {
    list.innerHTML = '<div class="no-docs" style="padding:10px 4px">No collections yet. Create one to get started.</div>';
    return;
  }

  for (const col of collections) {
    const chunks = await dbGetByIndex('chunks', 'collection', col.id);
    const filenames = [...new Set(chunks.map(c => c.filename))];

    const item = document.createElement('div');
    item.className = 'project-item';
    item.innerHTML =
      '<div class="project-header" onclick="toggleCollectionDocs(this,\'' + col.id + '\')">' +
        '<span class="project-chevron">&#9658;</span>' +
        '<span class="project-name">' + escapeHtml(col.name) + '</span>' +
        '<div class="project-actions" onclick="event.stopPropagation()">' +
          '<button title="Chat with this collection" onclick="selectChatCollection(\'' + col.id + '\')">&#128172;</button>' +
          '<button class="btn-grey" title="Delete collection" onclick="confirmDeleteCollection(\'' + col.id + '\',\'' + escapeHtml(col.name) + '\')">&#128465;</button>' +
        '</div>' +
      '</div>' +
      '<div class="project-docs" id="col-docs-' + col.id + '">' +
        (filenames.length === 0
          ? '<span class="no-docs">No documents yet.</span>'
          : filenames.map(f =>
              '<div class="doc-item" style="display:flex;justify-content:space-between;align-items:center">' +
                '<span>&#128196; ' + escapeHtml(f) + '</span>' +
                '<button class="btn-grey" style="padding:2px 7px;font-size:0.7em;flex-shrink:0;margin-left:6px"' +
                  ' onclick="confirmDeleteDocument(\'' + escapeHtml(f) + '\',\'' + col.id + '\')">&#10005;</button>' +
              '</div>'
            ).join('')) +
      '</div>';
    list.appendChild(item);
  }
}

function toggleCollectionDocs(headerEl, colId) {
  const docsEl = document.getElementById('col-docs-' + colId);
  const chevron = headerEl.querySelector('.project-chevron');
  const isOpen = docsEl.classList.contains('open');
  docsEl.classList.toggle('open', !isOpen);
  chevron.style.transform = isOpen ? '' : 'rotate(90deg)';
}

function updateChatSelector() {
  const sel = document.getElementById('chatCollection');
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = '<option value="all">All Collections</option>';
  for (const col of collections) {
    const opt = document.createElement('option');
    opt.value = col.id;
    opt.textContent = col.name;
    if (col.id === current) opt.selected = true;
    sel.appendChild(opt);
  }
}

function selectChatCollection(colId) {
  selectedCollection = colId;
  const sel = document.getElementById('chatCollection');
  if (sel) sel.value = colId;
  const col = collections.find(c => c.id === colId);
  chatHistory = [];
  document.getElementById('chatMessages').innerHTML = '';
  addMessage('assistant', 'Ready. Ask me anything about "' + (col ? col.name : 'this collection') + '".');
  showStatus('Switched to "' + (col ? col.name : colId) + '"', 'info');
}

function confirmDeleteCollection(id, name) {
  if (confirm('Delete collection "' + name + '" and all its documents? This cannot be undone.')) {
    deleteCollection(id);
  }
}

function confirmDeleteDocument(filename, colId) {
  if (confirm('Remove "' + filename + '" from this collection?')) {
    deleteDocument(filename, colId);
  }
}

// ── Export / Import ───────────────────────────────────────────────────────────
async function exportKB() {
  const allCollections = await dbGetAll('collections');
  const allChunks = await dbGetAll('chunks');
  const blob = new Blob(
    [JSON.stringify({ version: 1, collections: allCollections, chunks: allChunks }, null, 2)],
    { type: 'application/json' }
  );
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'knowledge-base-' + new Date().toISOString().slice(0, 10) + '.json';
  a.click();
  showStatus('Knowledge base exported', 'success');
}

async function importKB(file) {
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    if (!data.collections || !data.chunks) throw new Error('Invalid export file');
    for (const col of data.collections) await dbPut('collections', col);
    for (const chunk of data.chunks) await dbPut('chunks', chunk);
    await loadCollections();
    await rebuildBM25();
    showStatus('Imported ' + data.chunks.length + ' chunks across ' + data.collections.length + ' collections', 'success');
  } catch (e) {
    showStatus('Import failed: ' + e.message, 'error');
  }
}

function triggerImport() {
  const input = document.createElement('input');
  input.type = 'file'; input.accept = '.json';
  input.onchange = (e) => { if (e.target.files[0]) importKB(e.target.files[0]); };
  input.click();
}

// ── GitHub ingest ─────────────────────────────────────────────────────────────
function showGitHubModal() {
  if (!collections.length) { showStatus('Create a collection first', 'error'); return; }
  const sel = document.getElementById('githubCollection');
  sel.innerHTML = '';
  for (const col of collections) {
    const opt = document.createElement('option');
    opt.value = col.id; opt.textContent = col.name;
    if (col.id === selectedCollection) opt.selected = true;
    sel.appendChild(opt);
  }
  document.getElementById('githubUrl').value = '';
  document.getElementById('githubStatus').textContent = '';
  document.getElementById('githubEnhanced').checked = false;
  showModal('githubModal');
}

async function doIngestGitHub() {
  const colId = document.getElementById('githubCollection').value;
  const url = document.getElementById('githubUrl').value.trim();
  const enhanced = true;
  const statusEl = document.getElementById('githubStatus');
  const btn = document.getElementById('githubIngestBtn');

  if (!url) { showStatus('Please enter a repository URL', 'error'); return; }
  if (!colId) { showStatus('Select a collection', 'error'); return; }

  btn.disabled = true;
  btn.textContent = 'Cloning...';
  statusEl.textContent = 'Cloning repository — this may take a moment...';

  try {
    const res = await fetch(baseUrl + '/fetch/github', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    const files = data.files || [];
    statusEl.textContent = 'Indexing ' + files.length + ' files...';
    btn.textContent = 'Indexing...';

    let totalChunks = 0;
    for (const file of files) {
      totalChunks += await ingestText(file.text, file.filename, colId, false);
    }

    // enhanced indexing in one batch after all files are stored
    if (enhanced && files.length) {
      statusEl.textContent = 'Generating search questions...';
      const allChunks = await dbGetByIndex('chunks', 'collection', colId);
      const repoChunks = allChunks.filter(c => files.some(f => f.filename === c.filename));
      if (repoChunks.length) {
        try {
          const payload = { chunks: repoChunks.slice(0, 30).map(c => ({ id: c.id, text: c.text, source: c.source })) };
          const qRes = await fetch(baseUrl + '/ingest/questions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          if (qRes.ok) {
            const qData = await qRes.json();
            const chunkMap = new Map(repoChunks.map(c => [c.id, c]));
            for (const item of (qData.results || [])) {
              const chunk = chunkMap.get(item.id);
              if (chunk) { chunk.questions = item.questions; await dbPut('chunks', chunk); }
            }
            await rebuildBM25();
          }
        } catch (e) { /* enhanced indexing is optional */ }
      }
    }

    await renderCollections();
    updateChatSelector();
    statusEl.textContent = '';
    closeModal('githubModal');
    showStatus('Ingested ' + files.length + ' files (' + totalChunks + ' chunks) from ' + url, 'success');

  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
    showStatus('GitHub ingest failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Clone & Ingest';
  }
}

// ── Chat persistence ─────────────────────────────────────────────────────────
async function saveChatHistory() {
  const d = await openDB();
  await new Promise((resolve, reject) => {
    const tx = d.transaction('chat_sessions', 'readwrite');
    tx.objectStore('chat_sessions').put({ id: 'current', messages: chatHistory, savedAt: Date.now() });
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function loadChatHistory() {
  try {
    const d = await openDB();
    const saved = await new Promise((resolve, reject) => {
      const tx = d.transaction('chat_sessions', 'readonly');
      const req = tx.objectStore('chat_sessions').get('current');
      req.onsuccess = () => resolve(req.result);
      req.onerror  = () => resolve(null);
    });
    if (saved && Array.isArray(saved.messages) && saved.messages.length) {
      chatHistory = saved.messages;
      const messagesDiv = document.getElementById('chatMessages');
      messagesDiv.innerHTML = '';
      for (const msg of chatHistory) {
        addMessage(msg.role === 'user' ? 'user' : 'assistant', msg.content);
      }
    }
  } catch (e) { /* ignore, start fresh */ }
}

async function clearChat() {
  if (!confirm('Clear all chat history?')) return;
  chatHistory = [];
  const messagesDiv = document.getElementById('chatMessages');
  messagesDiv.innerHTML = '<div class="message assistant">Hello! Select a collection and ask me anything about your documents.</div>';
  await saveChatHistory();
}

// ── Chat ──────────────────────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  addMessage('user', message);
  input.value = '';

  const btn = document.getElementById('sendButton');
  btn.disabled = true;
  btn.textContent = '...';

  try {
    const { mode, chunks } = await retrieve(message, selectedCollection || null);

    const payload = {
      query: message,
      history: chatHistory.slice(-6),
    };

    const mapped = chunks.map(c => ({ id: c.id, text: c.text, source: c.source || c.filename }));
    if (mode === 'full') {
      payload.chunks = mapped;
    } else {
      payload.candidates = mapped;
    }

    const res = await fetch(baseUrl + '/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(res.status + ': ' + errText);
    }

    const data = await res.json();
    const answer = data.answer || 'No answer returned.';

    addMessage('assistant', answer);
    chatHistory.push({ role: 'user', content: message });
    chatHistory.push({ role: 'assistant', content: answer });
    if (chatHistory.length > 40) chatHistory = chatHistory.slice(-40);
    await saveChatHistory();

  } catch (e) {
    addMessage('assistant', 'Error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Send';
  }
}

function handleKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

function addMessage(type, content) {
  const messagesDiv = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'message ' + type;
  div.textContent = content;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ── Modals ────────────────────────────────────────────────────────────────────
function showModal(id) { document.getElementById(id).style.display = 'block'; }
function closeModal(id) { document.getElementById(id).style.display = 'none'; }

function showNewCollectionModal() {
  document.getElementById('newCollectionName').value = '';
  showModal('newCollectionModal');
}

function showAddDocModal() {
  if (!collections.length) { showStatus('Create a collection first', 'error'); return; }
  const sel = document.getElementById('addDocCollection');
  sel.innerHTML = '';
  for (const col of collections) {
    const opt = document.createElement('option');
    opt.value = col.id;
    opt.textContent = col.name;
    if (col.id === selectedCollection) opt.selected = true;
    sel.appendChild(opt);
  }
  document.getElementById('addDocFile').value = '';
  document.getElementById('addDocText').value = '';
  document.getElementById('addDocFilename').value = '';
  showModal('addDocModal');
}

async function doCreateCollection() {
  const name = document.getElementById('newCollectionName').value.trim();
  if (!name) { showStatus('Please enter a collection name', 'error'); return; }
  await createCollection(name);
  closeModal('newCollectionModal');
  showStatus('Collection "' + name + '" created', 'success');
}

async function doAddDocument() {
  const colId = document.getElementById('addDocCollection').value;
  const enhanced = true;
  const fileInput = document.getElementById('addDocFile');
  const textContent = document.getElementById('addDocText').value.trim();
  const manualFilename = document.getElementById('addDocFilename').value.trim();

  if (!colId) { showStatus('Select a collection', 'error'); return; }

  const btn = document.querySelector('#addDocModal .btn-green');
  btn.disabled = true;
  btn.textContent = 'Processing...';

  try {
    if (fileInput.files && fileInput.files.length > 0) {
      for (const file of fileInput.files) {
        const text = await file.text();
        await ingestText(text, file.name, colId, enhanced);
      }
    } else if (textContent) {
      const filename = manualFilename || ('note-' + new Date().toISOString().slice(0, 10) + '.txt');
      await ingestText(textContent, filename, colId, enhanced);
    } else {
      showStatus('Provide a file or paste text', 'error');
      btn.disabled = false;
      btn.textContent = 'Add Document';
      return;
    }

    await renderCollections();
    updateChatSelector();
    closeModal('addDocModal');
  } catch (e) {
    showStatus('Failed: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Add Document';
  }
}

// ── Status bar ────────────────────────────────────────────────────────────────
function showStatus(msg, type) {
  type = type || 'info';
  const bar = document.getElementById('statusBar');
  bar.textContent = msg;
  bar.className = type;
  clearTimeout(bar._timer);
  bar._timer = setTimeout(function() { bar.className = 'hidden'; bar.textContent = ''; }, 5000);
}

// ── Resizable panes ───────────────────────────────────────────────────────────
function setupResizablePanes() {
  const divider = document.getElementById('paneDivider');
  const leftPane = document.getElementById('leftPane');
  if (!divider || !leftPane) return;
  let dragging = false, startX = 0, startWidth = 0;
  divider.addEventListener('mousedown', function(e) {
    dragging = true; startX = e.clientX; startWidth = leftPane.getBoundingClientRect().width;
    divider.classList.add('dragging'); document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none'; e.preventDefault();
  });
  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    var w = Math.max(220, Math.min(600, startWidth + e.clientX - startX));
    leftPane.style.width = w + 'px'; leftPane.style.minWidth = w + 'px';
  });
  document.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false; divider.classList.remove('dragging');
    document.body.style.cursor = ''; document.body.style.userSelect = '';
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async function() {
  setupResizablePanes();

  document.getElementById('chatCollection').addEventListener('change', function(e) {
    var val = e.target.value;
    selectedCollection = val === 'all' ? null : val;
    var col = collections.find(function(c) { return c.id === selectedCollection; });
    chatHistory = [];
    document.getElementById('chatMessages').innerHTML = '';
    addMessage('assistant', selectedCollection
      ? 'Ready. Ask me anything about "' + (col ? col.name : 'this collection') + '".'
      : 'Ready. Ask me anything about your knowledge base.');
  });

  await loadCollections();
  await rebuildBM25();
  await loadChatHistory();

  if (chatHistory.length === 0) {
    addMessage('assistant', collections.length
      ? 'Welcome back! Select a collection and ask me anything.'
      : 'Welcome! Create a collection and add documents to get started.');
  }
});
