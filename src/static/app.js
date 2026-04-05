const baseUrl = (window.location.origin.includes('8100') ? window.location.origin : 'http://127.0.0.1:8100');

function show(msg) {
  document.getElementById('output').textContent = msg;
}

async function doIngest() {
  const team = document.getElementById('team').value;
  const project = document.getElementById('project').value;
  const source = document.getElementById('source').value;
  const doc_type = document.getElementById('doc_type').value || 'mixed';
  if (!source) { show('Please set source folder path'); return; }
  const body = {team, project, source, doc_type};
  show('Ingesting...');
  const response = await fetch(`${baseUrl}/ingest`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const json = await response.json();
  show(JSON.stringify(json, null, 2));
}

async function doUpload() {
  const team = document.getElementById('team').value;
  const project = document.getElementById('project').value;
  const filename = document.getElementById('filename').value;
  const content = document.getElementById('content').value;
  const body = {team, project, filename, content, doc_type:'uploaded'};
  show('Uploading...');
  const response = await fetch(`${baseUrl}/upload`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const json = await response.json();
  show(JSON.stringify(json, null, 2));
}

async function doQuery() {
  const team = document.getElementById('team').value;
  const project = document.getElementById('project').value;
  const queryText = document.getElementById('queryText').value;
  const top_k = parseInt(document.getElementById('topK').value || '5');
  const use_llm = document.getElementById('useLLM').value === 'true';
  const body = {team, project, query: queryText, top_k, use_llm};
  show('Querying...');
  const response = await fetch(`${baseUrl}/query`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const json = await response.json();
  show(JSON.stringify(json, null, 2));
}
