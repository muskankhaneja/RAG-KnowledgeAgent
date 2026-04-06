const baseUrl = window.location.origin;
console.log('RAG Agent UI baseUrl:', baseUrl);

// Global state
let currentProjects = {};
let selectedProject = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
  loadProjects();
  setupChat();
  setupResizablePanes();
});

function setupResizablePanes() {
  const divider = document.getElementById('paneDivider');
  const leftPane = document.getElementById('leftPane');
  if (!divider || !leftPane) return;

  let dragging = false;
  let startX = 0;
  let startWidth = 0;

  divider.addEventListener('mousedown', function(e) {
    dragging = true;
    startX = e.clientX;
    startWidth = leftPane.getBoundingClientRect().width;
    divider.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    const delta = e.clientX - startX;
    const newWidth = Math.max(200, Math.min(600, startWidth + delta));
    leftPane.style.width = newWidth + 'px';
    leftPane.style.minWidth = newWidth + 'px';
  });

  document.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false;
    divider.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
}

// Load and display ingested projects
async function loadProjects() {
  try {
    const response = await fetch(`${baseUrl}/projects`);
    const data = await response.json();
    currentProjects = data.projects || {};

    updateProjectLists();
    showStatus('Projects loaded successfully', 'success');
  } catch (error) {
    console.error('Error loading projects:', error);
    showStatus('Error loading projects: ' + error.message, 'error');
  }
}

// Update project dropdowns and lists
function updateProjectLists() {
  const projectList = document.getElementById('projectList');
  const actionProject = document.getElementById('actionProject');
  const chatProject = document.getElementById('chatProject');

  // Clear existing options
  projectList.innerHTML = '';
  actionProject.innerHTML = '<option value="">Select project...</option>';
  chatProject.innerHTML = '<option value="all">All Projects</option>';

  // Add projects to lists
  for (const [team, projects] of Object.entries(currentProjects)) {
    projects.forEach(project => {
      // Project tile with expand/collapse
      const item = document.createElement('div');
      item.className = 'project-item';
      item.innerHTML = `
        <div class="project-header" onclick="toggleProjectDocs(this, '${team}', '${project}')">
          <span class="project-chevron">▶</span>
          <span class="project-name">${team}/${project}</span>
          <div class="project-actions" onclick="event.stopPropagation()">
            <button title="Chat with this project" onclick="selectProject('${team}', '${project}')">💬</button>
            <button class="btn-grey" title="Rename project" onclick="showRenameDialog('${team}', '${project}')">✏️</button>
          </div>
        </div>
        <div class="project-docs" id="docs-${team}-${project}">
          <span class="no-docs">Click to load documents…</span>
        </div>
      `;
      projectList.appendChild(item);

      // Dropdown options
      const option = document.createElement('option');
      option.value = `${team}/${project}`;
      option.textContent = `${team}/${project}`;
      actionProject.appendChild(option.cloneNode(true));
      chatProject.appendChild(option.cloneNode(true));
    });
  }
}

// Select a project for chat
function selectProject(team, project) {
  selectedProject = { team, project };
  document.getElementById('chatProject').value = `${team}/${project}`;

  // Clear chat and show welcome message
  document.getElementById('chatMessages').innerHTML = '';
  addMessage('assistant', `Ready to chat about ${team}/${project}. Ask me anything about the documents!`);

  showStatus(`Selected project: ${team}/${project}`, 'info');
}

// Ingest from local path
async function doIngest() {
  const team = document.getElementById('team').value;
  const project = document.getElementById('project').value;
  const source = document.getElementById('source').value;

  if (!project || !source) {
    showStatus('Please provide project name and source path', 'error');
    return;
  }

  const body = { team, project, source, doc_type: 'mixed' };

  showStatus('Ingesting project...', 'info');
  disableButtons(true);

  try {
    const response = await fetch(`${baseUrl}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`${response.status}: ${error}`);
    }

    const result = await response.json();
    showStatus(`Project ingested successfully: ${result.count} documents`, 'success');
    loadProjects(); // Refresh project list

  } catch (error) {
    showStatus('Ingestion failed: ' + error.message, 'error');
  } finally {
    disableButtons(false);
  }
}

// Ingest from GitHub
async function doIngestGitHub() {
  const team = document.getElementById('team').value;
  const project = document.getElementById('project').value;
  const githubUrl = document.getElementById('githubUrl').value;

  if (!project || !githubUrl) {
    showStatus('Please provide project name and GitHub URL', 'error');
    return;
  }

  const body = { team, project, source: githubUrl, doc_type: 'mixed' };

  showStatus('Cloning and ingesting from GitHub...', 'info');
  disableButtons(true);

  try {
    const response = await fetch(`${baseUrl}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`${response.status}: ${error}`);
    }

    const result = await response.json();
    showStatus(`GitHub project ingested successfully: ${result.count} documents`, 'success');
    loadProjects(); // Refresh project list

  } catch (error) {
    showStatus('GitHub ingestion failed: ' + error.message, 'error');
  } finally {
    disableButtons(false);
  }
}

// Upload document
async function doUpload() {
  const projectPath = document.getElementById('uploadProject').value;
  const filename = document.getElementById('filename').value;
  const content = document.getElementById('content').value;

  if (!projectPath || !filename || !content) {
    showStatus('Please select project, provide filename, and content', 'error');
    return;
  }

  const [team, project] = projectPath.split('/');
  const body = { team, project, filename, content, doc_type: 'uploaded' };

  showStatus('Uploading document...', 'info');
  disableButtons(true);

  try {
    const response = await fetch(`${baseUrl}/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`${response.status}: ${error}`);
    }

    const result = await response.json();
    showStatus(`Document uploaded and reindexed: ${filename}`, 'success');

  } catch (error) {
    showStatus('Upload failed: ' + error.message, 'error');
  } finally {
    disableButtons(false);
  }
}

// Setup chat functionality
function setupChat() {
  document.getElementById('chatProject').addEventListener('change', function(e) {
    const projectPath = e.target.value;
    if (projectPath === 'all') {
      selectedProject = null;
      document.getElementById('chatMessages').innerHTML = '';
      addMessage('assistant', 'Ready to chat about all documents. Ask me anything!');
      showStatus('Selected all projects', 'info');
    } else if (projectPath) {
      const [team, project] = projectPath.split('/');
      selectProject(team, project);
    }
  });
}

// Handle chat message sending
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();

  if (!message) return;

  // Add user message to chat
  addMessage('user', message);
  input.value = '';

  // Disable send button during processing
  const sendButton = document.getElementById('sendButton');
  sendButton.disabled = true;
  sendButton.textContent = 'Thinking...';

  try {
    const body = {
      query: message,
      top_k: 5,
      use_llm: true
    };
    if (selectedProject) {
      body.team = selectedProject.team;
      body.project = selectedProject.project;
    }

    const response = await fetch(`${baseUrl}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`${response.status}: ${error}`);
    }

    const result = await response.json();

    // Format response
    let responseText = '';

    if (result.llm_error) {
      responseText = `Error: ${result.llm_error}`;
    } else if (result.answer) {
      responseText = result.answer;
    } else if (result.retrieved) {
      // Show retrieved documents
      responseText = 'Retrieved documents:\n\n';
      for (const [proj, hits] of Object.entries(result.retrieved)) {
        hits.forEach(hit => {
          responseText += `📄 ${hit.source}\n${hit.text.substring(0, 200)}...\n\n`;
        });
      }
    } else {
      responseText = 'No results found.';
    }

    addMessage('assistant', responseText);

  } catch (error) {
    addMessage('assistant', `Error: ${error.message}`);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = 'Send';
  }
}

// Handle Enter key in chat input
function handleKeyPress(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

// Add message to chat
function addMessage(type, content) {
  const messagesDiv = document.getElementById('chatMessages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${type}`;
  messageDiv.textContent = content;
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Toggle project docs expand/collapse
async function toggleProjectDocs(headerEl, team, project) {
  const docsEl = document.getElementById(`docs-${team}-${project}`);
  const chevron = headerEl.querySelector('.project-chevron');
  const isOpen = docsEl.classList.contains('open');

  if (isOpen) {
    docsEl.classList.remove('open');
    chevron.style.transform = '';
    return;
  }

  chevron.style.transform = 'rotate(90deg)';
  docsEl.classList.add('open');
  docsEl.innerHTML = '<span class="no-docs">Loading…</span>';

  try {
    const resp = await fetch(`${baseUrl}/projects/${team}/${project}/docs`);
    const data = await resp.json();
    const docs = data.docs || [];
    if (docs.length === 0) {
      docsEl.innerHTML = '<span class="no-docs">No documents ingested yet.</span>';
    } else {
      docsEl.innerHTML = docs.map(d => {
        const isUrl = d.startsWith('http://') || d.startsWith('https://');
        return `<div class="doc-item">${isUrl ? `<a href="${d}" target="_blank">${d}</a>` : `📄 ${d}`}</div>`;
      }).join('');
    }
  } catch (e) {
    docsEl.innerHTML = `<span class="no-docs">Error loading docs: ${e.message}</span>`;
  }
}

// Show status message
function showStatus(message, type = 'info') {
  const bar = document.getElementById('statusBar');
  bar.textContent = message;
  bar.className = type;
  clearTimeout(bar._timer);
  bar._timer = setTimeout(() => { bar.className = 'hidden'; bar.textContent = ''; }, 5000);
}

// Disable/enable buttons during operations
function disableButtons(disabled) {
  const buttons = document.querySelectorAll('button');
  buttons.forEach(button => {
    if (button.id !== 'sendButton') {
      button.disabled = disabled;
    }
  });
}

// Modal functions
function showModal(modalId) {
  document.getElementById(modalId).style.display = 'block';
}

function closeModal(modalId) {
  document.getElementById(modalId).style.display = 'none';
}

// Dialog show functions
function showCreateProjectDialog() {
  showModal('createProjectModal');
}

function showUploadDialog() {
  showModal('uploadModal');
}

function showAddWebUrlDialog() {
  showModal('webUrlModal');
}

function showIngestGitHubDialog() {
  showModal('githubModal');
}

// Action functions
async function doCreateProject() {
  const team = document.getElementById('newTeam').value;
  const project = document.getElementById('newProject').value;
  if (!team || !project) {
    showStatus('Please provide team and project name', 'error');
    return;
  }
  try {
    const response = await fetch(`${baseUrl}/projects/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team, project })
    });
    if (!response.ok) throw new Error(`${response.status}`);
    showStatus('Project created successfully', 'success');
    loadProjects();
    closeModal('createProjectModal');
  } catch (error) {
    showStatus('Failed to create project: ' + error.message, 'error');
  }
}

async function doUploadDocument() {
  const projectPath = document.getElementById('actionProject').value;
  const filename = document.getElementById('uploadFilename').value;
  const content = document.getElementById('uploadContent').value;
  if (!projectPath || !filename || !content) {
    showStatus('Please select project, provide filename, and content', 'error');
    return;
  }
  const [team, project] = projectPath.split('/');
  try {
    const response = await fetch(`${baseUrl}/upload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team, project, filename, content, doc_type: 'uploaded' })
    });
    if (!response.ok) throw new Error(`${response.status}`);
    showStatus('Document uploaded successfully', 'success');
    closeModal('uploadModal');
  } catch (error) {
    showStatus('Upload failed: ' + error.message, 'error');
  }
}

async function doAddWebUrl() {
  const projectPath = document.getElementById('actionProject').value;
  const url = document.getElementById('webUrl').value;
  if (!projectPath || !url) {
    showStatus('Please select project and provide URL', 'error');
    return;
  }
  const [team, project] = projectPath.split('/');
  try {
    const response = await fetch(`${baseUrl}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team, project, source: url, doc_type: 'web' })
    });
    if (!response.ok) throw new Error(`${response.status}`);
    showStatus('Web URL ingested successfully', 'success');
    loadProjects();
    closeModal('webUrlModal');
  } catch (error) {
    showStatus('Ingestion failed: ' + error.message, 'error');
  }
}

async function doIngestGitHubModal() {
  const projectPath = document.getElementById('actionProject').value;
  const githubUrl = document.getElementById('githubUrl').value;
  if (!projectPath || !githubUrl) {
    showStatus('Please select project and provide GitHub URL', 'error');
    return;
  }
  const [team, project] = projectPath.split('/');
  try {
    const response = await fetch(`${baseUrl}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team, project, source: githubUrl, doc_type: 'mixed' })
    });
    if (!response.ok) throw new Error(`${response.status}`);
    showStatus('GitHub ingested successfully', 'success');
    loadProjects();
    closeModal('githubModal');
  } catch (error) {
    showStatus('Ingestion failed: ' + error.message, 'error');
  }
}

// Rename project
let _renameTarget = null;

function showRenameDialog(team, project) {
  _renameTarget = { team, project };
  document.getElementById('renameCurrentLabel').textContent = `Current: ${team}/${project}`;
  document.getElementById('renameNewName').value = project;
  showModal('renameModal');
}

async function doRenameProject() {
  if (!_renameTarget) return;
  const newName = document.getElementById('renameNewName').value.trim();
  if (!newName || newName === _renameTarget.project) {
    showStatus('Please enter a different project name', 'error');
    return;
  }
  try {
    const response = await fetch(`${baseUrl}/projects/rename`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ team: _renameTarget.team, old_project: _renameTarget.project, new_project: newName })
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || response.status);
    }
    showStatus(`Renamed to ${_renameTarget.team}/${newName}`, 'success');
    if (selectedProject && selectedProject.team === _renameTarget.team && selectedProject.project === _renameTarget.project) {
      selectedProject = null;
    }
    _renameTarget = null;
    loadProjects();
    closeModal('renameModal');
  } catch (error) {
    showStatus('Rename failed: ' + error.message, 'error');
  }
}
