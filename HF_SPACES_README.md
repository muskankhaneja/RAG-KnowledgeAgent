---
title: RAG Knowledge Agent
emoji: 🧠
colorFrom: orange
colorTo: gray
sdk: docker
port: 7860
pinned: false
---

# RAG Knowledge Agent — Personal Knowledge Base

A personal, browser-first Retrieval-Augmented Generation system.
All your document data lives in your **browser** (IndexedDB) — the server is 100% stateless.

## Features
- 📁 Ingest documents (file upload, paste text, or GitHub repo URL)
- 🔍 BM25 retrieval running entirely in-browser
- 🤖 LLM-powered answers via HuggingFace Inference API
- 💾 All data stored in IndexedDB — no server-side storage
- 📤 Export / import your full knowledge base as JSON
- 🖼️ Resizable chat, collapsible sidebar

## Setup (HuggingFace Spaces)
Add `HF_ACCESS_TOKEN` as a Space **secret** with the **"Make calls to Inference Providers"** permission enabled.


## Setup

### Local Development

1. Clone the repo:
```bash
git clone https://github.com/muskankhaneja/RAG-KnowledgeAgent.git
cd RAG-KnowledgeAgent
```

2. Create `.env` file:
```bash
HF_ACCESS_TOKEN=your_hf_token_here
HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python -m src.agent.app serve
```

Visit `http://localhost:8000`

### Hugging Face Spaces Deployment

This space is already configured to run on HF Spaces. Just set the environment variable:

1. Go to Space **Settings** → **Variables and Secrets**
2. Add secret: `HF_ACCESS_TOKEN` = your Hugging Face API token
3. Restart the space

## Usage

1. **Create a Project**: Select a team name and project name, click "+ New"
2. **Ingest Documents**: 
   - Upload a file directly
   - Paste a GitHub repo link
   - Add a web URL
3. **Chat**: Select your project and ask questions—the LLM will answer using retrieved documents

## Architecture

- **Backend**: FastAPI + uvicorn
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS (CPU)
- **LLM**: Hugging Face Inference API
- **Frontend**: Vanilla JS with dark theme

## Environment Variables

- `HF_ACCESS_TOKEN`: Hugging Face API token (required)
- `HF_MODEL`: LLM model name (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `HF_TIMEOUT_SECONDS`: LLM inference timeout (default: 40)
- `HF_MAX_TOKENS`: Max output tokens (default: 256)
- `MAX_CONTEXT_PROJECTS`: Max projects to search (default: 3)
- `MAX_CONTEXT_HITS_PER_PROJECT`: Max docs per project (default: 2)

## License

MIT
