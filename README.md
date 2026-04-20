---
sdk: docker
port: 7860
---

# RAG Agent — Personal Knowledge Base

A personal, browser-first Retrieval-Augmented Generation (RAG) system. All your document data lives in your browser (IndexedDB); the server is **100% stateless** — it only receives context and a query, then returns an LLM-generated answer. No database, no user accounts, no server-side storage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (SPA)                       │
│                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  IndexedDB  │◄───│  BM25 Index  │◄───│  Ingest   │  │
│  │  (chunks,   │    │  (in-memory, │    │  Engine   │  │
│  │  questions) │    │  rebuilt on  │    │           │  │
│  └─────────────┘    │  load/ingest)│    └───────────┘  │
│         │           └──────────────┘          ▲        │
│         │                   │                 │        │
│         └─────────┬─────────┘        File / Paste /    │
│                   │                  GitHub URL         │
│             Retrieval                                   │
│          (BM25 or full context)                         │
│                   │                                     │
└───────────────────┼─────────────────────────────────────┘
                    │  POST /chat  { query, context, history }
                    ▼
         ┌──────────────────┐
         │  FastAPI Server  │  ← stateless, no storage
         │  (stateless)     │
         └────────┬─────────┘
                  │
                  ▼
         HuggingFace Inference API
         (Qwen2.5-1.5B-Instruct)
```

### Data Flow

1. **Ingest** — User adds documents (file upload, paste, or GitHub repo URL). The browser chunks the text, stores chunks in IndexedDB, and optionally calls `POST /ingest/questions` to generate additional BM25-indexed search questions per chunk via LLM.
2. **Retrieve** — On each query, the browser auto-selects a retrieval strategy:
   - **Full-context mode**: if total KB < ~60 k tokens, all chunks are sent directly to the LLM as context.
   - **BM25 mode**: otherwise, the in-memory BM25 index returns the top-20 candidate chunks, which are sent to the server for LLM reranking before the final answer is generated.
3. **Answer** — The server builds a chat-history-aware prompt and returns the LLM response.

---

## Techniques

| Technique | Detail |
|---|---|
| **BM25 retrieval** | Pure-JS BM25 (k1=1.5, b=0.75) runs entirely in the browser. Rebuilt from IndexedDB on page load and after each ingest. |
| **Enhanced indexing** | For each chunk, the LLM generates 3–5 synthetic search questions. These are stored alongside the chunk and included in the BM25 index, dramatically improving recall for paraphrased queries. |
| **Full-context fallback** | When the total knowledge base is small enough to fit in one LLM context window, all chunks are passed directly — no retrieval loss. |
| **Auto-detection** | The browser estimates total tokens and picks between full-context and BM25 mode transparently. |
| **LLM reranking** | In BM25 mode the server can optionally rerank the top candidates before answering. |
| **Chat history** | The last N conversation turns are appended to each prompt to support follow-up questions. |
| **Chunking with overlap** | Documents are split into ~200-word chunks with a 40-word overlap to avoid context boundary artifacts. |
| **IndexedDB persistence** | All data survives page reload and browser restart. Only explicit "Clear site data" removes it. |
| **Export / Import** | The entire knowledge base can be serialised to a single JSON file and restored on another device. |
| **GitHub ingest** | The server clones a public GitHub repo to a temp directory, extracts all text files, returns them to the browser for chunking and indexing, then deletes the temp dir. |
| **Mock mode** | Set `MOCK_LLM=1` to run the server without an LLM token — `/chat` returns a debug response showing exactly what context was retrieved. |

---

## Project Structure

```
src/
  agent/
    app.py        # FastAPI — stateless endpoints: /chat, /ingest/questions, /fetch/github
    model.py      # HuggingFace InferenceClient wrapper
    ingest.py     # Git clone helper (reused by /fetch/github)
    utils.py      # File reading helpers (reused by /fetch/github)
  static/
    index.html    # Single-page UI (collections, chat, modals)
    app.js        # All client-side logic (IndexedDB, BM25, ingest, chat)
    config.js     # Runtime config (API base URL)
```

---

## Quick Start (local)

### Prerequisites

- Python 3.9+
- A HuggingFace account with a token that has **"Make calls to Inference Providers"** permission enabled at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rag-agent.git
cd rag-agent

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Create a .env file
echo "HF_ACCESS_TOKEN=hf_your_token_here" > .env
echo "HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct"  >> .env

# 4. Start the server
python -m uvicorn src.agent.app:app --host 127.0.0.1 --port 8007
```

Open [http://127.0.0.1:8007](http://127.0.0.1:8007) in your browser.

### Mock mode (no token required)

```bash
MOCK_LLM=1 python -m uvicorn src.agent.app:app --host 127.0.0.1 --port 8007
```

The chat endpoint will return a debug response showing retrieved context instead of calling the LLM.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Stateless chat — browser sends `{query, chunks[], history[]}`, server returns LLM answer |
| `POST` | `/ingest/questions` | Generate 3–5 search questions per chunk for enhanced indexing |
| `POST` | `/fetch/github` | Clone a public GitHub repo and return all text file contents |

---

## HuggingFace Spaces Deployment

1. Push the repository to a HuggingFace Space (Docker SDK).
2. Add `HF_ACCESS_TOKEN` as a Space secret.
3. The `Dockerfile` is already configured — the server binds to `0.0.0.0:7860` as required by Spaces.

> **Token permission note**: The token must have **"Make calls to Inference Providers"** enabled. Standard read tokens are insufficient.

---

## Browser Compatibility

Requires a modern browser with IndexedDB support (Chrome 80+, Firefox 79+, Edge 80+, Safari 15+).
