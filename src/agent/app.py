import argparse
import os
import json
import time
import collections
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Response, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .ingest import build_index
from .retriever import query, list_projects
from .model import call_hf_chat


class IngestPayload(BaseModel):
    team: Optional[str] = "default"
    project: str
    source: str
    doc_type: Optional[str] = "mixed"


class UploadPayload(BaseModel):
    team: Optional[str] = "default"
    project: str
    filename: str
    content: str
    doc_type: Optional[str] = "uploaded"


class QueryPayload(BaseModel):
    query: str
    team: Optional[str] = None
    project: Optional[str] = None
    top_k: int = 5
    use_llm: bool = False


# ── New local-first endpoints ──────────────────────────────────────────────────

class ChunkItem(BaseModel):
    id: str
    text: str
    source: Optional[str] = "unknown"


class ChatPayload(BaseModel):
    query: str
    candidates: Optional[List[ChunkItem]] = None   # BM25 results from browser
    chunks: Optional[List[ChunkItem]] = None        # full-context mode
    history: Optional[List[Dict[str, str]]] = None


class QuestionsPayload(BaseModel):
    chunks: List[ChunkItem]


class GitHubFetchPayload(BaseModel):
    url: str                             # GitHub repo URL
    max_files: Optional[int] = 200      # safety cap


class UrlFetchPayload(BaseModel):
    url: str                             # Web page URL


app = FastAPI(title="RAG Agent API")

# ── Rate limiter (in-memory, per IP) ──────────────────────────────────────────
_RATE_LIMIT  = int(os.environ.get("RATE_LIMIT_RPM", "20"))   # max requests per window
_RATE_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60")) # window size in seconds
_rate_store: Dict[str, collections.deque] = {}

def _check_rate_limit(ip: str) -> None:
    """Raise 429 if the IP has exceeded RATE_LIMIT_RPM in the last RATE_WINDOW seconds."""
    now = time.time()
    if ip not in _rate_store:
        _rate_store[ip] = collections.deque()
    dq = _rate_store[ip]
    # drop timestamps outside the window
    while dq and now - dq[0] > _RATE_WINDOW:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_RATE_LIMIT} requests per {_RATE_WINDOW}s. Please wait and try again."
        )
    dq.append(now)

# Serve static front-end UI from /static
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/projects/{team}/{project}/docs")
def get_project_docs(team: str, project: str):
    """Return list of documents/sources ingested for a project."""
    import glob
    base_dir = os.path.join("vectorstores", team, project)
    config_path = os.path.join(base_dir, "config.json")
    docs = []
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # config may have a 'source' field or 'documents' list
        if "documents" in cfg and isinstance(cfg["documents"], list):
            docs = cfg["documents"]
        elif "source" in cfg:
            docs = [cfg["source"]]
    # also look for a data directory with files
    data_dir = os.path.join("data", team, project)
    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath) and fname not in docs:
                docs.append(fname)
    return {"team": team, "project": project, "docs": docs}


# Endpoint to create a new project
@app.post("/projects/create")
def create_project(team: str = Body(...), project: str = Body(...)):
    base_dir = os.path.join("vectorstores", team, project)
    os.makedirs(base_dir, exist_ok=True)
    # Optionally, create an empty config.json
    config_path = os.path.join(base_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"team": team, "project": project, "documents": []}, f)
    return {"created": True, "team": team, "project": project}


@app.post("/projects/rename")
def rename_project(team: str = Body(...), old_project: str = Body(...), new_project: str = Body(...)):
    old_dir = os.path.join("vectorstores", team, old_project)
    new_dir = os.path.join("vectorstores", team, new_project)
    if not os.path.exists(old_dir):
        raise HTTPException(status_code=404, detail="Project not found")
    if os.path.exists(new_dir):
        raise HTTPException(status_code=400, detail="A project with that name already exists")
    os.rename(old_dir, new_dir)
    # Update config.json if it exists
    config_path = os.path.join(new_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["project"] = new_project
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    return {"renamed": True, "team": team, "old_project": old_project, "new_project": new_project}


@app.get("/projects/{project_name}/config")
def get_project_config(project_name: str):
    config_path = f"vectorstores/{project_name}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return JSONResponse(content=json.load(f))
    else:
        return JSONResponse(content={"documents": []}, status_code=404)


def _is_git_url(src: str) -> bool:
    return src.startswith("git@") or src.startswith("https://") or src.startswith("http://")


# Allow CORS for development frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
def ingest(payload: IngestPayload):
    if not _is_git_url(payload.source) and not os.path.exists(payload.source):
        raise HTTPException(status_code=400, detail="source path not found")
    cfg = build_index(payload.team or "default", payload.project, payload.source, doc_type=payload.doc_type or "mixed")
    return cfg


@app.post("/upload")
def upload_doc(payload: UploadPayload):
    base_dir = os.path.join("data", payload.team or "default", payload.project)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, payload.filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(payload.content)
    # Rebuild index from combined project directory so upload persists
    cfg = build_index(payload.team or "default", payload.project, base_dir, doc_type=payload.doc_type or "uploaded")
    return {
        "saved_to": file_path,
        "project": payload.project,
        "team": payload.team,
        "index_config": cfg,
    }


@app.get("/projects")
def get_projects():
    return {"projects": list_projects()}


@app.post("/query")
def api_query(p: QueryPayload):
    top_k = max(1, min(int(p.top_k or 5), 5))
    results = query(p.team, p.project, p.query, top_k)
    if p.use_llm:
        api_key = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
        if not api_key:
            # return raw docs if no key
            return {"retrieved": results, "llm": None, "note": "HF_ACCESS_TOKEN not set"}
        # build context
        max_projects = int(os.environ.get("MAX_CONTEXT_PROJECTS", "3"))
        max_hits_per_project = int(os.environ.get("MAX_CONTEXT_HITS_PER_PROJECT", "2"))
        max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS", "6000"))
        context_texts = []
        for proj_idx, (proj, hits) in enumerate(results.items()):
            if proj_idx >= max_projects:
                break
            for h in hits[:max_hits_per_project]:
                context_texts.append(f"Project: {proj}\nSource: {h.get('source')}\nText: {h.get('text')}\nScore: {h.get('score')}\n---\n")
        context = "\n".join(context_texts)
        if len(context) > max_context_chars:
            context = context[:max_context_chars]
        system = "You are an assistant that uses the provided context to answer the user's question. If context contradicts itself, prefer the most relevant extracts. Keep answers concise and actionable."
        prompt = f"Context:\n{context}\nUser question:\n{p.query}\nProvide an answer, cite sources if helpful."
        model = os.environ.get("HF_MODEL", "google/flan-t5-large")
        try:
            answer = call_hf_chat(system, prompt, api_key, model)
        except Exception as e:
            return {"retrieved": results, "llm_error": str(e)}
        return {"retrieved": results, "answer": answer}
    return {"retrieved": results}


# ── Local-first: stateless chat (context sent from browser) ───────────────────

@app.post("/chat")
def chat_endpoint(payload: ChatPayload, request: Request):
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    _check_rate_limit(client_ip)

    api_key = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    mock_mode = not api_key or os.environ.get("MOCK_LLM", "").lower() in ("1", "true", "yes")

    # Accept either full-context chunks or BM25 candidate chunks
    context_items = payload.chunks or payload.candidates or []

    max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS", "12000"))
    context_parts = []
    total = 0
    for c in context_items[:25]:
        entry = f"[Source: {c.source}]\n{c.text}"
        if total + len(entry) > max_context_chars:
            break
        context_parts.append(entry)
        total += len(entry)

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No documents provided."

    system = (
        "You are a helpful personal knowledge base assistant. "
        "Answer questions using only the provided document context. "
        "If the answer is not found in the context, say so clearly. "
        "Be concise. Cite the source filename when relevant."
    )

    history = payload.history or []
    history_text = ""
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prefix = "User" if role == "user" else "Assistant"
        history_text += f"{prefix}: {content}\n"

    prompt = (
        f"Document context:\n{context_str}\n\n"
        + (f"Conversation so far:\n{history_text}\n" if history_text else "")
        + f"User: {payload.query}\n\nAnswer:"
    )

    model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

    if mock_mode:
        sources = list({c.source for c in context_items[:5]})
        snippet = context_parts[0][:200] if context_parts else "no context"
        answer = (
            f"[MOCK MODE — no HF_ACCESS_TOKEN set]\n\n"
            f"Query: {payload.query}\n\n"
            f"Retrieved {len(context_items)} chunk(s) from: {', '.join(sources) or 'none'}.\n\n"
            f"First chunk preview: \"{snippet}...\""
        )
        return {"answer": answer}

    try:
        answer = call_hf_chat(system, prompt, api_key, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"answer": answer}


# ── Local-first: generate search-index questions per chunk ────────────────────

@app.post("/ingest/questions")
def generate_questions_endpoint(payload: QuestionsPayload):
    api_key = os.environ.get("HF_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    if not api_key:
        return {"results": [{"id": c.id, "questions": []} for c in payload.chunks]}

    model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    results = []

    for chunk in payload.chunks[:30]:          # cap to avoid rate limiting
        try:
            prompt = (
                f"Read this passage and write exactly 3 short questions it answers. "
                f"Return only the questions, one per line, no numbering or bullets.\n\n"
                f"Passage: {chunk.text[:800]}\n\nQuestions:"
            )
            raw = call_hf_chat(
                "You write concise search-index questions for document retrieval.",
                prompt, api_key, model,
            )
            questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in raw.strip().split("\n")
                if q.strip() and len(q.strip()) > 8
            ][:5]
        except Exception:
            questions = []

        results.append({"id": chunk.id, "questions": questions})

    return {"results": results}


# ── Local-first: fetch GitHub repo contents → return to browser for local ingest

@app.post("/fetch/github")
def fetch_github(payload: GitHubFetchPayload):
    import tempfile, shutil
    from .ingest import _clone_repo
    from .utils import list_source_files, read_text_file

    url = payload.url.strip()
    if not (url.startswith("https://") or url.startswith("http://") or url.startswith("git@")):
        raise HTTPException(status_code=400, detail="Invalid git URL")

    tmp = tempfile.mkdtemp(prefix="rag_gh_")
    try:
        try:
            _clone_repo(url, tmp)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"git clone failed: {e}")

        files = list_source_files(tmp)[:payload.max_files]
        results = []
        for fpath in files:
            try:
                text = read_text_file(fpath)
                if not text.strip():
                    continue
                rel = os.path.relpath(fpath, tmp).replace("\\", "/")
                results.append({"filename": rel, "text": text})
            except Exception:
                continue

        if not results:
            raise HTTPException(status_code=422, detail="No readable text files found in repo")

        return {"url": url, "files": results}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── Local-first: fetch a web page → return clean text to browser for local ingest

@app.post("/fetch/url")
def fetch_url(payload: UrlFetchPayload):
    import re
    import html as html_module

    url = payload.url.strip()
    if not (url.startswith("https://") or url.startswith("http://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        import requests as req_lib
        resp = req_lib.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Agent/1.0)"},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to fetch URL: {e}")

    content_type = resp.headers.get("content-type", "")
    if "text/" not in content_type and "application/xhtml" not in content_type:
        raise HTTPException(status_code=422, detail=f"Unsupported content type: {content_type}")

    raw = resp.text

    # Strip <script>, <style>, <head>, <nav>, <footer> blocks entirely
    for tag in ("script", "style", "head", "nav", "footer", "header", "aside"):
        raw = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", raw, flags=re.DOTALL | re.IGNORECASE)

    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", raw)

    # Decode HTML entities
    text = html_module.unescape(text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) < 100:
        raise HTTPException(status_code=422, detail="Page returned too little readable text")

    # Derive a filename from the URL
    from urllib.parse import urlparse
    parsed = urlparse(url)
    slug = (parsed.netloc + parsed.path).strip("/").replace("/", "_")
    slug = re.sub(r"[^\w\-.]", "_", slug)[:80] or "webpage"
    filename = slug + ".txt"

    return {"url": url, "filename": filename, "text": text}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--team", default="default", help="Team name for this project")
    p_ingest.add_argument("--project", required=True)
    p_ingest.add_argument("--source", required=True)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))

    args = parser.parse_args()
    if args.cmd == "ingest":
        team = getattr(args, "team", "default")
        print(build_index(team, args.project, args.source))
    elif args.cmd == "serve":
        uvicorn.run("src.agent.app:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
