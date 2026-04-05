import argparse
import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .ingest import build_index
from .retriever import query, list_projects
from .model import call_openai_chat


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


app = FastAPI(title="RAG Agent API")

# Serve static front-end UI
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

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
    if not os.path.exists(payload.source):
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
    results = query(p.team, p.project, p.query, p.top_k)
    if p.use_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # return raw docs if no key
            return {"retrieved": results, "llm": None, "note": "OPENAI_API_KEY not set"}
        # build context
        context_texts = []
        for proj, hits in results.items():
            for h in hits:
                context_texts.append(f"Project: {proj}\nSource: {h.get('source')}\nText: {h.get('text')}\nScore: {h.get('score')}\n---\n")
        context = "\n".join(context_texts)
        system = "You are an assistant that uses the provided context to answer the user's question. If context contradicts itself, prefer the most relevant extracts. Keep answers concise and actionable."
        prompt = f"Context:\n{context}\nUser question:\n{p.query}\nProvide an answer, cite sources if helpful."
        try:
            answer = call_openai_chat(system, prompt, api_key)
        except Exception as e:
            return {"retrieved": results, "llm_error": str(e)}
        return {"retrieved": results, "answer": answer}
    return {"retrieved": results}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--team", default="default", help="Team name for this project")
    p_ingest.add_argument("--project", required=True)
    p_ingest.add_argument("--source", required=True)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

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
