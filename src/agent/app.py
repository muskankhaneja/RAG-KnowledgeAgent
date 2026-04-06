import argparse
import os
import json
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Response, Body
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


app = FastAPI(title="RAG Agent API")

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
    results = query(p.team, p.project, p.query, p.top_k)
    if p.use_llm:
        api_key = os.environ.get("HF_ACCESS_TOKEN")
        if not api_key:
            # return raw docs if no key
            return {"retrieved": results, "llm": None, "note": "HF_ACCESS_TOKEN not set"}
        # build context
        context_texts = []
        for proj, hits in results.items():
            for h in hits:
                context_texts.append(f"Project: {proj}\nSource: {h.get('source')}\nText: {h.get('text')}\nScore: {h.get('score')}\n---\n")
        context = "\n".join(context_texts)
        system = "You are an assistant that uses the provided context to answer the user's question. If context contradicts itself, prefer the most relevant extracts. Keep answers concise and actionable."
        prompt = f"Context:\n{context}\nUser question:\n{p.query}\nProvide an answer, cite sources if helpful."
        model = os.environ.get("HF_MODEL", "google/flan-t5-large")
        try:
            answer = call_hf_chat(system, prompt, api_key, model)
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
