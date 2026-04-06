# RAG-Agent (Team Assistant)

Minimal end-to-end Retrieval-Augmented-Generation (RAG) agent scaffold.

Quick start

1. Create and activate a Python environment (recommended).

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set `HF_ACCESS_TOKEN` in your environment for LLM responses.
   - Optionally set `HF_MODEL` to a Hugging Face model name, e.g. `google/flan-t5-large`.

4. Ingest a project directory (each project is a folder with code/docs), or a GitHub repo URL:

```bash
python -m src.agent.app ingest --team analytics --project myproj --source path/to/myproj
# or ingest directly from GitHub:
python -m src.agent.app ingest --team analytics --project myproj --source https://github.com/user/repo.git
```

5. Save uploaded docs persistently (optional):

```bash
curl -X POST "http://127.0.0.1:8000/upload" -H "Content-Type: application/json" -d '{"team":"analytics","project":"myproj","filename":"notes.txt","content":"Analysis notes..."}'
```

5. Run the API server:

```bash
python -m src.agent.app serve
```

6. Query the agent (example using curl):

```bash
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"How does auth work?","project":"myproj","top_k":5,"use_llm":true}'
```

Design notes

- Each ingested project produces a vectorstore under `vectorstores/<project>`.
- The retriever can search per-project or across all projects.
- LLM calls use Hugging Face inference if `HF_ACCESS_TOKEN` is provided; otherwise the API returns retrieved documents only.
