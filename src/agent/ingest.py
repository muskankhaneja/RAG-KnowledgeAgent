import os
import json
import pickle
import shutil
import subprocess
import tempfile
from typing import List, Dict

# numpy and embedding imports
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

try:
    import faiss
except Exception:
    faiss = None

# Lightweight fallback embeddings using TF-IDF + TruncatedSVD
def _compute_fallback_embeddings(texts, dim=384):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(texts)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return np.zeros((len(texts), dim), dtype=np.float32)

    Xdense = X.toarray().astype(np.float32)
    if Xdense.shape[0] == 1 or Xdense.shape[1] == 1:
        Xpad = np.zeros((Xdense.shape[0], dim), dtype=np.float32)
        Xpad[:, : Xdense.shape[1]] = Xdense
        return Xpad

    n_comp = min(dim, X.shape[1], X.shape[0] - 1)
    if n_comp <= 0:
        Xpad = np.zeros((Xdense.shape[0], dim), dtype=np.float32)
        Xpad[:, : Xdense.shape[1]] = Xdense
        return Xpad

    svd = TruncatedSVD(n_components=n_comp)
    Xred = svd.fit_transform(X)
    if Xred.ndim != 2:
        raise ValueError("Unexpected embeddings shape from SVD: %s" % (Xred.shape,))
    if Xred.shape[1] < dim:
        # pad with zeros up to target dimension
        Xpad = np.zeros((Xred.shape[0], dim), dtype=np.float32)
        Xpad[:, : Xred.shape[1]] = Xred
        Xred = Xpad
    elif Xred.shape[1] > dim:
        # truncate if needed
        Xred = Xred[:, :dim]
    return Xred.astype(np.float32)

from .utils import read_text_file, list_source_files


def chunk_text(text: str, max_words: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def _is_git_url(src: str) -> bool:
    return src.startswith("git@") or src.startswith("https://") or src.startswith("http://")


def _clone_repo(git_url: str, dest: str) -> None:
    # Uses system `git` to clone; this avoids adding GitPython as a dependency.
    subprocess.check_call(["git", "clone", "--depth", "1", git_url, dest])


def build_index(team: str, project: str, source_dir: str, persist_dir: str = "vectorstores", model_name: str = "all-MiniLM-L6-v2", doc_type: str = "mixed") -> Dict:
    os.makedirs(persist_dir, exist_ok=True)
    target = os.path.join(persist_dir, team, project)
    os.makedirs(target, exist_ok=True)

    # If source_dir is a git URL, clone into a temporary directory
    temp_dir = None
    source_root = source_dir
    if _is_git_url(source_dir):
        temp_dir = tempfile.mkdtemp(prefix="rag_clone_")
        _clone_repo(source_dir, temp_dir)
        source_root = temp_dir

    try:
        model = None
        if _HAS_ST:
            try:
                model = SentenceTransformer(model_name)
            except Exception:
                model = None

        files = list_source_files(source_root)
        docs = []
        for f in files:
            try:
                txt = read_text_file(f)
            except Exception:
                continue
            for i, chunk in enumerate(chunk_text(txt)):
                docs.append({
                    "text": chunk,
                    "source": f,
                    "chunk": i,
                    "team": team,
                    "project": project,
                    "doc_type": doc_type,
                })

        if len(docs) == 0:
            raise ValueError("No documents found to ingest")

        texts = [d["text"] for d in docs]
        if model is not None:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        else:
            embeddings = _compute_fallback_embeddings(texts)
        # normalize for cosine similarity with inner-product index
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        if faiss is None:
            raise RuntimeError("faiss is required but not available")

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        index_path = os.path.join(target, "index.faiss")
        faiss.write_index(index, index_path)

        meta_path = os.path.join(target, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(docs, f)

        config = {
            "team": team,
            "project": project,
            "model": model_name,
            "dim": dim,
            "count": len(docs),
            "source": source_dir,
            "doc_type": doc_type,
        }
        with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        return config
    finally:
        if temp_dir is not None:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
