import os
import pickle
from typing import List, Dict, Optional

import numpy as np
try:
_EMBED_MODEL = None

try:
    import faiss
except Exception:
    faiss = None

_HAS_ST = None

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
        Xpad = np.zeros((Xred.shape[0], dim), dtype=np.float32)
        Xpad[:, : Xred.shape[1]] = Xred
        Xred = Xpad
    elif Xred.shape[1] > dim:
        Xred = Xred[:, :dim]
    return Xred.astype(np.float32)


def _index_paths(project: str, team: Optional[str] = None, persist_dir: str = "vectorstores"):
    if team:
        base = os.path.join(persist_dir, team, project)
    else:
        base = os.path.join(persist_dir, project)
        if not os.path.exists(base):
            # fallback to team-nested path when no top-level project exists
            for t in os.listdir(persist_dir):
                tpath = os.path.join(persist_dir, t, project)
                if os.path.isdir(tpath):
                    base = tpath
                    break
    return os.path.join(base, "index.faiss"), os.path.join(base, "metadata.pkl"), os.path.join(base, "config.json")


def list_projects(persist_dir: str = "vectorstores") -> Dict[str, List[str]]:
    if not os.path.exists(persist_dir):
        return {}
    teams = {}
    for team_name in os.listdir(persist_dir):
        team_path = os.path.join(persist_dir, team_name)
        if not os.path.isdir(team_path):
            continue
        projects = [p for p in os.listdir(team_path) if os.path.isdir(os.path.join(team_path, p))]
        if projects:
            teams[team_name] = projects
    # include any top-level ungrouped projects for backward compatibility
    ungrouped = []
    for p in os.listdir(persist_dir):
        ppath = os.path.join(persist_dir, p)
        if os.path.isdir(ppath) and p not in teams:
            # already listed as a team folder; skip
            continue
    if ungrouped:
        teams["ungrouped"] = ungrouped
    return teams


def load_index(project: str, team: Optional[str] = None, persist_dir: str = "vectorstores"):
    index_path, meta_path, _ = _index_paths(project, team, persist_dir)
    if not os.path.exists(index_path):
        raise FileNotFoundError("Index not found for project: %s" % project)
    if faiss is None:
        raise RuntimeError("faiss is required but not available")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def _get_embedder(model_name: str):
    """Load and cache one embedding model per process to avoid OOM churn."""
    global _EMBED_MODEL, _HAS_ST
    disable_local = os.environ.get("DISABLE_LOCAL_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
    if disable_local:
        return None

    if _HAS_ST is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _HAS_ST = SentenceTransformer
        except Exception:
            _HAS_ST = False

    if not _HAS_ST:
        return None

    if _EMBED_MODEL is None:
        _EMBED_MODEL = _HAS_ST(model_name)
    return _EMBED_MODEL


def query(team: Optional[str], project: Optional[str], query_text: str, top_k: int = 5, persist_dir: str = "vectorstores", model_name: str = "all-MiniLM-L6-v2") -> Dict:
    model = None
    try:
        model = _get_embedder(model_name)
    except Exception:
        model = None
    if model is not None:
        q_emb = model.encode([query_text], convert_to_numpy=True)
    else:
        q_emb = _compute_fallback_embeddings([query_text])
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    results = {}

    def _walk_targets():
        if team and project:
            yield team, project
            return
        if team:
            team_path = os.path.join(persist_dir, team)
            if os.path.isdir(team_path):
                for p in os.listdir(team_path):
                    ppath = os.path.join(team_path, p)
                    if os.path.isdir(ppath):
                        yield team, p
            return
        if project:
            for t in os.listdir(persist_dir):
                tpath = os.path.join(persist_dir, t)
                if os.path.isdir(tpath):
                    candidate = os.path.join(tpath, project)
                    if os.path.isdir(candidate):
                        yield t, project
            # fallback top-level project
            top = os.path.join(persist_dir, project)
            if os.path.isdir(top):
                yield None, project
            return
        # all
        for t in os.listdir(persist_dir):
            tpath = os.path.join(persist_dir, t)
            if os.path.isdir(tpath):
                for p in os.listdir(tpath):
                    ppath = os.path.join(tpath, p)
                    if os.path.isdir(ppath):
                        yield t, p
        # top-level ungrouped
        for p in os.listdir(persist_dir):
            ppath = os.path.join(persist_dir, p)
            if os.path.isdir(ppath) and not os.listdir(ppath):
                # top-level project folders with no team (legacy) are not expected in this structure; include if there
                yield None, p

    visited = set()
    for t, p in _walk_targets():
        key = (t or "ungrouped", p)
        if key in visited:
            continue
        visited.add(key)
        try:
            index, meta = load_index(p, t, persist_dir)
        except Exception:
            continue
        D, I = index.search(q_emb, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            item = meta[int(idx)].copy()
            item.update({"score": float(score)})
            hits.append(item)
        results[f"{t if t else 'ungrouped'}/{p}"] = hits
    return results
