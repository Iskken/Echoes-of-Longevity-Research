# cross_assign_btm.py
# Portable cross-assignment (BTM-style) between two BERTopic models
# - Prefers doc→topic similarity (topic embeddings; else c-TF-IDF)
# - Falls back automatically; avoids UMAP/HDBSCAN pitfalls
# - Reuses precomputed native embeddings if provided

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# =========================
# Small utilities
# =========================
def _as_float32(E):
    E = np.asarray(E)
    if E.dtype != np.float32:
        E = E.astype(np.float32, copy=False)
    return E

def _unit_norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)

def is_unit_norm(E, tol=1e-3):
    norms = np.linalg.norm(_as_float32(E), axis=1)
    return abs(np.median(norms) - 1.0) <= tol

def ensure_unit_norm(E):
    E = _as_float32(E)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.clip(n, 1e-12, None)

def detect_training_policy_from_file(path, sample=2000):
    E = np.load(path, mmap_mode="r")
    n = min(sample, len(E))
    return is_unit_norm(E[:n])

def _has_umap_and_cluster(model) -> bool:
    has_umap = getattr(model, "umap_model", None) is not None
    has_hdb  = getattr(model, "hdbscan_model", None) is not None or getattr(model, "cluster_model", None) is not None
    return has_umap and has_hdb


# =========================
# Topic order helpers (robust to version quirks)
# =========================
def _topics_order_for_ctfidf(model):
    """
    Return a list of topic IDs whose order matches rows of model.c_tf_idf_.
    Tries the internal mapper first; falls back to heuristics.
    """
    tm = getattr(model, "topic_mapper_", None)
    if tm is not None:
        topic_ids = getattr(tm, "topic_ids", None)
        if topic_ids is None and hasattr(tm, "__dict__"):
            topic_ids = tm.__dict__.get("topic_ids", None)
        if topic_ids:
            return [t for t in topic_ids if t != -1]
        for attr in ("map_index_to_id", "index_to_id", "_index_to_id"):
            map_it = getattr(tm, attr, None)
            if isinstance(map_it, (list, tuple)):
                return [t for t in map_it if t != -1]

    topics_dict = getattr(model, "get_topics", lambda: {})()
    if isinstance(topics_dict, dict) and topics_dict:
        cand = sorted([t for t in topics_dict.keys() if t != -1])
        ctf = getattr(model, "c_tf_idf_", None)
        if ctf is not None and len(cand) == ctf.shape[0]:
            return cand

    info = model.get_topic_info()
    return info.loc[info["Topic"] != -1, "Topic"].tolist()

def _topics_order_for_topic_embeddings(model):
    """
    Return topic IDs aligned with rows of model.topic_embeddings_.
    Prefer topic_info() without -1; validate length and fall back to get_topics order.
    """
    Te = getattr(model, "topic_embeddings_", None)
    if Te is None:
        raise RuntimeError("topic_embeddings_ not available")
    k = len(Te)

    info = model.get_topic_info()
    topic_ids = info.loc[info["Topic"] != -1, "Topic"].tolist()
    if len(topic_ids) == k:
        return topic_ids

    topics_dict = getattr(model, "get_topics", lambda: {})()
    if isinstance(topics_dict, dict) and topics_dict:
        cand = sorted([t for t in topics_dict.keys() if t != -1])
        if len(cand) == k:
            return cand

    return list(range(k))  # synthetic ids aligned to rows


# =========================
# Encoders / similarity assigners
# =========================
def _batched_encode(embedder_id_or_dir, docs, want_unit_norm=True, batch_size=256, device=None):
    st = SentenceTransformer(embedder_id_or_dir, device=device)
    # modest seq length for throughput; adjust if you want maximum quality
    try:
        st.max_seq_length = 256
    except Exception:
        pass
    n = len(docs)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        E = st.encode(
            docs[i:j],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=want_unit_norm,
            show_progress_bar=False,
        ).astype(np.float32)
        if want_unit_norm:
            E = _unit_norm_rows(E)
        yield i, j, E

def _cosine_argmax(E_docs: np.ndarray, E_topics: np.ndarray) -> np.ndarray:
    S = E_docs @ E_topics.T  # cosine for unit-norm inputs
    return np.argmax(S, axis=1).astype(np.int32)

def _assign_via_topic_embeddings(
    model, docs, embedder, want_unit_norm=True, batch_size=256, device=None,
    E_docs: Optional[np.ndarray] = None
):
    Te = getattr(model, "topic_embeddings_", None)
    if Te is None:
        raise RuntimeError("topic_embeddings_ not available on model")

    Te = _unit_norm_rows(_as_float32(Te))
    topic_ids = _topics_order_for_topic_embeddings(model)
    if len(topic_ids) != Te.shape[0] and topic_ids != list(range(Te.shape[0])):
        raise RuntimeError(f"topic_embeddings_ rows ({Te.shape[0]}) != topic_ids ({len(topic_ids)})")

    if E_docs is not None:
        E_docs = _unit_norm_rows(_as_float32(E_docs)) if want_unit_norm else _as_float32(E_docs)
        idx = _cosine_argmax(E_docs, Te)
        return np.asarray([topic_ids[k] for k in idx], dtype=np.int32)

    out = np.empty(len(docs), dtype=np.int32)
    for i, j, E in _batched_encode(embedder, docs, want_unit_norm, batch_size, device):
        idx = _cosine_argmax(E, Te)
        out[i:j] = np.asarray([topic_ids[k] for k in idx], dtype=np.int32)
    return out

def _assign_via_ctfidf(model, docs, batch_size=4096):
    vec = model.vectorizer_model
    Ct = model.c_tf_idf_
    if Ct is None or vec is None:
        raise RuntimeError("Model lacks c_tf_idf_ or vectorizer_model.")
    if not sparse.issparse(Ct):
        Ct = sparse.csr_matrix(Ct)

    topics_order = _topics_order_for_ctfidf(model)
    n_rows = Ct.shape[0]
    if len(topics_order) != n_rows:
        print(f"[WARN] c_tf_idf_ rows ({n_rows}) != derived topic ids ({len(topics_order)}); using indices 0..{n_rows-1}")
        topics_order = list(range(n_rows))

    CtT = Ct.T  # (vocab, n_rows)
    n = len(docs)
    out = np.empty(n, dtype=np.int32)

    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        X = vec.transform(docs[i:j])               # (m, vocab)
        S = X @ CtT                                # (m, n_rows)
        if sparse.issparse(S):
            S = S.toarray()

        empty = (X.getnnz(axis=1) == 0) if sparse.issparse(X) else (X.sum(axis=1) == 0)
        best_idx = np.argmax(S, axis=1).astype(np.int32)
        mapped = np.take(np.asarray(topics_order, dtype=np.int32), best_idx, mode="clip")
        if np.ndim(empty) > 1: empty = np.asarray(empty).ravel()
        mapped = np.where(empty, -1, mapped)
        out[i:j] = mapped
    return out

def _safe_assign(
    model, docs, embedder, want_unit_norm=True, batch_size=256, device=None,
    E_docs: Optional[np.ndarray] = None, prefer_similarity: bool = True
):
    """
    Unified assignment:
      - If prefer_similarity or model lacks UMAP+HDBSCAN -> similarity path
      - Else try native .transform(embeddings=...), then fallbacks
    """
    if prefer_similarity or not _has_umap_and_cluster(model):
        try:
            return _assign_via_topic_embeddings(model, docs, embedder, want_unit_norm, batch_size, device, E_docs)
        except Exception as e_te:
            print(f"[INFO] topic_embeddings_ fallback unavailable: {e_te} -> c-TF-IDF")
            return _assign_via_ctfidf(model, docs, batch_size=max(2048, batch_size))

    # Native path
    if E_docs is not None:
        try:
            E_docs = _unit_norm_rows(_as_float32(E_docs)) if want_unit_norm else _as_float32(E_docs)
            out = np.empty(len(docs), dtype=np.int32)
            bs = batch_size
            for i in range(0, len(docs), bs):
                j = min(i + bs, len(docs))
                topics_chunk, _ = model.transform(docs[i:j], embeddings=E_docs[i:j])
                out[i:j] = np.asarray(topics_chunk, dtype=np.int32)
            return out
        except Exception as e:
            print(f"[INFO] .transform(precomputed E) failed: {e} -> falling back")

    try:
        out = np.empty(len(docs), dtype=np.int32)
        for i, j, E in _batched_encode(embedder, docs, want_unit_norm, batch_size, device):
            topics_chunk, _ = model.transform(docs[i:j], embeddings=E)
            out[i:j] = np.asarray(topics_chunk, dtype=np.int32)
        return out
    except Exception as e:
        print(f"[INFO] .transform() failed: {e} -> falling back to similarity")

    try:
        return _assign_via_topic_embeddings(model, docs, embedder, want_unit_norm, batch_size, device, E_docs)
    except Exception as e_te:
        print(f"[INFO] topic_embeddings_ fallback unavailable: {e_te} -> c-TF-IDF")
    return _assign_via_ctfidf(model, docs, batch_size=max(2048, batch_size))


# =========================
# Pairing + metrics
# =========================
def pairing_strength(native_topics: np.ndarray,
                     cross_topics: np.ndarray) -> Tuple[pd.DataFrame, Counter]:
    assert native_topics.shape == cross_topics.shape
    counts_native = Counter(native_topics)
    pair_counts = Counter(zip(native_topics, cross_topics))

    native_ids = sorted(counts_native.keys())
    cross_ids  = sorted(set(cross_topics.tolist()))

    rows = []
    for ti in native_ids:
        denom = counts_native[ti]
        row = {tj: pair_counts.get((ti, tj), 0) / denom for tj in cross_ids}
        rows.append(pd.Series(row, name=ti))

    S = pd.DataFrame(rows).fillna(0.0)
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    return S[cols], counts_native

def pairing_strength_weighted(native_topics, cross_topics, weights=None):
    if weights is None:
        return pairing_strength(native_topics, cross_topics)
    counts_native, pair_counts = {}, {}
    for ti, tj, w in zip(native_topics, cross_topics, weights):
        counts_native[ti] = counts_native.get(ti, 0.0) + float(w)
        pair_counts[(ti, tj)] = pair_counts.get((ti, tj), 0.0) + float(w)
    native_ids = sorted(counts_native.keys())
    cross_ids  = sorted({tj for (_, tj) in pair_counts.keys()})
    rows = []
    for ti in native_ids:
        denom = counts_native[ti]
        row = {tj: pair_counts.get((ti, tj), 0.0) / denom for tj in cross_ids}
        rows.append(pd.Series(row, name=ti))
    S = pd.DataFrame(rows).fillna(0.0)
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    return S[cols], Counter({k:int(v) for k,v in counts_native.items()})

def corpus_metrics(S: pd.DataFrame, counts_native: Counter) -> Dict[str, Union[float, pd.Series]]:
    T = S.shape[0]
    weights = np.array([counts_native[i] for i in S.index])

    non_out = [c for c in S.columns if c != -1]
    out_col = -1 if -1 in S.columns else None

    C  = S[non_out].to_numpy().sum() / T
    Cw = (weights[:, None] * S[non_out].to_numpy()).sum() / weights.sum()

    U_series = S[out_col] if out_col is not None else pd.Series(0.0, index=S.index)
    U  = U_series.sum() / T
    Uw = (weights * U_series.to_numpy()).sum() / weights.sum()

    SA = S[non_out].max(axis=1)
    A  = SA.mean()
    Aw = (weights * SA.to_numpy()).sum() / weights.sum()

    return {
        "C": C, "Cw": Cw, "theta": Cw - C,
        "U": U, "Uw": Uw,
        "A": A, "Aw": Aw, "Aw_minus_A": Aw - A,
        "SA": SA
    }

def adjusted_metrics(S, counts_native):
    non_out = [c for c in S.columns if c != -1]
    out = S[-1] if -1 in S.columns else pd.Series(0.0, index=S.index)
    w = np.array([counts_native[i] for i in S.index])
    denom = (1.0 - out).replace(0, 1.0)
    Snorm = S[non_out].div(denom, axis=0).fillna(0.0)
    Cp  = Snorm.to_numpy().sum() / S.shape[0]
    Cpw = (w[:, None] * Snorm.to_numpy()).sum() / w.sum()
    Ap  = Snorm.max(axis=1).mean()
    Apw = (w * Snorm.max(axis=1).to_numpy()).sum() / w.sum()
    return {"C'": Cp, "Cw'": Cpw, "A'": Ap, "Aw'": Apw}

def top_matches(S, k=3):
    out = []
    for ti, row in S.drop(columns=[-1], errors="ignore").iterrows():
        out.append((ti, row.sort_values(ascending=False).head(k)))
    rows = []
    for ti, s in out:
        for tj, val in s.items():
            rows.append({"topic_native": ti, "topic_cross": tj, "S": float(val)})
    return pd.DataFrame(rows)


# =========================
# Specs + I/O
# =========================
@dataclass
class CorpusSpec:
    """One corpus/model config."""
    name: str
    model_path: str
    embedder: Optional[str]
    docs_path: str
    docs_text_column: Optional[str] = None
    # Optional assets
    train_embeddings_path: Optional[str] = None     # for unit-norm detection only
    native_topics_path: Optional[str] = None        # per-doc topic ids to skip native assignment
    embeddings_path: Optional[str] = None           # precomputed *native* embeddings (same order as docs)
    embedding_model_on_load: Optional[str] = None   # forwarded to BERTopic.load(...)

@dataclass
class RunSpec:
    output_dir: str
    batch_size: int = 256
    device: Optional[str] = None
    top_k_matches: int = 3
    make_output_dir: bool = True
    sample_n: Optional[int] = None
    prefer_similarity: bool = True   # default to BTM-style portable assignment

def load_docs(path: str, text_col: Optional[str] = None, sample_n: Optional[int] = None):
    lower = path.lower()
    if lower.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            docs = pickle.load(f)
        docs = [str(x).strip() for x in docs if isinstance(x, str) or pd.notna(x)]
        docs = [d for d in docs if d]
        return docs[:sample_n] if sample_n else docs

    if lower.endswith(".csv"):
        df = pd.read_csv(path)
        col = text_col if (text_col and text_col in df.columns) else (
            "text" if "text" in df.columns else
            next((c for c in df.columns if df[c].dtype == "object"), df.columns[0])
        )
        s = df[col].astype(str).str.strip()
        s = s[s.notna() & (s != "")]
        docs = s.tolist()
        return docs[:sample_n] if sample_n else docs

    raise ValueError(f"Unsupported docs file type: {path}")

def load_embeddings(path: Optional[str], n_docs: Optional[int], unit_norm: bool) -> Optional[np.ndarray]:
    if not path:
        return None
    E = np.load(path)
    if n_docs is not None and E.shape[0] != n_docs:
        raise ValueError(f"Embeddings rows ({E.shape[0]}) != n_docs ({n_docs}) for {path}")
    E = _as_float32(E)
    return _unit_norm_rows(E) if unit_norm else E

def load_model(spec: CorpusSpec) -> BERTopic:
    return BERTopic.load(spec.model_path, embedding_model=spec.embedding_model_on_load)

def load_topic_ids(path: str, n_docs: Optional[int] = None) -> np.ndarray:
    lower = path.lower()

    if lower.endswith((".npy", ".npz")):
        arr = np.load(path)
        arr = np.ravel(arr).astype(np.int32)
        return arr[:n_docs] if n_docs is not None else arr

    if lower.endswith(".csv"):
        df = pd.read_csv(path)
        if "topic" in df.columns:
            s = df["topic"]
        else:
            num_cols = [c for c in df.columns
                        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])]
            s = df[num_cols[0]] if num_cols else df.iloc[:, -1]
        arr = s.to_numpy(dtype=np.int32)
        return arr[:n_docs] if n_docs is not None else arr

    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
        cols = {c.lower(): c for c in df.columns}
        doc_col   = cols.get("doc_idx") or cols.get("doc_id") or cols.get("index")
        topic_col = cols.get("topic_id") or cols.get("topic") or cols.get("label")
        if doc_col is None or topic_col is None:
            raise ValueError(f"Parquet at {path} must contain 'doc_idx' and 'topic_id' (or compatible) columns; found: {list(df.columns)}")
        df = df[[doc_col, topic_col]].dropna().sort_values(doc_col)
        if n_docs is None:
            return df[topic_col].to_numpy(dtype=np.int32)
        topics = np.full((n_docs,), -1, dtype=np.int32)
        idx = df[doc_col].astype(int).to_numpy()
        mask = (idx >= 0) & (idx < n_docs)
        topics[idx[mask]] = df[topic_col].astype(np.int32).to_numpy()[mask]
        return topics

    raise ValueError(f"Unsupported topics file type: {path}")

def _assert_doc_label_match(docs, native_topics_path: Optional[str], label_name: str):
    if native_topics_path:
        arr = load_topic_ids(native_topics_path, n_docs=len(docs))
        if len(docs) != arr.shape[0]:
            raise AssertionError(
                f"{label_name} docs vs labels mismatch ({len(docs)} vs {arr.shape[0]}). "
                f"Ensure per-DOC labels and matching order."
            )


# =========================
# Cross-assign orchestration (BTM)
# =========================
def cross_assign_bt(
    model_A, docs_A, embedder_A, train_emb_path_A=None, native_topics_A_path=None, embeddings_A_path=None,
    model_B=None, docs_B=None, embedder_B=None, train_emb_path_B=None, native_topics_B_path=None, embeddings_B_path=None,
    batch_size=256, device=None, prefer_similarity=True
):
    """
    Returns (A_native, A_cross, B_native, B_cross), all int32 ndarrays.
    """
    want_unit_A = True if train_emb_path_A is None else detect_training_policy_from_file(train_emb_path_A)
    want_unit_B = True if train_emb_path_B is None else detect_training_policy_from_file(train_emb_path_B)

    # Native A
    if native_topics_A_path:
        print(f"[INFO] Using provided native topics for A from {native_topics_A_path}")
        A_native = load_topic_ids(native_topics_A_path, n_docs=len(docs_A))
    else:
        E_A_native = load_embeddings(embeddings_A_path, n_docs=len(docs_A), unit_norm=want_unit_A)
        print(f"[INFO] Assigning native A via {'similarity' if prefer_similarity or not _has_umap_and_cluster(model_A) else 'transform'}")
        A_native = _safe_assign(model_A, docs_A, embedder_A, want_unit_A, batch_size, device,
                                E_docs=E_A_native, prefer_similarity=prefer_similarity)

    # Native B
    if native_topics_B_path:
        print(f"[INFO] Using provided native topics for B from {native_topics_B_path}")
        B_native = load_topic_ids(native_topics_B_path, n_docs=len(docs_B))
    else:
        E_B_native = load_embeddings(embeddings_B_path, n_docs=len(docs_B), unit_norm=want_unit_B)
        print(f"[INFO] Assigning native B via {'similarity' if prefer_similarity or not _has_umap_and_cluster(model_B) else 'transform'}")
        B_native = _safe_assign(model_B, docs_B, embedder_B, want_unit_B, batch_size, device,
                                E_docs=E_B_native, prefer_similarity=prefer_similarity)

    # Cross A→B
    print(f"[INFO] Cross-assign A→B via {'similarity' if prefer_similarity or not _has_umap_and_cluster(model_B) else 'transform'}")
    A_cross = _safe_assign(model_B, docs_A, embedder_B, want_unit_B, batch_size, device,
                           E_docs=None, prefer_similarity=prefer_similarity)

    # Cross B→A
    print(f"[INFO] Cross-assign B→A via {'similarity' if prefer_similarity or not _has_umap_and_cluster(model_A) else 'transform'}")
    B_cross = _safe_assign(model_A, docs_B, embedder_A, want_unit_A, batch_size, device,
                           E_docs=None, prefer_similarity=prefer_similarity)

    return (
        np.asarray(A_native, dtype=np.int32),
        np.asarray(A_cross,  dtype=np.int32),
        np.asarray(B_native, dtype=np.int32),
        np.asarray(B_cross,  dtype=np.int32),
    )

def run_cross_assignment(corpus_A: CorpusSpec, corpus_B: CorpusSpec, run: RunSpec):
    if run.make_output_dir:
        os.makedirs(run.output_dir, exist_ok=True)

    # Load models
    model_A = load_model(corpus_A)  # keep as-is; no pickled shims needed for similarity route
    model_B = load_model(corpus_B)

    # Load docs
    docs_A = load_docs(corpus_A.docs_path, text_col=corpus_A.docs_text_column, sample_n=run.sample_n)
    docs_B = load_docs(corpus_B.docs_path, text_col=corpus_B.docs_text_column, sample_n=run.sample_n)
    print(f"[INFO] Loaded {len(docs_A)} docs for {corpus_A.name}, {len(docs_B)} for {corpus_B.name}")

    # Optional checks if native topics provided
    _assert_doc_label_match(docs_A, corpus_A.native_topics_path, corpus_A.name)
    _assert_doc_label_match(docs_B, corpus_B.native_topics_path, corpus_B.name)

    # Cross-assign
    A_native, A_cross, B_native, B_cross = cross_assign_bt(
        model_A=model_A, docs_A=docs_A, embedder_A=corpus_A.embedder,
        train_emb_path_A=corpus_A.train_embeddings_path, native_topics_A_path=corpus_A.native_topics_path,
        embeddings_A_path=corpus_A.embeddings_path,
        model_B=model_B, docs_B=docs_B, embedder_B=corpus_B.embedder,
        train_emb_path_B=corpus_B.train_embeddings_path, native_topics_B_path=corpus_B.native_topics_path,
        embeddings_B_path=corpus_B.embeddings_path,
        batch_size=run.batch_size, device=run.device, prefer_similarity=run.prefer_similarity
    )

    # S matrices + metrics
    S_A, counts_A = pairing_strength(A_native, A_cross)   # rows=A topics, cols=B topics (+ maybe -1)
    S_B, counts_B = pairing_strength(B_native, B_cross)

    metrics_A = corpus_metrics(S_A, counts_A)
    metrics_B = corpus_metrics(S_B, counts_B)

    # Persist
    S_A.to_csv(os.path.join(run.output_dir, f"BTM_S_{corpus_A.name}_vs_{corpus_B.name}.csv"))
    S_B.to_csv(os.path.join(run.output_dir, f"BTM_S_{corpus_B.name}_vs_{corpus_A.name}.csv"))

    pd.Series(metrics_A).drop(labels=["SA"]).to_csv(os.path.join(run.output_dir, f"BTM_metrics_{corpus_A.name}.csv"))
    pd.Series(metrics_B).drop(labels=["SA"]).to_csv(os.path.join(run.output_dir, f"BTM_metrics_{corpus_B.name}.csv"))

    sa_A = metrics_A["SA"].rename("SA").to_frame()
    names_A = model_A.get_topic_info().set_index("Topic")["Name"]
    sa_A["topic_name"] = sa_A.index.map(names_A)
    sa_A["n_docs"] = sa_A.index.map(lambda t: counts_A.get(t, 0))
    sa_A.sort_values("SA", ascending=True).to_csv(os.path.join(run.output_dir, f"BTM_SA_{corpus_A.name}_by_topic.csv"), index_label="topic")

    sa_B = metrics_B["SA"].rename("SA").to_frame()
    names_B = model_B.get_topic_info().set_index("Topic")["Name"]
    sa_B["topic_name"] = sa_B.index.map(names_B)
    sa_B["n_docs"] = sa_B.index.map(lambda t: counts_B.get(t, 0))
    sa_B.sort_values("SA", ascending=True).to_csv(os.path.join(run.output_dir, f"BTM_SA_{corpus_B.name}_by_topic.csv"), index_label="topic")

    pd.Series(adjusted_metrics(S_A, counts_A)).to_csv(os.path.join(run.output_dir, f"BTM_metrics_{corpus_A.name}_adjusted.csv"))
    pd.Series(adjusted_metrics(S_B, counts_B)).to_csv(os.path.join(run.output_dir, f"BTM_metrics_{corpus_B.name}_adjusted.csv"))

    topA = top_matches(S_A, k=run.top_k_matches)
    topA.to_csv(os.path.join(run.output_dir, f"top_matches_{corpus_A.name}.csv"), index=False)
    topB = top_matches(S_B, k=run.top_k_matches)
    topB.to_csv(os.path.join(run.output_dir, f"top_matches_{corpus_B.name}.csv"), index=False)

    print(f"[OK] Saved outputs under: {run.output_dir}")

    return {
        "S_A": S_A, "S_B": S_B,
        "counts_A": counts_A, "counts_B": counts_B,
        "metrics_A": metrics_A, "metrics_B": metrics_B,
        "A_native": A_native, "A_cross": A_cross,
        "B_native": B_native, "B_cross": B_cross,
    }


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Fill these with YOUR paths
    cross_assign_round = 2
    out_dir = f"reddit-youtube/round_{cross_assign_round}"

    youtube = CorpusSpec(
        name="youtube",
        model_path=REDDIT_MODELS / "round_11" / "mpnet_topic_model_no_umap",
        embedder="all-mpnet-base-v2",
        docs_path=REDDIT_MODELS / "round_11" / "preprocessed_data.pkl",
        train_embeddings_path=REDDIT_MODELS / "round_11" / "embeddings.npy",
        native_topics_path=REDDIT_MODELS / "round_11" / "text_topic.csv",
        embeddings_path=REDDIT_MODELS / "round_11" / "embeddings.npy",
        embedding_model_on_load=None
    )

    reddit = CorpusSpec(
        name="reddit",
        model_path="../topic_modelling_v2/round_10/bertopic_no_embed",
        embedder="../topic_modelling_v2/round_10/embedder",
        docs_path="../topic_modelling_v2/round_4/unique_docs.pkl",
        train_embeddings_path="../topic_modelling_v2/round_10/embeddings_fp32_l2.npy",
        native_topics_path="../topic_modelling_v2/round_10/train_topics_unique.npy",
        embeddings_path="../topic_modelling_v2/round_10/embeddings_fp32_l2.npy",
        embedding_model_on_load=None
    )

    run = RunSpec(
        output_dir=out_dir,
        batch_size=256,
        device=None,                 # "cuda:0" / "mps" / None
        top_k_matches=3,
        make_output_dir=True,
        sample_n=None,               # set e.g. 1000 for a quick test
        prefer_similarity=True       # BTM-style portable assignment (recommended)
    )

    _ = run_cross_assignment(youtube, reddit, run)