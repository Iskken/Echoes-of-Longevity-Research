# topic_pair_postprocess.py (non-CLI, importable)
import os, pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from bertopic import BERTopic

# ----------------- Specs -----------------
@dataclass
class CorpusSpec:
    name: str
    model_path: str
    embedding_model_on_load: Optional[str] = None   # HF id or local dir; or None
    topics_csv: Optional[str] = None
    native_labels_path: Optional[str] = None        # npy/npz/csv/parquet
    docs_path: Optional[str] = None                 # pkl(list[str]) or csv
    docs_text_column: Optional[str] = None

@dataclass
class RunSpec:
    outdir: str
    s_A_vs_B_path: str          # rows=A topics, cols=B topics (+ -1)
    s_B_vs_A_path: str
    min_topic_size: int = 40
    thresh_A_to_B: float = 0.0  # TA
    thresh_B_to_A: float = 0.0  # TB
    topk: int = 3
    n_pairs_to_sample: int = 10
    n_docs_per_topic_sample: int = 10

# ----------------- I/O helpers -----------------
def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def load_S(path: str) -> pd.DataFrame:
    S = pd.read_csv(path, index_col=0)
    S.columns = [int(c) for c in S.columns]
    S.index = S.index.astype(int)
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    S = S[cols]
    rs = S.sum(axis=1)
    if not (rs <= 1.0 + 1e-6).all():
        raise ValueError(f"Row sums exceed 1.0 in {path}.")
    return S

def load_topic_ids(path: str, n_docs: Optional[int] = None) -> np.ndarray:
    lower = path.lower()
    if lower.endswith((".npy", ".npz")):
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr).ravel().astype(np.int32)
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
        if "topic" in df.columns:
            s = df["topic"]
        else:
            num_cols = [c for c in df.columns
                        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])]
            s = df[num_cols[0]] if num_cols else df.iloc[:, -1]
        return s.to_numpy(dtype=np.int32)
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
        cols = {c.lower(): c for c in df.columns}
        doc_col   = cols.get("doc_idx") or cols.get("doc_id") or cols.get("index")
        topic_col = cols.get("topic_id") or cols.get("topic")  or cols.get("label")
        if doc_col is None or topic_col is None:
            raise ValueError(f"{path} needs doc_idx/doc_id and topic_id/topic.")
        df = df[[doc_col, topic_col]].dropna().sort_values(doc_col)
        if n_docs is None:
            return df[topic_col].to_numpy(dtype=np.int32)
        out = np.full((n_docs,), -1, dtype=np.int32)
        idx = df[doc_col].astype(int).to_numpy()
        if idx.min() < 0 or idx.max() >= n_docs:
            raise ValueError(f"doc_idx out of range 0..{n_docs-1} in {path}")
        out[idx] = df[topic_col].astype(np.int32).to_numpy()
        return out
    raise ValueError(f"Unsupported topic-id file type: {path}")

def load_docs(path: str, text_col: Optional[str] = None) -> List[str]:
    lower = path.lower()
    if lower.endswith((".pkl", ".pickle")):
        docs = pickle.load(open(path, "rb"))
        docs = [str(x).strip() for x in docs if isinstance(x, str) or pd.notna(x)]
        return [d for d in docs if d]
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
        if text_col and text_col in df.columns:
            col = text_col
        elif "text" in df.columns:
            col = "text"
        else:
            obj_cols = [c for c in df.columns if df[c].dtype == "object"]
            col = obj_cols[0] if obj_cols else df.columns[0]
        s = df[col].astype(str).str.strip()
        s = s[s.notna() & (s != "")]
        return s.tolist()
    raise ValueError(f"Unsupported docs file type: {path}")

def load_model_for_meta(path: str, embedding_model_on_load: Optional[str]) -> BERTopic:
    return BERTopic.load(path, embedding_model=embedding_model_on_load)

# ----------------- Logic helpers (unchanged math) -----------------
import numpy as np
import pandas as pd

import re
from typing import Set, Tuple, Iterable

def unified_topic_centroids(
    docs: List[str],
    native_labels: np.ndarray,
    embedder_id: str,
    per_topic: int = 50,
    device: Optional[str] = None
) -> dict[int, np.ndarray]:
    """
    Compute per-topic centroids in a single embedding space.
    Uses up to 'per_topic' randomly sampled docs per topic.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    st = SentenceTransformer(embedder_id, device=device)
    try:
        st.max_seq_length = 256
    except Exception:
        pass

    rng = np.random.default_rng(0)
    centroids = {}

    unique_topics = sorted(set(int(t) for t in native_labels.tolist()))
    for t in unique_topics:
        if t == -1:
            continue
        idx = np.where(native_labels == t)[0]
        if len(idx) == 0:
            continue
        take = idx[rng.choice(len(idx), size=min(per_topic, len(idx)), replace=False)]
        batch_texts = [docs[i] for i in take]
        E = st.encode(batch_texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        if E.size == 0:
            continue
        c = E.mean(axis=0)
        n = np.linalg.norm(c)
        centroids[t] = (c / max(n, 1e-12)).astype(np.float32)
    return centroids

def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def row_max_excl_minus1(S: pd.DataFrame) -> pd.Series:
    non_out = [c for c in S.columns if c != -1]
    if not non_out:
        return pd.Series(0.0, index=S.index)
    return S[non_out].max(axis=1)

# Simple tokenizer + small stopword set (extend if you like)
_BASIC_STOP = {
    "the","a","an","and","or","for","to","of","in","on","with","by","from",
    "at","as","is","are","be","was","were","this","that","these","those",
    "it","its","we","you","they","their","our","your"
}

_token_re = re.compile(r"[A-Za-z0-9+#]+")  # keeps tokens like B12, mTOR, CRISPR

def _normalize_tokens(text: str) -> Set[str]:
    return {t.lower() for t in _token_re.findall(text) if t and t.lower() not in _BASIC_STOP}

def topic_word_sets(model: BERTopic, k: int = 15) -> dict[int, Set[str]]:
    out = {}
    for t in model.get_topics().keys():
        if t == -1:
            continue
        words = [w for w, _ in model.get_topic(t)[:k]]
        toks = set()
        for w in words:
            toks |= _normalize_tokens(w)
        out[int(t)] = toks
    return out

def jaccard(A: Set[str], B: Set[str]) -> float:
    if not A or not B:
        return 0.0
    i = len(A & B)
    u = len(A | B)
    return i / u

def common_terms(A: Set[str], B: Set[str], k: int = 10) -> str:
    # top-k by alphabetical just to keep deterministic; adjust if you want frequency-weighted
    return ", ".join(sorted(A & B)[:k])

def cos(a: np.ndarray, b: np.ndarray) -> float:
    # assumes a,b are unit length; if not, this still works as cosine similarity
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def _topk_membership(S: pd.DataFrame, k: int) -> dict[int, set[int]]:
    """For each row i, return the set of top-k column ids among non-(-1) columns."""
    non_out = [c for c in S.columns if c != -1]
    out: dict[int, set[int]] = {}
    if not non_out:
        return out
    # Ensure numeric ids
    non_out = [int(c) for c in non_out]
    # Work on a copy with int columns so idxmax/sort are consistent
    S2 = S.copy()
    S2.columns = [int(c) for c in S2.columns]
    for i, row in S2[non_out].iterrows():
        i = int(i)
        # Handle all-NaN or empty rows gracefully
        if row.notna().any():
            top_js = row.sort_values(ascending=False).head(k).index.astype(int).tolist()
            out[i] = set(top_js)
        else:
            out[i] = set()
    return out

def _bidirectional_score(S_A: pd.DataFrame, S_B: pd.DataFrame, i: int, j: int) -> float:
    """Geometric mean of the two directions; returns 0.0 if either side missing."""
    if (i not in S_A.index) or (j not in S_A.columns): return 0.0
    if (j not in S_B.index) or (i not in S_B.columns): return 0.0
    return float(np.sqrt(float(S_A.loc[i, j]) * float(S_B.loc[j, i])))

def soft_pairs_one_to_one(
    S_A: pd.DataFrame, S_B: pd.DataFrame,
    TA: float = 0.12, TB: float = 0.12,
    gmin: float = 0.18,
    require_topk: bool = True, topk: int = 3,
    # optional extra filters
    jaccard_min: float = 0.0,
    jaccard_lookup: Optional[Dict[Tuple[int,int], float]] = None,
    cos_min: float = 0.0,
    cos_lookup: Optional[Dict[Tuple[int,int], float]] = None,
    # scoring weights
    w_g: float = 0.6, w_c: float = 0.3, w_j: float = 0.1,
    # optional per-topic size filters
    min_size_A: Optional[dict[int, int]] = None,
    min_size_B: Optional[dict[int, int]] = None,
    min_topic_size: int = 0
) -> list[tuple[int, int, float, float, float, float, float]]:
    """
    One-to-one matching with multi-signal gating.
    Returns (i, j, S_ij, S_ji, gmean, cos_sim, jacc).
    """
    from scipy.optimize import linear_sum_assignment as hungarian

    # Normalize ids
    S_A = S_A.copy()
    S_B = S_B.copy()
    S_A.index = S_A.index.astype(int); S_A.columns = [int(c) for c in S_A.columns]
    S_B.index = S_B.index.astype(int); S_B.columns = [int(c) for c in S_B.columns]

    A_topk = _topk_membership(S_A, topk) if require_topk else {}
    B_topk = _topk_membership(S_B, topk) if require_topk else {}

    A_ids = [int(i) for i in S_A.index.tolist() if i != -1]
    B_ids = [int(c) for c in S_A.columns if c != -1]

    candidates = []
    for i in A_ids:
        for j in B_ids:
            S_ij = float(S_A.loc[i, j])
            S_ji = float(S_B.loc[j, i]) if (j in S_B.index and i in S_B.columns) else 0.0
            if S_ij < TA or S_ji < TB:
                continue
            g = float(np.sqrt(S_ij * S_ji))
            if g < gmin:
                continue
            if require_topk:
                in_top = (j in A_topk.get(i, set())) or (i in B_topk.get(j, set()))
                if not in_top:
                    continue
            if min_size_A is not None and min_size_A.get(i, min_topic_size) < min_topic_size:
                continue
            if min_size_B is not None and min_size_B.get(j, min_topic_size) < min_topic_size:
                continue

            # optional lexical / embedding gates
            jc = jaccard_lookup.get((i,j), 0.0) if jaccard_lookup else 0.0
            cs = cos_lookup.get((i,j), 0.0)     if cos_lookup     else 0.0
            if jc < jaccard_min:
                continue
            if cs < cos_min:
                continue

            # score for Hungarian
            score = w_g * g + w_c * cs + w_j * jc
            candidates.append((i, j, S_ij, S_ji, g, cs, jc, score))

    if not candidates:
        return []

    A_list = sorted({i for (i, *_ ) in candidates})
    B_list = sorted({j for (_, j, *_ ) in candidates})
    a_index = {i: idx for idx, i in enumerate(A_list)}
    b_index = {j: idx for idx, j in enumerate(B_list)}

    cost = np.full((len(A_list), len(B_list)), 1e3, dtype=np.float64)
    best: Dict[Tuple[int,int], Tuple[float,float,float,float,float]] = {}

    for i, j, S_ij, S_ji, g, cs, jc, score in candidates:
        key = (i, j)
        # keep max score per (i,j)
        if (key not in best) or (score > best[key][-1]):
            best[key] = (S_ij, S_ji, g, cs, jc, score)

    for (i, j), (S_ij, S_ji, g, cs, jc, score) in best.items():
        cost[a_index[i], b_index[j]] = -score

    row_ind, col_ind = hungarian(cost)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e3:
            continue
        i, j = A_list[r], B_list[c]
        S_ij, S_ji, g, cs, jc, score = best[(i, j)]
        pairs.append((i, j, float(S_ij), float(S_ji), float(g), float(cs), float(jc)))

    pairs.sort(key=lambda x: (0.999*x[4] + 0.7*x[5] + 0.2*x[6]), reverse=True)  # g, cos, jacc
    return pairs

def top1_map(S: pd.DataFrame) -> pd.Series:
    non_out = [c for c in S.columns if c != -1]
    if not non_out: return pd.Series(dtype=int)
    return S[non_out].idxmax(axis=1).astype(int)

def build_mutual_pairs(S_A: pd.DataFrame, S_B: pd.DataFrame):
    A2B, B2A = top1_map(S_A), top1_map(S_B)
    return [(int(i), int(j)) for i, j in A2B.items() if j in B2A.index and B2A.loc[j] == i]

def pairs_df(pairs, S_A, S_B, a_label="A", b_label="B") -> pd.DataFrame:
    rows = []
    for i, j in pairs:
        s_ij = float(S_A.loc[i, j]) if (i in S_A.index and j in S_A.columns) else 0.0
        s_ji = float(S_B.loc[j, i]) if (j in S_B.index and i in S_B.columns) else 0.0
        g    = (s_ij * s_ji) ** 0.5
        rows.append({a_label: i, b_label: j, "S_ij": s_ij, "S_ji": s_ji, "gmean": g})
    return pd.DataFrame(rows).sort_values("gmean", ascending=False)

def topk_long(S: pd.DataFrame, k=3, native_label="row", cross_label="col", val_label="S") -> pd.DataFrame:
    non_out = [c for c in S.columns if c != -1]
    out = []
    for i, row in S[non_out].iterrows():
        s = row.sort_values(ascending=False).head(k)
        for j, val in s.items():
            out.append((int(i), int(j), float(val)))
    return pd.DataFrame(out, columns=[native_label, cross_label, val_label])

def topic_topwords(model: BERTopic, k=10) -> pd.Series:
    out = {}
    for t in model.get_topics().keys():
        if t == -1: continue
        words = [w for w, _ in model.get_topic(t)[:k]]
        out[t] = ", ".join(words)
    return pd.Series(out)

def load_topic_meta(model: BERTopic, topics_csv_path: Optional[str]):
    if topics_csv_path and os.path.exists(topics_csv_path):
        df = pd.read_csv(topics_csv_path)
    else:
        df = model.get_topic_info()
    df = df.set_index("Topic")
    names = df["Name"]
    counts = df["Count"] if "Count" in df.columns else None
    return names, counts

def sample_docs(docs: List[str], native_labels: np.ndarray, topic_id: int, k=10, seed=0) -> List[str]:
    idx = np.where(native_labels == topic_id)[0]
    if len(idx) == 0: return []
    rng = np.random.default_rng(seed)
    take = idx[rng.choice(len(idx), size=min(k, len(idx)), replace=False)]
    return [docs[i] for i in take]

# ----------------- Main entrypoint -----------------
def run_topic_pair_postprocess(a: CorpusSpec, b: CorpusSpec, run: RunSpec) -> Dict[str, object]:
    """Run post-processing; returns key DataFrames for inspection in IDE."""
    ensure_dir(run.outdir)
    tp_dir = os.path.join(run.outdir, "topic-pairs")
    ensure_dir(tp_dir)

    # Load S matrices
    S_A = load_S(run.s_A_vs_B_path)  # rows=a, cols=b (+ -1)
    S_B = load_S(run.s_B_vs_A_path)  # rows=b, cols=a (+ -1)

    S_A.index = S_A.index.astype(int)
    S_A.columns = [int(c) for c in S_A.columns]
    S_B.index = S_B.index.astype(int)
    S_B.columns = [int(c) for c in S_B.columns]

    # Mutual top-1 pairs
    pairs = build_mutual_pairs(S_A, S_B)
    pairs_df_ab = pairs_df(pairs, S_A, S_B, a_label=a.name, b_label=b.name)

    # Load models for names/topwords (no transforms)
    model_a = load_model_for_meta(a.model_path, a.embedding_model_on_load)
    model_b = load_model_for_meta(b.model_path, b.embedding_model_on_load)

    names_a, counts_a = load_topic_meta(model_a, a.topics_csv)
    names_b, counts_b = load_topic_meta(model_b, b.topics_csv)

    # Add names/sizes
    pairs_df_ab[f"{a.name}_name"]   = pairs_df_ab[a.name].map(names_a)
    pairs_df_ab[f"{b.name}_name"]   = pairs_df_ab[b.name].map(names_b)
    pairs_df_ab[f"n_docs_{a.name}"] = pairs_df_ab[a.name].map(counts_a) if counts_a is not None else np.nan
    pairs_df_ab[f"n_docs_{b.name}"] = pairs_df_ab[b.name].map(counts_b) if counts_b is not None else np.nan
    pairs_df_ab.to_csv(os.path.join(tp_dir, "pair_quality_mutual_top1_with_meta.csv"), index=False)

    # --- Soft, high-quality 1-1 pairs (recommended) ---
    # Pull per-topic sizes if available (for optional min size filtering)
    # sizes
    info_a = model_a.get_topic_info().set_index("Topic")
    info_b = model_b.get_topic_info().set_index("Topic")
    sizes_a = {int(k): int(v) for k, v in info_a["Count"].dropna().items()} if "Count" in info_a else {}
    sizes_b = {int(k): int(v) for k, v in info_b["Count"].dropna().items()} if "Count" in info_b else {}

    # peakedness (row-wise max of S excluding -1); use to drop diffuse topics
    SA = row_max_excl_minus1(S_A)
    SB = row_max_excl_minus1(S_B)
    # example: keep topics with SA/SB >= 0.25 (tune)
    keep_A = set(SA[SA >= 0.25].index.astype(int).tolist())
    keep_B = set(SB[SB >= 0.25].index.astype(int).tolist())
    S_A = S_A.loc[S_A.index.isin(keep_A), [c for c in S_A.columns if (c == -1 or c in keep_B)]]
    S_B = S_B.loc[S_B.index.isin(keep_B), [c for c in S_B.columns if (c == -1 or c in keep_A)]]

    # lexical overlap lookup
    twset_a = topic_word_sets(model_a, k=15)
    twset_b = topic_word_sets(model_b, k=15)

    jacc_lookup: Dict[Tuple[int, int], float] = {}
    for i in twset_a:
        for j in twset_b:
            jacc_lookup[(int(i), int(j))] = jaccard(twset_a[i], twset_b[j])

    cos_lookup: Dict[Tuple[int, int], float] = {}  # safe default

    try:
        native_a = load_topic_ids(a.native_labels_path)
        native_b = load_topic_ids(b.native_labels_path)
        docs_a = load_docs(a.docs_path, text_col=a.docs_text_column)
        docs_b = load_docs(b.docs_path, text_col=b.docs_text_column)

        cent_a = unified_topic_centroids(docs_a, native_a, "all-mpnet-base-v2", per_topic=50, device=None)
        cent_b = unified_topic_centroids(docs_b, native_b, "all-mpnet-base-v2", per_topic=50, device=None)

        for i, ca in cent_a.items():
            for j, cb in cent_b.items():
                cos_lookup[(int(i), int(j))] = cos(ca, cb)
    except Exception as e:
        print(f"[WARN] Unified centroids not computed: {e}")

    soft = soft_pairs_one_to_one(
        S_A, S_B,
        TA=0.12, TB=0.12, gmin=0.18,
        require_topk=True, topk=3,
        jaccard_min=0.08,  # require modest word overlap
        jaccard_lookup=jacc_lookup,
        cos_min=0.35 if cos_lookup else 0.0,  # only enforce if available
        cos_lookup=cos_lookup,
        w_g=0.6, w_c=0.3, w_j=0.1,
        min_size_A=sizes_a, min_size_B=sizes_b,
        min_topic_size=max(30, 0)  # tune if needed
    )

    soft_df = pd.DataFrame(soft, columns=[a.name, b.name, "S_ij", "S_ji", "gmean", "cos_unified", "jaccard"])
    soft_df[f"{a.name}_name"] = soft_df[a.name].map(info_a["Name"])
    soft_df[f"{b.name}_name"] = soft_df[b.name].map(info_b["Name"])
    soft_df[f"{a.name}_words"] = soft_df[a.name].map(lambda t: ", ".join(sorted(twset_a.get(int(t), set()))))
    soft_df[f"{b.name}_words"] = soft_df[b.name].map(lambda t: ", ".join(sorted(twset_b.get(int(t), set()))))
    soft_df["common_terms"] = [
        common_terms(
            twset_a.get(int(row[a.name]), set()),
            twset_b.get(int(row[b.name]), set()),
            k=12
        )
        for _, row in soft_df.iterrows()
    ]

    soft_df = soft_df.sort_values(["gmean", "cos_unified", "jaccard"], ascending=False)
    soft_df.to_csv(os.path.join(tp_dir, "pairs_soft_one_to_one_multi_signal.csv"), index=False)

    # Stable shortlist
    stable = pairs_df_ab.copy()
    if f"n_docs_{a.name}" in stable.columns and f"n_docs_{b.name}" in stable.columns:
        stable = stable[
            (stable.S_ij >= run.thresh_A_to_B) &
            (stable.S_ji >= run.thresh_B_to_A) &
            (stable[f"n_docs_{a.name}"] >= run.min_topic_size) &
            (stable[f"n_docs_{b.name}"] >= run.min_topic_size)
        ]
    else:
        stable = stable[(stable.S_ij >= run.thresh_A_to_B) & (stable.S_ji >= run.thresh_B_to_A)]
    stable.to_csv(os.path.join(tp_dir, "stable_pairs_mutual_top1.csv"), index=False)

    # Top-k bidirectional
    topk_a = topk_long(S_A, k=run.topk, native_label=a.name, cross_label=b.name, val_label="S_ij")
    topk_b = topk_long(S_B, k=run.topk, native_label=b.name, cross_label=a.name, val_label="S_ji")
    topk_merged = topk_a.merge(topk_b, on=[a.name, b.name], how="outer").fillna(0.0)
    topk_merged["gmean"] = np.sqrt(topk_merged.S_ij * topk_merged.S_ji)
    topk_merged = topk_merged.sort_values("gmean", ascending=False)
    topk_merged.to_csv(os.path.join(tp_dir, "topk_pairs_bidir.csv"), index=False)

    # Descriptors
    tw_a = topic_topwords(model_a, k=10)
    tw_b = topic_topwords(model_b, k=10)
    def add_descriptors(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[f"{a.name}_words"] = out[a.name].map(tw_a)
        out[f"{b.name}_words"] = out[b.name].map(tw_b)
        out[f"{a.name}_name"]  = out[a.name].map(names_a)
        out[f"{b.name}_name"]  = out[b.name].map(names_b)
        return out
    stable_desc = add_descriptors(stable)
    stable_desc.to_csv(os.path.join(tp_dir, "stable_pairs_readable.csv"), index=False)

    # Qualitative samples (optional)
    if run.n_pairs_to_sample > 0 and a.native_labels_path and b.native_labels_path and a.docs_path and b.docs_path:
        native_a = load_topic_ids(a.native_labels_path)
        native_b = load_topic_ids(b.native_labels_path)
        docs_a   = load_docs(a.docs_path, text_col=a.docs_text_column)
        docs_b   = load_docs(b.docs_path, text_col=b.docs_text_column)
        to_sample = min(run.n_pairs_to_sample, len(stable_desc))
        for _, row in stable_desc.head(to_sample).iterrows():
            ia, ib = int(row[a.name]), int(row[b.name])
            dA = sample_docs(docs_a, native_a, ia, k=run.n_docs_per_topic_sample, seed=0)
            dB = sample_docs(docs_b, native_b, ib, k=run.n_docs_per_topic_sample, seed=0)
            with open(os.path.join(tp_dir, f"pair_{ia}_{ib}_samples.txt"), "w") as f:
                f.write(f"[{a.name.upper()} {ia}] {names_a.get(ia,'')}\n{tw_a.get(ia,'')}\n\n")
                f.write("\n--- samples ---\n" + "\n\n".join(dA))
                f.write(f"\n\n[{b.name.upper()} {ib}] {names_b.get(ib,'')}\n{tw_b.get(ib,'')}\n\n")
                f.write("\n--- samples ---\n" + "\n\n".join(dB))

    return {
        "S_A": S_A, "S_B": S_B,
        "pairs": pairs_df_ab,
        "stable": stable,
        "topk_pairs": topk_merged,
        "stable_desc": stable_desc,
    }

A = CorpusSpec(
    name="youtube",
    model_path=REDDIT_MODELS / "round_11" / "mpnet_topic_model_no_umap",
    topics_csv=REDDIT_MODELS / "round_11" / "mpnet_topics.csv",
    native_labels_path=REDDIT_MODELS / "round_11" / "text_topic.csv",
    docs_path=REDDIT_MODELS / "round_11" / "preprocessed_data.pkl",
)

B = CorpusSpec(
    name="reddit",
    model_path="../topic_modelling_v2/round_10/bertopic_no_embed",
    topics_csv="../topic_modelling_v2/round_10/topics.csv",
    native_labels_path="../topic_modelling_v2/round_10/train_topics_unique.npy",
    docs_path="../topic_modelling_v2/round_4/unique_docs.pkl",
)

R = RunSpec(
    outdir="reddit-youtube/round_2",
    s_A_vs_B_path="reddit-youtube/round_2/BTM_S_youtube_vs_reddit.csv",
    s_B_vs_A_path="reddit-youtube/round_2/BTM_S_reddit_vs_youtube.csv",
    min_topic_size=0, thresh_A_to_B=0.0, thresh_B_to_A=0.0, topk=3,
    n_pairs_to_sample=10, n_docs_per_topic_sample=10,
)

artifacts = run_topic_pair_postprocess(A, B, R)
# Inspect in IDE:
pairs = artifacts["pairs"]
stable = artifacts["stable"]