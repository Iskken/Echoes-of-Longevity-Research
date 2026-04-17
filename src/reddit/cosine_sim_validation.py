# === Cosine-similarity validation for BTM pairs ===
import os, json
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score

# ---- config / paths (reuse your earlier OUTDIR etc.)
ROUND = 1
OUTDIR = f"reddit-telegram/round_{ROUND}"
os.makedirs(OUTDIR, exist_ok=True)

NEWS_MODEL_PATH = TELEGRAM_OUTPUT / "archive" / "take_2" / "topic_modeling/bertopic_base.model"
SOC_MODEL_PATH  = "../topic_modelling_v2/round_11/bertopic_no_embed"
SHARED_EMBEDDER = "all-mpnet-base-v2"     # both topic models were trained with this family

S_NEWS_PATH   = f"{OUTDIR}/BTM_S_news_vs_social.csv"  # rows=news, cols=social (+ -1)
S_SOCIAL_PATH = f"{OUTDIR}/BTM_S_social_vs_news.csv"  # rows=social, cols=news (+ -1)

# ---- 1) Load S matrices (reuse your helper) ----
def load_S(path: str) -> pd.DataFrame:
    S = pd.read_csv(path, index_col=0)
    S.columns = [int(c) for c in S.columns]
    S.index = S.index.astype(int)
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    S = S[cols]
    return S

S_news   = load_S(S_NEWS_PATH)
S_social = load_S(S_SOCIAL_PATH)

# BTM top-1 mapping from NEWS->SOCIAL (ignore outlier col)
non_out_cols_news = [c for c in S_news.columns if c != -1]
btm_top_social_for_news = S_news[non_out_cols_news].idxmax(axis=1).astype(int)

# Optionally restrict evaluation to rows with some non-outlier mass:
valid_news_rows = (S_news[non_out_cols_news].sum(axis=1) > 0)
btm_top_social_for_news = btm_top_social_for_news[valid_news_rows]

# ---- 2) Build topic text representations for each model ----
# We’ll embed *topic descriptors* in the shared SBERT space to get topic embeddings.
def topic_texts(model: BERTopic, topn: int = 10) -> pd.Series:
    # topic_id -> "w1, w2, ... wN"  (skip -1)
    out = {}
    for t in model.get_topics().keys():
        if t == -1:
            continue
        words = [w for w, _ in model.get_topic(t)[:topn]]
        out[t] = ", ".join(words) if words else f"topic_{t}"
    return pd.Series(out, dtype=str)

model_news   = BERTopic.load(NEWS_MODEL_PATH, embedding_model=SHARED_EMBEDDER)
model_social = BERTopic.load(SOC_MODEL_PATH,  embedding_model=None)

texts_news   = topic_texts(model_news,   topn=12)
texts_social = topic_texts(model_social, topn=12)

news_ids   = texts_news.index.to_list()
social_ids = texts_social.index.to_list()

# ---- 3) Encode topic texts in the shared space and L2-normalize ----
st = SentenceTransformer(SHARED_EMBEDDER)
E_news   = st.encode(texts_news.tolist(),   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
E_social = st.encode(texts_social.tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

# ---- 4) Cosine similarity matrix (NEWS x SOCIAL) ----
# Since embeddings are unit-norm, cos = dot(E_news, E_social^T)
COS = (E_news @ E_social.T).astype(np.float32)
cos_df = pd.DataFrame(COS, index=news_ids, columns=social_ids)
cos_df.to_csv(f"{OUTDIR}/cosine-validation/cosine_news_vs_social.csv")

# Cosine top-1 mapping NEWS->SOCIAL
cos_top_social_for_news = pd.Series(
    data=[social_ids[int(j)] for j in COS.argmax(axis=1)],
    index=news_ids,
    dtype=int
)

# Align both mappings to the same set of news topic ids (valid rows only)
cos_aligned = cos_top_social_for_news.loc[btm_top_social_for_news.index]

# ---- 5) Agreement metrics (Cohen’s κ and simple accuracy) ----
kappa = cohen_kappa_score(btm_top_social_for_news.to_numpy(), cos_aligned.to_numpy())
acc   = float((btm_top_social_for_news.to_numpy() == cos_aligned.to_numpy()).mean())

# Also report the cosine value at each BTM-chosen pair (diagnostic)
btm_pairs_cos = pd.Series(
    [cos_df.loc[i, j] if (i in cos_df.index and j in cos_df.columns) else np.nan
     for i, j in zip(btm_top_social_for_news.index, btm_top_social_for_news.values)],
    index=btm_top_social_for_news.index,
    name="cos_at_BTM_pair"
)
btm_pairs_cos.to_csv(f"{OUTDIR}/cosine-validation/cosine_at_BTM_pairs_per_news_topic.csv", header=True)

# Save a tidy comparison table
cmp = pd.DataFrame({
    "btm_match": btm_top_social_for_news,
    "cos_match": cos_aligned,
    "agree": (btm_top_social_for_news == cos_aligned)
})
# Add both scores for context
cmp["btm_S_max"] = S_news[non_out_cols_news].max(axis=1).loc[cmp.index]
cmp["cos_max"]   = cos_df.loc[cmp.index][cos_df.columns].max(axis=1).values
cmp.to_csv(f"{OUTDIR}/cosine-validation/btm_vs_cosine_news_side.csv", index_label="news_topic")

with open(f"{OUTDIR}/cosine-validation/btm_vs_cosine_agreement.json", "w") as f:
    json.dump({"cohen_kappa": float(kappa), "percent_agreement": acc}, f, indent=2)

print(f"[Validation] Cohen's kappa (news-side mapping): {kappa:.3f} | agreement: {acc:.3f}")

# ---- 6) (Optional) Do the reverse angle SOCIAL->NEWS as well ----
non_out_cols_soc = [c for c in S_social.columns if c != -1]
btm_top_news_for_social = S_social[non_out_cols_soc].idxmax(axis=1).astype(int)
valid_social_rows = (S_social[non_out_cols_soc].sum(axis=1) > 0)
btm_top_news_for_social = btm_top_news_for_social[valid_social_rows]

# Cosine top-1 SOCIAL->NEWS
COS_T = COS.T  # SOCIAL x NEWS
cos_top_news_for_social = pd.Series(
    data=[news_ids[int(i)] for i in COS_T.argmax(axis=1)],
    index=social_ids,
    dtype=int
)
cos_aligned_rev = cos_top_news_for_social.loc[btm_top_news_for_social.index]
kappa_rev = cohen_kappa_score(btm_top_news_for_social.to_numpy(), cos_aligned_rev.to_numpy())
acc_rev   = float((btm_top_news_for_social.to_numpy() == cos_aligned_rev.to_numpy()).mean())
with open(f"{OUTDIR}/cosine-validation/btm_vs_cosine_agreement_social_side.json", "w") as f:
    json.dump({"cohen_kappa": float(kappa_rev), "percent_agreement": acc_rev}, f, indent=2)
print(f"[Validation] Cohen's kappa (social-side mapping): {kappa_rev:.3f} | agreement: {acc_rev:.3f}")