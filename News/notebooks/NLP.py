import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Union
import re, unicodedata, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

def analyze_bertopic(
        df: pd.DataFrame,
        v_min_df: int = 2,
        v_max_df: Union[int, float] = 0.6,
        text_col: str = "abstract",
        embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-mpnet-base-v2",
        max_features: int = 5000,
        top_n_words: int = 10,
        save_csv: str | None = None,
) -> Tuple[BERTopic, pd.DataFrame]:
    """
    Fit BERTopic on `df[text_col]`.

    Returns
    -------
    topic_model : fitted BERTopic object
    topic_info  : DataFrame with topic frequencies & keywords
    """

    # Only non-empty texts must be modeled
    mask = df[text_col].fillna("").str.strip() != ""    # True where text exists
    docs = df.loc[mask, text_col].tolist()              # only non‑empty docs

    # ---- 2. Embedding & vectorizer ----
    if isinstance(embedding_model, str):
        embedding_model = SentenceTransformer(embedding_model)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=v_min_df, #keep terms that appear in ≥ v_min_df docs
        max_df=v_max_df, #drops terms seen in ≥ v_max_df*100% of docs.
        max_features=max_features, #cap vocabulary size, larger = richer topics
    )

    # ---- 3. Fit BERTopic ----
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        nr_topics="auto", #let's bertopic optionally merge topics
        top_n_words=top_n_words, #controls how many words you store per topic
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs)

    
    # ---- 4. Attach results back to original DataFrame ----
    df_out = df.copy()
    docs = df[text_col].fillna("").str.strip() != ""

    df_out.loc[docs, "topic"]      = topics
    df_out.loc[docs, "topic_prob"] = [max(p) for p in probs]

    
    df_out.loc[docs, "topic_keywords"] = df_out.loc[docs, "topic"].apply(
        lambda t: [w for w, _ in topic_model.get_topic(int(t))]
    )

    
    # ---- 5. Optional: save topic summary ----
    topic_info = topic_model.get_topic_info()
    if save_csv:
        topic_info.to_csv(save_csv, index=False)

    return topic_model, topic_info, df_out


def basic_clean(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"http\S+|www\.\S+", " ", t)   # URLs
    t = re.sub(r"\d+", " ", t)                # digits
    return re.sub(r"\s+", " ", t).lower().strip()

def top_tfidf_terms(texts, top_k=50, min_df=3, max_df=0.8,
                    ngram_range=(1,3), extra_stop=None):
    docs = [basic_clean(x) for x in texts if isinstance(x, str) and x.strip()]
    stop = set(ENGLISH_STOP_WORDS) | set(extra_stop or [])
    vec = TfidfVectorizer(stop_words=list(stop),
                          ngram_range=ngram_range,
                          min_df=min_df, max_df=max_df,
                          token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
                          sublinear_tf=True)
    X = vec.fit_transform(docs)
    scores = X.max(axis=0).toarray().ravel()          # importance per term
    terms  = np.array(vec.get_feature_names_out())
    idx = scores.argsort()[-top_k:][::-1]
    return pd.DataFrame({"term": terms[idx], "tfidf": scores[idx]})
