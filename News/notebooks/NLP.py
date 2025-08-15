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
        v_min_df: Union[int, float] = 1,
        v_max_df: Union[int, float] = 0.8,
        text_col: str = "abstract",
        embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-mpnet-base-v2",
        max_features: int = 5000,
        top_n_words: int = 10,
        topic_col_name: str = 'topic',
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

    df_out.loc[docs, topic_col_name]      = topics
    df_out.loc[docs, f"{topic_col_name}_prob"] = [max(p) for p in probs]

    
    df_out.loc[docs, "topic_keywords"] = df_out.loc[docs, topic_col_name].apply(
        lambda t: [w for w, _ in topic_model.get_topic(int(t))]
    )

    
    # ---- 5. Optional: save topic summary ----
    topic_info = topic_model.get_topic_info()
    if save_csv:
        topic_info.to_csv(save_csv, index=False)

    return topic_model, topic_info, df_out

def full_keyword_getter(df):
    # 1. Only keep the rows where we actually assigned keywords
    df_map = df.dropna(subset=["topic_keywords"])
    
    # 2. Drop duplicates so each topic ID appears exactly once
    df_map = df_map.drop_duplicates(subset=["topic"])
    
    # 3. Build a mapping: topic_id → keyword list
    topic_to_keywords = (
        df_map.set_index("topic")["topic_keywords"]
              .astype(object)               # ensure lists aren’t cast to strings
              .to_dict()
    )
    
    # 4. Print them in ID order
    for tid, kws in sorted(topic_to_keywords.items()):
        print(f"{tid} → {kws}")


def clean_tags(df: pd.DataFrame, new_keywords) -> pd.DataFrame:
    keywords = [
        "aging", "ageing", "longevity",
        "healthy aging", "healthy ageing",
        "anti-aging", "anti ageing",
        "living longer", 
        "ageing well", "well ageing", "aging well", "well aging"
    ]
    keywords.extend(new_keywords)

    escaped_keywords = [
        r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
        for kw in keywords
    ]

    pattern = "|".join(escaped_keywords)

    mask = df['tags'].str.contains(pattern, case=False, na=False, regex=True)
    tag_clean_df = df[mask].reset_index(drop=True)

    print(f"Kept {len(tag_clean_df)} of {len(df)} articles related to ageing.")
    return tag_clean_df