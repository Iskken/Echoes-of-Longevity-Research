import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple, Union
import re, unicodedata, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import umap, hdbscan

# def analyze_bertopic(
#         df: pd.DataFrame,
#         v_min_df: Union[int, float] = 10,
#         v_max_df: Union[int, float] = 0.8,
#         text_col: str = "abstract",
#         embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-mpnet-base-v2",
#         embed_filename: str | None = None,
#         max_features: int = 8000,
#         top_n_words: int = 10,
#         topic_col_name: str = 'topic',
#         use_embedding: bool = False,
#         save_csv: str | None = None,
# ):
#     # 1) texts & mask
#     mask = df[text_col].fillna("").str.strip() != ""
#     docs = df.loc[mask, text_col].astype(str).tolist()
    
#     if use_embedding:
#         X = np.load(f"../ProQuest/Processed/{embed_filename}")
#     else:
#         # 2) embedding model
#         if isinstance(embedding_model, str):
#             embedding_model = SentenceTransformer(embedding_model)
    
#         # 3) compute embeddings ON THE SAME DOCS and save them
#         X = embedding_model.encode(
#             docs,
#             batch_size=128,
#             normalize_embeddings=True,
#             convert_to_numpy=True,
#             show_progress_bar=True,
#         ).astype("float32")
#         np.save(f"../ProQuest/Processed/embeddings_{embed_filename}", X)
#         # optionally also save the model itself:
#         # if (embed_filename):
#         #     embedding_model.save(f"../ProQuest/Processed/{embed_filename}_encoder")

#     # 4) vectorizer + UMAP + HDBSCAN (as you had)
#     vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2),
#                                  min_df=v_min_df, max_df=v_max_df,
#                                  max_features=max_features)
#     umap_model = umap.UMAP(n_neighbors=40, min_dist=0.0, metric="cosine", random_state=42)
#     hdb_model  = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=None,
#                                  metric="euclidean", prediction_data=False)

#     topic_model = BERTopic(embedding_model=None,   # we pass embeddings explicitly
#                            vectorizer_model=vectorizer,
#                            umap_model=umap_model,
#                            hdbscan_model=hdb_model,
#                            nr_topics="auto",
#                            top_n_words=top_n_words,
#                            calculate_probabilities=False,
#                            verbose=True)

#     # 5) USE the saved embeddings here
#     topics, probs = topic_model.fit_transform(docs, embeddings=X)

#     # 6) write back using the original mask (not `docs`)
#     df_out = df.copy()
#     df_out.loc[mask, topic_col_name] = topics
#     df_out.loc[mask, f"{topic_col_name}_prob"] = [float(max(p)) if len(p) else 0.0 for p in probs]
#     df_out.loc[mask, "topic_keywords"] = df_out.loc[mask, topic_col_name].apply(
#         lambda t: [] if int(t) == -1 else [w for w, _ in (topic_model.get_topic(int(t)) or [])]
#     )

#     topic_info = topic_model.get_topic_info()
#     if save_csv:
#         topic_info.to_csv(save_csv, index=False)
#     return topic_model, topic_info, df_out

import numpy as np
import pandas as pd
import torch                                  # CHANGED: needed to detect GPU
from typing import Tuple, Union
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
import hdbscan
from bertopic import BERTopic

from src.project_paths import PROQUEST_PROCESSED_DIR

def analyze_bertopic(
    df: pd.DataFrame,
    v_min_df: Union[int, float] = 1,
    v_max_df: Union[int, float] = 0.8,
    text_col: str = "abstract",
    embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-mpnet-base-v2",
    embed_filename: str | None = None,
    max_features: int = 8000,
    top_n_words: int = 10,
    topic_col_name: str = "topic",
    use_embedding: bool = False,
    save_csv: str | None = None,
    use_gpu: bool = True,                      # CHANGED: toggle GPU
    batch_size_gpu: int = 256,                 # CHANGED: larger batch on GPU
    batch_size_cpu: int = 16,                  # CHANGED
    calc_probs: bool = False,                  # CHANGED: speed/memory
) -> Tuple[BERTopic, pd.DataFrame, pd.DataFrame]:

    # 1) texts & mask
    mask = df[text_col].fillna("").str.strip() != ""
    docs = df.loc[mask, text_col].astype(str).tolist()

    if use_embedding:
        X = np.load(PROQUEST_PROCESSED_DIR / f"{embed_filename}.npy")
    else:
        # 2) embedding model  --------------------------- CHANGED (GPU here)
        if isinstance(embedding_model, str):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = SentenceTransformer(embedding_model, device=device)
        else:
            device = "cuda" if next(embedding_model.auto_model.parameters()).is_cuda else "cpu"
        
        batch_size = 16
        
        X = embedding_model.encode(
            docs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        np.save(PROQUEST_PROCESSED_DIR / f"{embed_filename}.npy", X)

    # 4) vectorizer + UMAP + HDBSCAN
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=v_min_df,
        max_df=v_max_df,
        max_features=max_features,
    )

    umap_model = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)

    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=None,                      # CHANGED: usually better default
        metric="euclidean",
        prediction_data=False,                 # CHANGED: faster/lower memory
    )

    topic_model = BERTopic(
        embedding_model=None,                  # passing embeddings explicitly
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        nr_topics=None,
        top_n_words=top_n_words,
        calculate_probabilities=calc_probs,    # CHANGED
        verbose=True,
    )

    # 5) use the saved embeddings here
    topics, probs = topic_model.fit_transform(docs, embeddings=X)

    # 6) write back using the original mask (not `docs`)
    df_out = df.copy()
    df_out.loc[mask, topic_col_name] = topics
    if calc_probs and probs is not None:
        df_out.loc[mask, f"{topic_col_name}_prob"] = np.max(probs, axis=1)
    else:
        df_out.loc[mask, f"{topic_col_name}_prob"] = np.nan

    df_out.loc[mask, "topic_keywords"] = df_out.loc[mask, topic_col_name].apply(
        lambda t: [] if int(t) == -1 else [w for w, _ in (topic_model.get_topic(int(t)) or [])]
    )

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
