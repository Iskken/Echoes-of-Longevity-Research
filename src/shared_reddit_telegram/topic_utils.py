"""
BERTopic utilities for the Echoes of Longevity project.

Canonical implementations extracted from:
- Echoes-of-Longevity-Research/News/notebooks/NLP.py (analyze_bertopic, full_keyword_getter)
- to_csv/scripts/topic_cleaning_and_analysis/subtopic_modelling.py (explore_subtopics)
- to_csv/scripts/topic-matching/*.py (load_similarity_matrix)
"""

import os
import pickle
from collections import defaultdict
from typing import Tuple, Union

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer


def analyze_bertopic(
    df: pd.DataFrame,
    text_col: str = "abstract",
    embedding_model: Union[str, SentenceTransformer] = "sentence-transformers/all-mpnet-base-v2",
    v_min_df: Union[int, float] = 1,
    v_max_df: Union[int, float] = 0.8,
    max_features: int = 5000,
    top_n_words: int = 10,
    topic_col_name: str = "topic",
    save_csv: str = None,
) -> Tuple[BERTopic, pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic on df[text_col] and attach results back to the DataFrame.

    Args:
        df: Input DataFrame with a text column.
        text_col: Column containing documents to model.
        embedding_model: SentenceTransformer model name or instance.
        v_min_df: CountVectorizer min_df (terms must appear in >= this many docs).
        v_max_df: CountVectorizer max_df (drop terms in >= this fraction of docs).
        max_features: Maximum vocabulary size.
        top_n_words: Number of words per topic.
        topic_col_name: Name of the output topic column.
        save_csv: If provided, save topic_info to this CSV path.

    Returns:
        Tuple of (topic_model, topic_info, df_with_topics).
    """
    mask = df[text_col].fillna("").str.strip() != ""
    docs = df.loc[mask, text_col].tolist()

    if isinstance(embedding_model, str):
        embedding_model = SentenceTransformer(embedding_model)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=v_min_df,
        max_df=v_max_df,
        max_features=max_features,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        nr_topics=None,
        top_n_words=top_n_words,
        calculate_probabilities=True,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(docs)

    # Attach results back to DataFrame
    df_out = df.copy()
    docs_mask = df[text_col].fillna("").str.strip() != ""

    df_out.loc[docs_mask, topic_col_name] = topics
    df_out.loc[docs_mask, f"{topic_col_name}_prob"] = [max(p) for p in probs]
    df_out.loc[docs_mask, "topic_keywords"] = df_out.loc[docs_mask, topic_col_name].apply(
        lambda t: [w for w, _ in topic_model.get_topic(int(t))]
    )

    topic_info = topic_model.get_topic_info()
    if save_csv:
        topic_info.to_csv(save_csv, index=False)

    return topic_model, topic_info, df_out


def full_keyword_getter(df: pd.DataFrame, topic_col: str = "topic", keywords_col: str = "topic_keywords"):
    """Print topic_id -> keyword list mapping from a DataFrame with topic assignments.

    Args:
        df: DataFrame with topic and topic_keywords columns.
        topic_col: Name of the topic ID column.
        keywords_col: Name of the keywords column.
    """
    df_map = df.dropna(subset=[keywords_col])
    df_map = df_map.drop_duplicates(subset=[topic_col])

    topic_to_keywords = (
        df_map.set_index(topic_col)[keywords_col].astype(object).to_dict()
    )

    for tid, kws in sorted(topic_to_keywords.items()):
        print(f"{tid} -> {kws}")


def explore_subtopics(
    topic_model_path: str,
    topic_doc_map_path: str,
    topic_id: int,
    save_dir: str,
    embedding_model_name: str = "all-mpnet-base-v2",
    n_subtopics: int = 10,
    top_n_words: int = 10,
    visualize: bool = False,
) -> BERTopic:
    """Generate subtopics for a given parent topic.

    Uses PCA + MiniBatchKMeans for sub-clustering (more stable for
    small topic-level corpora than UMAP + HDBSCAN).

    Args:
        topic_model_path: Path to saved BERTopic model.
        topic_doc_map_path: Path to topic-to-documents mapping pickle.
        topic_id: Parent topic ID to decompose.
        save_dir: Directory to save subtopic model and outputs.
        embedding_model_name: SentenceTransformer model name.
        n_subtopics: Target number of subtopics.
        top_n_words: Words per subtopic for visualization.
        visualize: If True, generate and open HTML visualizations.

    Returns:
        Fitted BERTopic subtopic model.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading model from {topic_model_path} ...")
    topic_model = BERTopic.load(topic_model_path)

    with open(topic_doc_map_path, "rb") as f:
        topic_doc_map = pickle.load(f)

    if topic_id not in topic_doc_map or len(topic_doc_map[topic_id]) == 0:
        raise ValueError(f"No documents found for topic {topic_id}")

    docs = topic_doc_map[topic_id]
    print(f"Found {len(docs)} documents for topic {topic_id}.")

    n_subtopics = max(2, min(n_subtopics, len(docs)))

    embedding_model = SentenceTransformer(embedding_model_name)
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    subtopic_model = BERTopic(
        embedding_model=None,
        umap_model=PCA(n_components=25),
        hdbscan_model=MiniBatchKMeans(n_clusters=n_subtopics, random_state=42),
        vectorizer_model=OnlineCountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            max_features=5000,
        ),
        calculate_probabilities=True,
        verbose=True,
    )

    subtopic_model.fit(docs, embeddings=embeddings)

    # Build subtopic -> docs mapping
    subtopics, probs = subtopic_model.transform(docs, embeddings=embeddings)
    max_probs = probs.max(axis=1) if probs is not None else np.full(len(docs), np.nan)

    subtopic_doc_map = defaultdict(list)
    for st_id, doc in zip(subtopics, docs):
        subtopic_doc_map[int(st_id)].append(doc)

    # Save mapping
    map_path = os.path.join(save_dir, f"subtopic_doc_map_topic{topic_id}.pkl")
    with open(map_path, "wb") as f:
        pickle.dump(subtopic_doc_map, f)

    # Save assignments CSV
    df_assign = pd.DataFrame({
        "parent_topic": topic_id,
        "subtopic": np.array(subtopics, dtype=int),
        "doc": docs,
        "max_prob": max_probs,
    })
    df_assign.to_csv(os.path.join(save_dir, f"subtopic_assignments_topic{topic_id}.csv"), index=False)

    # Save topic info CSV
    subtopic_info_df = subtopic_model.get_topic_info()
    subtopic_info_df["TopWords"] = subtopic_info_df["Topic"].apply(
        lambda t: ", ".join([w for w, _ in (subtopic_model.get_topic(t) or [])]) if t != -1 else ""
    )
    subtopic_info_df.to_csv(os.path.join(save_dir, f"subtopic_info_topic{topic_id}.csv"), index=False)

    # Save model
    subtopic_model.save(os.path.join(save_dir, f"subtopic_model_topic{topic_id}.model"))

    if visualize:
        import webbrowser
        fig_map = subtopic_model.visualize_topics()
        fig_map.write_html(os.path.join(save_dir, "topic_map.html"))
        fig_bar = subtopic_model.visualize_barchart(n_words=top_n_words)
        fig_bar.write_html(os.path.join(save_dir, "subtopic_barchart.html"))

    return subtopic_model


def load_similarity_matrix(path: str, validate_row_sums: bool = False) -> pd.DataFrame:
    """Load a topic similarity CSV with proper index/column coercion.

    Reads a CSV where rows and columns are topic IDs, coerces them to int,
    and moves topic -1 (outlier) to the last column if present.

    Args:
        path: Path to the similarity CSV file.
        validate_row_sums: If True, raise ValueError if any row sum exceeds 1.0.

    Returns:
        DataFrame with int-typed index and columns.
    """
    S = pd.read_csv(path, index_col=0)
    S.columns = [int(c) for c in S.columns]
    S.index = S.index.astype(int)

    # Move outlier topic -1 to last column
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    S = S[cols]

    if validate_row_sums:
        rs = S.sum(axis=1)
        if not (rs <= 1.0 + 1e-6).all():
            raise ValueError(f"Row sums exceed 1.0 in {path}.")

    return S
