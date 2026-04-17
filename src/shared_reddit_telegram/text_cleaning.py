"""
Text cleaning and preprocessing utilities for the Echoes of Longevity project.

Canonical implementations extracted from:
- to_csv/scripts/topic_modelling_v2/topic_modelling_on_smaller_corpus.py (v2, refined)
- Telegram-Data-Colllection/hybrid/topic-modeling.py

The v2 clean_text preserves numbers and supplement names (B12, NAD+, 5-HTP)
which is critical for biohacking/longevity discourse.
"""

import re
import logging
import pickle

import emoji
import numpy as np
import pandas as pd
import spacy
from langdetect import detect, DetectorFactory
from tqdm import tqdm

# Deterministic language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


def is_english(text, min_len=20):
    """Check if text is English using langdetect.

    Args:
        text: Input string.
        min_len: Minimum character length to attempt detection.
                 Shorter texts are unreliable for language detection.

    Returns:
        True if detected as English, False otherwise.
    """
    if not text or len(text) < min_len:
        return False
    try:
        return detect(text) == "en"
    except Exception:
        return False


def clean_text(text, strip_numbers=False, min_len=5):
    """Clean a single document string.

    Removes URLs, emojis, Reddit-specific syntax (u/user, r/subreddit,
    blockquotes), and special characters. Lowercases the result.

    Args:
        text: Raw input string.
        strip_numbers: If True, removes all digits and keeps only letters
                       (v1 behavior, suitable for PubMed abstracts).
                       If False (default), preserves digits and characters
                       like hyphens, plus, hash, apostrophe (v2 behavior,
                       preserves supplement names like B12, NAD+, 5-HTP).
        min_len: Minimum character length; shorter texts return None.

    Returns:
        Cleaned lowercase string, or None if input is too short/empty.
    """
    if not isinstance(text, str) or len(text) < min_len:
        return None

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove Reddit-specific syntax
    text = re.sub(r"\b[ru]/\w+\b", " ", text)  # r/subreddit, u/username
    text = re.sub(r"(^|\n)>\s.*", " ", text, flags=re.M)  # blockquotes

    # Character filtering
    if strip_numbers:
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
    else:
        text = re.sub(r"[^\w\-\+/#']", " ", text)

    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()

    return text if text else None


def process_docs(df, title_col="title", text_col="selftext"):
    """Concatenate title + text fields into document strings.

    Args:
        df: DataFrame with text columns.
        title_col: Name of the title column.
        text_col: Name of the body text column.

    Returns:
        List of concatenated strings (title + " " + text).
    """
    for col in (title_col, text_col):
        if col not in df.columns:
            df[col] = ""

    return (
        df[title_col].fillna("").astype(str)
        + " "
        + df[text_col].fillna("").astype(str)
    ).tolist()


def lemmatize_texts(texts, spacy_model="en_core_web_sm", batch_size=1000):
    """Lemmatize texts using spaCy with parser/NER disabled.

    Removes stopwords, punctuation, whitespace tokens, and tokens
    without any alphanumeric characters.

    Args:
        texts: List of cleaned text strings.
        spacy_model: Name of the spaCy model to use.
        batch_size: Batch size for spaCy's nlp.pipe().

    Returns:
        Tuple of (lemmatized_texts, kept_indices) where kept_indices
        are the indices into the input list of texts that produced
        non-empty lemmatized output.
    """
    nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    lemmatized = []
    kept_indices = []

    for i, doc in enumerate(
        tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="Lemmatizing")
    ):
        tokens = []
        for t in doc:
            if t.is_space or t.is_punct or t.is_stop:
                continue
            txt = t.text.lower()
            if not re.search(r"[a-z0-9]", txt):
                continue
            tokens.append(txt)

        if tokens:
            lemmatized.append(" ".join(tokens))
            kept_indices.append(i)

    logger.info(f"Lemmatization complete. Kept {len(lemmatized)}/{len(texts)} docs.")
    return lemmatized, kept_indices


def dedupe_strings(strings):
    """Deduplicate a list of strings while preserving order.

    Returns:
        Tuple of (unique_strings, map_orig_to_unique, groups) where:
        - unique_strings: deduplicated list
        - map_orig_to_unique: numpy array mapping original index -> unique index
        - groups: list of lists, groups[j] = original indices that map to unique[j]
    """
    first = {}
    unique, map_o2u = [], []
    for s in strings:
        j = first.get(s)
        if j is None:
            j = len(unique)
            first[s] = j
            unique.append(s)
        map_o2u.append(j)

    groups = [[] for _ in range(len(unique))]
    for i, j in enumerate(map_o2u):
        groups[j].append(i)

    return unique, np.asarray(map_o2u, np.int32), groups


def preprocess_pipeline(
    texts,
    strip_numbers=False,
    min_tokens=3,
    spacy_model="en_core_web_sm",
    save_path=None,
):
    """Full preprocessing pipeline: clean -> English filter -> min-length -> lemmatize -> deduplicate.

    Args:
        texts: List of raw text strings.
        strip_numbers: Passed to clean_text().
        min_tokens: Minimum word count after cleaning.
        spacy_model: spaCy model for lemmatization.
        save_path: If provided, save preprocessed docs as pickle to this path.

    Returns:
        Tuple of (preprocessed_docs, kept_indices, unique_docs, map_orig_to_unique).
    """
    logger.info(f"Starting preprocessing of {len(texts)} texts...")

    # Step 1: Clean
    cleaned = []
    clean_indices = []
    for i, text in enumerate(tqdm(texts, desc="Cleaning")):
        result = clean_text(text, strip_numbers=strip_numbers)
        if result and len(result.split()) > min_tokens:
            cleaned.append(result)
            clean_indices.append(i)

    logger.info(f"Retained {len(cleaned)}/{len(texts)} texts after cleaning.")

    # Step 2: Lemmatize
    lemmatized, lem_kept = lemmatize_texts(cleaned, spacy_model=spacy_model)

    # Map back to original indices
    kept_indices = [clean_indices[j] for j in lem_kept]

    # Step 3: Deduplicate
    unique_docs, map_o2u, _groups = dedupe_strings(lemmatized)
    logger.info(f"Unique docs: {len(unique_docs)} (from {len(lemmatized)} after dedup).")

    # Optional save
    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(lemmatized, f)
        logger.info(f"Saved preprocessed docs to {save_path}")

    return lemmatized, kept_indices, unique_docs, map_o2u
