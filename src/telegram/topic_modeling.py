import json
import numpy as np
import hdbscan
import pandas as pd
import umap
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import logging
import re
import emoji
from langdetect import detect
import spacy
import sys
from tqdm import tqdm
import os
import pickle
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def clean_text(text):
    if not isinstance(text, str) or len(text) < 5:
        return None

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"u/\w+|r/\w+|>\s.*", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()

    if not is_english(text):
        return None

    return text.strip()

def preprocess_texts(texts):
    logger.info(f"Starting preprocessing of {len(texts)} texts...")

    # Clean texts with progress bar
    cleaned = []
    for text in tqdm(texts, desc="Cleaning"):
        result = clean_text(text)
        if result and len(result.split()) > 3:
            cleaned.append(result)

    logger.info(f"Retained {len(cleaned)} texts after cleaning.")

    # Lemmatize with spaCy using progress-aware batching
    lemmatized = []
    logger.info("Starting lemmatization...")
    for doc in tqdm(nlp.pipe(cleaned, batch_size=1000), total=len(cleaned), desc="Lemmatizing"):
        tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
        if tokens:
            lemmatized.append(" ".join(tokens))

    logger.info("Lemmatization complete.")
    return lemmatized

def process_docs(chunk):
    chunk = chunk['text'].fillna('').astype(str)
    texts = chunk.tolist()
    return preprocess_texts(texts)


def preprocess_and_save_documents(
        input_path,
        output_path,
        chunk_size=100_000,
        sample_size=None
):
    """
    Preprocess all documents from a JSONL file and save them to a file.

    Parameters:
    -----------
    input_path : str
        Path to the input JSONL file
    output_path : str
        Path to save the preprocessed documents
    chunk_size : int, optional
        Number of documents to process at once
    sample_size : int, optional
        If provided, only process this many documents
    """

    logger.info(f"Starting document preprocessing from {input_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize counters
    total_processed = 0
    total_retained = 0
    all_processed_docs = []

    # Process in chunks
    with open(input_path, 'r') as f:
        messages = json.load(f)
    messages_df = pd.DataFrame(messages)

    # Process the chunk
    processed_docs = process_docs(messages_df)

    # Update counts
    total_processed += len(messages_df)
    total_retained += len(processed_docs)

    # Add to our collection
    all_processed_docs.extend(processed_docs)


    # Save final results
    logger.info(f"Saving {len(all_processed_docs)} preprocessed documents to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(all_processed_docs, f)

    # Report statistics
    logger.info(f"Preprocessing complete. Total documents processed: {total_processed}")
    logger.info(f"Documents retained after preprocessing: {total_retained} ({total_retained / total_processed:.2%})")

    return all_processed_docs

# all_processed_docs = preprocess_and_save_documents(
#     input_path="../take_2/messages.json",
#     output_path="../take_2/topic_modeling/preprocessed-docs.pkl",
# )

with open("../take_2/topic_modeling/preprocessed-docs.pkl", "rb") as f:
    all_processed_docs = pickle.load(f)

# ---- UMAP (dimensionality reduction) ----
umap_model = umap.UMAP(
    n_neighbors=20,
    n_components=10,
    metric="cosine",
    random_state=42,
    low_memory=True,
    verbose=True
)

# ---- HDBSCAN (density clustering) ----
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=80,             # n_docs * 0.0020 (0.20%)
    min_samples=None,                 # ties to min_cluster_size (robust)
    metric="euclidean",               # operate in UMAP space
    cluster_selection_method="eom",   # stable, avoids tiny leaflets
    approx_min_span_tree=True,
    prediction_data=True
)

vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,  # Lowered from 10 to 2 for compatibility
    max_df=0.8,  # Keep as is, or set to 1.0 for all terms
    max_features=60_000,
)

# ---- 3. Fit BERTopic ----
topic_model = BERTopic(
    embedding_model="all-mpnet-base-v2",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    nr_topics="auto",
    top_n_words=10,
    calculate_probabilities=True,
    verbose=True,
)
topics, probs = topic_model.fit_transform(all_processed_docs)
np.save("../take_2/topic_modeling/train_topics_x.npy", np.asarray(topics, dtype=np.int32))

train_emb_news_path = "../take_2/topic_modeling/news_probe_fp32.npy"
st = SentenceTransformer("all-mpnet-base-v2")
E = st.encode(
    all_processed_docs,
    batch_size=128,
    convert_to_numpy=True,
    normalize_embeddings=False,   # <-- match training policy
    show_progress_bar=True,
).astype("float32")

np.save(train_emb_news_path, E)


# ---- 4. Save the topics ----
df = topic_model.get_topic_info()
df.to_csv("../take_2/topic_modeling/topics.csv", index=False)

# Save the model
final_model_path = "../take_2/topic_modeling/bertopic_base.model"
topic_model.save(final_model_path,
                 serialization="safetensors",
                 save_ctfidf=False,
                 save_embedding_model="all-mpnet-base-v2")