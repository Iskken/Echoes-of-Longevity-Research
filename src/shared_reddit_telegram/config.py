"""
Centralized path configuration for the Echoes of Longevity project.

All input and output data files live on Google Drive in a well-defined
folder structure (see README.md). Notebooks and helper modules should
import constants from here rather than hard-coding paths.

Usage:
    from src.shared_reddit_telegram.config import DATA_ROOT, REDDIT_INPUT, REDDIT_OUTPUT

The `DATA_ROOT` env var (loaded from `.env`) points at the project's
Google Drive folder. Set it to your local sync folder or Drive File
Stream mount point. See `.env.example` for the template.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project root: this file lives at `src/shared_reddit_telegram/config.py`, so
# the repo root is three parents up. Used for importing sibling modules like
# `from src.reddit.cross_assign import ...`.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Google Drive data root: all input corpora and output artifacts live here.
# Override via `DATA_ROOT=...` in .env.
# ---------------------------------------------------------------------------
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "GoogleDrive" / "Echoes-of-Longevity")).expanduser()

# ---------------------------------------------------------------------------
# Per-platform input / output folders on Google Drive.
# Layout:
#     <DATA_ROOT>/<Platform>/input/     -- raw data
#     <DATA_ROOT>/<Platform>/output/    -- derived artifacts (models, CSVs)
# ---------------------------------------------------------------------------
REDDIT_INPUT = DATA_ROOT / "Reddit" / "input"
REDDIT_OUTPUT = DATA_ROOT / "Reddit" / "output"
REDDIT_MODELS = REDDIT_OUTPUT / "bertopic"

PUBMED_INPUT = DATA_ROOT / "PubMed" / "input"
PUBMED_OUTPUT = DATA_ROOT / "PubMed" / "output"
PUBMED_MODELS = PUBMED_OUTPUT / "bertopic"

TELEGRAM_INPUT = DATA_ROOT / "Telegram" / "input"
TELEGRAM_OUTPUT = DATA_ROOT / "Telegram" / "output"
TELEGRAM_MODELS = TELEGRAM_OUTPUT / "bertopic"

# Cross-platform topic-matching outputs (consumed by RQ2 and other cross-cutting notebooks)
TOPIC_MATCHINGS = DATA_ROOT / "topic_matchings"

# ---------------------------------------------------------------------------
# Back-compat aliases: older notebooks import these names directly.
# Keep them as thin aliases so imports continue to resolve without edits.
# The Small-Data / Medium-Data splits from the original local filesystem no
# longer exist separately on Google Drive; they all map to REDDIT_OUTPUT.
# ---------------------------------------------------------------------------
REDDIT_DATA = REDDIT_OUTPUT
REDDIT_SMALL_DATA = REDDIT_OUTPUT  # deprecated: historical sampling split
REDDIT_MEDIUM_DATA = REDDIT_OUTPUT  # deprecated: historical sampling split
PUBMED_DATA = PUBMED_OUTPUT
TELEGRAM_DATA = TELEGRAM_OUTPUT
TELEGRAM_PIPELINE = TELEGRAM_OUTPUT  # older notebooks import this name

# ---------------------------------------------------------------------------
# Convenience aliases for specific outputs.
# ---------------------------------------------------------------------------
REDDIT_MERGED_JSONL = REDDIT_OUTPUT / "merged_submissions.jsonl"
REDDIT_MERGED_CSV = REDDIT_OUTPUT / "merged_submissions.csv"
REDDIT_PREPROCESSED = REDDIT_OUTPUT / "preprocessed-docs.pkl"
TELEGRAM_MESSAGES = TELEGRAM_OUTPUT / "messages.json"
TELEGRAM_GRAPH = TELEGRAM_OUTPUT / "tg_channel_network.gexf"
