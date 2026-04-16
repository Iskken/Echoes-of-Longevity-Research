from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RESEARCH_QUESTIONS_DIR = ROOT / "Research Questions"
RQ1_DIR = RESEARCH_QUESTIONS_DIR / "RQ1"
RQ4_DIR = RESEARCH_QUESTIONS_DIR / "RQ4"

NEWS_DIR = ROOT / "News"
NEWS_DATA_DIR = NEWS_DIR / "data"
NEWS_OUTPUT_DIR = NEWS_DIR / "outputs"
PROQUEST_RAW_DIR = NEWS_DATA_DIR / "raw" / "proquest"
PROQUEST_UNPROCESSED_DIR = PROQUEST_RAW_DIR / "unprocessed"
PROQUEST_PROCESSED_DIR = NEWS_DATA_DIR / "processed" / "proquest"
GUARDIAN_DATA_DIR = NEWS_DATA_DIR / "raw" / "guardian_api"
EXTERNAL_NEWS_DIR = NEWS_DATA_DIR / "raw" / "external_sources"
TOPIC_MATCHING_DIR = NEWS_DATA_DIR / "processed" / "topic_matching"
LLM_CLASSIFICATION_DIR = NEWS_DATA_DIR / "processed" / "llm_classification"
RQ1_FIGURES_DIR = RQ1_DIR / "figures"
RQ4_PLOTS_DIR = RQ4_DIR / "plots"
NEWS_HTML_DIR = NEWS_OUTPUT_DIR / "html"
CHROMA_DIR = NEWS_OUTPUT_DIR / "vector_store" / "chroma_store"

PUBLICATIONS_DIR = ROOT / "Publications"
PUBLICATIONS_DATA_DIR = PUBLICATIONS_DIR / "data"
PUBLICATIONS_OUTPUT_DIR = PUBLICATIONS_DIR / "outputs"
PUBMED_RAW_DIR = PUBLICATIONS_DATA_DIR / "raw" / "pubmed"
PUBMED_PROCESSED_DIR = PUBLICATIONS_DATA_DIR / "processed" / "pubmed"
ARXIV_RAW_DIR = PUBLICATIONS_DATA_DIR / "raw" / "arxiv"
PREPRINT_RAW_DIR = PUBLICATIONS_DATA_DIR / "raw" / "preprints"
PUBLICATIONS_TABLE_DIR = PUBLICATIONS_OUTPUT_DIR / "tables"

SOCIALMEDIA_DIR = ROOT / "SocialMedia"
REDDIT_DIR = SOCIALMEDIA_DIR / "Reddit"
REDDIT_DATA_DIR = REDDIT_DIR / "data" / "processed"
YOUTUBE_DIR = SOCIALMEDIA_DIR / "Youtube"
YOUTUBE_DATA_DIR = YOUTUBE_DIR / "data" / "processed"
