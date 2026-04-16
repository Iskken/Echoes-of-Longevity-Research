# Repository Structure

This repo is now organized around a simple split:

- `src/`: reusable Python helpers that notebooks import.
- `Research Questions/`: top-level home for research-question notebooks, with one folder per RQ.
- `News/`, `Publications/`, `SocialMedia/`: domain folders.
- `data/`: raw or processed datasets within each domain.
- `notebooks/`: active notebooks grouped by workflow stage.
- `outputs/`: generated figures, HTML, vector stores, and similar artifacts.

Current notebook conventions:

- `01_*`: collection or cleaning
- `02_*` / `03_*`: exploration and modeling
- `04_*`: topic matching or cross-platform alignment
- `06_*`: LLM or experimental workflows
- `archive/`: legacy, duplicate, or symposium-specific notebooks kept for reference

Research question convention:

- `Research Questions/RQ1/`, `Research Questions/RQ2/`, etc.
- each RQ folder can contain the notebook plus any question-specific figures or tables

Shared path constants live in [src/project_paths.py](/home/iskender/longevity_research/Echoes-of-Longevity-Research/src/project_paths.py).
