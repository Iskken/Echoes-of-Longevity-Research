#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract the 'Full text' column from a CSV and save the documents to a .pkl file.

- Case-insensitive match for the column name ("Full text")
- Keeps order, drops NaNs, converts to plain strings, strips whitespace
- Optional: de-duplicate while preserving order (toggle DEDUPE)
"""

import os
import sys
import pickle
import pandas as pd

# ====== CONFIG ======
INPUT_CSV  = DATA_ROOT / "News" / "Data/topic_aligment_v3/doc_info_v3.csv"       # <- change me
OUTPUT_PKL = DATA_ROOT / "News" / "Data/topic_aligment_v3/documents_full_text.pkl"     # <- change me
DEDUPE     = False                         # set True to remove duplicates while preserving order
MIN_LEN    = 0                             # drop docs shorter than this many characters (0 to disable)
# ====================

def find_full_text_col(columns) -> str:
    """Find a column named 'Full text' (case-insensitive, forgiving on spaces/underscores)."""
    norm = {c: "".join(str(c).lower().split()).replace("_","") for c in columns}
    for original, lowered in norm.items():
        if lowered in {"fulltext", "fulltextcol"} or lowered == "fulltext":  # main target
            return original
        if lowered == "fulltext":  # redundancy
            return original
    # Fallback: best-effort match containing 'full' and 'text'
    for original, lowered in norm.items():
        if "full" in lowered and "text" in lowered:
            return original
    raise ValueError("Could not find a 'Full text' column (case-insensitive).")

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    # Read CSV (robust to large files; change chunksize if needed)
    df = pd.read_csv(INPUT_CSV)

    # Locate the 'Full text' column (case-insensitive)
    ft_col = find_full_text_col(df.columns)

    # Build clean list of documents
    docs = (
        df[ft_col]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )

    if MIN_LEN > 0:
        docs = docs[docs.map(len) >= MIN_LEN]

    if DEDUPE:
        seen = set()
        ordered_unique = []
        for d in docs:
            if d not in seen:
                seen.add(d)
                ordered_unique.append(d)
        docs_list = ordered_unique
    else:
        docs_list = docs.tolist()

    # Save to pickle (list of strings)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(docs_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(docs_list):,} documents to: {OUTPUT_PKL}")

if __name__ == "__main__":
    main()