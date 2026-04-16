import re, pandas as pd

from src.project_paths import PROQUEST_UNPROCESSED_DIR

def read_file(filename:str):
    data= {
        "Author":[],
        "Publication info":[],
        "Abstract":[],
        "Full text":[],
        "Subject":[],
        "Product name":[],
        "Title":[],
        "Publication title":[],
        "Pages":[],
        "Publication year":[],
        "Publication date":[],
        "Section":[],
        "Publisher":[],
        "Place of publication":[],
        "Country of publication":[],
        "Publication subject":[],
        "ISSN":[],
        "Source type":[],
        "Language of publication":[],
        "Document type":[],
        "ProQuest document ID":[],
        "Document URL":[],
        "Copyright":[],
        "Last updated":[],
        "Database":[]
    }

    delimiter = '____________________________________________________________'
    headers = list(data.keys())
    unprocessed_headers = headers.copy()
    
    with open(filename) as file:
        content = file.read()
        article_bodies = content.split(delimiter)
    
    articles_processed = 0
    for article in article_bodies:
        article = article.split("\n\n")
        for paragraph in article:
            for prefix in headers:
                if (paragraph.startswith(prefix)) & (prefix in unprocessed_headers):
                    data[prefix].append(paragraph[len(prefix)+1:])
                    unprocessed_headers.remove(prefix)
        
        articles_processed += 1
        unprocessed_headers = headers.copy()
        
        for prefix in headers:
            if len(data[prefix]) < articles_processed:
                data[prefix].append(None)

    df = pd.DataFrame(data)
    df.drop(columns=['Copyright', 'Source type', 'Database', 'Publication info', 'Pages','Product name'], inplace=True)
    df = df[['Title', 'Abstract', 'Full text', 'Author', 'Subject', 'Publication title', 'Publication year', 'Publication date', 
            'Section', 'Publisher', 'Place of publication', 'Country of publication', 'Publication subject', 'ISSN', 'Language of publication',
            'Document type', 'ProQuest document ID', 'Document URL', 'Last updated']]
    
    
    return df

def assemble_df():
    df = pd.DataFrame()
    for year in range(2010,2025+1):
        df_year = read_file(PROQUEST_UNPROCESSED_DIR / f"longevity_{year}.txt")
        print(year)
        df_dedup = dedup_by_prefix_any(df_year, by = ["Title", "Full text"], n_words=10, keep="longest", scope_pub_month = False)    
        df = pd.concat([df, df_dedup], ignore_index=True)
        print(len(df), "→", len(df_year),"\n")
    return df

# Deduplicating function

from typing import Iterable, List

# --- helpers ---
def normalize(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r"['’`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def first_n_words(s: str, n: int) -> str:
    s = normalize(s)
    if not s: return ""
    return " ".join(s.split()[:n])

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def dedup_by_prefix_any(
    df: pd.DataFrame,
    by: list[str] = ["Title", "Abstract", "Full text"],
    n_words: int = 10,
    keep: str = "longest",
    scope_pub_month: bool = False,   # <- constrain matches within pub+month
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    # Optional scope: normalize publication + month
    if scope_pub_month:
        def pub_base(s):
            s = (s or "")
            s = s.split("\n")[0].split(";")[0].strip().lower()
            return s
        df["_pub"]   = df.get("Publication title", "").map(pub_base)
        df["_date"]  = pd.to_datetime(df.get("Publication date"), errors="coerce")
        df["_month"] = df["_date"].dt.to_period("M")

    cols = []
    if "Title" in by:
        df["_t_fp"] = df.get("Title", "").map(lambda x: first_n_words(x, n_words))
        cols.append("_t_fp")
    if "Abstract" in by:
        df["_a_fp"] = df.get("Abstract", "").map(lambda x: first_n_words(x, n_words))
        cols.append("_a_fp")
    if "Full text" in by:
        df["_f_fp"] = df.get("Full text", "").map(lambda x: first_n_words(x, n_words))
        cols.append("_f_fp")

    # Union-Find
    uf = UnionFind(len(df))
    seen = {}
    for col in cols:
        for i, fp in enumerate(df[col].values):
            if not fp:
                continue
            key = (col, fp)
            if scope_pub_month:
                key += (df.loc[i, "_pub"], df.loc[i, "_month"])
            j = seen.get(key)
            if j is None:
                seen[key] = i
            else:
                uf.union(i, j)

    df["_rep"] = [uf.find(i) for i in range(len(df))]

    # tie-break
    df["_text_len"] = df.get("Full text", "").fillna("").astype(str).str.len()
    if "_date" not in df.columns:
        df["_date"] = pd.to_datetime(df.get("Publication date"), errors="coerce")

    if keep == "longest":
        order = ["_rep", "_text_len"]
        ascending = [True, False]
    elif keep == "earliest":
        order = ["_rep", "_date"]
        ascending = [True, True]
    else:
        raise ValueError("keep must be 'longest' or 'earliest'")

    out = (
        df.sort_values(order, ascending=ascending)
          .drop_duplicates(subset=["_rep"], keep="first")
    )

    drop_cols = cols + ["_rep", "_text_len", "_date"]
    if scope_pub_month:
        drop_cols += ["_pub", "_month"]
    return out.drop(columns=[c for c in drop_cols if c in out.columns]).reset_index(drop=True)
