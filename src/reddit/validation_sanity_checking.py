import pandas as pd
import numpy as np

cross_assign_round = "reddit-telegram/round_1"

def load_S(path):
    S = pd.read_csv(path, index_col=0)
    # coerce column names to int (CSV may load them as strings)
    S.columns = [int(c) for c in S.columns]
    S.index = S.index.astype(int)
    # put -1 as last column if present
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    return S[cols]

def quality_from_S(S):
    non_out = [c for c in S.columns if c != -1]
    S_non_out = S[non_out]
    SA = S_non_out.max(axis=1)                                  # per-topic alignment
    out_col = S[-1] if -1 in S.columns else pd.Series(0.0, index=S.index)
    # corpus-level quick checks (unweighted)
    C  = S_non_out.to_numpy().sum() / S.shape[0]                # closeness
    U  = out_col.mean()                                         # avg uniqueness via outlier share
    A  = SA.mean()                                              # average alignment
    # outlier-adjusted variants (ignores -1 mass)
    denom = (1.0 - out_col).replace(0, 1.0)
    S_norm = S_non_out.div(denom, axis=0).fillna(0.0)
    C_adj = S_norm.to_numpy().sum() / S.shape[0]
    A_adj = S_norm.max(axis=1).mean()
    return SA, out_col, {"C": C, "U": U, "A": A, "C_adj": C_adj, "A_adj": A_adj}

# === Load your S-matrices ===
S_news   = load_S(f"{cross_assign_round}/BTM_S_news_vs_social.csv")
S_social = load_S(f"{cross_assign_round}/BTM_S_social_vs_news.csv")

# === Recompute SA and quick metrics ===
SA_news,   out_news,   metr_news   = quality_from_S(S_news)
SA_social, out_social, metr_social = quality_from_S(S_social)

print("NEWS  ->  C={C:.3f}  U={U:.3f}  A={A:.3f}  |  C'={C_adj:.3f}  A'={A_adj:.3f}".format(**metr_news))
print("SOCIAL->  C={C:.3f}  U={U:.3f}  A={A:.3f}  |  C'={C_adj:.3f}  A'={A_adj:.3f}".format(**metr_social))

# Worst-aligned topics (sanity check)
print("Worst-aligned NEWS topics:\n", SA_news.nsmallest(10))
print("Worst-aligned SOCIAL topics:\n", SA_social.nsmallest(10))

# Save per-topic SA tables (so you have them next time)
SA_news.rename("SA").to_frame().assign(outlier_share=out_news).to_csv(
    f"{cross_assign_round}/BTM_SA_news_by_topic.csv", index_label="topic"
)
SA_social.rename("SA").to_frame().assign(outlier_share=out_social).to_csv(
    f"{cross_assign_round}/BTM_SA_social_by_topic.csv", index_label="topic"
)
