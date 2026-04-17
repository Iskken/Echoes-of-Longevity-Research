import pandas as pd

def load_S(path):
    S = pd.read_csv(path, index_col=0)
    # coerce column names to int (CSV may load them as strings)
    S.columns = [int(c) for c in S.columns]
    S.index = S.index.astype(int)
    # put -1 as last column if present
    cols = [c for c in S.columns if c != -1] + ([-1] if -1 in S.columns else [])
    return S[cols]

def unique_topics(S, thresh=0.5):
    return S.index[( -1 in S.columns) & (S[-1] >= thresh)].tolist()

S_A   = load_S("reddit-telegram/round_1/BTM_S_news_vs_social.csv")
S_B = load_S("reddit-telegram/round_1/BTM_S_social_vs_news.csv")
uniq_A = unique_topics(S_A); uniq_B = unique_topics(S_B)
print (f"Unique topics in A: {len(uniq_A)}")
print (uniq_A)
print (f"Unique topics in B: {len(uniq_B)}")
print (uniq_B)

topics_A = pd.read_csv(TELEGRAM_OUTPUT / "archive" / "take_2" / "topic_modeling/topics.csv")
topics_B = pd.read_csv("../topic_modelling_v2/round_11/topics.csv")

filtered_A = topics_A[topics_A['Topic'].isin(uniq_A)]
filtered_B = topics_B[topics_B['Topic'].isin(uniq_B)]

filtered_A.to_csv("reddit-telegram/round_1/unique-topics/uniq_topics_A.csv", index=False)
filtered_B.to_csv("reddit-telegram/round_1/unique-topics/uniq_topics_B.csv", index=False)