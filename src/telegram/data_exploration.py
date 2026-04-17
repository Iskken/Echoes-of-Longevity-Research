import json, pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

with open("../take_2/messages.json", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'], errors="coerce")
df = df.dropna(subset=['date'])
df = df.loc[df['date'].dt.year >= 2010].copy()

print("Total records (2010+):", len(df))
print(df['source'].value_counts().head(10))

df.set_index('date', inplace=True)
df.resample('W').size().plot(title="Weekly Message Volume (2010–present)")
plt.xlabel("Date"); plt.ylabel("Messages")
plt.show()

counts = df.groupby(df.index.year).size()
years = list(range(2010, datetime.now().year + 1))
counts = counts.reindex(years, fill_value=0)

counts.plot(kind='bar', figsize=(12,6), color='skyblue')
plt.title(f"Telegram Messages per Year (2010–{datetime.now().year})")
plt.xlabel("Year"); plt.ylabel("Number of Messages")
plt.xticks(rotation=45); plt.tight_layout()
plt.show()