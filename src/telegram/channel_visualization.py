import matplotlib.pyplot as plt
from collections import Counter
import json

with open('../take_2/messages.json', 'r') as f:
    messages = json.load(f)
print(len(messages))

# Count messages per channel
channel_counts = Counter(msg['source'] for msg in messages)

# Prepare data for plotting
channels = list(channel_counts.keys())
counts = list(channel_counts.values())

# Plot
plt.figure(figsize=(10, 6))
plt.bar(channels, counts, color='skyblue')
plt.xlabel('Channel')
plt.ylabel('Number of Messages')
plt.title('Distribution of Messages per Channel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

