import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gexf("../take_2/tg_channel_network.gexf")
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
plt.show()