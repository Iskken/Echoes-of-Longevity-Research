from bertopic import BERTopic

model = BERTopic.load("../take_2/topic_modeling/bertopic_base.model")

# Visualize topics (overview)
fig = model.visualize_topics()
fig.show()