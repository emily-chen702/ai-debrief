from src.topic_model import TopicModel
from src.summary_model import SummaryModel

example_sentence = "Met with product to define the scope of our project."

topic_model = TopicModel(model_name="all-MiniLM-L6-v2")

theme_scores = topic_model.compute_theme_scores(threshold=0.5, sentence=example_sentence)
print(theme_scores)

summary_model = SummaryModel()
summary_result = summary_model.compute(example_sentence)
summary_result['top_themes'] = theme_scores

print(summary_result)
