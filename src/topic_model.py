import json
from sentence_transformers import SentenceTransformer

class TopicModel:
    def __init__(self, model_name: str, themes: dict):
        # self.model_name = model_name
        self.model = self.create_sentence_transformer(model_name)
        if themes:
            self.themes = themes
        else:
            self.themes = self.load_example_topics()
    
    def create_sentence_transformer(self, model_name: str) -> SentenceTransformer:
        try: 
            return SentenceTransformer(model_name)
        except Exception as e:
            raise ValueError(f"Please input a valid model name for HuggingFace Sentence Transformers: {e}")

    def load_example_topics(self) -> dict:
        f = open('resources/example_themes.json') 
        topic_sentence = json.load(f)
        theme_dict = {}
        for item in topic_sentence:
            theme_dict[item['name']] = item['description']
        return theme_dict

    def compute_theme_scores(self, threshold: float, sentence: str) -> str | list[str]:
        # Compute embeddings for both lists
        sentence_embedding = self.model.encode(sentence)
        topic_embeddings = self.model.encode(self.themes.values())

        # Compute cosine similarities
        similarities = self.model.similarity(sentence_embedding, topic_embeddings)[0]


        # Determine the themes to return based on threshold or maximum similarity 
        result_themes = []
        if threshold:
            for i, score in enumerate(similarities):
                if score >= threshold:
                    result_themes.append(list(self.themes.keys())[i])
        else:
            max_score = max(similarities)
            for i, score in enumerate(similarities):
                if score == max_score:
                    result_themes.append(list(self.themes.keys())[i])
        return result_themes