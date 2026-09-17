import numpy as np
from collections import defaultdict


class PrototypeIntentClassifier:
    """
    Intent classifier using semantic intent prototypes.

    Each intent is represented by the centroid of the
    embeddings of its labeled training examples.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.prototypes = {}
        self.labels = []

    def train(self, texts, labels):
        """
        Create one semantic prototype for each intent.
        """

        embeddings = self.embedding_model.encode(
            list(texts),
            normalize_embeddings=True
        )

        grouped = defaultdict(list)

        for embedding, label in zip(
            embeddings,
            labels
        ):
            grouped[label].append(embedding)

        self.prototypes = {}

        for label, vectors in grouped.items():

            centroid = np.mean(
                vectors,
                axis=0
            )

            # Normalize the centroid so that
            # dot product behaves like cosine similarity.
            centroid = centroid / np.linalg.norm(
                centroid
            )

            self.prototypes[label] = centroid

        self.labels = list(
            self.prototypes.keys()
        )

    def predict_one(self, text):
        """
        Predict the intent whose prototype
        is most similar to the input message.
        """

        query_embedding = self.embedding_model.encode(
            [text],
            normalize_embeddings=True
        )[0]

        scores = {}

        for label, prototype in self.prototypes.items():

            scores[label] = np.dot(
                query_embedding,
                prototype
            )

        return max(
            scores,
            key=scores.get
        )

    def predict(self, texts):
        """
        Predict intents for multiple messages.
        """

        return [
            self.predict_one(text)
            for text in texts
        ]