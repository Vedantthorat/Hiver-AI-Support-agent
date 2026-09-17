from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class IntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=10000,
            ngram_range=(1, 2)
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    def train(self, texts, labels):
        """
        Train the classifier.
        """

        X = self.vectorizer.fit_transform(texts)

        self.model.fit(
            X,
            labels
        )

    def predict(self, texts):
        """
        Predict intents for multiple messages.
        """

        X = self.vectorizer.transform(texts)

        return self.model.predict(X)

    def predict_one(self, text):
        """
        Predict the intent for one customer message.
        """

        return self.predict([text])[0]