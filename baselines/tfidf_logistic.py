from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_tfidf_logistic(X_train, y_train):
    """
    Train a TF-IDF + Logistic Regression intent classifier.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


def predict_tfidf_logistic(vectorizer, model, X_test):
    """
    Predict intents for new messages.
    """
    X_test_tfidf = vectorizer.transform(X_test)

    return model.predict(X_test_tfidf)