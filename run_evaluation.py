import pandas as pd

from src.semantic_intents import PrototypeIntentClassifier
from evaluation.classification_metrics import evaluate_classification

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Load training and golden evaluation data
# ---------------------------------------------------------

training_data = pd.read_csv(
    "data/processed/amazonhelp_training_135.csv"
)

golden_data = pd.read_csv(
    "data/golden/amazonhelp_golden.csv"
)


# ---------------------------------------------------------
# 2. Load the sentence embedding model
# ---------------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 3. Train the semantic intent classifier
# ---------------------------------------------------------

classifier = PrototypeIntentClassifier(
    embedding_model=embedding_model
)

classifier.train(
    texts=training_data["text"].fillna("").tolist(),
    labels=training_data["intent"].tolist()
)


# ---------------------------------------------------------
# 4. Predict intents for the golden evaluation set
# ---------------------------------------------------------

predictions = classifier.predict(
    golden_data["text"].fillna("").tolist()
)


# ---------------------------------------------------------
# 5. Calculate evaluation metrics
# ---------------------------------------------------------

metrics = evaluate_classification(
    y_true=golden_data["final_intent"].tolist(),
    y_pred=predictions
)


# ---------------------------------------------------------
# 6. Display results
# ---------------------------------------------------------

print("\nAmazonHelp Intent Classification")
print("--------------------------------")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Macro F1:    {metrics['macro_f1']:.4f}")
print(f"Weighted F1: {metrics['weighted_f1']:.4f}")