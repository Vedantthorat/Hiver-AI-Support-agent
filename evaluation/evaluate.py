import pandas as pd

from evaluation.classification_metrics import (
    evaluate_classification,
    get_classification_report,
)


def evaluate_classifier(
    classifier,
    golden_path,
):
    """
    Evaluate an intent classifier on the golden evaluation set.

    Args:
        classifier: trained classifier with predict() method.
        golden_path: path to the golden CSV file.

    Returns:
        metrics: dictionary containing evaluation metrics.
        report: per-class classification report.
        predictions: dataframe containing predictions.
    """

    # Load golden evaluation set
    golden = pd.read_csv(golden_path)

    # Customer messages
    texts = golden["text"].fillna("").tolist()

    # Human-labelled intents
    y_true = golden["final_intent"].tolist()

    # Model predictions
    y_pred = classifier.predict(texts)

    # Calculate metrics
    metrics = evaluate_classification(
        y_true,
        y_pred
    )

    # Generate per-class report
    report = get_classification_report(
        y_true,
        y_pred
    )

    # Store predictions for later error analysis
    predictions = golden.copy()

    predictions["predicted_intent"] = y_pred

    predictions["correct"] = (
        predictions["final_intent"]
        == predictions["predicted_intent"]
    )

    return (
        metrics,
        report,
        predictions
    )