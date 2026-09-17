from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


def evaluate_classification(
    y_true,
    y_pred
):
    """
    Calculate classification metrics.

    Returns:
        dict containing accuracy and macro/weighted F1.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def get_classification_report(
    y_true,
    y_pred
):
    """
    Generate a per-class classification report.
    """

    return classification_report(
        y_true,
        y_pred,
        zero_division=0
    )