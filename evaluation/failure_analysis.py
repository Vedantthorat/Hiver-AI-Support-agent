from collections import Counter


def find_confusion_pairs(predictions):
    """
    Find the most common incorrect intent predictions.

    Args:
        predictions: DataFrame containing:
            - final_intent
            - predicted_intent

    Returns:
        List of confusion pairs sorted by frequency.
    """

    errors = predictions[
        predictions["final_intent"]
        != predictions["predicted_intent"]
    ]

    confusion_pairs = Counter(
        zip(
            errors["final_intent"],
            errors["predicted_intent"]
        )
    )

    return confusion_pairs.most_common()


def get_failure_examples(
    predictions,
    true_intent,
    predicted_intent,
    n=5
):
    """
    Get real customer examples for a specific
    confusion pair.

    Args:
        predictions: Evaluation predictions DataFrame.
        true_intent: Human-labelled intent.
        predicted_intent: Model prediction.
        n: Number of examples to return.

    Returns:
        DataFrame containing failure examples.
    """

    failures = predictions[
        (predictions["final_intent"] == true_intent)
        & (
            predictions["predicted_intent"]
            == predicted_intent
        )
    ].copy()

    return failures[
        [
            "tweet_id",
            "text",
            "final_intent",
            "predicted_intent"
        ]
    ].head(n)


def get_top_failure_modes(
    predictions,
    top_n=5,
    examples_per_mode=3
):
    """
    Identify the top failure modes and return
    representative real examples.

    Returns:
        List of dictionaries containing:
            - true_intent
            - predicted_intent
            - error_count
            - examples
    """

    confusion_pairs = find_confusion_pairs(
        predictions
    )

    failure_modes = []

    for (
        (true_intent, predicted_intent),
        error_count
    ) in confusion_pairs[:top_n]:

        examples = get_failure_examples(
            predictions,
            true_intent,
            predicted_intent,
            n=examples_per_mode
        )

        failure_modes.append(
            {
                "true_intent": true_intent,
                "predicted_intent": predicted_intent,
                "error_count": error_count,
                "examples": examples,
            }
        )

    return failure_modes