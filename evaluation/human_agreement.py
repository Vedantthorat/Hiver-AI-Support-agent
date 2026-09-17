from sklearn.metrics import cohen_kappa_score


def convert_score_to_label(score, threshold=3):
    """
    Convert the LLM judge's numeric overall score
    into an acceptance label.
    """

    if score is None:
        return None

    if score >= threshold:
        return "acceptable"

    return "not_acceptable"


def calculate_human_agreement(
    human_labels,
    judge_scores,
    threshold=3
):
    """
    Compare human evaluation labels with LLM judge labels.

    LLM scores >= threshold are treated as acceptable.
    """

    judge_labels = [
        convert_score_to_label(
            score,
            threshold=threshold
        )
        for score in judge_scores
    ]

    valid_pairs = [
        (human, judge)
        for human, judge in zip(
            human_labels,
            judge_labels
        )
        if human is not None
        and judge is not None
    ]

    if not valid_pairs:
        return {
            "judge_labels": [],
            "agreement": None,
            "cohen_kappa": None
        }

    human_valid = [
        pair[0]
        for pair in valid_pairs
    ]

    judge_valid = [
        pair[1]
        for pair in valid_pairs
    ]

    agreement = sum(
        human == judge
        for human, judge in valid_pairs
    ) / len(valid_pairs)

    kappa = cohen_kappa_score(
        human_valid,
        judge_valid
    )

    return {
        "judge_labels": judge_valid,
        "agreement": agreement,
        "cohen_kappa": kappa
    }