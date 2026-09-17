import pandas as pd

from evaluation.human_agreement import (
    calculate_human_agreement
)


def evaluate_human_agreement(
    evaluation_path,
    output_path,
    threshold=3
):
    """
    Evaluate agreement between human labels
    and LLM judge scores.
    """

    data = pd.read_csv(
        evaluation_path
    )

    human_labels = data[
        "human_label"
    ].tolist()

    judge_scores = data[
        "overall_score"
    ].tolist()

    results = calculate_human_agreement(
        human_labels=human_labels,
        judge_scores=judge_scores,
        threshold=threshold
    )

    summary = pd.DataFrame(
        [
            {
                "num_examples": len(human_labels),
                "agreement": results["agreement"],
                "cohen_kappa": results["cohen_kappa"],
                "judge_threshold": threshold
            }
        ]
    )

    summary.to_csv(
        output_path,
        index=False
    )

    return summary


if __name__ == "__main__":

    summary = evaluate_human_agreement(
        evaluation_path="../results/reply_quality_human_eval.csv",
        output_path="../results/human_agreement_summary.csv"
    )

    print(summary.to_string(index=False))