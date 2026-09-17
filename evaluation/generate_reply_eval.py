import pandas as pd


def generate_evaluation_replies(
    sample_path,
    output_path,
    intent_classifier,
    embedding_model,
    index,
    retrieval_data,
    gemini_client,
    run_support_agent
):
    """
    Generate support replies for the reply-quality
    evaluation sample.

    Each customer message is passed through the
    complete support-agent pipeline.
    """

    sample = pd.read_csv(
        sample_path
    )

    results = []

    for _, row in sample.iterrows():

        result = run_support_agent(
            customer_message=row["text"],
            intent_classifier=intent_classifier,
            embedding_model=embedding_model,
            index=index,
            retrieval_data=retrieval_data,
            gemini_client=gemini_client,
            top_k=3
        )

        results.append(
            {
                "tweet_id": row["tweet_id"],
                "customer_message": row["text"],
                "gold_intent": row["final_intent"],
                "predicted_intent": result["intent"],
                "similarity_score": result["similarity_score"],
                "generated_reply": result["generated_reply"],
                "decision": result["decision"],
                "decision_reason": result["reason"]
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    return results_df