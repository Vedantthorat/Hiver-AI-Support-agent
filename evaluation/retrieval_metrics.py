import numpy as np


def recall_at_k(
    evaluation_data,
    retrieval_data,
    embedding_model,
    index,
    k=5
):
    """
    Calculate Recall@K for historical case retrieval.

    A retrieval is counted as correct when the historically linked
    response tweet is present in the top-K retrieved cases.

    Args:
        evaluation_data: Golden examples containing:
            - text
            - response_tweet_id

        retrieval_data: Historical retrieval dataset containing:
            - response_tweet_id

        embedding_model: SentenceTransformer model.

        index: FAISS similarity index.

        k: Number of retrieved cases.

    Returns:
        Recall@K as a float.
    """

    hits = 0
    total = len(evaluation_data)

    if total == 0:
        return 0.0

    for _, row in evaluation_data.iterrows():

        query_embedding = embedding_model.encode(
            [row["text"]],
            normalize_embeddings=True
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = index.search(
            query_embedding,
            k
        )

        retrieved_ids = (
            retrieval_data.iloc[indices[0]]
            ["response_tweet_id"]
            .tolist()
        )

        if row["response_tweet_id"] in retrieved_ids:
            hits += 1

    return hits / total


def evaluate_retrieval(
    evaluation_data,
    retrieval_data,
    embedding_model,
    index,
    ks=(1, 3, 5)
):
    """
    Evaluate retrieval performance at multiple K values.

    Returns:
        Dictionary containing Recall@K values.
    """

    results = {}

    for k in ks:

        results[f"recall@{k}"] = recall_at_k(
            evaluation_data=evaluation_data,
            retrieval_data=retrieval_data,
            embedding_model=embedding_model,
            index=index,
            k=k
        )

    return results