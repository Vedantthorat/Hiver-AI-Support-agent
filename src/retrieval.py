import numpy as np


def retrieve_similar_cases(
    query,
    embedding_model,
    index,
    retrieval_data,
    top_k=5
):
    """
    Retrieve historically similar customer-support cases.
    """

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = retrieval_data.iloc[indices[0]].copy()
    results["similarity_score"] = scores[0]

    return results