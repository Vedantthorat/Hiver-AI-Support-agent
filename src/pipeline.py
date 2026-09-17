from src.retrieval import retrieve_similar_cases
from src.generation import generate_grounded_reply
from src.escalation import decide_escalation


def run_support_agent(
    customer_message,
    intent_classifier,
    embedding_model,
    index,
    retrieval_data,
    gemini_client,
    top_k=3
):
    """
    Run the complete Amazon customer-support pipeline.

    Steps:
    1. Predict intent.
    2. Retrieve similar historical cases.
    3. Generate a grounded reply.
    4. Decide auto-handle vs human escalation.
    """

    # 1. Intent classification
    intent = intent_classifier.predict_one(
        customer_message
        
    )

    # 2. Historical retrieval
    retrieved_cases = retrieve_similar_cases(
        customer_message,
        embedding_model,
        index,
        retrieval_data,
        top_k=top_k
    )

    similarity_score = float(
        retrieved_cases["similarity_score"].iloc[0]
    )

    # 3. Generate grounded reply
    generated_reply = generate_grounded_reply(
        gemini_client,
        customer_message,
        intent,
        retrieved_cases
    )

    # 4. Escalation decision
    escalation = decide_escalation(
        intent,
        similarity_score,
        generated_reply
    )

    return {
        "customer_message": customer_message,
        "intent": intent,
        "retrieved_cases": retrieved_cases,
        "similarity_score": similarity_score,
        "generated_reply": generated_reply,
        "decision": escalation["decision"],
        "reason": escalation["reason"]
    }