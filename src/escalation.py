def decide_escalation(
    intent,
    similarity_score,
    generated_reply
):
    """
    Decide whether a customer message should be
    auto-handled or escalated to a human.
    """

    high_risk_intents = {
        "complaint_or_escalation",
        "payment_or_billing",
        "refund",
        "product_issue",
    }

    if intent in high_risk_intents:
        return {
            "decision": "human",
            "reason": f"Intent '{intent}' requires human review."
        }

    if similarity_score < 0.65:
        return {
            "decision": "human",
            "reason": "Historical evidence is not sufficiently similar."
        }

    if not generated_reply or len(generated_reply.strip()) < 20:
        return {
            "decision": "human",
            "reason": "Generated response is insufficient."
        }

    return {
        "decision": "auto",
        "reason": "Intent is low-risk and relevant historical evidence was found."
    }