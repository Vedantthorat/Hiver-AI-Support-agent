import re

from google import genai


def sanitize_historical_response(text):
    """
    Remove case-specific details from historical responses
    before using them as guidance for the LLM.
    """

    text = str(text)

    # Remove times such as:
    # 8 PM, 8:30 AM, 21:00
    text = re.sub(
        r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b",
        "",
        text
    )

    # Remove common relative dates
    text = re.sub(
        r"\b(today|tomorrow|yesterday|tonight)\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove long numeric IDs such as tracking/order numbers
    text = re.sub(
        r"\b\d{6,}\b",
        "",
        text
    )

    # Clean extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def sanitize_generated_reply(text, customer_message):
    """
    Remove unsupported case-specific details from the
    generated customer-facing reply.
    """

    text = str(text)

    customer_lower = customer_message.lower()

    # ---------------------------------------------------------
    # 1. Remove sentences containing unsupported relative dates
    # ---------------------------------------------------------

    if not re.search(
        r"\b(?:today|tomorrow|yesterday|tonight)\b",
        customer_lower
    ):
        text = re.sub(
            r"[^.!?]*\b(?:today|tomorrow|yesterday|tonight)\b[^.!?]*[.!?]?",
            "",
            text,
            flags=re.IGNORECASE
        )

    # ---------------------------------------------------------
    # 2. Remove unsupported deadline phrases
    # ---------------------------------------------------------

    if "tomorrow" not in customer_lower:
        text = re.sub(
            r"\bby\s+(?:the\s+end\s+of\s+)?tomorrow\b",
            "",
            text,
            flags=re.IGNORECASE
        )

    if "today" not in customer_lower:
        text = re.sub(
            r"\bby\s+today\b",
            "",
            text,
            flags=re.IGNORECASE
        )

    # ---------------------------------------------------------
    # 3. Remove unsupported delivery-date references
    # ---------------------------------------------------------

    if not re.search(
        r"\b(?:today|tomorrow|yesterday|tonight)\b",
        customer_lower
    ):
        text = re.sub(
            r"\bby\s+(?:the\s+)?(?:expected\s+)?delivery\s+date\b",
            "",
            text,
            flags=re.IGNORECASE
        )

    # ---------------------------------------------------------
    # 4. Remove standalone unsupported times
    # ---------------------------------------------------------

    customer_times = re.findall(
        r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b",
        customer_message
    )

    generated_times = re.findall(
        r"\b\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)\b",
        text
    )

    for time_value in generated_times:
        if time_value not in customer_times:
            text = re.sub(
                rf"\b{re.escape(time_value)}\b",
                "",
                text,
                flags=re.IGNORECASE
            )

    # ---------------------------------------------------------
    # 5. Remove long numeric IDs if hallucinated
    # ---------------------------------------------------------

    customer_ids = re.findall(
        r"\b\d{6,}\b",
        customer_message
    )

    generated_ids = re.findall(
        r"\b\d{6,}\b",
        text
    )

    for id_value in generated_ids:
        if id_value not in customer_ids:
            text = re.sub(
                rf"\b{re.escape(id_value)}\b",
                "",
                text
            )

    # ---------------------------------------------------------
    # 6. Clean whitespace and punctuation
    # ---------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    text = re.sub(
        r"\s+([,.!?])",
        r"\1",
        text
    )

    text = re.sub(
        r"\(\s*\)",
        "",
        text
    )

    text = re.sub(
        r"\s{2,}",
        " ",
        text
    ).strip()

    return text


def generate_grounded_reply(
    client,
    customer_message,
    intent,
    retrieved_cases
):
    """
    Generate a customer-support reply grounded in
    historical AmazonHelp responses.
    """

    historical_examples = []

    for _, row in retrieved_cases.iterrows():

        sanitized_response = sanitize_historical_response(
            row["response_text"]
        )

        # IMPORTANT:
        # We intentionally provide only the historical
        # AmazonHelp response, not the historical customer's
        # message. This reduces the risk of copying
        # customer-specific information.
        historical_examples.append(
            f"""
Historical AmazonHelp response:

{sanitized_response}

Similarity:

{row["similarity_score"]:.3f}
"""
        )

    evidence = "\n---\n".join(
        historical_examples
    )

    prompt = f"""
You are an Amazon customer-support assistant.

Current customer message:

{customer_message}

Detected intent:

{intent}

Here are historical AmazonHelp responses from similar
support cases:

{evidence}

Write a helpful draft reply to the CURRENT customer.

Rules:

1. Use historical responses only to understand the general
   way AmazonHelp handles similar issues.

2. Do not copy information that belongs to a historical
   customer or historical case.

3. Never copy dates, times, tracking numbers, order numbers,
   delivery estimates, or deadlines from historical cases.

4. Never make a delivery promise or guarantee based on a
   historical case.

5. Do not combine case-specific information from multiple
   historical responses.

6. If the current customer message does not provide a specific
   date, time, tracking number, order number, or other
   case-specific detail, do not invent one.

7. Do not introduce words such as "today", "tomorrow",
   "yesterday", or "tonight" unless they appear in the
   current customer's message.

8. Do not introduce phrases such as "by tomorrow",
   "by the end of tomorrow", "by today", or any other
   deadline unless the current customer explicitly provided
   that deadline.

9. Do not introduce specific delivery times such as
   "8 PM" unless the current customer explicitly provided
   that time.

10. Give general, actionable guidance appropriate for the
    CURRENT customer's message.

11. Do not invent policies, refunds, delivery dates,
    tracking information, guarantees, or account information.

12. Do not claim that you performed an action that you cannot
    actually perform.

13. If the historical evidence is insufficient, provide a
    cautious response asking for the information needed to
    investigate rather than making up an answer.

14. If the customer message is simply a thank-you, acknowledgement,
    or other non-problem message, respond naturally and do not
    invent a support issue.

15. Keep the response concise and professional.

16. Do not mention that you are an AI.

17. Return only the customer-facing reply.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    reply = response.text.strip()

    # Final safety check before returning the response.
    reply = sanitize_generated_reply(
        reply,
        customer_message
    )

    return reply