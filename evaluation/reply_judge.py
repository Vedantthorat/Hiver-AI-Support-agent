import json

from google import genai


def build_judge_prompt(
    customer_message,
    generated_reply,
    retrieved_cases
):
    """
    Build the prompt used by the LLM-as-a-judge.
    """

    evidence = []

    for _, row in retrieved_cases.iterrows():
        evidence.append(
            f"""
Historical customer:
{row["customer_text"]}

Historical AmazonHelp response:
{row["response_text"]}

Similarity:
{row["similarity_score"]:.3f}
"""
        )

    evidence_text = "\n---\n".join(evidence)

    prompt = f"""
You are evaluating an AI customer-support reply.

Customer message:
{customer_message}

AI-generated reply:
{generated_reply}

Historical evidence used by the system:
{evidence_text}

Important evaluation rule:

Historical conversations are examples of how AmazonHelp handled
similar issues. However, information that was specific to the
historical customer must NOT be transferred to the current customer.

Treat the following as NON-TRANSFERABLE historical details:
- specific dates
- relative dates such as today, tomorrow, yesterday, or Monday
- specific times
- tracking numbers
- order numbers
- customer names
- case-specific delivery estimates
- case-specific deadlines
- actions that were performed for the historical customer

The AI reply should instead use the GENERAL resolution pattern
shown by the historical responses.

For example:

Historical response:
"Please let us know if your order doesn't arrive tomorrow."

General resolution pattern:
"Ask the customer to contact support again if the package still
has not arrived."

The word "tomorrow" is specific to the historical case and should
NOT be considered necessary evidence for the current customer.

Evaluate the AI-generated reply using the following criteria.

1. Helpfulness:
Does the reply appropriately address the customer's issue and
provide useful guidance?

2. Groundedness:
Does the reply reflect the general resolution approach supported
by the historical evidence, without incorrectly transferring
case-specific information?

3. Unsupported claims:
Does the reply invent facts, policies, dates, times, deadlines,
tracking information, guarantees, or actions?

A safe reply that deliberately avoids transferring historical
customer-specific information should NOT be penalized for doing so.

4. Professional tone:
Is the reply concise, respectful, and appropriate for customer support?

Give each criterion a score from 1 to 5.

Scoring:

1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

For unsupported claims:

5 = no unsupported claims
4 = very minor concern
3 = some questionable information
2 = significant unsupported information
1 = major hallucination or fabricated information

Return ONLY valid JSON in exactly this structure:

{{
    "helpfulness": 1,
    "groundedness": 1,
    "unsupported_claims": 1,
    "professional_tone": 1,
    "overall_score": 1,
    "reason": "Brief explanation of the scores."
}}
"""

    return prompt


def judge_reply(
    client,
    customer_message,
    generated_reply,
    retrieved_cases
):
    """
    Evaluate a generated customer-support reply
    using an LLM judge.
    """

    prompt = build_judge_prompt(
        customer_message=customer_message,
        generated_reply=generated_reply,
        retrieved_cases=retrieved_cases
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    raw_output = response.text.strip()

    # Remove Markdown JSON fences if the model adds them.
    raw_output = raw_output.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    try:
        result = json.loads(raw_output)

    except json.JSONDecodeError:
        return {
            "helpfulness": None,
            "groundedness": None,
            "unsupported_claims": None,
            "professional_tone": None,
            "overall_score": None,
            "reason": "Judge returned invalid JSON.",
            "raw_output": raw_output
        }

    return result