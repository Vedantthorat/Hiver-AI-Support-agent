## Top 5 Failure Modes

The prototype intent classifier achieved 54.5% accuracy and
0.404 macro F1 on the 200-example golden set. The largest
confusion patterns were:

### 1. order_status → other (7 errors)

Examples:
- `493427`: "Monday I think"
- `1464284`: "8-9 nov"
- `2954521`: "29/11/2017 Votre colis est en cours de livraison"

**Hypothesis:** Very short or context-dependent messages contain
insufficient semantic information for reliable intent classification.
Date-only messages are particularly difficult because their meaning
depends heavily on the preceding conversation.

### 2. delivery_issue → other (7 errors)

Examples:
- `930108`: "Ich soll mich morgen nochmal melden, wenn die Sendung weiterhin nicht ausgeliefert werden sollte."
- `140066`: "Voici : C20025897313 Parceque j'ai beau me plaindre à chaque fois..."
- `133756`: "Ja, die wissen auch nicht mehr weiter. Der Mitarbeiter gerade hat gesagt, dass er nicht weiß, warum es nicht verschickt wird."

**Hypothesis:** Indirect delivery complaints and multilingual
messages may be harder for the semantic prototype representation
to distinguish from the broad `other` category.

### 3. complaint_or_escalation → other (6 errors)

Examples:
- `1388465`: "Ça fait un moment que je remonte des dysfonctionnements... Pourtant aucun changement :/"
- `970515`: "Sorry, I do not want you to access my account .. I just want an email through which I communicate with the management of the site .."
- `843568`: "Yes but I had to email as I can phone so it'll be to late"

**Hypothesis:** Complaint and escalation language can overlap with
general follow-ups, requests for assistance, and unresolved-status
messages.

### 4. other → complaint_or_escalation (5 errors)

Examples:
- `117906`: "submitted my response."
- `1751594`: "4. the seller by trying to locate the Contact the Seller button..."
- `38799`: "No reply to my last tweet. Thank you for replying now tho :)"

**Hypothesis:** Some messages labelled `other` contain frustration,
lack-of-response language, or partial conversational context that
makes them semantically similar to escalation messages.

### 5. order_status → complaint_or_escalation (4 errors)

Examples:
- `182983`: "What is the status still no response"
- `921197`: "i want to know when u r going to resolve it"
- `1022898`: "I emailed amazon and spoke with someone named George and he said to reply to the email with my order number..."

**Hypothesis:** Customers asking for order status may simultaneously
express frustration about delays or previous support interactions,
causing overlap between status and escalation intents.

## What Is Misleading About My Headline Number?

The headline number of 54.5% accuracy should not be interpreted as
a fully independent estimate of production performance.

The 200-example golden set was also used during the exploratory
intent-taxonomy discovery process. The final prototype classifier
was trained on 135 labeled examples that were kept separate from
the golden examples, so there was no direct training-example overlap.
However, the intent definitions themselves were developed by
inspecting the golden sample.

Therefore, the 54.5% accuracy is best treated as an exploratory
benchmark rather than a clean, independently held-out estimate.

The result is also affected by the class distribution. The
`other` and `delivery_issue` intents are much more frequent than
several smaller intents, which makes accuracy alone insufficient
for judging performance across all classes. This is why macro F1
(40.4%) is reported alongside accuracy.

Finally, the golden set contains only 200 examples. Several intents
have fewer than 10 examples, so per-class metrics for these intents
are statistically noisy.

A stronger evaluation would create a new, independently sampled and
labelled test set after the taxonomy is frozen, with enough examples
per intent to support reliable per-class comparisons.

## What Good Means

For this prototype, a good support agent should satisfy four
requirements:

1. **Intent accuracy**
   - Correctly identify the customer's primary support intent.
   - Macro F1 is important because the intent distribution is
     imbalanced.

2. **Relevant historical evidence**
   - Retrieve previous AmazonHelp cases that are semantically
     relevant to the current customer message.
   - Avoid transferring customer-specific information from
     historical cases.

3. **Safe and useful replies**
   - Draft concise, professional responses.
   - Ground the response in the general resolution patterns found
     in historical AmazonHelp responses.
   - Avoid invented dates, times, order numbers, tracking numbers,
     guarantees, policies, or actions.

4. **Appropriate escalation**
   - Automatically handle low-risk cases when sufficiently
     relevant historical evidence is available.
   - Escalate higher-risk intents or cases with weak evidence
     to a human.

## What We Did Not Build

This project is a prototype rather than a production customer-support
system. The following capabilities were intentionally out of scope:

- Direct access to Amazon customer accounts or order systems.
- Real-time order, shipment, refund, or payment information.
- Ability to actually execute refunds, cancellations, or account
  changes.
- Authentication or customer identity verification.
- Full conversation-state tracking across long multi-turn threads.
- Production-grade monitoring and observability.
- Human-agent workflow integration.
- Automatic policy verification against a live Amazon policy source.
- Large-scale production deployment.
- Fully independent evaluation after freezing the intent taxonomy.

The system therefore produces **support drafts and escalation
decisions**, rather than performing real customer-account actions.

## Results

### Intent Classification

The intent-classification models were evaluated on the 200-example
golden set. The semantic prototype classifier was trained on 135
labeled examples, while the majority and TF-IDF baselines were
evaluated using the same evaluation framework.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 24.5% | — | — |
| TF-IDF + Logistic Regression | 38.0% | 11.0% | — |
| Semantic Prototype Classifier | **54.5%** | **40.4%** | **52.8%** |

The semantic prototype approach improved accuracy over the majority
baseline by 30.0 percentage points and over the TF-IDF baseline by
16.5 percentage points.

Macro F1 is substantially lower than accuracy because performance is
uneven across the 11 intents, particularly for intents with fewer
examples.

### Retrieval

The retrieval system used `all-MiniLM-L6-v2` sentence embeddings
with a FAISS inner-product index over 90,708 filtered historical
customer-support pairs.

| Metric | Result |
|---|---:|
| Recall@1 | 78.89% |
| Recall@3 | 81.67% |
| Recall@5 | 82.22% |

Recall@3 is close to Recall@5, so the pipeline uses the top 3
historical cases as generation evidence.

The retrieval evaluation contained 180 examples because messages
below the minimum text-length threshold were excluded. The metric
also assumes that the exact historically linked response is the
correct target, although multiple historical responses may be
reasonable for a similar customer issue. Therefore, retrieval recall
should be interpreted as an approximate diagnostic rather than a
complete measure of evidence quality.

### Reply Quality Pilot

A 4-example pilot was used to verify the LLM-as-judge and
human-agreement evaluation pipeline.

The LLM judge evaluates:

- Helpfulness
- Groundedness
- Unsupported claims
- Professional tone
- Overall quality

The pilot achieved 75% raw human–LLM agreement. Cohen's kappa was
0.0, but this result is not treated as a reliable estimate because
the pilot contains only four examples and has a highly skewed label
distribution.

Therefore, the pilot demonstrates that the evaluation harness works,
but it is not presented as a statistically strong estimate of
production reply quality.

## System Architecture

The system follows a four-stage support pipeline:

```text
Customer Message
       |
       v
+----------------------+
| Intent Classification|
+----------------------+
       |
       v
+----------------------+
| Historical Retrieval |
| Sentence Embeddings  |
| + FAISS              |
+----------------------+
       |
       v
+----------------------+
| Grounded Reply       |
| Generation           |
| Gemini 2.5 Flash     |
+----------------------+
       |
       v
+----------------------+
| Escalation Decision  |
| Auto / Human         |
+----------------------+
       |
       v
Customer-support draft