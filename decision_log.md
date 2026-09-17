# Decision Log

## 1. Selected AmazonHelp as the target brand
Decision:
Selected AmazonHelp instead of another brand in the dataset.

Why:
AmazonHelp had the largest number of customer messages directly connected to its support tweets among the candidate brands. This provides a larger historical support corpus and better coverage of different customer issues.

---

## 2. Defined a small intent taxonomy from AmazonHelp data
Decision:
Used 11 intents rather than a large predefined taxonomy.

Why:
The assignment asks for a small set of intents defined from the data. The taxonomy was created by manually reviewing a sample of AmazonHelp customer messages and grouping recurring support issues.

---

## 3. Added an `other` intent
Decision:
Created an `other` category.

Why:
Some customer messages were too ambiguous, short, contextual, or did not clearly belong to another support category. Forcing these messages into a specific intent would create misleading labels.

---

## 4. Merged delivery-date-change into delivery_issue
Decision:
Mapped `delivery_date_change` to `delivery_issue`.

Why:
Both categories concern delivery-related problems, and the 200-example sample did not justify maintaining a separate class for delivery-date changes.

---

## 5. Merged payment/account and price/billing
Decision:
Mapped `payment_or_account` and `price_or_billing` to `payment_or_billing`.

Why:
The two categories had substantial overlap in the sampled messages and were not sufficiently distinct for a small intent taxonomy.

---

## 6. Used a 200-example golden evaluation set
Decision:
Created a 200-example manually labelled golden set.

Why:
The assignment requires 150–250 hand-labelled examples. 200 provides a reasonable balance between evaluation coverage and manual labelling effort.

---

## 7. Used TF-IDF + Logistic Regression as the simple ML baseline
Decision:
Used TF-IDF with bigrams and Logistic Regression.

Why:
This is a simple, interpretable text-classification baseline that provides a meaningful comparison against the semantic classifier.

---

## 8. Included a majority-class baseline
Decision:
Used the majority-class predictor as a trivial baseline.

Why:
It establishes the minimum performance level that a classifier must beat. The prototype classifier substantially outperformed this baseline.

---

## 9. Chose semantic prototypes for the main intent classifier
Decision:
Used sentence embeddings and one centroid/prototype per intent.

Why:
Customer-support messages can express the same issue using different words. Semantic embeddings allow messages with similar meaning to be compared even when their exact vocabulary differs.

---

## 10. Tested and rejected weighted KNN
Decision:
Did not use similarity-weighted KNN as the final classifier.

Why:
Weighted KNN performed worse than the prototype approach on the evaluation set. The weighted version achieved approximately 46.5% accuracy compared with 54.5% for the prototype classifier.

---

## 11. Filtered very short retrieval messages
Decision:
Removed customer messages with fewer than 20 characters from the retrieval corpus.

Why:
Very short messages often provide insufficient context for useful semantic retrieval. This reduced the retrieval corpus from 100,503 to 90,708 cases.

---

## 12. Used FAISS for historical case retrieval
Decision:
Used normalized sentence embeddings with a FAISS inner-product index.

Why:
FAISS provides efficient similarity search over the large historical support corpus. Normalized embeddings allow inner product to represent cosine similarity.

---

## 13. Used top-3 historical cases
Decision:
The support agent retrieves the top 3 historical cases.

Why:
Recall@3 was 81.67%, compared with 82.22% for Recall@5. The small improvement from adding two more cases was not large enough to justify adding more evidence to the generation prompt.

---

## 14. Sanitized historical responses before generation
Decision:
Removed case-specific information such as dates, times, and long numeric IDs from historical responses before passing them to the LLM.

Why:
Historical responses may contain information specific to the original customer, such as delivery times or order/tracking numbers. Passing these details directly to the generator can cause the model to incorrectly reuse them for a new customer.

---

## 15. Added an escalation layer
Decision:
High-risk intents and low-similarity cases are escalated to a human.

Why:
A support agent should not automatically handle every message. Complaints, billing, refunds, and product issues can require human judgement, while weak retrieval evidence indicates that the system may not have sufficient historical support to answer safely.

---

## Evaluation Caveat

The 200-example golden set was initially used during intent discovery and taxonomy development. Although the classifier training examples used for the reported prototype evaluation exclude the golden tweet IDs, the taxonomy itself was informed by the same sample.

Therefore, the reported 54.5% accuracy should be treated as an evaluation result with a methodology caveat rather than as a perfectly independent benchmark.