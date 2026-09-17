# Amazon Customer Support AI Agent

An AI customer-support agent built using real customer-support
conversations from the Customer Support on Twitter dataset.

The system is designed around four stages:

1. Classify the incoming customer message into a small set of
   support intents.
2. Retrieve historically similar AmazonHelp customer-support cases.
3. Generate a grounded customer-facing reply based on historical
   AmazonHelp resolution patterns.
4. Decide whether the request should be auto-handled or escalated
   to a human.

## Architecture

```text
Customer Message
       |
       v
Intent Classification
       |
       v
Historical Case Retrieval
       |
       v
Grounded Reply Generation
       |
       v
Escalation Decision
       |
       v
Support Draft + Decision