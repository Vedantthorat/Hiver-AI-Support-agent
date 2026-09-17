import pandas as pd


def build_customer_response_pairs(amazon_customers, amazon_tweets):
    """
    Build customer -> AmazonHelp response pairs.

    A pair is created when a customer tweet directly replies
    to an AmazonHelp tweet.
    """

    amazon_responses = amazon_tweets[
        amazon_tweets["author_id"] == "AmazonHelp"
    ][
        ["tweet_id", "text"]
    ].copy()

    amazon_responses = amazon_responses.rename(
        columns={
            "tweet_id": "response_tweet_id",
            "text": "response_text"
        }
    )

    pairs = amazon_customers[
        ["tweet_id", "text", "in_response_to_tweet_id"]
    ].copy()

    pairs = pairs.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "text": "customer_text",
            "in_response_to_tweet_id": "response_tweet_id"
        }
    )

    pairs["response_tweet_id"] = pd.to_numeric(
        pairs["response_tweet_id"],
        errors="coerce"
    )

    pairs = pairs.merge(
        amazon_responses,
        on="response_tweet_id",
        how="inner"
    )

    pairs = pairs.dropna(
        subset=["customer_text", "response_text"]
    )

    return pairs.reset_index(drop=True)