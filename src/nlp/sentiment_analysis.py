from transformers import pipeline

# Load sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def get_sentiment_score(text):
    """
    Calculate sentiment score for text
    """

    result = sentiment_model(text[:512])[0]

    score = result["score"]

    if result["label"] == "NEGATIVE":
        score = -score

    return score