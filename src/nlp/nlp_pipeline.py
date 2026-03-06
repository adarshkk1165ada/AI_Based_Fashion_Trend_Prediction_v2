import pandas as pd

from .text_preprocessing import clean_text
from .sentiment_analysis import get_sentiment_score
from .keywords import extract_keywords
from .topic_modeling import run_lda


def calculate_engagement(row):
    likes = row["likes"]
    replies = row["replies"]
    score = (likes * 0.7 + replies * 0.3) / 5000
    return round(score, 3)


def run_nlp_pipeline():

    print("Loading dataset...")

    df = pd.read_csv("data/nlp_raw/fashion_trend_dataset_2000_nlp_sithin.csv")

    print("Cleaning text...")
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    df["text_length"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["hashtag_count"] = df["hashtags"].apply(lambda x: len(str(x).split()))

    print("Running sentiment analysis...")
    df["sentiment_score"] = df["clean_text"].apply(get_sentiment_score)

    print("Extracting keywords...")
    keywords = extract_keywords(df["clean_text"].tolist())
    keyword_list = [k[0] for k in keywords]

    df["keyword_list"] = df["clean_text"].apply(
        lambda x: ",".join([w for w in keyword_list if w in x][:5])
    )

    df["keyword_count"] = df["keyword_list"].apply(
        lambda x: len(x.split(",")) if x else 0
    )

    print("Running topic modeling...")
    topics, topic_matrix = run_lda(df["clean_text"].tolist())
    df["topic_id"] = topic_matrix.argmax(axis=1)

    print("Calculating engagement score...")
    df["engagement_score"] = df.apply(calculate_engagement, axis=1)

    print("Calculating trend score...")
    df["trend_score"] = (
        0.35 * df["sentiment_score"]
        + 0.35 * df["engagement_score"]
        + 0.30 * df["keyword_count"]
    )

    print("Saving processed dataset...")
    df.to_csv("data/nlp_processed/nlp_features.csv", index=False)

    print("NLP pipeline completed.")


if __name__ == "__main__":
    run_nlp_pipeline()