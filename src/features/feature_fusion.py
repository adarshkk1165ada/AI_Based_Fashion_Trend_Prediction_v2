import pandas as pd
import os


def fuse_cv_nlp():

    print("Loading NLP features...")
    nlp_path = "data/processed_data/nlp_data/nlp_features.csv"

    print("Loading CV features...")
    cv_path = "data/processed_data/visual_data/processed_visual_features/final_cv_features.csv"

    nlp_df = pd.read_csv(nlp_path)
    cv_df = pd.read_csv(cv_path)

    print("Preparing NLP aggregation...")

    # Aggregate NLP signals by category
    nlp_agg = nlp_df.groupby("category").agg({
        "sentiment_score": "mean",
        "engagement_score": "mean",
        "keyword_count": "mean",
        "trend_score": "mean"
    }).reset_index()

    print("Normalizing CV category names...")

    # Normalize CV category names to match NLP categories
    cv_df["category"] = cv_df["category"].replace({
        "kurti_pastel": "kurti",
        "saree_fashion": "saree",
        "streetwear_india": "streetwear"
    })

    print("Merging CV and NLP features...")

    fused_df = pd.merge(
        cv_df,
        nlp_agg,
        on="category",
        how="inner"
    )

    os.makedirs("data/fused_features", exist_ok=True)

    output_path = "data/fused_features/fashion_trend_fused_dataset.csv"

    fused_df.to_csv(output_path, index=False)

    print("Feature fusion completed.")
    print(f"Fused dataset saved to: {output_path}")


if __name__ == "__main__":
    fuse_cv_nlp()