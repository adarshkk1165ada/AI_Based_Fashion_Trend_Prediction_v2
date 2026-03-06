import pandas as pd


def fuse_cv_nlp():

    print("Loading NLP features...")
    nlp = pd.read_csv("data/nlp_processed/nlp_features.csv")

    print("Loading CV features...")
    cv = pd.read_csv(
        "data/images/features_extracted_images/final_cv_features.csv"
    )

    print("Combining CV and NLP signals...")

    # cross join so both signals appear together
    nlp["key"] = 1
    cv["key"] = 1

    df = pd.merge(nlp, cv, on="key").drop("key", axis=1)

    print("Saving fused dataset...")

    df.to_csv(
        "data/fused_features/fashion_trend_fused_dataset.csv",
        index=False
    )

    print("Feature fusion complete.")


if __name__ == "__main__":
    fuse_cv_nlp()