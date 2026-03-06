from src.vision.color_analysis import run_color_analysis
from src.vision.image_analyser_v2 import run_deep_feature_extraction
import pandas as pd


def run_cv_pipeline():
    base_path = "data/images"
    output_path = "data/images/features_extracted_images/"

    # 1️⃣ Color Features
    color_df = run_color_analysis(base_path)
    color_df.to_csv(output_path + "image_color_features.csv", index=False)

    # 2️⃣ Deep Features
    deep_df = run_deep_feature_extraction(base_path)
    deep_df.to_csv(output_path + "image_deep_features.csv", index=False)

    # 3️⃣ Merge
    final_df = pd.merge(color_df, deep_df, on="category")
    final_df.to_csv(output_path + "final_cv_features.csv", index=False)

    print("CV pipeline completed.")


if __name__ == "__main__":
    run_cv_pipeline()


# Feature fusion will be run separately after both CV and NLP pipelines have completed.
from src.features.feature_fusion import fuse_cv_nlp

if __name__ == "__main__":

    print("Running feature fusion...")

    fuse_cv_nlp()

    print("Pipeline finished.")    