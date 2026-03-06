import os
import pandas as pd

from src.vision.color_analysis import run_color_analysis
from src.vision.image_analyser_v2 import run_deep_feature_extraction
from src.features.feature_fusion import fuse_cv_nlp


RAW_IMAGE_PATH = "data/raw_data/visual_data/raw_training_images"
PROCESSED_VISUAL_PATH = "data/processed_data/visual_data/processed_visual_features"


def run_cv_pipeline():

    print("Running CV pipeline...")

    os.makedirs(PROCESSED_VISUAL_PATH, exist_ok=True)

    # 1️⃣ Color Features
    color_df = run_color_analysis(RAW_IMAGE_PATH)

    color_output = os.path.join(
        PROCESSED_VISUAL_PATH,
        "image_color_features.csv"
    )

    color_df.to_csv(color_output, index=False)

    # 2️⃣ Deep Features
    deep_df = run_deep_feature_extraction(RAW_IMAGE_PATH)

    deep_output = os.path.join(
        PROCESSED_VISUAL_PATH,
        "image_deep_features.csv"
    )

    deep_df.to_csv(deep_output, index=False)
    print("Color DF columns:", color_df.columns)
    print("Deep DF columns:", deep_df.columns)
    # 3️⃣ Merge both visual features
    final_df = pd.merge(color_df, deep_df, on="category")

    final_output = os.path.join(
        PROCESSED_VISUAL_PATH,
        "final_cv_features.csv"
    )

    final_df.to_csv(final_output, index=False)

    print("CV pipeline completed.")


if __name__ == "__main__":

    run_cv_pipeline()

    print("Running feature fusion...")

    fuse_cv_nlp()

    print("Full pipeline finished.")    