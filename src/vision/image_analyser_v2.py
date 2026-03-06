import os
import torch
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms


# Load pretrained EfficientNet (feature extractor)
model = timm.create_model("efficientnet_b0", pretrained=True)
model.reset_classifier(0)
model.eval()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def extract_deep_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.squeeze().numpy()


def run_deep_feature_extraction(
        base_path="data/raw_data/visual_data/raw_training_images"
):

    results = []

    for category in os.listdir(base_path):

        folder_path = os.path.join(base_path, category)

        if not os.path.isdir(folder_path):
            continue

        feature_list = []

        for file in os.listdir(folder_path):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(folder_path, file)

                feat = extract_deep_features(image_path)

                feature_list.append(feat)

        if len(feature_list) == 0:
            continue

        mean_features = np.mean(feature_list, axis=0)

        row = {"category": category}

        for i, val in enumerate(mean_features):
            row[f"deep_feat_{i}"] = val

        results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":

    df = run_deep_feature_extraction(
        "data/raw_data/visual_data/raw_training_images"
    )

    os.makedirs(
        "data/processed_data/visual_data/processed_visual_features",
        exist_ok=True
    )

    df.to_csv(
        "data/processed_data/visual_data/processed_visual_features/image_deep_features.csv",
        index=False
    )

    print("Deep visual features saved.")