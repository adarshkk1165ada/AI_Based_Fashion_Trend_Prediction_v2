import os
import cv2
import numpy as np
import pandas as pd


def analyze_image_colors(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(image)

    s = s / 255.0
    v = v / 255.0

    pastel_pixels = np.logical_and(v > 0.7, s < 0.4)
    dark_pixels = v < 0.3
    bright_pixels = np.logical_and(v > 0.5, s > 0.6)

    total_pixels = v.size

    return (
        np.sum(pastel_pixels) / total_pixels,
        np.sum(dark_pixels) / total_pixels,
        np.sum(bright_pixels) / total_pixels
    )


def analyze_category(folder_path, category_name):
    pastel_total = 0
    dark_total = 0
    bright_total = 0
    image_count = 0

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, file)

            result = analyze_image_colors(image_path)

            if result:
                pastel, dark, bright = result
                pastel_total += pastel
                dark_total += dark
                bright_total += bright
                image_count += 1

    if image_count == 0:
        return None

    return {
        "category": category_name,
        "pastel_ratio": pastel_total / image_count,
        "dark_ratio": dark_total / image_count,
        "bright_ratio": bright_total / image_count,
        "image_count": image_count
    }


def run_color_analysis(base_path="data/images"):
    results = []

    for category in os.listdir(base_path):
        folder_path = os.path.join(base_path, category)

        if os.path.isdir(folder_path):
            features = analyze_category(folder_path, category)

            if features:
                results.append(features)

    return pd.DataFrame(results)
if __name__ == "__main__":
    df = run_color_analysis("data/images")
    df.to_csv("data/images/features_extracted_images/image_color_features.csv", index=False)
    print("Color features saved.")