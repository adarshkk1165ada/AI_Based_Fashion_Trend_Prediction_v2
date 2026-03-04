import os
import requests
import pandas as pd
from tqdm import tqdm

CSV_PATH = "data/raw/pinterest_data.csv"
SAVE_DIR = "data/images"
MAX_IMAGES_PER_KEYWORD = 50

def download_images():
    if not os.path.exists(CSV_PATH):
        print("Pinterest CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    os.makedirs(SAVE_DIR, exist_ok=True)

    grouped = df.groupby("keyword")

    for keyword, group in grouped:
        print(f"\nDownloading images for: {keyword}")

        folder_name = keyword.replace(" ", "_")
        keyword_folder = os.path.join(SAVE_DIR, folder_name)
        os.makedirs(keyword_folder, exist_ok=True)

        count = 0

        for _, row in tqdm(group.iterrows(), total=len(group)):
            if count >= MAX_IMAGES_PER_KEYWORD:
                break

            img_url = row["image_url"]

            try:
                response = requests.get(img_url, timeout=5)
                if response.status_code == 200:
                    file_path = os.path.join(keyword_folder, f"{folder_name}_{count}.jpg")
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    count += 1
            except:
                continue

        print(f"Downloaded {count} images for {keyword}")

if __name__ == "__main__":
    download_images()