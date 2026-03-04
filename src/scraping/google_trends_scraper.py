from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# CONFIG
# -----------------------------
KEYWORDS = ["saree", "kurti", "jeans", "tshirt", "hoodie", "streetwear"]
GEO = "IN"  # India
TIMEFRAME = "now 7-d"  # last 7 days (run daily)
SAVE_PATH = "data/raw/google_trends_data.csv"

# -----------------------------
# SCRAPER FUNCTION
# -----------------------------
def fetch_google_trends():
    pytrends = TrendReq(hl="en-IN", tz=330)

    all_data = []

    for keyword in KEYWORDS:
        print(f"Fetching Google Trends for: {keyword}")

        pytrends.build_payload([keyword], geo=GEO, timeframe=TIMEFRAME)
        df = pytrends.interest_over_time()

        if df.empty:
            continue

        df = df.reset_index()
        df["keyword"] = keyword
        df = df.rename(columns={keyword: "search_index"})

        all_data.append(df[["date", "keyword", "search_index"]])

    if not all_data:
        print("No data fetched.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    # Add metadata
    final_df["scrape_date"] = datetime.now().strftime("%Y-%m-%d")

    # Save
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(SAVE_PATH):
        existing = pd.read_csv(SAVE_PATH)
        final_df = pd.concat([existing, final_df], ignore_index=True)

    final_df.to_csv(SAVE_PATH, index=False)
    print("Google Trends data saved successfully.")


if __name__ == "__main__":
    fetch_google_trends()