from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
from datetime import datetime
import os

KEYWORDS = ["saree fashion", "kurti pastel", "streetwear india"]
SCROLLS = 15
SAVE_PATH = "data/raw/pinterest_data.csv"

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--log-level=3")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver

def scrape_pinterest():
    driver = setup_driver()
    all_data = []

    for keyword in KEYWORDS:
        print(f"\nScraping Pinterest for: {keyword}")

        search_url = f"https://www.pinterest.com/search/pins/?q={keyword.replace(' ', '%20')}"
        driver.get(search_url)
        time.sleep(5)

        for _ in range(SCROLLS):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

        pins = driver.find_elements(By.CSS_SELECTOR, "img[src*='pinimg.com']")
        print(f"Found {len(pins)} images for {keyword}")

        for pin in pins:
            img_url = pin.get_attribute("src")
            if img_url:
                all_data.append({
                    "keyword": keyword,
                    "image_url": img_url,
                    "scrape_date": datetime.now().strftime("%Y-%m-%d")
                })

    driver.quit()

    if not all_data:
        print("No data collected.")
        return

    df = pd.DataFrame(all_data)
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(SAVE_PATH, index=False)
    print("\nPinterest data saved successfully.")

if __name__ == "__main__":
    scrape_pinterest()