from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from datetime import datetime


def scrape_category(driver, wait, keyword, pages=3):
    all_products = []

    for page in range(1, pages + 1):
        url = f"https://www.myntra.com/{keyword}?p={page}"
        print(f"🔍 Scraping {keyword} | Page {page}")

        driver.get(url)

        try:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "product-base")))
        except:
            print("⚠️ Retry loading...")
            time.sleep(5)

        products = driver.find_elements(By.CLASS_NAME, "product-base")

        for product in products:
            try:
                brand = product.find_element(By.CLASS_NAME, "product-brand").text
                name = product.find_element(By.CLASS_NAME, "product-product").text
                price = product.find_element(By.CLASS_NAME, "product-discountedPrice").text

                try:
                    rating = product.find_element(By.CLASS_NAME, "product-ratingsContainer").text
                except:
                    rating = None

                scrape_date = datetime.now().strftime("%Y-%m-%d")

                all_products.append([
                    scrape_date,
                    brand,
                    name,
                    price,
                    rating,
                    keyword
                ])

            except:
                continue

        time.sleep(2)

    return all_products


def scrape_myntra_full():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    wait = WebDriverWait(driver, 15)

    categories = [
        "kurta",
        "saree",
        "tshirt",
        "jeans",
        "hoodie",
        "sneakers"
    ]

    all_data = []

    for category in categories:
        category_data = scrape_category(driver, wait, category, pages=3)
        all_data.extend(category_data)

    driver.quit()

    df = pd.DataFrame(all_data, columns=[
        "scrape_date",
        "brand",
        "product_name",
        "price",
        "rating",
        "category"
    ])

    return df


if __name__ == "__main__":
    df = scrape_myntra_full()

    output_file = "data/raw/myntra_full_dataset.csv"

    # Append if file exists (to build historical dataset)
    try:
        existing = pd.read_csv(output_file)
        df = pd.concat([existing, df], ignore_index=True)
    except:
        pass

    df.to_csv(output_file, index=False)

    print("✅ Myntra dataset updated")
    print(f"📊 Total rows now: {len(df)}")