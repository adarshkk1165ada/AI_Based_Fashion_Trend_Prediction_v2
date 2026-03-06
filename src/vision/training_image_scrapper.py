from icrawler.builtin import BingImageCrawler
import os

def scrape_images(query, folder, max_num=300):
    os.makedirs(folder, exist_ok=True)
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=query, max_num=max_num)

if __name__ == "__main__":
    scrape_images("indian saree fashion", "data/training_images/saree", 300)
    scrape_images("indian kurti fashion", "data/training_images/kurti", 300)
    scrape_images("indian streetwear fashion", "data/training_images/streetwear", 300)