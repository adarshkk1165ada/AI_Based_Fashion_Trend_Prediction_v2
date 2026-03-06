import os
from collections import Counter, defaultdict
from .predict_clothing import predict_clothing, extract_color_features

IMAGE_FOLDER = "data/raw_data/visual_data/raw_training_images"

clothing_counter = Counter()
category_color = defaultdict(lambda: {"pastel":0,"dark":0,"bright":0})
category_counts = Counter()

image_results = []

for root, dirs, files in os.walk(IMAGE_FOLDER):

    for file in files:

        if file.endswith((".jpg",".jpeg",".png")):

            path = os.path.join(root,file)

            clothing, conf = predict_clothing(path)

            pastel, dark, bright = extract_color_features(path)

            clothing_counter[clothing] += 1
            category_counts[clothing] += 1

            category_color[clothing]["pastel"] += pastel
            category_color[clothing]["dark"] += dark
            category_color[clothing]["bright"] += bright

            image_results.append({
                "image":file,
                "clothing":clothing,
                "confidence":conf,
                "pastel":pastel,
                "dark":dark,
                "bright":bright
            })

total_images = len(image_results)

report = []

report.append("Visual Trend Summary")
report.append("--------------------\n")

report.append("Color style definitions")
report.append("Pastel  : soft/light colors (high brightness, low saturation)")
report.append("Dark    : low brightness colors")
report.append("Bright  : vivid colors with high saturation\n")

report.append("Clothing distribution\n")

for k,v in clothing_counter.items():
    percent=(v/total_images)*100
    report.append(f"{k:12}: {percent:.0f}%")

report.append("\nColor style trends per category\n")

for cat in category_color:

    count = category_counts[cat]

    pastel = (category_color[cat]["pastel"]/count)*100
    dark = (category_color[cat]["dark"]/count)*100
    bright = (category_color[cat]["bright"]/count)*100

    report.append(cat.upper())
    report.append(f"pastel : {pastel:.0f}%")
    report.append(f"dark   : {dark:.0f}%")
    report.append(f"bright : {bright:.0f}%\n")

report.append("\nRecent Image Analysis (latest 20 images)")
report.append("----------------------------------------\n")

for item in image_results[-20:]:

    report.append(f"Image: {item['image']}")
    report.append(f"Clothing type : {item['clothing']}")
    report.append(f"Confidence    : {item['confidence']:.2f}")

    report.append("Color style")
    report.append(f"pastel ratio : {item['pastel']:.2f}")
    report.append(f"dark ratio   : {item['dark']:.2f}")
    report.append(f"bright ratio : {item['bright']:.2f}")
    report.append("----------------------------------------")

report_text = "\n".join(report)

os.makedirs("reports", exist_ok=True)

with open("reports/visual_trend_summary.txt", "w") as f:
    f.write(report_text)

print(report_text)
print("\nSaved to reports/visual_trend_summary.txt")