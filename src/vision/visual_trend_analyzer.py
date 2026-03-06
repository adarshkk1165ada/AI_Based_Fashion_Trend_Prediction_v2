import os
from collections import Counter, defaultdict
import cv2
import numpy as np
from predict_clothing import predict_clothing, extract_color_features

IMAGE_FOLDER = "data/images"

clothing_counter = Counter()
category_color = defaultdict(lambda: {"pastel":0,"dark":0,"bright":0})
category_counts = Counter()

image_results = []


def dominant_color_rgb(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(150,150))

    pixels = img.reshape((-1,3))
    pixels = np.float32(pixels)

    _, labels, palette = cv2.kmeans(
        pixels,3,None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,200,.1),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    dominant = palette[np.argmax(np.bincount(labels.flatten()))]
    b,g,r = dominant
    return int(r),int(g),int(b)


def rgb_to_color_name(r,g,b):

    if r>200 and g>200 and b>200:
        return "white"

    if r<60 and g<60 and b<60:
        return "black"

    if r>g and r>b:
        if r<120:
            return "dark red"
        return "red"

    if g>r and g>b:
        if g<120:
            return "dark green"
        return "green"

    if b>r and b>g:
        if b<120:
            return "dark blue"
        return "blue"

    if r>150 and g>150 and b<80:
        return "yellow"

    if r>150 and b>150:
        return "purple"

    return "mixed color"


for root,dirs,files in os.walk(IMAGE_FOLDER):
    for file in files:

        if file.endswith((".jpg",".jpeg",".png")):

            path = os.path.join(root,file)

            clothing,conf = predict_clothing(path)

            pastel,dark,bright = extract_color_features(path)

            r,g,b = dominant_color_rgb(path)

            color_name = rgb_to_color_name(r,g,b)

            clothing_counter[clothing]+=1
            category_counts[clothing]+=1

            category_color[clothing]["pastel"]+=pastel
            category_color[clothing]["dark"]+=dark
            category_color[clothing]["bright"]+=bright

            image_results.append({
                "image":file,
                "clothing":clothing,
                "confidence":conf,
                "r":r,"g":g,"b":b,
                "color":color_name,
                "pastel":pastel,
                "dark":dark,
                "bright":bright
            })


total_images=len(image_results)

print("\nVisual Trend Summary")
print("--------------------")

print("\nColor style definitions")
print("Pastel  : soft/light colors (high brightness, low saturation)")
print("Dark    : low brightness colors")
print("Bright  : vivid colors with high saturation\n")

print("Clothing distribution")

for k,v in clothing_counter.items():
    percent=(v/total_images)*100
    print(f"{k:12}: {percent:.0f}%")

print("\nColor style trends per category\n")

for cat in category_color:

    count=category_counts[cat]

    pastel=(category_color[cat]["pastel"]/count)*100
    dark=(category_color[cat]["dark"]/count)*100
    bright=(category_color[cat]["bright"]/count)*100

    print(cat.upper())
    print(f"pastel : {pastel:.0f}%")
    print(f"dark   : {dark:.0f}%")
    print(f"bright : {bright:.0f}%\n")


print("\nRecent Image Analysis (latest 20 images)")
print("----------------------------------------\n")

for item in image_results[-20:]:

    print(f"Image: {item['image']}")
    print(f"Clothing type : {item['clothing']}")
    print(f"Confidence    : {item['confidence']:.2f}")

    print(f"\nDominant RGB  : ({item['r']},{item['g']},{item['b']})")
    print(f"Dominant color: {item['color']}")

    print("\nColor style")
    print(f"pastel ratio : {item['pastel']:.2f}")
    print(f"dark ratio   : {item['dark']:.2f}")
    print(f"bright ratio : {item['bright']:.2f}")

    print("\n----------------------------------------\n")