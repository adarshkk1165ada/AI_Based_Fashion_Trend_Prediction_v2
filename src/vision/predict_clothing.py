import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import cv2

classes = ['kurti', 'saree', 'streetwear']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, len(classes))
model.load_state_dict(torch.load("models/clothing_classifier.pth", map_location=device))
model.to(device)
model.eval()


def predict_clothing(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return classes[predicted.item()], confidence.item()


def extract_color_features(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    s = s / 255.0
    v = v / 255.0

    pastel = np.logical_and(v > 0.7, s < 0.4)
    dark = v < 0.3
    bright = np.logical_and(v > 0.5, s > 0.6)

    total_pixels = v.size

    pastel_ratio = np.sum(pastel) / total_pixels
    dark_ratio = np.sum(dark) / total_pixels
    bright_ratio = np.sum(bright) / total_pixels

    return pastel_ratio, dark_ratio, bright_ratio


def dominant_color(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (150, 150))
    pixels = image.reshape((-1, 3))

    pixels = np.float32(pixels)

    _, labels, palette = cv2.kmeans(
        pixels, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    dominant = palette[np.argmax(np.bincount(labels.flatten()))]

    b, g, r = dominant
    return int(r), int(g), int(b)


if __name__ == "__main__":

    image_path = "test.jpg"

    clothing, conf = predict_clothing(image_path)

    pastel, dark, bright = extract_color_features(image_path)

    r, g, b = dominant_color(image_path)

    print("\nPrediction Results")
    print("-------------------")

    print(f"Clothing type : {clothing}")
    print(f"Confidence    : {conf:.2f}")

    print("\nDominant Color (RGB)")
    print(f"R:{r}  G:{g}  B:{b}")

    print("\nColor Style")
    print(f"Pastel ratio : {pastel:.2f}")
    print(f"Dark ratio   : {dark:.2f}")
    print(f"Bright ratio : {bright:.2f}")