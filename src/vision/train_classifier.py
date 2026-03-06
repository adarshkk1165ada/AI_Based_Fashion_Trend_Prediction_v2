import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure model folder exists
os.makedirs("models", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# IMPORTANT: Correct dataset path
dataset = datasets.ImageFolder("data/training_images", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = timm.create_model("efficientnet_b0", pretrained=True)

# Replace classifier head for our number of classes
model.classifier = nn.Linear(model.classifier.in_features, len(dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(3):  # demo training
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed")

torch.save(model.state_dict(), "models/clothing_classifier.pth")
print("Model saved successfully")