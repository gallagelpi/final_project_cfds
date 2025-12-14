import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================================================
# CONFIG
# ==========================================================
DATASET_DIR = "dataset"   # where your raw images are
WORK_DIR = "data"         # train/val will be created here

TRAIN_RATIO = 0.8
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 5
IMG_SIZE = 224

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==========================================================
# CLEAN + RECREATE SPLIT FOLDERS
# ==========================================================
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

os.makedirs(f"{WORK_DIR}/train/alpaca", exist_ok=True)
os.makedirs(f"{WORK_DIR}/train/not_alpaca", exist_ok=True)
os.makedirs(f"{WORK_DIR}/val/alpaca", exist_ok=True)
os.makedirs(f"{WORK_DIR}/val/not_alpaca", exist_ok=True)

# Map original folder names to cleaned class names
CLASS_MAP = {
    "alpaca": "alpaca",
    "not alpaca": "not_alpaca"
}

# ==========================================================
# SPLIT DATASET
# ==========================================================
print("Splitting dataset...")

for class_name in os.listdir(DATASET_DIR):
    src = os.path.join(DATASET_DIR, class_name)

    if class_name not in CLASS_MAP:
        print(f"Skipping unknown folder: {class_name}")
        continue

    dest_name = CLASS_MAP[class_name]

    images = os.listdir(src)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Copy train images
    for img in train_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(WORK_DIR, "train", dest_name, img)
        )

    # Copy val images
    for img in val_imgs:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(WORK_DIR, "val", dest_name, img)
        )

print("Dataset split complete!")
print("Training folders:", os.listdir(f"{WORK_DIR}/train"))
print("Validation folders:", os.listdir(f"{WORK_DIR}/val"))

# ==========================================================
# TRANSFORMS
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================================
# LOAD DATA
# ==========================================================
train_dataset = datasets.ImageFolder(os.path.join(WORK_DIR, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(WORK_DIR, "val"),   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

print("Classes detected:", train_dataset.classes)

# ==========================================================
# MODEL
# ==========================================================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================================
# TRAINING LOOP
# ==========================================================
for epoch in range(EPOCHS):
    model.train()
    train_correct, train_total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

# ==========================================================
# SAVE MODEL
# ==========================================================
torch.save(model.state_dict(), "alpaca_classifier.pt")
print("Model saved as alpaca_classifier.pt 🎉")
