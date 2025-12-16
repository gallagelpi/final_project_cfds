import os
import torch
from src.best_library.split.split_train_test import split_dataset
from src.best_library.preprocessing.preprocessing import Preprocessing
from src.best_library.data.load_data import load_data
from src.best_library.model.model_resnet18 import ModelResnet18
from src.best_library.features.feature_engineering import compute_dataset_stats

# CONFIG
DATASET_DIR = "dataset"   # where your raw images are
WORK_DIR = "data"         # train/val will be created here
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 5
IMG_SIZE = 224

def main():
    print("Starting pipeline...")
    
    # 1. Split Data
    split_dataset(DATASET_DIR, WORK_DIR)
    
    # 2. Feature Engineering (Optional - just to show usage)
    # We use the 'train' folder created by split_dataset
    train_dir = os.path.join(WORK_DIR, "train")
    if os.path.exists(train_dir):
        compute_dataset_stats(train_dir, img_size=IMG_SIZE)
    
    # 3. Preprocessing
    preprocessing = Preprocessing(img_size=IMG_SIZE)
    transform = preprocessing.get_transform()
    
    # 4. Load Data
    try:
        train_loader, val_loader, class_names = load_data(WORK_DIR, BATCH_SIZE, transform)
    except FileNotFoundError as e:
        print(e)
        return

    # 5. Build Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model_api = ModelResnet18(device=device, class_names=class_names)
    model = model_api.build_model()

    # 6. Train Model
    model_api.train_model(model, train_loader, val_loader, EPOCHS, LR)

if __name__ == "__main__":
    main()
