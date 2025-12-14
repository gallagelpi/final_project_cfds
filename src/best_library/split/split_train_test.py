import os
import shutil
import random

def split_dataset(dataset_dir, work_dir, train_ratio=0.8):
    """
    Splits the dataset into train and validation sets.
    
    Args:
        dataset_dir (str): Path to the raw dataset.
        work_dir (str): Path where the split dataset will be stored.
        train_ratio (float): Ratio of images to use for training.
    """
    # Clean up existing work directory
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)

    # Define class mapping
    class_map = {
        "alpaca": "alpaca",
        "not alpaca": "not_alpaca"
    }

    # Create directories
    for category in ["train", "val"]:
        for class_name in class_map.values():
            os.makedirs(os.path.join(work_dir, category, class_name), exist_ok=True)

    print("Splitting dataset...")

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist!")
        return

    for class_name in os.listdir(dataset_dir):
        src = os.path.join(dataset_dir, class_name)
        
        if not os.path.isdir(src):
            continue

        if class_name not in class_map:
            print(f"Skipping unknown folder: {class_name}")
            continue

        dest_name = class_map[class_name]

        images = [img for img in os.listdir(src) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy train images
        for img in train_imgs:
            shutil.copy(
                os.path.join(src, img),
                os.path.join(work_dir, "train", dest_name, img)
            )

        # Copy val images
        for img in val_imgs:
            shutil.copy(
                os.path.join(src, img),
                os.path.join(work_dir, "val", dest_name, img)
            )

    print("Dataset split complete!")
    if os.path.exists(os.path.join(work_dir, "train")):
        print("Training folders:", os.listdir(os.path.join(work_dir, "train")))
    if os.path.exists(os.path.join(work_dir, "val")):
        print("Validation folders:", os.listdir(os.path.join(work_dir, "val")))