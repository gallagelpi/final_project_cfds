import os
import shutil
import random

def create_split_dirs(work_dir: str, class_map: dict):
    """Create train/val directory structure."""
    for split in ["train", "val"]:
        for class_name in class_map.values():
            os.makedirs(os.path.join(work_dir, split, class_name), exist_ok=True)


def list_images(directory: str):
    """List image files in a directory."""
    return [
        img for img in os.listdir(directory)
        if img.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def copy_images(image_list, src_dir, dst_dir):
    """Copy images from source to destination."""
    for img in image_list:
        shutil.copy(
            os.path.join(src_dir, img),
            os.path.join(dst_dir, img)
        )

class DatasetSplitter:
    """Class to split image dataset into train and validation sets."""

    def __init__(self, dataset_dir: str, work_dir: str, train_ratio: float = 0.8, seed: int = 42):
        """
        Initialize the dataset splitter.

        Args:
            dataset_dir (str): Path to raw dataset.
            work_dir (str): Output directory for split dataset.
            train_ratio (float): Ratio of training data.
        """
        self.seed = seed
        self.dataset_dir = dataset_dir
        self.work_dir = work_dir
        self.train_ratio = train_ratio

        self.class_map = {
            "alpaca": "alpaca",
            "not alpaca": "not_alpaca"
        }

    def split(self):
        """Split dataset into train and validation sets."""
        print("Splitting dataset...")

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                f"Dataset directory {self.dataset_dir} does not exist!"
            )

        # Clean output directory
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)

        create_split_dirs(self.work_dir, self.class_map)

        for class_name in os.listdir(self.dataset_dir):
            src_dir = os.path.join(self.dataset_dir, class_name)

            if not os.path.isdir(src_dir):
                continue

            if class_name not in self.class_map:
                print(f"Skipping unknown folder: {class_name}")
                continue

            dst_class = self.class_map[class_name]
            images = list_images(src_dir)
            random.seed(self.seed)
            random.shuffle(images)

            split_idx = int(len(images) * self.train_ratio)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            copy_images(
                train_imgs,
                src_dir,
                os.path.join(self.work_dir, "train", dst_class)
            )

            copy_images(
                val_imgs,
                src_dir,
                os.path.join(self.work_dir, "val", dst_class)
            )

        print("Dataset split complete!")
        print("Training folders:", os.listdir(os.path.join(self.work_dir, "train")))
        print("Validation folders:", os.listdir(os.path.join(self.work_dir, "val")))
