import os
from torch.utils.data import DataLoader
from torchvision import datasets


def load_datasets(work_dir: str, transform):
    """Load training and validation datasets from directory."""
    train_dir = os.path.join(work_dir, "train")
    val_dir = os.path.join(work_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"Train or Val directory not found in {work_dir}. "
            "Please run split_dataset first."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size: int):
    """Create PyTorch DataLoaders for training and validation."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


class LoadData:
    """Class to load image datasets and create DataLoaders."""

    def __init__(self, work_dir: str, transform):
        """Initialize with dataset directory and image transforms."""
        self.work_dir = work_dir
        self.transform = transform

    def load_and_split(self, batch_size: int):
        """
        Load datasets and create DataLoaders.

        Returns:
            tuple: (train_loader, val_loader, class_names)
        """
        train_dataset, val_dataset = load_datasets(self.work_dir, self.transform)
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, batch_size
        )

        print("Classes detected:", train_dataset.classes)

        return train_loader, val_loader, train_dataset.classes
