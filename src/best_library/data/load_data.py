import os
from torch.utils.data import DataLoader
from torchvision import datasets

def load_data(work_dir, batch_size, transform):
    """
    Loads the training and validation data.

    Args:
        work_dir (str): Path to the split dataset.
        batch_size (int): Batch size for the dataloaders.
        transform (callable): Transformations to apply to the images.

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    train_dir = os.path.join(work_dir, "train")
    val_dir = os.path.join(work_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Train or Val directory not found in {work_dir}. Please run split_dataset first.")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    print("Classes detected:", train_dataset.classes)
    
    return train_loader, val_loader, train_dataset.classes
