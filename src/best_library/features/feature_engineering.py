import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def compute_dataset_stats(dataset_dir, img_size=224, batch_size=32):
    """
    Computes the mean and standard deviation of the dataset.
    This helps in normalizing the data better.
    
    Args:
        dataset_dir (str): Path to the dataset.
        img_size (int): Size to resize images to.
        batch_size (int): Batch size for loading.
        
    Returns:
        tuple: (mean, std)
    """
    print(f"Computing statistics for dataset at {dataset_dir}...")
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found.")
        return None, None

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # We can use the training set for this
    try:
        dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    except Exception as e:
        print(f"Error loading dataset for stats: {e}")
        return None, None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
        
    if total_images > 0:
        mean /= total_images
        std /= total_images
        
        print(f"Computed Mean: {mean}")
        print(f"Computed Std: {std}")
        return mean, std
    else:
        print("No images found.")
        return None, None
