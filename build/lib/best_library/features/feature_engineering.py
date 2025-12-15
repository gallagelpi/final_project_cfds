import torch
import numpy as np
import torch.nn.functional as F
from scipy.stats import entropy

def color_features(image: torch.Tensor):
    """Mean and std per RGB channel."""
    mean = image.mean(dim=(1, 2))
    std = image.std(dim=(1, 2))
    return torch.cat([mean, std])


def grayscale_histogram(image: torch.Tensor, bins: int = 16):
    """Histogram of grayscale intensities."""
    gray = image.mean(dim=0).numpy()
    hist, _ = np.histogram(gray, bins=bins, range=(0, 1), density=True)
    return torch.tensor(hist, dtype=torch.float32)


import torch
import torch.nn.functional as F

def edge_density(image: torch.Tensor):
    """Edge density using Sobel filters (no cv2)."""
    # image: (C, H, W)
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=torch.float32,
        device=image.device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=torch.float32,
        device=image.device
    ).unsqueeze(0)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return magnitude.mean()



def image_entropy(image: torch.Tensor):
    """Entropy of grayscale image."""
    gray = image.mean(dim=0).numpy()
    hist, _ = np.histogram(gray, bins=256, range=(0, 1), density=True)
    return torch.tensor([entropy(hist + 1e-8)], dtype=torch.float32)


def frequency_features(image: torch.Tensor):
    """Low vs high frequency energy ratio."""
    gray = image.mean(dim=0).numpy()
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(fft)

    h, w = magnitude.shape
    center = magnitude[h//4:3*h//4, w//4:3*w//4]

    low_freq = center.mean()
    high_freq = (magnitude.mean() - low_freq)

    return torch.tensor([low_freq, high_freq], dtype=torch.float32)

class FeatureBuilder:
    """Class to build handcrafted feature sets from images."""

    def __init__(self):
        """Initialize feature builder."""
        pass

    def extract_features(self, image: torch.Tensor):
        """
        Extract all feature sets from a single image.

        Returns:
            torch.Tensor: concatenated feature vector
        """
        features = [
            color_features(image),
            grayscale_histogram(image),
            edge_density(image),
            image_entropy(image),
            frequency_features(image)
        ]

        return torch.cat(features)
