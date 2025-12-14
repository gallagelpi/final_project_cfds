import torch
import torch.nn as nn
from torchvision import models

def build_model(device, num_classes=2, weights="IMAGENET1K_V1"):
    """
    Builds the ResNet18 model.

    Args:
        device (str): Device to put the model on ('cpu' or 'cuda').
        num_classes (int): Number of output classes.
        weights (str): Pretrained weights to use.

    Returns:
        model: The PyTorch model.
    """
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    return model
