import torch.nn as nn
from torchvision import models


def build_model(num_classes, weights="IMAGENET1K_V1"):
    """Build a ResNet18 model."""
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ModelBuilder:
    """Class to build and initialize models."""

    def __init__(self, device: str, num_classes: int = 2):
        self.device = device
        self.num_classes = num_classes

    def build(self):
        model = build_model(self.num_classes)
        return model.to(self.device)
