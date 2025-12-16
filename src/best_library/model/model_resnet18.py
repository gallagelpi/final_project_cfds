from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models

from best_library.evaluation.evaluate import evaluate_model


class ModelResnet18:
    """
    Single class (ResNet18) that centralizes:
    - build_model
    - train_model
    - load_trained_model
    - predict_image
    - predict

    It contains all logic that previously lived in separate modules.
    """

    device: str
    class_names: List[str]

    def __init__(self, device: str, class_names: Iterable[str]):
        self.device = device
        self.class_names = list(class_names)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def build_model(self, num_classes: Optional[int] = None, weights: Optional[str] = "IMAGENET1K_V1"):
        """
        Build a ResNet18 with the final layer adjusted to num_classes and move it to self.device.
        """
        n = self.num_classes if num_classes is None else int(num_classes)
        # Compatibility: torchvision prefers a WeightsEnum; some environments may pass a string.
        resolved_weights = weights
        if isinstance(weights, str):
            try:
                resolved_weights = models.ResNet18_Weights[weights]
            except Exception:
                resolved_weights = weights

        model = models.resnet18(weights=resolved_weights)
        model.fc = nn.Linear(model.fc.in_features, n)
        return model.to(self.device)

    def train_model(self, model, train_loader, val_loader, epochs: int, lr: float) -> float:
        """
        Train using the project's training loop.
        Returns the best validation accuracy achieved during training.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            correct, total = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total if total else 0.0
            val_acc = evaluate_model(model, val_loader, self.device)

            print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        return best_val_acc

    def load_trained_model(self, model_path: str, num_classes: Optional[int] = None):
        """
        Load a trained checkpoint and place the model on self.device.
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # If not provided, try to read from checkpoint; otherwise fall back to the instance num_classes.
        if num_classes is None:
            n = int(checkpoint.get("num_classes", self.num_classes))
        else:
            n = int(num_classes)

        model = self.build_model(num_classes=n, weights=None)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image_path: str, model, transform) -> Tuple[str, float]:
        """
        Predict for an image on disk, returning (label, confidence).
        """
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            idx = torch.argmax(probs).item()

        return self.class_names[idx], probs[idx].item()

    def predict(self, image_path: str, model, transform) -> Tuple[str, float]:
        """
        Alias for predict_image (shorter API).
        """
        return self.predict_image(image_path=image_path, model=model, transform=transform)


