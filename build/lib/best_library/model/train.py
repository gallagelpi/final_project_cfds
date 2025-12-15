import torch
import torch.optim as optim
import torch.nn as nn
from best_library.evaluation.evaluate import evaluate_model


def train_model(model, train_loader, val_loader, epochs: int, lr: float, device: str):
    """
    Train a model and return the best validation accuracy.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        best_val_acc (float): Best validation accuracy achieved during training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


class Trainer:
    """Class to train models with a fixed device."""

    def __init__(self, device: str):
        self.device = device

    def train(self, model, train_loader, val_loader, epochs: int, lr: float):
        """
        Train a model using the device of the Trainer.

        Returns:
            best_val_acc (float): Best validation accuracy.
        """
        return train_model(model, train_loader, val_loader, epochs, lr, self.device)
