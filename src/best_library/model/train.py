import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.best_library.evaluation.evaluate import evaluate_model

def train_model(model, train_loader, val_loader, epochs, lr, device, save_path=None):
    """
    Trains the model.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        device (str): Device to use.
        save_path (str, optional): Path to save the trained model. If None, model is not saved.
    
    Returns:
        float: Best validation accuracy achieved.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        # Validation
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Model saved as {save_path} (New Best: {best_val_acc:.3f})")

    return best_val_acc
