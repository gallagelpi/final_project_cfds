import torch

def evaluate_model(model, val_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        val_loader: DataLoader for validation data.
        device (str): Device to use.

    Returns:
        float: Validation accuracy.
    """
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0
    return val_acc
