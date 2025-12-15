import torch

def evaluate_model(model, val_loader, device: str):
    
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0

class Evaluator:
    """
    Class to evaluate a PyTorch model.
    """

    def __init__(self, device: str):
    
        self.device = device

    def evaluate(self, model, val_loader):
    
        return evaluate_model(model, val_loader, self.device)
