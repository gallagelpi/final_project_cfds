import torch
import torch.nn as nn
from torchvision import models
from PIL import Image

def load_trained_model(model_path, device, num_classes=2):
    """
    Loads a trained ResNet18 model.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, transform, device, class_names=["alpaca", "not_alpaca"]):
    """
    Predicts the class of an image.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class_idx = torch.argmax(probs).item()

    label = class_names[pred_class_idx]
    confidence = probs[pred_class_idx].item()

    return label, confidence
