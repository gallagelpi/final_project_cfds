from PIL import Image
import torch
from best_library.model.model_definition import build_model


def load_trained_model(model_path, device, num_classes: int = 2):
    checkpoint = torch.load(model_path, map_location=device)

    # If you don't know num_classes, get it from checkpoint
    if num_classes is None:
        num_classes = checkpoint["num_classes"]

    model = build_model(num_classes=num_classes, weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()
    return model



def predict_image(image_path: str, model, transform, device: str, class_names):
    
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        idx = torch.argmax(probs).item()

    return class_names[idx], probs[idx].item()


class Predictor:
    """
    Class to handle loading a trained model and making predictions.
    """

    def __init__(self, device: str, class_names):
        self.device = device
        self.class_names = class_names

    def load_model(self, model_path: str):
        """Load trained model onto the specified device."""
        return load_trained_model(model_path, self.device, num_classes=len(self.class_names))

    def predict(self, image_path: str, model, transform):
        """Predict class and confidence for a single image."""
        return predict_image(
            image_path=image_path,
            model=model,
            transform=transform,
            device=self.device,
            class_names=self.class_names
        )
