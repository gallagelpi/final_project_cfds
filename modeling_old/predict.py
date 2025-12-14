import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

IMG_SIZE = 224

# Load model
def load_model(model_path="alpaca_classifier.pt"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# Preprocessing for 1 image
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Predict function
def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probs).item()

    classes = ["alpaca", "not_alpaca"]
    confidence = probs[pred_class].item()

    return classes[pred_class], confidence

# CLI usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        exit()

    image_path = sys.argv[1]
    model = load_model()
    label, conf = predict(image_path, model)

    print(f"\nPrediction: {label}  (confidence: {conf:.2%})\n")
