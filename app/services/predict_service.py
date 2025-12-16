import os
import shutil
import torch
from best_library.model.model_resnet18 import ModelResnet18

DEVICE = "cuda"  # or detect automatically
MODEL_PATH = "models/best_model.pth"


class_names = ["alpaca", "not_alpaca"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_api = ModelResnet18(device=DEVICE, class_names=class_names)
model = model_api.load_trained_model(MODEL_PATH)


def predict_image(image_file, transform):
    """
    Receives an uploaded image file, saves it temporarily,
    and runs ModelResnet18.predict
    """
    temp_path = "temp_image.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)

    label, confidence = model_api.predict(image_path=temp_path, model=model, transform=transform)

    os.remove(temp_path)

    return label, confidence
