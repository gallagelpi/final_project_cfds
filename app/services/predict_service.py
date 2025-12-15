import os
import shutil
import torch
from best_library.model.predict import Predictor
from best_library.model.predict import load_trained_model

DEVICE = "cuda"  # or detect automatically
MODEL_PATH = "models/best_model.pth"


class_names = ["alpaca", "not_alpaca"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_trained_model(
    MODEL_PATH,
    device=DEVICE,
    num_classes=len(class_names)
)

predictor = Predictor(DEVICE, class_names)


def predict_image(image_file, transform):
    """
    Receives an uploaded image file, saves it temporarily,
    and runs Predictor.predict
    """
    temp_path = "temp_image.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image_file.file, buffer)

    label, confidence = predictor.predict(
        image_path=temp_path,
        model=model,
        transform=transform
    )

    os.remove(temp_path)

    return label, confidence
