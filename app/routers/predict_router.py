from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.predict_schema import PredictionResponse
from app.services.predict_service import predict_image
from best_library.preprocessing.preprocessing import Preprocessing

router = APIRouter()

# Same preprocessing as training
preprocessor = Preprocessing(img_size=224)
transform = preprocessor.get_transform()


@router.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    label, confidence = predict_image(file, transform)

    return PredictionResponse(
        label=label,
        confidence=confidence
    )

