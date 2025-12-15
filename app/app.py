from fastapi import FastAPI
from app.routers import predict_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(predict_router.router)

