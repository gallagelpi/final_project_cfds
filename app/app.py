from fastapi import FastAPI
from app.routers import predict_router

app = FastAPI(title="Image Classification API")

app.include_router(predict_router.router)

