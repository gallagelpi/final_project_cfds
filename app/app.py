from fastapi import FastAPI
from app.routers.predict_router import router

app = FastAPI(title="Image Classification API")

app.include_router(router)

