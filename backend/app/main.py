"""
Main FastAPI application.
"""
from fastapi import FastAPI
from .config import get_settings
from .api.v1 import predict, health

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Prediction"])

@app.get("/")
def root():
    return {"message": "Sinhala Agentic Fake News Detection API"}
