"""
Main FastAPI application.
"""
from fastapi import FastAPI
from .config import get_settings
from .api.v1 import predict, health, news

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Prediction"])
app.include_router(news.router, prefix="/v1", tags=["News"])


@app.get("/")
def root():
    return {"message": "Sinhala Agentic Fake News Detection API"}
