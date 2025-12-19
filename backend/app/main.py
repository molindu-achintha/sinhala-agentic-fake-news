"""
main.py - Main FastAPI Application

This file is the entry point for the backend server.
It creates the FastAPI application and includes all API routes.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from .config import get_settings
from .api.v1 import predict, health, news, evaluate
from dotenv import load_dotenv

# Explicitly load .env file to ensure os.getenv works everywhere
load_dotenv()

# Load settings from .env file
settings = get_settings()

# Create the FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="A multi-agent system for detecting fake news in Sinhala language",
    version=settings.VERSION
)

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health.router, prefix="/v1", tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Prediction"])
app.include_router(news.router, prefix="/v1", tags=["News"])
app.include_router(evaluate.router, prefix="/v1", tags=["Evaluation"])


@app.get("/")
def root():
    """Root endpoint - returns welcome message."""
    print("[main] Root endpoint called")
    return {
        "message": "Sinhala Agentic Fake News Detection API",
        "version": settings.VERSION,
        "docs": "/docs"
    }


@app.on_event("startup")
async def startup_event():
    """Runs when the application starts."""
    print("=" * 50)
    print("Sinhala Fake News Detection API")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print("Endpoints available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Runs when the application shuts down."""
    print("Shutting down Sinhala Fake News Detection API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
