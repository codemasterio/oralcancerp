"""
Oral Cancer Detection - Backend API
Main application entry point for the FastAPI backend.
"""

import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app modules
from backend.app.routes import prediction, health
from backend.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backend.log")
    ]
)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="Oral Cancer Detection API",
    description="API for detecting oral cancer from images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    logger.info("Starting Oral Cancer Detection API")
    # Ensure required directories exist
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Model path: {settings.MODEL_PATH}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Actions to perform on application shutdown.
    """
    logger.info("Shutting down Oral Cancer Detection API")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
