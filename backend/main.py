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
import os
from pathlib import Path
from backend.app.routes import prediction, health
from backend.config import get_settings

def check_model_files():
    """Check if required model files exist."""
    model_dir = Path("model/checkpoints")
    required_files = [
        model_dir / "latest_svm_model.pkl",
        model_dir / "feature_scaler.pkl",
        model_dir / "svm_20250524_121655_metadata.pkl"
    ]
    
    missing_files = [str(f) for f in required_files if not f.exists()]
    if missing_files:
        error_msg = (
            "The following required model files are missing.\n"
            "Please download them and place them in the model/checkpoints/ directory.\n\n"
            f"Missing files: {', '.join(missing_files)}\n\n"
            "See README.md for instructions on how to download the model files."
        )
        raise FileNotFoundError(error_msg)

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
    expose_headers=["*"]
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])

@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on application startup.
    """
    logger.info("Starting up Oral Cancer Detection API...")
    
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    
    # Ensure model directory exists
    os.makedirs("model/checkpoints", exist_ok=True)
    
    # Check for required model files
    try:
        check_model_files()
        logger.info("All required model files found.")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("API will not function properly without the model files.")
    
    logger.info(f"CORS allowed origins: {settings.CORS_ORIGINS}")
    logger.info("API started successfully")

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
