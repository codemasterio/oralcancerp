"""
Oral Cancer Detection - Health Check Routes
Routes for health checks and system status.
"""

import os
import time
from fastapi import APIRouter, Depends
from typing import Dict, Any
import platform
import psutil
import logging

from backend.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/health")

# Start time of the application
START_TIME = time.time()

@router.get("/", summary="Health check")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.
    
    Returns:
        dict: Status information
    """
    return {"status": "ok", "message": "API is running"}

@router.get("/status", summary="Detailed system status")
async def system_status() -> Dict[str, Any]:
    """
    Get detailed system status.
    
    Returns:
        dict: Detailed status information
    """
    settings = get_settings()
    
    # Calculate uptime
    uptime_seconds = time.time() - START_TIME
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Get system information
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check if model file exists
    model_exists = os.path.exists(settings.MODEL_PATH)
    model_size_mb = 0
    if model_exists:
        model_size_mb = os.path.getsize(settings.MODEL_PATH) / (1024 * 1024)
    
    # Check upload directory
    upload_dir_exists = os.path.exists(settings.UPLOAD_DIR)
    
    return {
        "status": "ok",
        "uptime": uptime,
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "disk_usage_percent": disk.percent
        },
        "api": {
            "version": settings.API_VERSION,
            "debug_mode": settings.DEBUG
        },
        "model": {
            "path": settings.MODEL_PATH,
            "exists": model_exists,
            "size_mb": round(model_size_mb, 2) if model_exists else 0,
            "image_size": settings.MODEL_IMAGE_SIZE
        },
        "storage": {
            "upload_dir": settings.UPLOAD_DIR,
            "upload_dir_exists": upload_dir_exists,
            "max_upload_size_mb": settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        }
    }

@router.get("/model-info", summary="Model information")
async def model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        dict: Model information
    """
    settings = get_settings()
    
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from model.sklearn_inference import OralCancerPredictor
    
    try:
        # Check if model file exists
        if not os.path.exists(settings.MODEL_PATH):
            return {
                "status": "error",
                "message": "Model file not found",
                "path": settings.MODEL_PATH
            }
        
        # Create predictor
        predictor = OralCancerPredictor(settings.MODEL_PATH, settings.MODEL_IMAGE_SIZE)
        
        # Get model info
        model_info = predictor.get_model_info()
        
        return {
            "status": "ok",
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
