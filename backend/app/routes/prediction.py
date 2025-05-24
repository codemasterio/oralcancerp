"""
Oral Cancer Detection - Prediction Routes
Routes for image upload and cancer prediction.
"""

import os
import uuid
import time
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, Any, List, Optional
import shutil
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import model inference
from model.sklearn_inference import OralCancerPredictor
from backend.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/predict")

# Initialize predictor at module level
predictor = None

def get_predictor():
    """
    Get or initialize the predictor.
    
    Returns:
        OralCancerPredictor: Predictor instance
    """
    global predictor
    settings = get_settings()
    
    if predictor is None:
        try:
            predictor = OralCancerPredictor(
                settings.MODEL_PATH, 
                settings.MODEL_IMAGE_SIZE
            )
            logger.info("Predictor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize model: {str(e)}"
            )
    
    return predictor

def is_valid_image(filename: str) -> bool:
    """
    Check if the file has a valid image extension.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    settings = get_settings()
    allowed_extensions = settings.ALLOWED_EXTENSIONS
    
    if not filename:
        return False
    
    extension = filename.split('.')[-1].lower()
    return extension in allowed_extensions

def cleanup_old_files(background_tasks: BackgroundTasks):
    """
    Schedule cleanup of old uploaded files.
    
    Args:
        background_tasks: FastAPI background tasks
    """
    def _cleanup():
        settings = get_settings()
        upload_dir = settings.UPLOAD_DIR
        
        # Skip if directory doesn't exist
        if not os.path.exists(upload_dir):
            return
        
        # Current time
        now = time.time()
        
        # Delete files older than 1 hour
        max_age = 3600  # 1 hour in seconds
        
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check file age
            file_age = now - os.path.getmtime(file_path)
            
            if file_age > max_age:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
    
    background_tasks.add_task(_cleanup)

@router.post("/", summary="Predict oral cancer from image")
async def predict_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Upload an image and get oral cancer prediction.
    
    Args:
        background_tasks: FastAPI background tasks
        file: Uploaded image file
        settings: Application settings
        
    Returns:
        dict: Prediction results
    """
    # Schedule cleanup of old files
    cleanup_old_files(background_tasks)
    
    # Validate file
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file extension
    if not is_valid_image(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    original_extension = file.filename.split('.')[-1]
    unique_filename = f"upload_{timestamp}_{unique_id}.{original_extension}"
    
    # Save file
    file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    try:
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved: {file_path}")
        
        # Get predictor
        predictor = get_predictor()
        
        # Make prediction
        result = predictor.predict(file_path, return_visualization=True)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Save visualization
        if 'visualization' in result:
            viz_filename = f"viz_{timestamp}_{unique_id}.jpg"
            viz_path = os.path.join(settings.UPLOAD_DIR, viz_filename)
            predictor.save_visualization(result['visualization'], viz_path)
            
            # Add visualization URL to result
            result['visualization_url'] = f"/predict/visualization/{viz_filename}"
            
            # Remove visualization image from response
            del result['visualization']
        
        # Add timestamp
        result['timestamp'] = timestamp
        
        # Schedule file deletion after 1 hour
        def _delete_file():
            try:
                time.sleep(3600)  # 1 hour
                if os.path.exists(file_path):
                    os.remove(file_path)
                if 'visualization_url' in result and os.path.exists(viz_path):
                    os.remove(viz_path)
            except Exception as e:
                logger.error(f"Error deleting file: {e}")
        
        background_tasks.add_task(_delete_file)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        # Clean up file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/visualization/{filename}", summary="Get prediction visualization")
async def get_visualization(
    filename: str,
    settings = Depends(get_settings)
) -> FileResponse:
    """
    Get visualization image by filename.
    
    Args:
        filename: Visualization image filename
        settings: Application settings
        
    Returns:
        FileResponse: Image file
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(file_path, media_type="image/jpeg")

@router.post("/batch", summary="Batch predict multiple images")
async def batch_predict(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Upload multiple images and get predictions for all.
    
    Args:
        background_tasks: FastAPI background tasks
        files: List of uploaded image files
        settings: Application settings
        
    Returns:
        dict: Batch prediction results
    """
    # Schedule cleanup of old files
    cleanup_old_files(background_tasks)
    
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Limit number of files
    max_files = 10
    if len(files) > max_files:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Maximum allowed: {max_files}"
        )
    
    results = []
    saved_files = []
    
    try:
        # Get predictor
        predictor = get_predictor()
        
        # Process each file
        for file in files:
            # Check file extension
            if not is_valid_image(file.filename):
                results.append({
                    "filename": file.filename,
                    "error": f"Invalid file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
                })
                continue
            
            # Create unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            original_extension = file.filename.split('.')[-1]
            unique_filename = f"batch_{timestamp}_{unique_id}.{original_extension}"
            
            # Save file
            file_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
            
            # Ensure upload directory exists
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            
            # Save uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            saved_files.append(file_path)
            
            # Make prediction
            result = predictor.predict(file_path)
            
            if 'error' in result:
                results.append({
                    "filename": file.filename,
                    "error": result['error']
                })
            else:
                result['original_filename'] = file.filename
                results.append(result)
        
        # Schedule file deletion after 1 hour
        def _delete_files():
            try:
                time.sleep(3600)  # 1 hour
                for file_path in saved_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                logger.error(f"Error deleting files: {e}")
        
        background_tasks.add_task(_delete_files)
        
        return {
            "batch_size": len(files),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "failed_predictions": len([r for r in results if 'error' in r]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        # Clean up files
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
