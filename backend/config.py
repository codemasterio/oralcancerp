"""
Oral Cancer Detection - Backend Configuration
Configuration settings for the backend API.
"""

import os
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application settings.
    
    These settings can be overridden with environment variables.
    """
    # API settings
    API_TITLE: str = "Oral Cancer Detection API"
    API_DESCRIPTION: str = "API for detecting oral cancer from images"
    API_VERSION: str = "1.0.0"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS settings
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:8000"
    ).split(",") if os.getenv("CORS_ORIGINS") else ["*"]
    
    # File upload settings
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    
    # Model settings
    MODEL_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model", "checkpoints", "latest_svm_model.pkl"
    )
    MODEL_IMAGE_SIZE: tuple = (224, 224)
    
    # Security settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: str = ""  # Set this via environment variable in production
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()
