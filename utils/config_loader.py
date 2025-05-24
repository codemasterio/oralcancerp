"""
Oral Cancer Detection - Configuration Loader
Utility for loading and parsing configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration as a dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def get_model_config(model_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name (str): Name of the model (e.g., 'resnet50')
        config_path (str, optional): Path to the configuration file
        
    Returns:
        dict: Model configuration
    """
    if config_path is None:
        # Use default config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'model_config.yaml'
        )
    
    # Load the configuration
    config = load_yaml_config(config_path)
    
    # Get model-specific configuration
    model_config = config.get('models', {}).get(model_name)
    
    if model_config is None:
        logger.warning(f"No configuration found for model '{model_name}'. Using default values.")
        model_config = {}
    
    # Merge with general configuration
    result = {
        'data': config.get('data', {}),
        'training': config.get('training', {}),
        'evaluation': config.get('evaluation', {}),
        'inference': config.get('inference', {}),
        'model': model_config
    }
    
    return result

def get_data_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get data processing configuration.
    
    Args:
        config_path (str, optional): Path to the configuration file
        
    Returns:
        dict: Data configuration
    """
    if config_path is None:
        # Use default config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'model_config.yaml'
        )
    
    # Load the configuration
    config = load_yaml_config(config_path)
    
    # Return data configuration
    return config.get('data', {})

def get_training_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get training configuration.
    
    Args:
        config_path (str, optional): Path to the configuration file
        
    Returns:
        dict: Training configuration
    """
    if config_path is None:
        # Use default config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'model_config.yaml'
        )
    
    # Load the configuration
    config = load_yaml_config(config_path)
    
    # Return training configuration
    return config.get('training', {})

def get_inference_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get inference configuration.
    
    Args:
        config_path (str, optional): Path to the configuration file
        
    Returns:
        dict: Inference configuration
    """
    if config_path is None:
        # Use default config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'model_config.yaml'
        )
    
    # Load the configuration
    config = load_yaml_config(config_path)
    
    # Return inference configuration
    return config.get('inference', {})

if __name__ == "__main__":
    # Example usage
    model_config = get_model_config('resnet50')
    print(f"ResNet50 Configuration: {model_config}")
    
    data_config = get_data_config()
    print(f"Data Configuration: {data_config}")
    
    training_config = get_training_config()
    print(f"Training Configuration: {training_config}")
    
    inference_config = get_inference_config()
    print(f"Inference Configuration: {inference_config}")
