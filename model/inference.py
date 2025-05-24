"""
Oral Cancer Detection - Model Inference Module
This module handles model inference for new images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import logging
import time
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OralCancerPredictor:
    """
    Class to handle predictions for oral cancer detection.
    """
    
    def __init__(self, model_path, image_size=(224, 224)):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to the saved model
            image_size (tuple): Input image size for the model
        """
        self.model_path = model_path
        self.image_size = image_size
        self.model = self._load_model()
        self.class_names = ['Non-Cancer', 'Cancer']
        
        # Get model metadata
        self.model_metadata = self._get_model_metadata()
        
    def _load_model(self):
        """
        Load the model from disk.
        
        Returns:
            tf.keras.Model: Loaded model
        """
        logger.info(f"Loading model from {self.model_path}")
        try:
            model = load_model(self.model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_model_metadata(self):
        """
        Get model metadata.
        
        Returns:
            dict: Model metadata
        """
        # Extract model name from path
        model_name = os.path.basename(self.model_path).split('_')[0]
        
        # Get model creation time from file timestamp
        creation_time = time.ctime(os.path.getmtime(self.model_path))
        
        # Get model size
        model_size_bytes = os.path.getsize(self.model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Get input shape
        input_shape = self.model.input_shape[1:]
        
        metadata = {
            'model_name': model_name,
            'creation_time': creation_time,
            'model_size_mb': model_size_mb,
            'input_shape': input_shape,
            'class_names': self.class_names
        }
        
        return metadata
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction.
        
        Args:
            image: Image as numpy array or path to image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # If image is a file path, load it
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Failed to load image: {image}")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # If image is already a numpy array, make a copy
            img = image.copy()
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image, return_visualization=False):
        """
        Make a prediction for an image.
        
        Args:
            image: Image as numpy array or path to image file
            return_visualization (bool): Whether to return visualization
            
        Returns:
            dict: Prediction results
        """
        # Preprocess the image
        preprocessed_img = self.preprocess_image(image)
        if preprocessed_img is None:
            return {'error': 'Failed to preprocess image'}
        
        # Make prediction
        try:
            start_time = time.time()
            prediction = self.model.predict(preprocessed_img)[0][0]
            inference_time = time.time() - start_time
            
            # Convert to binary prediction
            binary_prediction = 1 if prediction > 0.5 else 0
            
            # Create result
            result = {
                'prediction': int(binary_prediction),
                'class_name': self.class_names[binary_prediction],
                'confidence': float(prediction if binary_prediction == 1 else 1 - prediction),
                'probability': float(prediction),
                'inference_time': inference_time
            }
            
            logger.info(f"Prediction: {result['class_name']} with {result['confidence']:.4f} confidence")
            
            # Generate visualization if requested
            if return_visualization:
                # Get the original image for visualization
                if isinstance(image, str):
                    original_img = cv2.imread(image)
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                else:
                    original_img = image.copy()
                    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
                        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Create visualization
                visualization = self._create_prediction_visualization(original_img, result)
                result['visualization'] = visualization
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {'error': str(e)}
    
    def _create_prediction_visualization(self, image, prediction_result):
        """
        Create a visualization of the prediction.
        
        Args:
            image: Original image
            prediction_result (dict): Prediction result
            
        Returns:
            numpy.ndarray: Visualization image
        """
        # Resize image for display if needed
        if image.shape[0] > 800 or image.shape[1] > 800:
            scale = min(800 / image.shape[0], 800 / image.shape[1])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)
        
        # Create a copy for drawing
        viz_img = image.copy()
        
        # Add prediction information
        class_name = prediction_result['class_name']
        confidence = prediction_result['confidence']
        
        # Set color based on prediction (red for cancer, green for non-cancer)
        color = (0, 0, 255) if class_name == 'Cancer' else (0, 255, 0)
        
        # Add text background
        text = f"{class_name}: {confidence:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(viz_img, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10), (0, 0, 0), -1)
        
        # Add text
        cv2.putText(viz_img, text, (15, 15 + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add border based on prediction
        border_thickness = 10
        h, w = viz_img.shape[:2]
        viz_img = cv2.copyMakeBorder(viz_img, border_thickness, border_thickness, 
                                     border_thickness, border_thickness, 
                                     cv2.BORDER_CONSTANT, value=color)
        
        return viz_img
    
    def batch_predict(self, image_paths):
        """
        Make predictions for multiple images.
        
        Args:
            image_paths (list): List of paths to images
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path)
            if 'error' not in result:
                result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return self.model_metadata
    
    def save_prediction_to_file(self, prediction_result, output_path):
        """
        Save prediction result to a file.
        
        Args:
            prediction_result (dict): Prediction result
            output_path (str): Path to save the result
            
        Returns:
            bool: Success status
        """
        try:
            # Create a copy of the result to avoid modifying the original
            result_to_save = prediction_result.copy()
            
            # Remove visualization if present
            if 'visualization' in result_to_save:
                del result_to_save['visualization']
            
            # Add timestamp
            result_to_save['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(result_to_save, f, indent=4)
            
            logger.info(f"Prediction saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def save_visualization(self, visualization, output_path):
        """
        Save visualization image to a file.
        
        Args:
            visualization: Visualization image
            output_path (str): Path to save the image
            
        Returns:
            bool: Success status
        """
        try:
            # Convert RGB to BGR for OpenCV
            if len(visualization.shape) == 3 and visualization.shape[2] == 3:
                visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(output_path, visualization)
            
            logger.info(f"Visualization saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            return False

def predict_image(model_path, image_path, output_dir=None):
    """
    Convenience function to predict a single image.
    
    Args:
        model_path (str): Path to the saved model
        image_path (str): Path to the image
        output_dir (str): Directory to save results
        
    Returns:
        dict: Prediction result
    """
    # Create predictor
    predictor = OralCancerPredictor(model_path)
    
    # Make prediction
    result = predictor.predict(image_path, return_visualization=True)
    
    # Save results if output directory is provided
    if output_dir and 'error' not in result:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction result
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        predictor.save_prediction_to_file(
            result,
            os.path.join(output_dir, f"{base_filename}_prediction.json")
        )
        
        # Save visualization
        if 'visualization' in result:
            predictor.save_visualization(
                result['visualization'],
                os.path.join(output_dir, f"{base_filename}_visualization.jpg")
            )
    
    return result

if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/resnet50_20250524-123456_best.h5"
    image_path = "data/test_images/sample.jpg"
    output_dir = "predictions"
    
    result = predict_image(model_path, image_path, output_dir)
