"""
Oral Cancer Detection - Simple Backend Server
A simplified Flask-based backend server for oral cancer detection.
"""

import os
import sys
import logging
import json
import numpy as np
import cv2
import pickle
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
from datetime import datetime
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "checkpoints", "latest_svm_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "checkpoints", "feature_scaler.pkl")
METADATA_PATH = MODEL_PATH.replace('_model.pkl', '_metadata.pkl')
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model and metadata
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
            class_names = metadata.get('class_names', ['Cancer', 'Normal'])
    else:
        class_names = ['Cancer', 'Normal']
    
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Class names: {class_names}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None
    class_names = ['Cancer', 'Normal']

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(img):
    """
    Extract features from an image.
    
    Args:
        img: Image as numpy array
        
    Returns:
        numpy.ndarray: Extracted features
    """
    # Initialize features array
    features = np.zeros(512)  # 512 features
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Extract histogram features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Extract Haralick texture features
    haralick = np.zeros(13)
    try:
        haralick = np.mean(cv2.calcHist([gray], [0], None, [256], [0, 256]), axis=0)[:13]
    except:
        pass
    
    # Extract color features
    color_features = []
    for channel in range(3):
        hist = cv2.calcHist([img], [channel], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_features.extend(hist)
    
    # Combine features
    combined_features = np.concatenate([
        hist,  # 256 features
        haralick,  # 13 features
        np.array(color_features)  # 192 features (64*3)
    ])
    
    # Ensure we have exactly 512 features (pad or truncate)
    if combined_features.shape[0] < 512:
        combined_features = np.pad(combined_features, (0, 512 - combined_features.shape[0]))
    else:
        combined_features = combined_features[:512]
    
    return combined_features

def create_prediction_visualization(image, prediction_result):
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'message': 'API is running',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict oral cancer from uploaded image."""
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.'
        }), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file part in the request'
        }), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        original_extension = file.filename.split('.')[-1]
        unique_filename = f"upload_{timestamp}_{unique_id}.{original_extension}"
        
        # Save file
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        file.save(file_path)
        
        logger.info(f"File saved: {file_path}")
        
        # Read and preprocess image
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({
                'error': 'Failed to read uploaded image'
            }), 500
        
        # Convert from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, IMAGE_SIZE)
        
        # Extract features
        features = extract_features(img)
        
        # Scale features
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Map numeric prediction to class name
        class_idx = int(prediction)
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
        else:
            class_name = f"Class {class_idx}"
        
        # Get probability for the predicted class
        probability = probabilities[class_idx]
        
        # For binary classification, ensure probability is for cancer class
        if len(class_names) == 2:
            cancer_idx = class_names.index('Cancer') if 'Cancer' in class_names else 0
            probability = probabilities[cancer_idx]
            
            # If normal is predicted, invert the probability for consistency
            if class_name != 'Cancer':
                probability = 1 - probability
        
        # Create visualization
        viz_img = create_prediction_visualization(img, {
            'class_name': class_name,
            'confidence': float(probability)
        })
        
        # Save visualization
        viz_filename = f"viz_{timestamp}_{unique_id}.jpg"
        viz_path = os.path.join(UPLOAD_DIR, viz_filename)
        cv2.imwrite(viz_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
        
        # Create result
        result = {
            'prediction': int(class_idx),
            'class_name': class_name,
            'confidence': float(probability),
            'probability': float(probability),
            'visualization_url': f"/visualization/{viz_filename}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/visualization/<filename>', methods=['GET'])
def get_visualization(filename):
    """Get visualization image by filename."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        return jsonify({
            'error': 'Visualization not found'
        }), 404
    
    return send_file(file_path, mimetype='image/jpeg')

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    # Get model metadata
    model_name = os.path.basename(MODEL_PATH).split('_')[0]
    creation_time = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S')
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    
    return jsonify({
        'model_name': model_name,
        'creation_time': creation_time,
        'model_size_mb': round(model_size_mb, 2),
        'class_names': class_names,
        'image_size': IMAGE_SIZE
    })

if __name__ == '__main__':
    logger.info("Starting Oral Cancer Detection API")
    app.run(host='0.0.0.0', port=8000, debug=True)
