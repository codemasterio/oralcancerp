from flask import Flask, send_from_directory, request, jsonify
import os
import logging
import json
import datetime
import uuid
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "checkpoints", "svm_model_numpy_1243.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "checkpoints", "feature_scaler_numpy_1243.pkl")
METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "checkpoints", "svm_20250524_121655_metadata.pkl")
IMAGE_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model and metadata
try:
    import cloudpickle
    import numpy as np
    import sys
    import joblib  # Keep joblib for potential fallback
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Cloudpickle version: {cloudpickle.__version__}")
    
    # Function to try loading with cloudpickle first, then fallback to joblib
    def safe_load(path, description):
        try:
            logger.info(f"Loading {description} with cloudpickle from: {path}")
            with open(path, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e1:
            logger.warning(f"Failed to load {description} with cloudpickle: {str(e1)}")
            try:
                logger.info(f"Trying to load {description} with joblib...")
                return joblib.load(path)
            except Exception as e2:
                logger.error(f"Failed to load {description} with joblib: {str(e2)}")
                raise
    
    # Load model
    model = safe_load(MODEL_PATH, "model")
    logger.info("Model loaded successfully")
    
    # Load scaler
    scaler = safe_load(SCALER_PATH, "scaler")
    logger.info("Scaler loaded successfully")
    
    # Load metadata if it exists
    class_names = ['Cancer', 'Normal']
    if os.path.exists(METADATA_PATH):
        try:
            metadata = safe_load(METADATA_PATH, "metadata")
            class_names = metadata.get('class_names', class_names)
            logger.info("Metadata loaded successfully")
        except Exception as e:
            logger.warning(f"Error loading metadata: {str(e)}")
            logger.warning("Using default class names")
    
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

@app.route('/')
def serve_frontend():
    """Serve the frontend HTML."""
    return send_from_directory('.', 'simple_frontend.html')

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        
        # Create result
        result = {
            'prediction': int(class_idx),
            'class_name': class_name,
            'confidence': float(probability),
            'probability': float(probability)
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting Oral Cancer Detection API")
    app.run(host='0.0.0.0', port=8000, debug=True)
