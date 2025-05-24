"""
Oral Cancer Detection - Training Script using scikit-learn
This script handles feature extraction and model training using scikit-learn.
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)  # Standard size for feature extraction
RANDOM_STATE = 42  # For reproducibility

def load_images_from_directory(directory):
    """
    Load images from a directory structure where each subdirectory is a class.
    
    Args:
        directory (str): Directory containing class subdirectories with images
        
    Returns:
        tuple: X (image features) and y (labels)
    """
    X = []
    y = []
    class_names = []
    
    # Get class names (subdirectories)
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
    
    # Sort class names for consistent label assignment
    class_names.sort()
    logger.info(f"Found classes: {class_names}")
    
    # Create a mapping from class names to numeric labels
    class_to_label = {class_name: i for i, class_name in enumerate(class_names)}
    
    # Load images from each class
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        label = class_to_label[class_name]
        
        logger.info(f"Loading images from class '{class_name}' (label: {label})...")
        
        # Get all images for this class
        images = [img for img in os.listdir(class_dir) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Failed to load image: {img_path}")
                continue
            
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, IMAGE_SIZE)
            
            # Add to dataset
            X.append(img)
            y.append(label)
    
    return np.array(X), np.array(y), class_names

def extract_features(X):
    """
    Extract features from images using simple techniques.
    
    Args:
        X (numpy.ndarray): Array of images
        
    Returns:
        numpy.ndarray: Extracted features
    """
    logger.info("Extracting features from images...")
    
    # Initialize features array
    n_samples = X.shape[0]
    features = np.zeros((n_samples, 512))  # 512 features per image
    
    for i, img in enumerate(X):
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
        
        features[i] = combined_features
        
        # Log progress
        if (i + 1) % 20 == 0 or (i + 1) == n_samples:
            logger.info(f"Processed {i + 1}/{n_samples} images")
    
    return features

def train_model(X_train, y_train, model_type='svm'):
    """
    Train a machine learning model.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        model_type (str): Type of model to train ('svm' or 'rf')
        
    Returns:
        object: Trained model
    """
    logger.info(f"Training {model_type.upper()} model...")
    
    if model_type == 'svm':
        # Define parameter grid for SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        }
        
        # Create SVM classifier
        base_model = SVC(probability=True, random_state=RANDOM_STATE)
        
    elif model_type == 'rf':
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Create Random Forest classifier
        base_model = RandomForestClassifier(random_state=RANDOM_STATE)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Use GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=5, 
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Train the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_model

def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate the trained model.
    
    Args:
        model: Trained model
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels
        class_names (list): Names of the classes
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Specificity (true negative rate)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    # Log results
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall/Sensitivity: {recall:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Create evaluation directory if it doesn't exist
    os.makedirs('model/evaluation/results', exist_ok=True)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('model/evaluation/results/confusion_matrix.png')
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('model/evaluation/results/roc_curve.png')
    
    # Compile metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(recall),  # Same as recall
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }
    
    return metrics

def save_model(model, class_names, metrics, model_type='svm'):
    """
    Save the trained model and related information.
    
    Args:
        model: Trained model
        class_names (list): Names of the classes
        metrics (dict): Evaluation metrics
        model_type (str): Type of model ('svm' or 'rf')
    """
    # Create directory if it doesn't exist
    os.makedirs('model/checkpoints', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = f'model/checkpoints/{model_type}_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'timestamp': timestamp,
        'class_names': class_names,
        'metrics': metrics,
        'image_size': IMAGE_SIZE
    }
    
    metadata_path = f'model/checkpoints/{model_type}_{timestamp}_metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Create a symlink to the latest model
    latest_model_path = f'model/checkpoints/latest_{model_type}_model.pkl'
    latest_metadata_path = f'model/checkpoints/latest_{model_type}_metadata.pkl'
    
    try:
        # Remove existing symlinks if they exist
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)
        if os.path.exists(latest_metadata_path):
            os.remove(latest_metadata_path)
        
        # Create copies instead of symlinks (more compatible with Windows)
        with open(model_path, 'rb') as src, open(latest_model_path, 'wb') as dst:
            dst.write(src.read())
        with open(metadata_path, 'rb') as src, open(latest_metadata_path, 'wb') as dst:
            dst.write(src.read())
        
        logger.info(f"Created latest model copy at {latest_model_path}")
    except Exception as e:
        logger.error(f"Error creating latest model copy: {e}")

def main():
    """
    Main function to run the training pipeline.
    """
    start_time = time.time()
    logger.info("Starting oral cancer detection model training...")
    
    # Load images from the splits directory
    train_dir = 'data/splits/train'
    test_dir = 'data/splits/test'
    
    # Load training data
    X_train_raw, y_train, class_names = load_images_from_directory(train_dir)
    logger.info(f"Loaded {X_train_raw.shape[0]} training images with {len(class_names)} classes")
    
    # Load test data
    X_test_raw, y_test, _ = load_images_from_directory(test_dir)
    logger.info(f"Loaded {X_test_raw.shape[0]} test images")
    
    # Extract features
    X_train = extract_features(X_train_raw)
    X_test = extract_features(X_test_raw)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    os.makedirs('model/checkpoints', exist_ok=True)
    with open('model/checkpoints/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Train SVM model
    svm_model = train_model(X_train_scaled, y_train, model_type='svm')
    
    # Evaluate SVM model
    svm_metrics = evaluate_model(svm_model, X_test_scaled, y_test, class_names)
    
    # Save SVM model
    save_model(svm_model, class_names, svm_metrics, model_type='svm')
    
    # Train Random Forest model
    rf_model = train_model(X_train_scaled, y_train, model_type='rf')
    
    # Evaluate Random Forest model
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, class_names)
    
    # Save Random Forest model
    save_model(rf_model, class_names, rf_metrics, model_type='rf')
    
    # Compare models
    logger.info("\nModel Comparison:")
    logger.info(f"SVM - Accuracy: {svm_metrics['accuracy']:.4f}, F1-Score: {svm_metrics['f1_score']:.4f}, Sensitivity: {svm_metrics['sensitivity']:.4f}, Specificity: {svm_metrics['specificity']:.4f}")
    logger.info(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, F1-Score: {rf_metrics['f1_score']:.4f}, Sensitivity: {rf_metrics['sensitivity']:.4f}, Specificity: {rf_metrics['specificity']:.4f}")
    
    # Determine the best model
    if svm_metrics['f1_score'] > rf_metrics['f1_score']:
        best_model = 'SVM'
        best_metrics = svm_metrics
    else:
        best_model = 'Random Forest'
        best_metrics = rf_metrics
    
    logger.info(f"\nBest model: {best_model}")
    logger.info(f"Best model metrics: Accuracy: {best_metrics['accuracy']:.4f}, F1-Score: {best_metrics['f1_score']:.4f}")
    
    # Calculate and log total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"\nTotal training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
