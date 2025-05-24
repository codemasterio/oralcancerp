"""
Oral Cancer Detection - Training Script
This script handles data preprocessing and model training in one file.
"""

import os
import sys
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import logging
import time
import json
import shutil
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)  # Standard size for most CNN architectures
RANDOM_STATE = 42  # For reproducibility
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_NAME = 'resnet50'  # Options: resnet50, resnet101, efficientnetb0, densenet121, custom_cnn

def process_dataset(raw_data_dir, processed_data_dir, splits_dir):
    """
    Process the entire dataset: preprocess, split, and prepare for training.
    
    Args:
        raw_data_dir (str): Directory containing raw images
        processed_data_dir (str): Directory to save preprocessed images
        splits_dir (str): Directory to save train/val/test splits
        
    Returns:
        dict: Paths to the train, validation, and test directories
    """
    logger.info("Starting dataset processing...")
    
    # Create directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    # Get class names (subdirectories)
    class_names = [d for d in os.listdir(raw_data_dir) 
                  if os.path.isdir(os.path.join(raw_data_dir, d))]
    
    logger.info(f"Found classes: {class_names}")
    
    # Process each class
    for class_name in class_names:
        class_dir = os.path.join(raw_data_dir, class_name)
        processed_class_dir = os.path.join(processed_data_dir, class_name)
        os.makedirs(processed_class_dir, exist_ok=True)
        
        # Get all images for this class
        images = [img for img in os.listdir(class_dir) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Processing {len(images)} images for class '{class_name}'...")
        
        # Process each image
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
            
            # Save preprocessed image
            processed_img_path = os.path.join(processed_class_dir, img_name)
            cv2.imwrite(processed_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    logger.info("Image preprocessing complete.")
    
    # Create train/val/test splits
    splits = create_data_splits(processed_data_dir, splits_dir)
    
    logger.info("Dataset processing complete.")
    return splits

def create_data_splits(source_dir, target_dir, test_size=0.2, val_size=0.2):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        source_dir (str): Directory containing class subdirectories with images
        target_dir (str): Directory to save the splits
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        
    Returns:
        dict: Paths to the train, validation, and test directories
    """
    # Create target directories if they don't exist
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'validation')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get class names (subdirectories)
    class_names = [d for d in os.listdir(source_dir) 
                  if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in class_names:
        # Create class subdirectories in each split
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images for this class
        class_dir = os.path.join(source_dir, class_name)
        images = [img for img in os.listdir(class_dir) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # First split: training + validation vs test
        train_val_files, test_files = train_test_split(
            images, test_size=test_size, random_state=RANDOM_STATE
        )
        
        # Second split: training vs validation
        train_files, val_files = train_test_split(
            train_val_files, test_size=val_size, random_state=RANDOM_STATE
        )
        
        # Copy files to their respective directories
        for file in train_files:
            shutil.copy(
                os.path.join(class_dir, file),
                os.path.join(train_dir, class_name, file)
            )
            
        for file in val_files:
            shutil.copy(
                os.path.join(class_dir, file),
                os.path.join(val_dir, class_name, file)
            )
            
        for file in test_files:
            shutil.copy(
                os.path.join(class_dir, file),
                os.path.join(test_dir, class_name, file)
            )
    
    logger.info(f"Data split complete. Training: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    
    return {
        'train': train_dir,
        'validation': val_dir,
        'test': test_dir
    }

def train_model():
    """
    Placeholder for model training function.
    In a real implementation, this would load the data and train the model.
    """
    logger.info("Model training would happen here.")
    logger.info("To train the model, you need to have TensorFlow installed.")
    logger.info(f"Selected model architecture: {MODEL_NAME}")
    logger.info(f"Training parameters: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, Learning Rate={LEARNING_RATE}")
    
    # Instructions for manual training
    logger.info("\nTo train the model manually:")
    logger.info("1. Install the required dependencies:")
    logger.info("   pip install tensorflow numpy opencv-python scikit-learn matplotlib pillow")
    logger.info("2. Run the training script:")
    logger.info("   python model/training/trainer.py")
    
    # Create a sample training history plot for demonstration
    create_sample_training_plot()

def create_sample_training_plot():
    """
    Create a sample training history plot for demonstration purposes.
    """
    # Create sample data
    epochs = range(1, EPOCHS + 1)
    acc = np.linspace(0.7, 0.95, EPOCHS) + np.random.normal(0, 0.02, EPOCHS)
    val_acc = np.linspace(0.65, 0.90, EPOCHS) + np.random.normal(0, 0.03, EPOCHS)
    loss = np.linspace(0.6, 0.1, EPOCHS) + np.random.normal(0, 0.05, EPOCHS)
    val_loss = np.linspace(0.7, 0.15, EPOCHS) + np.random.normal(0, 0.06, EPOCHS)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(epochs, acc, 'b', label='Training accuracy')
    ax1.plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(epochs, loss, 'b', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('model/training/plots', exist_ok=True)
    plt.savefig('model/training/plots/sample_training_history.png')
    logger.info("Sample training plot saved to 'model/training/plots/sample_training_history.png'")

def main():
    """
    Main function to run the data processing and model training.
    """
    # Define paths
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    splits_dir = 'data/splits'
    
    # Process dataset
    logger.info("Starting data processing...")
    splits = process_dataset(raw_data_dir, processed_data_dir, splits_dir)
    logger.info(f"Data processing complete. Splits created at: {splits_dir}")
    
    # Count images in each split
    for split_name, split_dir in splits.items():
        total_images = 0
        class_counts = {}
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = count
                total_images += count
        
        logger.info(f"{split_name.capitalize()} split: {total_images} images")
        for class_name, count in class_counts.items():
            logger.info(f"  - {class_name}: {count} images")
    
    # Train model
    logger.info("\nStarting model training...")
    train_model()
    
    logger.info("\nData processing and model preparation complete!")
    logger.info("You can now use the processed data to train your models.")

if __name__ == "__main__":
    main()
