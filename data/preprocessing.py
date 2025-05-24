"""
Oral Cancer Detection - Data Preprocessing Module
This module handles image preprocessing, data splitting, and augmentation for the oral cancer detection project.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
IMAGE_SIZE = (224, 224)  # Standard size for most CNN architectures
RANDOM_STATE = 42  # For reproducibility

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

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Preprocess a single image for the model.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        return None
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img / 255.0
    
    return img

def create_data_generators(batch_size=32):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        batch_size (int): Batch size for training
        
    Returns:
        tuple: Training, validation, and test data generators
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=lambda x: x / 255.0
    )
    
    # Validation and test data generators (no augmentation, only rescaling)
    valid_datagen = ImageDataGenerator(preprocessing_function=lambda x: x / 255.0)
    test_datagen = ImageDataGenerator(preprocessing_function=lambda x: x / 255.0)
    
    return train_datagen, valid_datagen, test_datagen

def load_data_generators(train_dir, val_dir, test_dir, batch_size=32, image_size=IMAGE_SIZE):
    """
    Create and configure data generators from directories.
    
    Args:
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        test_dir (str): Path to test data directory
        batch_size (int): Batch size
        image_size (tuple): Target image size
        
    Returns:
        tuple: Configured training, validation, and test generators
    """
    train_datagen, valid_datagen, test_datagen = create_data_generators(batch_size)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

def visualize_augmentation(image_path, num_augmentations=5):
    """
    Visualize data augmentation on a sample image.
    
    Args:
        image_path (str): Path to the image
        num_augmentations (int): Number of augmented samples to generate
    """
    img = preprocess_image(image_path)
    if img is None:
        return
    
    # Create a data generator with augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Reshape for the generator
    img = img.reshape((1,) + img.shape)
    
    # Generate augmented images
    augmented_images = []
    for batch in datagen.flow(img, batch_size=1):
        augmented_images.append(batch[0])
        if len(augmented_images) >= num_augmentations:
            break
    
    # Plot original and augmented images
    plt.figure(figsize=(12, 4))
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(img[0])
    plt.title('Original')
    plt.axis('off')
    
    for i, augmented_img in enumerate(augmented_images):
        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(augmented_img)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

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
            processed_img = preprocess_image(img_path)
            
            if processed_img is not None:
                # Save preprocessed image
                processed_img_path = os.path.join(processed_class_dir, img_name)
                cv2.imwrite(processed_img_path, cv2.cvtColor(
                    (processed_img * 255).astype(np.uint8), 
                    cv2.COLOR_RGB2BGR
                ))
    
    logger.info("Image preprocessing complete.")
    
    # Create train/val/test splits
    splits = create_data_splits(processed_data_dir, splits_dir)
    
    logger.info("Dataset processing complete.")
    return splits

if __name__ == "__main__":
    # Example usage
    RAW_DATA_DIR = "../data/raw"
    PROCESSED_DATA_DIR = "../data/processed"
    SPLITS_DIR = "../data/splits"
    
    splits = process_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR)
    
    # Example of loading data generators
    train_generator, validation_generator, test_generator = load_data_generators(
        splits['train'], 
        splits['validation'], 
        splits['test']
    )
    
    # Print class indices
    print("Class indices:", train_generator.class_indices)
