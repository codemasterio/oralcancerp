"""
Oral Cancer Detection - Model Training Module
This module handles model training, validation, and checkpointing.
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging
import sys
import json

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architectures.models import get_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OralCancerModelTrainer:
    """
    Class to handle model training for oral cancer detection.
    """
    
    def __init__(self, model_name, input_shape=(224, 224, 3), weights='imagenet',
                 learning_rate=0.001, batch_size=32, checkpoints_dir=None, logs_dir=None):
        """
        Initialize the trainer.
        
        Args:
            model_name (str): Name of the model architecture to use
            input_shape (tuple): Input shape for the model
            weights (str): Pre-trained weights to use
            learning_rate (float): Initial learning rate
            batch_size (int): Batch size for training
            checkpoints_dir (str): Directory to save model checkpoints
            logs_dir (str): Directory to save training logs
        """
        self.model_name = model_name
        self.input_shape = input_shape
        self.weights = weights
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Set up directories
        self.checkpoints_dir = checkpoints_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'checkpoints'
        )
        self.logs_dir = logs_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'logs'
        )
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize model
        self.model = self._build_model()
        
        # Training history
        self.history = None
        
    def _build_model(self):
        """
        Build and compile the model.
        
        Returns:
            tf.keras.Model: Compiled model
        """
        logger.info(f"Building model: {self.model_name}")
        
        # Get the model architecture
        model = get_model(
            self.model_name, 
            input_shape=self.input_shape, 
            weights=self.weights
        )
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        # Print model summary
        model.summary(print_fn=logger.info)
        
        return model
    
    def _create_callbacks(self, model_name_with_timestamp):
        """
        Create training callbacks.
        
        Args:
            model_name_with_timestamp (str): Model name with timestamp for file naming
            
        Returns:
            list: List of Keras callbacks
        """
        # Model checkpoint callback
        checkpoint_path = os.path.join(
            self.checkpoints_dir, 
            f"{model_name_with_timestamp}_best.h5"
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=10,
            verbose=1,
            mode='max',
            restore_best_weights=True
        )
        
        # Learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # TensorBoard callback
        tensorboard = TensorBoard(
            log_dir=os.path.join(self.logs_dir, model_name_with_timestamp),
            histogram_freq=1,
            write_graph=True
        )
        
        # CSV Logger
        csv_logger = CSVLogger(
            os.path.join(self.logs_dir, f"{model_name_with_timestamp}_training.csv")
        )
        
        return [checkpoint, early_stopping, reduce_lr, tensorboard, csv_logger]
    
    def train(self, train_generator, validation_generator, epochs=50, fine_tune=True,
              fine_tune_epochs=20, fine_tune_layers=10):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            fine_tune (bool): Whether to fine-tune the model after initial training
            fine_tune_epochs (int): Number of fine-tuning epochs
            fine_tune_layers (int): Number of layers to unfreeze for fine-tuning
            
        Returns:
            dict: Training history
        """
        # Create a timestamp for model naming
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name_with_timestamp = f"{self.model_name}_{timestamp}"
        
        # Create callbacks
        callbacks = self._create_callbacks(model_name_with_timestamp)
        
        logger.info("Starting model training...")
        
        # Initial training phase
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        if fine_tune and hasattr(self.model, 'layers') and len(self.model.layers) > fine_tune_layers:
            logger.info("Starting fine-tuning phase...")
            
            # Unfreeze the last few layers of the base model
            if self.model_name.lower() in ['resnet50', 'resnet101', 'efficientnetb0', 'efficientnetb3', 'densenet121']:
                # For transfer learning models, find the base model
                for layer in self.model.layers:
                    if hasattr(layer, 'layers'):  # This is likely the base model
                        base_model = layer
                        break
                else:
                    logger.warning("Could not identify base model for fine-tuning")
                    base_model = None
                
                if base_model:
                    # Unfreeze the last few layers
                    for layer in base_model.layers[-fine_tune_layers:]:
                        layer.trainable = True
                    
                    # Recompile the model with a lower learning rate
                    self.model.compile(
                        optimizer=Adam(learning_rate=self.learning_rate / 10),
                        loss='binary_crossentropy',
                        metrics=[
                            'accuracy',
                            tf.keras.metrics.Precision(name='precision'),
                            tf.keras.metrics.Recall(name='recall'),
                            tf.keras.metrics.AUC(name='auc')
                        ]
                    )
                    
                    # Create new callbacks for fine-tuning
                    fine_tune_callbacks = self._create_callbacks(f"{model_name_with_timestamp}_fine_tuned")
                    
                    # Continue training with fine-tuning
                    fine_tune_history = self.model.fit(
                        train_generator,
                        epochs=fine_tune_epochs,
                        validation_data=validation_generator,
                        callbacks=fine_tune_callbacks,
                        verbose=1
                    )
                    
                    # Combine histories
                    for key in history.history:
                        history.history[key].extend(fine_tune_history.history[key])
        
        # Save the final model
        final_model_path = os.path.join(
            self.checkpoints_dir, 
            f"{model_name_with_timestamp}_final.h5"
        )
        self.model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save model architecture
        model_json = self.model.to_json()
        with open(os.path.join(self.checkpoints_dir, f"{model_name_with_timestamp}_architecture.json"), 'w') as f:
            f.write(model_json)
        
        # Save training history
        with open(os.path.join(self.logs_dir, f"{model_name_with_timestamp}_history.json"), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, value in history.history.items():
                history_dict[key] = [float(x) for x in value]
            json.dump(history_dict, f)
        
        # Store history for later use
        self.history = history.history
        
        logger.info("Model training complete.")
        return history.history
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            logger.warning("No training history available to plot.")
            return
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axs[0, 0].plot(self.history['accuracy'])
        axs[0, 0].plot(self.history['val_accuracy'])
        axs[0, 0].set_title('Model Accuracy')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        axs[0, 1].plot(self.history['loss'])
        axs[0, 1].plot(self.history['val_loss'])
        axs[0, 1].set_title('Model Loss')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].legend(['Train', 'Validation'], loc='upper right')
        
        # Plot precision
        axs[1, 0].plot(self.history['precision'])
        axs[1, 0].plot(self.history['val_precision'])
        axs[1, 0].set_title('Model Precision')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].legend(['Train', 'Validation'], loc='lower right')
        
        # Plot recall
        axs[1, 1].plot(self.history['recall'])
        axs[1, 1].plot(self.history['val_recall'])
        axs[1, 1].set_title('Model Recall')
        axs[1, 1].set_ylabel('Recall')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].legend(['Train', 'Validation'], loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

def train_model(model_name, train_data_dir, validation_data_dir, epochs=50, 
                batch_size=32, input_shape=(224, 224, 3), learning_rate=0.001,
                fine_tune=True, fine_tune_epochs=20, fine_tune_layers=10):
    """
    Convenience function to train a model.
    
    Args:
        model_name (str): Name of the model architecture to use
        train_data_dir (str): Directory containing training data
        validation_data_dir (str): Directory containing validation data
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        input_shape (tuple): Input shape for the model
        learning_rate (float): Initial learning rate
        fine_tune (bool): Whether to fine-tune the model after initial training
        fine_tune_epochs (int): Number of fine-tuning epochs
        fine_tune_layers (int): Number of layers to unfreeze for fine-tuning
        
    Returns:
        tuple: Trained model and training history
    """
    # Import here to avoid circular imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.preprocessing import load_data_generators
    
    # Create data generators
    train_generator, validation_generator, _ = load_data_generators(
        train_data_dir,
        validation_data_dir,
        validation_data_dir,  # Placeholder, not used
        batch_size=batch_size,
        image_size=input_shape[:2]
    )
    
    # Create trainer
    trainer = OralCancerModelTrainer(
        model_name=model_name,
        input_shape=input_shape,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Train the model
    history = trainer.train(
        train_generator,
        validation_generator,
        epochs=epochs,
        fine_tune=fine_tune,
        fine_tune_epochs=fine_tune_epochs,
        fine_tune_layers=fine_tune_layers
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(
            trainer.logs_dir, 
            f"{model_name}_training_history.png"
        )
    )
    
    return trainer.model, history

if __name__ == "__main__":
    # Example usage
    model_name = "resnet50"
    train_data_dir = "../data/splits/train"
    validation_data_dir = "../data/splits/validation"
    
    model, history = train_model(
        model_name=model_name,
        train_data_dir=train_data_dir,
        validation_data_dir=validation_data_dir,
        epochs=30,
        batch_size=16
    )
