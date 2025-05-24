"""
Oral Cancer Detection - Model Architectures
This module defines various CNN architectures for oral cancer detection.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, 
    BatchNormalization, Flatten, Activation
)
from tensorflow.keras.applications import (
    ResNet50, ResNet101, EfficientNetB0, EfficientNetB3, DenseNet121
)

def create_resnet50_model(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Create a ResNet50 model with transfer learning.
    
    Args:
        input_shape (tuple): Input shape of images
        weights (str): Pre-trained weights to use
        
    Returns:
        tf.keras.Model: Compiled ResNet50 model
    """
    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_resnet101_model(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Create a ResNet101 model with transfer learning.
    
    Args:
        input_shape (tuple): Input shape of images
        weights (str): Pre-trained weights to use
        
    Returns:
        tf.keras.Model: Compiled ResNet101 model
    """
    base_model = ResNet101(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_efficientnet_model(input_shape=(224, 224, 3), weights='imagenet', version='B0'):
    """
    Create an EfficientNet model with transfer learning.
    
    Args:
        input_shape (tuple): Input shape of images
        weights (str): Pre-trained weights to use
        version (str): EfficientNet version (B0, B3, etc.)
        
    Returns:
        tf.keras.Model: Compiled EfficientNet model
    """
    if version == 'B0':
        base_model = EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    elif version == 'B3':
        base_model = EfficientNetB3(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported EfficientNet version: {version}")
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_densenet_model(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Create a DenseNet121 model with transfer learning.
    
    Args:
        input_shape (tuple): Input shape of images
        weights (str): Pre-trained weights to use
        
    Returns:
        tf.keras.Model: Compiled DenseNet model
    """
    base_model = DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def create_custom_cnn_model(input_shape=(224, 224, 3)):
    """
    Create a custom CNN architecture for oral cancer detection.
    
    Args:
        input_shape (tuple): Input shape of images
        
    Returns:
        tf.keras.Model: Compiled custom CNN model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    return model

def get_model(model_name, input_shape=(224, 224, 3), weights='imagenet'):
    """
    Factory function to create a model based on the model name.
    
    Args:
        model_name (str): Name of the model to create
        input_shape (tuple): Input shape of images
        weights (str): Pre-trained weights to use
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model_name = model_name.lower()
    
    if model_name == 'resnet50':
        return create_resnet50_model(input_shape, weights)
    elif model_name == 'resnet101':
        return create_resnet101_model(input_shape, weights)
    elif model_name == 'efficientnetb0':
        return create_efficientnet_model(input_shape, weights, 'B0')
    elif model_name == 'efficientnetb3':
        return create_efficientnet_model(input_shape, weights, 'B3')
    elif model_name == 'densenet121':
        return create_densenet_model(input_shape, weights)
    elif model_name == 'custom_cnn':
        return create_custom_cnn_model(input_shape)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
