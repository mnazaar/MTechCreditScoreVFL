#!/usr/bin/env python3
"""
Home Loans Feature Predictor Training Script

This script trains a neural network to predict the top 3 features, their direction,
and impact from the 16D intermediate representations of the home loans model.

The model predicts:
- Top 3 feature indices (from feature list)
- Feature directions (positive/negative)
- Feature impacts (Very High/High/Medium/Low)

Usage:
    python train_home_loans_feature_predictor.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging for the training script"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'VFLClientModels/logs/home_loans_feature_predictor_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset():
    """Load the home loans feature predictor dataset"""
    logger = setup_logging()
    logger.info("Loading home loans feature predictor dataset...")
    
    try:
        # Load dataset
        with open('VFLClientModels/models/explanations/data/home_loans_feature_predictor_dataset_sample.json', 'r') as f:
            dataset = json.load(f)
        
        # Load feature names
        with open('VFLClientModels/models/explanations/data/home_loans_feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        return dataset, feature_names
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def prepare_training_data(dataset, feature_names):
    """Prepare training data for the neural network"""
    logger = setup_logging()
    logger.info("Preparing training data...")
    
    # Extract features and targets
    X = []  # Intermediate representations
    y_feature1_idx = []  # Feature 1 index
    y_feature1_direction = []  # Feature 1 direction
    y_feature1_impact = []  # Feature 1 impact
    
    y_feature2_idx = []  # Feature 2 index
    y_feature2_direction = []  # Feature 2 direction
    y_feature2_impact = []  # Feature 2 impact
    
    y_feature3_idx = []  # Feature 3 index
    y_feature3_direction = []  # Feature 3 direction
    y_feature3_impact = []  # Feature 3 impact
    
    # Impact level mapping
    impact_mapping = {'Very High': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    direction_mapping = {'Positive': 0, 'Negative': 1}
    
    for sample in dataset:
        # Input: intermediate representation
        X.append(sample['intermediate_representation'])
        
        # Extract top 3 features
        top_features = sample['top_features']
        
        for i in range(3):
            if i < len(top_features):
                feature = top_features[i]
                
                # Feature index
                feature_idx = feature_names.index(feature['feature_name'])
                
                # Feature direction
                direction = direction_mapping.get(feature['direction'], 0)
                
                # Feature impact
                impact = impact_mapping.get(feature['impact'], 1)
                
                if i == 0:  # Feature 1
                    y_feature1_idx.append(feature_idx)
                    y_feature1_direction.append(direction)
                    y_feature1_impact.append(impact)
                elif i == 1:  # Feature 2
                    y_feature2_idx.append(feature_idx)
                    y_feature2_direction.append(direction)
                    y_feature2_impact.append(impact)
                elif i == 2:  # Feature 3
                    y_feature3_idx.append(feature_idx)
                    y_feature3_direction.append(direction)
                    y_feature3_impact.append(impact)
            else:
                # Pad with zeros if less than 3 features
                if i == 0:
                    y_feature1_idx.append(0)
                    y_feature1_direction.append(0)
                    y_feature1_impact.append(1)
                elif i == 1:
                    y_feature2_idx.append(0)
                    y_feature2_direction.append(0)
                    y_feature2_impact.append(1)
                elif i == 2:
                    y_feature3_idx.append(0)
                    y_feature3_direction.append(0)
                    y_feature3_impact.append(1)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    
    # Convert to one-hot encoding for categorical targets
    num_features = len(feature_names)
    num_directions = 2
    num_impacts = 4
    
    y_feature1_idx = tf.keras.utils.to_categorical(y_feature1_idx, num_classes=num_features).astype(np.float32)
    y_feature1_direction = tf.keras.utils.to_categorical(y_feature1_direction, num_classes=num_directions).astype(np.float32)
    y_feature1_impact = tf.keras.utils.to_categorical(y_feature1_impact, num_classes=num_impacts).astype(np.float32)
    
    y_feature2_idx = tf.keras.utils.to_categorical(y_feature2_idx, num_classes=num_features).astype(np.float32)
    y_feature2_direction = tf.keras.utils.to_categorical(y_feature2_direction, num_classes=num_directions).astype(np.float32)
    y_feature2_impact = tf.keras.utils.to_categorical(y_feature2_impact, num_classes=num_impacts).astype(np.float32)
    
    y_feature3_idx = tf.keras.utils.to_categorical(y_feature3_idx, num_classes=num_features).astype(np.float32)
    y_feature3_direction = tf.keras.utils.to_categorical(y_feature3_direction, num_classes=num_directions).astype(np.float32)
    y_feature3_impact = tf.keras.utils.to_categorical(y_feature3_impact, num_classes=num_impacts).astype(np.float32)
    
    logger.info(f"Training data prepared:")
    logger.info(f"  - X shape: {X.shape}")
    logger.info(f"  - Feature indices: {y_feature1_idx.shape}, {y_feature2_idx.shape}, {y_feature3_idx.shape}")
    logger.info(f"  - Feature directions: {y_feature1_direction.shape}, {y_feature2_direction.shape}, {y_feature3_direction.shape}")
    logger.info(f"  - Feature impacts: {y_feature1_impact.shape}, {y_feature2_impact.shape}, {y_feature3_impact.shape}")
    
    # Create target dictionary with proper data types
    targets = {
        'feature_1_idx': y_feature1_idx,
        'feature_1_direction': y_feature1_direction,
        'feature_1_impact': y_feature1_impact,
        'feature_2_idx': y_feature2_idx,
        'feature_2_direction': y_feature2_direction,
        'feature_2_impact': y_feature2_impact,
        'feature_3_idx': y_feature3_idx,
        'feature_3_direction': y_feature3_direction,
        'feature_3_impact': y_feature3_impact
    }
    
    # Ensure all targets are float32
    for key in targets:
        targets[key] = targets[key].astype(np.float32)
    
    return X, targets

def create_model(input_dim, num_features, num_directions, num_impacts):
    """Create the neural network model"""
    logger = setup_logging()
    logger.info("Creating neural network model...")
    
    # Input layer
    input_layer = keras.layers.Input(shape=(input_dim,), name='intermediate_representation')
    
    # Shared feature extraction layers
    x = keras.layers.Dense(128, activation='relu', name='dense_1')(input_layer)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu', name='dense_3')(x)
    
    # Output layers for each feature
    outputs = {}
    
    for feature_num in [1, 2, 3]:
        # Feature-specific layers
        feature_x = keras.layers.Dense(16, activation='relu', name=f'feature_{feature_num}_dense')(x)
        
        # Feature index (classification)
        feature_idx = keras.layers.Dense(num_features, activation='softmax', 
                                       name=f'feature_{feature_num}_idx')(feature_x)
        
        # Feature direction (classification)
        feature_direction = keras.layers.Dense(num_directions, activation='softmax', 
                                             name=f'feature_{feature_num}_direction')(feature_x)
        
        # Feature impact (classification)
        feature_impact = keras.layers.Dense(num_impacts, activation='softmax', 
                                          name=f'feature_{feature_num}_impact')(feature_x)
        
        outputs[f'feature_{feature_num}_idx'] = feature_idx
        outputs[f'feature_{feature_num}_direction'] = feature_direction
        outputs[f'feature_{feature_num}_impact'] = feature_impact
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=outputs, name='home_loans_feature_predictor')
    
    # Compile model with appropriate losses and metrics
    losses = {}
    metrics = {}
    
    for feature_num in [1, 2, 3]:
        # Classification losses for index, direction, impact
        losses[f'feature_{feature_num}_idx'] = 'categorical_crossentropy'
        losses[f'feature_{feature_num}_direction'] = 'categorical_crossentropy'
        losses[f'feature_{feature_num}_impact'] = 'categorical_crossentropy'
        
        # Metrics
        metrics[f'feature_{feature_num}_idx'] = 'accuracy'
        metrics[f'feature_{feature_num}_direction'] = 'accuracy'
        metrics[f'feature_{feature_num}_impact'] = 'accuracy'
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics
    )
    
    logger.info("Model created successfully")
    logger.info(f"Model parameters: {model.count_params():,}")
    
    return model

def train_model(model, X, y, feature_names):
    """Train the model"""
    logger = setup_logging()
    logger.info("Training the model...")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # Create indices for consistent splitting
    indices = np.arange(len(X))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split X and y using the same indices
    X_train, X_val = X[train_indices], X[val_indices]
    
    y_train = {}
    y_val = {}
    
    for key in y.keys():
        y_train[key] = y[key][train_indices]
        y_val[key] = y[key][val_indices]
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed successfully")
    
    return history

def save_model_and_metadata(model, feature_names, history):
    """Save the trained model and metadata"""
    logger = setup_logging()
    logger.info("Saving model and metadata...")
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Save model
    model_path = os.path.join('VFLClientModels','saved_models', 'home_loans_feature_predictor.keras')
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Save feature names
    feature_names_path = os.path.join('VFLClientModels','saved_models', 'home_loans_feature_predictor_feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Feature names saved: {feature_names_path}")
    
    # Save model info
    model_info = {
        'input_dim': 16,
        'num_features': len(feature_names),
        'num_directions': 2,
        'num_impacts': 4,
        'model_parameters': model.count_params(),
        'training_date': datetime.now().isoformat(),
        'architecture': 'Multi-output neural network for feature prediction'
    }
    
    info_path = os.path.join('VFLClientModels','saved_models', 'home_loans_feature_predictor_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model info saved: {info_path}")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Home Loans Feature Predictor Training History', fontsize=16)
    
    # Plot losses
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot feature index accuracy
    feature1_idx_acc = history.history['feature_1_idx_accuracy']
    feature2_idx_acc = history.history['feature_2_idx_accuracy']
    feature3_idx_acc = history.history['feature_3_idx_accuracy']
    
    axes[0, 1].plot(feature1_idx_acc, label='Feature 1 Index')
    axes[0, 1].plot(feature2_idx_acc, label='Feature 2 Index')
    axes[0, 1].plot(feature3_idx_acc, label='Feature 3 Index')
    axes[0, 1].set_title('Feature Index Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot direction accuracy
    feature1_dir_acc = history.history['feature_1_direction_accuracy']
    feature2_dir_acc = history.history['feature_2_direction_accuracy']
    feature3_dir_acc = history.history['feature_3_direction_accuracy']
    
    axes[1, 0].plot(feature1_dir_acc, label='Feature 1 Direction')
    axes[1, 0].plot(feature2_dir_acc, label='Feature 2 Direction')
    axes[1, 0].plot(feature3_dir_acc, label='Feature 3 Direction')
    axes[1, 0].set_title('Feature Direction Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot impact accuracy
    feature1_impact_acc = history.history['feature_1_impact_accuracy']
    feature2_impact_acc = history.history['feature_2_impact_accuracy']
    feature3_impact_acc = history.history['feature_3_impact_accuracy']
    
    axes[1, 1].plot(feature1_impact_acc, label='Feature 1 Impact')
    axes[1, 1].plot(feature2_impact_acc, label='Feature 2 Impact')
    axes[1, 1].plot(feature3_impact_acc, label='Feature 3 Impact')
    axes[1, 1].set_title('Feature Impact Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    plot_path = os.path.join('VFLClientModels','plots', 'home_loans_feature_predictor_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved: {plot_path}")
    
    plt.close()

def main():
    """Main function to train the home loans feature predictor"""
    logger = setup_logging()
    
    logger.info("🚀 HOME LOANS FEATURE PREDICTOR TRAINING")
    logger.info("=" * 80)
    logger.info("Training a neural network to predict top 3 features from intermediate representations")
    logger.info("")
    
    try:
        # Step 1: Load dataset
        logger.info("Step 1: Loading dataset...")
        dataset, feature_names = load_dataset()
        
        # Step 2: Prepare training data
        logger.info("Step 2: Preparing training data...")
        X, y = prepare_training_data(dataset, feature_names)
        
        # Step 3: Create model
        logger.info("Step 3: Creating model...")
        model = create_model(
            input_dim=16,
            num_features=len(feature_names),
            num_directions=2,
            num_impacts=4
        )
        
        # Step 4: Train model
        logger.info("Step 4: Training model...")
        history = train_model(model, X, y, feature_names)
        
        # Step 5: Save model and metadata
        logger.info("Step 5: Saving model and metadata...")
        save_model_and_metadata(model, feature_names, history)
        
        # Summary
        logger.info("")
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model trained on {len(X)} samples")
        logger.info(f"Input: 16D intermediate representations")
        logger.info(f"Output: Top 3 features with indices, directions, and impacts")
        logger.info(f"Model parameters: {model.count_params():,}")
        logger.info("")
        logger.info("📁 Generated Files:")
        logger.info("   - Model: saved_models/home_loans_feature_predictor.keras")
        logger.info("   - Feature names: saved_models/home_loans_feature_predictor_feature_names.json")
        logger.info("   - Model info: saved_models/home_loans_feature_predictor_info.json")
        logger.info("   - Training plots: plots/home_loans_feature_predictor_training_history.png")
        logger.info("")
        logger.info("🔧 Next Steps:")
        logger.info("   - Use the trained model to predict features from new intermediate representations")
        logger.info("   - Integrate with the home loans explainer system")
        logger.info("=" * 80)
        
        return model, feature_names, history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
