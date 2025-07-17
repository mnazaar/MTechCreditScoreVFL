#!/usr/bin/env python3
"""
Auto Loans Feature Predictor Training Script

This script trains a neural network to predict the top 3 features, their direction,
and impact from the 16D intermediate representations of the auto loans model.

The model predicts:
- Top 3 feature indices (from feature list)
- Feature directions (positive/negative)
- Feature impacts (Very High/High/Medium/Low)

Usage:
    python train_auto_loans_feature_predictor.py
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
            logging.FileHandler(f'VFLClientModels/logs/auto_loans_feature_predictor_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset():
    """Load the auto loans feature predictor dataset"""
    logger = setup_logging()
    logger.info("Loading auto loans feature predictor dataset.")
    
    try:
        # Load dataset
        with open('VFLClientModels/models/explanations/data/auto_loans_feature_predictor_dataset_sample.json', 'r') as f:
            dataset = json.load(f)
        
        # Load feature names
        with open('VFLClientModels/models/explanations/data/auto_loans_feature_names.txt', 'r') as f:
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
    logger.info("Preparing training data.")
    
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
    
    # Statistics tracking
    impact_counts = {'Very High': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    direction_counts = {'Positive': 0, 'Negative': 0}
    feature_usage_counts = {name: 0 for name in feature_names}
    
    logger.info(f"Processing {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"  Processed {i}/{len(dataset)} samples...")
            
        # Input: intermediate representation
        X.append(sample['intermediate_representation'])
        
        # Extract top 3 features
        top_features = sample['top_features']
        
        for j in range(3):
            if j < len(top_features):
                feature = top_features[j]
                
                # Feature index
                feature_idx = feature_names.index(feature['feature_name'])
                feature_usage_counts[feature['feature_name']] += 1
                
                # Feature direction
                direction = direction_mapping.get(feature['direction'], 0)
                direction_counts[feature['direction']] += 1
                
                # Feature impact
                impact = impact_mapping.get(feature['impact'], 1)
                impact_counts[feature['impact']] += 1
                
                if j == 0:  # Feature 1
                    y_feature1_idx.append(feature_idx)
                    y_feature1_direction.append(direction)
                    y_feature1_impact.append(impact)
                elif j == 1:  # Feature 2
                    y_feature2_idx.append(feature_idx)
                    y_feature2_direction.append(direction)
                    y_feature2_impact.append(impact)
                elif j == 2:  # Feature 3
                    y_feature3_idx.append(feature_idx)
                    y_feature3_direction.append(direction)
                    y_feature3_impact.append(impact)
            else:
                # Pad with zeros if less than 3 features
                if j == 0:
                    y_feature1_idx.append(0)
                    y_feature1_direction.append(0)
                    y_feature1_impact.append(1)
                elif j == 1:
                    y_feature2_idx.append(0)
                    y_feature2_direction.append(0)
                    y_feature2_impact.append(1)
                elif j == 2:
                    y_feature3_idx.append(0)
                    y_feature3_direction.append(0)
                    y_feature3_impact.append(1)
    
    logger.info(f"Data processing completed!")
    
    # Log statistics
    logger.info(f"📊 Data Preparation Statistics:")
    logger.info(f"  Total samples processed: {len(X)}")
    logger.info(f"  Intermediate representation dimension: {len(X[0])}")
    
    # Feature usage statistics
    top_10_features = sorted(feature_usage_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"  Top 10 most used features:")
    for i, (feature, count) in enumerate(top_10_features, 1):
        percentage = (count / (len(dataset) * 3)) * 100
        logger.info(f"    {i:2d}. {feature:<25} {count:>4} times ({percentage:>5.1f}%)")
    
    # Impact distribution
    total_impacts = sum(impact_counts.values())
    logger.info(f"  Impact level distribution:")
    for impact, count in impact_counts.items():
        percentage = (count / total_impacts) * 100
        logger.info(f"    {impact:<10}: {count:>4} ({percentage:>5.1f}%)")
    
    # Direction distribution
    total_directions = sum(direction_counts.values())
    logger.info(f"  Direction distribution:")
    for direction, count in direction_counts.items():
        percentage = (count / total_directions) * 100
        logger.info(f"    {direction:<10}: {count:>4} ({percentage:>5.1f}%)")
    
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
    logger.info("Creating neural network model.")
    
    # Input layer
    input_layer = keras.layers.Input(shape=(input_dim,), name='intermediate_representation')
    
    # Shared feature extraction layers
    x = keras.layers.Dense(1024, activation='relu', name='dense_1')(input_layer)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(512, activation='relu', name='dense_2')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(256, activation='relu', name='dense_3')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation='relu', name='dense_4')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu', name='dense_5')(x)
    
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
    model = keras.Model(inputs=input_layer, outputs=outputs, name='auto_loans_feature_predictor')
    
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
    logger.info("Training the model.")
    
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
    logger.info(f"Batch size: 32")
    logger.info(f"Learning rate: 0.001")
    logger.info(f"Max epochs: 100")
    logger.info("")
    
    # Custom callback for detailed logging
    class DetailedLoggingCallback(keras.callbacks.Callback):
        def __init__(self, logger):
            super().__init__()
            self.logger = logger
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            
        def on_epoch_begin(self, epoch, logs=None):
            self.logger.info(f"Epoch {epoch+1}/100 - Starting...")
            
        def on_epoch_end(self, epoch, logs=None):
            # Extract metrics
            train_loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            
            # Feature index accuracies
            feature1_idx_acc = logs.get('feature_1_idx_accuracy', 0)
            feature2_idx_acc = logs.get('feature_2_idx_accuracy', 0)
            feature3_idx_acc = logs.get('feature_3_idx_accuracy', 0)
            
            # Feature direction accuracies
            feature1_dir_acc = logs.get('feature_1_direction_accuracy', 0)
            feature2_dir_acc = logs.get('feature_2_direction_accuracy', 0)
            feature3_dir_acc = logs.get('feature_3_direction_accuracy', 0)
            
            # Feature impact accuracies
            feature1_impact_acc = logs.get('feature_1_impact_accuracy', 0)
            feature2_impact_acc = logs.get('feature_2_impact_accuracy', 0)
            feature3_impact_acc = logs.get('feature_3_impact_accuracy', 0)
            
            # Validation accuracies
            val_feature1_idx_acc = logs.get('val_feature_1_idx_accuracy', 0)
            val_feature2_idx_acc = logs.get('val_feature_2_idx_accuracy', 0)
            val_feature3_idx_acc = logs.get('val_feature_3_idx_accuracy', 0)
            
            val_feature1_dir_acc = logs.get('val_feature_1_direction_accuracy', 0)
            val_feature2_dir_acc = logs.get('val_feature_2_direction_accuracy', 0)
            val_feature3_dir_acc = logs.get('val_feature_3_direction_accuracy', 0)
            
            val_feature1_impact_acc = logs.get('val_feature_1_impact_accuracy', 0)
            val_feature2_impact_acc = logs.get('val_feature_2_impact_accuracy', 0)
            val_feature3_impact_acc = logs.get('val_feature_3_impact_accuracy', 0)
            
            # Check if this is the best epoch
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                improvement = "⭐ NEW BEST!"
            else:
                improvement = ""
            
            # Log detailed epoch information
            self.logger.info(f"Epoch {epoch+1}/100 - Completed {improvement}")
            self.logger.info(f"  📊 Loss: Train={train_loss:.4f}, Val={val_loss:.4f}")
            self.logger.info(f"  🎯 Feature Index Accuracy:")
            self.logger.info(f"    Train: F1={feature1_idx_acc:.3f}, F2={feature2_idx_acc:.3f}, F3={feature3_idx_acc:.3f}")
            self.logger.info(f"    Val:   F1={val_feature1_idx_acc:.3f}, F2={val_feature2_idx_acc:.3f}, F3={val_feature3_idx_acc:.3f}")
            self.logger.info(f"  📈 Feature Direction Accuracy:")
            self.logger.info(f"    Train: F1={feature1_dir_acc:.3f}, F2={feature2_dir_acc:.3f}, F3={feature3_dir_acc:.3f}")
            self.logger.info(f"    Val:   F1={val_feature1_dir_acc:.3f}, F2={val_feature2_dir_acc:.3f}, F3={val_feature3_dir_acc:.3f}")
            self.logger.info(f"  💪 Feature Impact Accuracy:")
            self.logger.info(f"    Train: F1={feature1_impact_acc:.3f}, F2={feature2_impact_acc:.3f}, F3={feature3_impact_acc:.3f}")
            self.logger.info(f"    Val:   F1={val_feature1_impact_acc:.3f}, F2={val_feature2_impact_acc:.3f}, F3={val_feature3_impact_acc:.3f}")
            
            # Calculate average accuracies
            avg_train_idx_acc = (feature1_idx_acc + feature2_idx_acc + feature3_idx_acc) / 3
            avg_val_idx_acc = (val_feature1_idx_acc + val_feature2_idx_acc + val_feature3_idx_acc) / 3
            avg_train_dir_acc = (feature1_dir_acc + feature2_dir_acc + feature3_dir_acc) / 3
            avg_val_dir_acc = (val_feature1_dir_acc + val_feature2_dir_acc + val_feature3_dir_acc) / 3
            avg_train_impact_acc = (feature1_impact_acc + feature2_impact_acc + feature3_impact_acc) / 3
            avg_val_impact_acc = (val_feature1_impact_acc + val_feature2_impact_acc + val_feature3_impact_acc) / 3
            
            self.logger.info(f"  📊 Average Accuracies:")
            self.logger.info(f"    Index:   Train={avg_train_idx_acc:.3f}, Val={avg_val_idx_acc:.3f}")
            self.logger.info(f"    Direction: Train={avg_train_dir_acc:.3f}, Val={avg_val_dir_acc:.3f}")
            self.logger.info(f"    Impact:   Train={avg_train_impact_acc:.3f}, Val={avg_val_impact_acc:.3f}")
            self.logger.info(f"  🏆 Best Validation Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
            self.logger.info("-" * 80)
    
    # Callbacks
    callbacks = [
        DetailedLoggingCallback(logger),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    logger.info("🚀 Starting training...")
    logger.info("=" * 80)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0  # Set to 0 since we have custom logging
    )
    
    # Final training summary
    logger.info("")
    logger.info("🎉 TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Total epochs trained: {len(history.history['loss'])}")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    logger.info(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
    
    # Final accuracies
    final_epoch = len(history.history['loss']) - 1
    logger.info(f"Final Training Accuracies (Epoch {final_epoch + 1}):")
    logger.info(f"  Feature Index: {history.history['feature_1_idx_accuracy'][-1]:.3f}, {history.history['feature_2_idx_accuracy'][-1]:.3f}, {history.history['feature_3_idx_accuracy'][-1]:.3f}")
    logger.info(f"  Feature Direction: {history.history['feature_1_direction_accuracy'][-1]:.3f}, {history.history['feature_2_direction_accuracy'][-1]:.3f}, {history.history['feature_3_direction_accuracy'][-1]:.3f}")
    logger.info(f"  Feature Impact: {history.history['feature_1_impact_accuracy'][-1]:.3f}, {history.history['feature_2_impact_accuracy'][-1]:.3f}, {history.history['feature_3_impact_accuracy'][-1]:.3f}")
    
    logger.info(f"Final Validation Accuracies (Epoch {final_epoch + 1}):")
    logger.info(f"  Feature Index: {history.history['val_feature_1_idx_accuracy'][-1]:.3f}, {history.history['val_feature_2_idx_accuracy'][-1]:.3f}, {history.history['val_feature_3_idx_accuracy'][-1]:.3f}")
    logger.info(f"  Feature Direction: {history.history['val_feature_1_direction_accuracy'][-1]:.3f}, {history.history['val_feature_2_direction_accuracy'][-1]:.3f}, {history.history['val_feature_3_direction_accuracy'][-1]:.3f}")
    logger.info(f"  Feature Impact: {history.history['val_feature_1_impact_accuracy'][-1]:.3f}, {history.history['val_feature_2_impact_accuracy'][-1]:.3f}, {history.history['val_feature_3_impact_accuracy'][-1]:.3f}")
    
    # Calculate and log average final accuracies
    avg_final_train_idx = np.mean([history.history['feature_1_idx_accuracy'][-1], history.history['feature_2_idx_accuracy'][-1], history.history['feature_3_idx_accuracy'][-1]])
    avg_final_val_idx = np.mean([history.history['val_feature_1_idx_accuracy'][-1], history.history['val_feature_2_idx_accuracy'][-1], history.history['val_feature_3_idx_accuracy'][-1]])
    
    avg_final_train_dir = np.mean([history.history['feature_1_direction_accuracy'][-1], history.history['feature_2_direction_accuracy'][-1], history.history['feature_3_direction_accuracy'][-1]])
    avg_final_val_dir = np.mean([history.history['val_feature_1_direction_accuracy'][-1], history.history['val_feature_2_direction_accuracy'][-1], history.history['val_feature_3_direction_accuracy'][-1]])
    
    avg_final_train_impact = np.mean([history.history['feature_1_impact_accuracy'][-1], history.history['feature_2_impact_accuracy'][-1], history.history['feature_3_impact_accuracy'][-1]])
    avg_final_val_impact = np.mean([history.history['val_feature_1_impact_accuracy'][-1], history.history['val_feature_2_impact_accuracy'][-1], history.history['val_feature_3_impact_accuracy'][-1]])
    
    logger.info(f"Final Average Accuracies:")
    logger.info(f"  Index:   Train={avg_final_train_idx:.3f}, Val={avg_final_val_idx:.3f}")
    logger.info(f"  Direction: Train={avg_final_train_dir:.3f}, Val={avg_final_val_dir:.3f}")
    logger.info(f"  Impact:   Train={avg_final_train_impact:.3f}, Val={avg_final_val_impact:.3f}")
    
    logger.info("Training completed successfully")
    
    return history

def save_model_and_metadata(model, feature_names, history):
    """Save the trained model and metadata"""
    logger = setup_logging()
    logger.info("Saving model and metadata.")
    

    # Save model
    model_path = os.path.join('VFLClientModels','saved_models', 'auto_loans_feature_predictor.keras')
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Save feature names
    feature_names_path = os.path.join('VFLClientModels','saved_models', 'auto_loans_feature_predictor_feature_names.json')
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
    
    info_path = os.path.join('VFLClientModels', 'saved_models', 'auto_loans_feature_predictor_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model info saved: {info_path}")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Auto Loans Feature Predictor Training History', fontsize=16)
    
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
    
    plot_path = os.path.join('VFLClientModels','plots', 'auto_loans_feature_predictor_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved: {plot_path}")
    
    plt.close()

def main():
    """Main function to train the auto loans feature predictor"""
    logger = setup_logging()
    
    logger.info("🚀 AUTO LOANS FEATURE PREDICTOR TRAINING")
    logger.info("=" * 80)
    logger.info("Training a neural network to predict top 3 features from intermediate representations")
    logger.info("")
    
    try:
        # Step 1: Load dataset
        logger.info("Step 1: Loading dataset.")
        dataset, feature_names = load_dataset()
        
        # Log dataset statistics
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"  Total samples: {len(dataset)}")
        logger.info(f"  Number of features: {len(feature_names)}")
        logger.info(f"  Intermediate representation dimension: {len(dataset[0]['intermediate_representation'])}")
        logger.info(f"  Top features per sample: {len(dataset[0]['top_features'])}")
        
        # Analyze feature distribution
        all_features = []
        for sample in dataset:
            for feature in sample['top_features']:
                all_features.append(feature['feature_name'])
        
        from collections import Counter
        feature_counts = Counter(all_features)
        top_5_features = feature_counts.most_common(5)
        
        logger.info(f"  Top 5 most common features:")
        for i, (feature, count) in enumerate(top_5_features, 1):
            percentage = (count / len(dataset)) * 100
            logger.info(f"    {i}. {feature}: {count} times ({percentage:.1f}%)")
        
        logger.info("")
        
        # Step 2: Prepare training data
        logger.info("Step 2: Preparing training data.")
        X, y = prepare_training_data(dataset, feature_names)
        
        # Log training data details
        logger.info(f"📈 Training Data Details:")
        logger.info(f"  Input shape: {X.shape}")
        logger.info(f"  Number of output targets: {len(y)}")
        for key, value in y.items():
            logger.info(f"  {key}: {value.shape}")
        
        # Calculate class distribution for each target
        logger.info(f"  Target Class Distributions:")
        for key, value in y.items():
            if 'idx' in key:
                # For feature indices
                class_counts = np.sum(value, axis=0)
                logger.info(f"    {key}: {len(class_counts)} classes")
            elif 'direction' in key:
                # For directions (2 classes: Positive/Negative)
                class_counts = np.sum(value, axis=0)
                logger.info(f"    {key}: Positive={class_counts[0]}, Negative={class_counts[1]}")
            elif 'impact' in key:
                # For impacts (4 classes: Very High/High/Medium/Low)
                class_counts = np.sum(value, axis=0)
                impact_names = ['Very High', 'High', 'Medium', 'Low']
                logger.info(f"    {key}: {', '.join([f'{name}={count}' for name, count in zip(impact_names, class_counts)])}")
        
        logger.info("")
        
        # Step 3: Create model
        logger.info("Step 3: Creating model.")
        model = create_model(
            input_dim=16,
            num_features=len(feature_names),
            num_directions=2,
            num_impacts=4
        )
        
        # Log model architecture details
        logger.info(f"🏗️  Model Architecture:")
        logger.info(f"  Input dimension: 16")
        logger.info(f"  Number of features: {len(feature_names)}")
        logger.info(f"  Number of directions: 2 (Positive/Negative)")
        logger.info(f"  Number of impacts: 4 (Very High/High/Medium/Low)")
        logger.info(f"  Total parameters: {model.count_params():,}")
        
        # Calculate model size
        model_size_mb = model.count_params() * 4 / (1024 * 1024)  # Assuming float32
        logger.info(f"  Estimated model size: {model_size_mb:.2f} MB")
        
        # Show layer summary
        logger.info(f"  Layer summary:")
        for i, layer in enumerate(model.layers):
            output_shape = layer.output_shape if hasattr(layer, 'output_shape') else 'N/A'
            params = layer.count_params() if hasattr(layer, 'count_params') else 0
            logger.info(f"    {i+1:2d}. {layer.name:<20} {str(output_shape):<20} {params:>8,} params")
        
        logger.info("")
        
        # Step 4: Train model
        logger.info("Step 4: Training model.")
        history = train_model(model, X, y, feature_names)
        
        # Step 5: Save model and metadata
        logger.info("Step 5: Saving model and metadata.")
        save_model_and_metadata(model, feature_names, history)
        
        # Summary
        logger.info("")
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model trained on {len(X)} samples")
        logger.info(f"Input: 16D intermediate representations")
        logger.info(f"Output: Top 3 features with indices, directions, and impacts")
        logger.info(f"Model parameters: {model.count_params():,}")
        
        # Training performance summary
        final_epoch = len(history.history['loss'])
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = min(history.history['val_loss'])
        
        logger.info(f"")
        logger.info(f"📊 Training Performance Summary:")
        logger.info(f"  Total epochs: {final_epoch}")
        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  Best validation loss: {best_val_loss:.4f}")
        
        # Final accuracies
        avg_final_val_idx = np.mean([history.history['val_feature_1_idx_accuracy'][-1], 
                                   history.history['val_feature_2_idx_accuracy'][-1], 
                                   history.history['val_feature_3_idx_accuracy'][-1]])
        avg_final_val_dir = np.mean([history.history['val_feature_1_direction_accuracy'][-1], 
                                   history.history['val_feature_2_direction_accuracy'][-1], 
                                   history.history['val_feature_3_direction_accuracy'][-1]])
        avg_final_val_impact = np.mean([history.history['val_feature_1_impact_accuracy'][-1], 
                                      history.history['val_feature_2_impact_accuracy'][-1], 
                                      history.history['val_feature_3_impact_accuracy'][-1]])
        
        logger.info(f"  Final validation accuracies:")
        logger.info(f"    Feature Index: {avg_final_val_idx:.3f}")
        logger.info(f"    Feature Direction: {avg_final_val_dir:.3f}")
        logger.info(f"    Feature Impact: {avg_final_val_impact:.3f}")
        
        logger.info("")
        logger.info("📁 Generated Files:")
        logger.info("   - Model: VFLClientModels/saved_models/auto_loans_feature_predictor.keras")
        logger.info("   - Feature names: VFLClientModels/saved_models/auto_loans_feature_predictor_feature_names.json")
        logger.info("   - Model info: VFLClientModels/saved_models/auto_loans_feature_predictor_info.json")
        logger.info("   - Training plots: VFLClientModels/plots/auto_loans_feature_predictor_training_history.png")
        logger.info("")
        logger.info("🔧 Next Steps:")
        logger.info("   - Use the trained model to predict features from new intermediate representations")
        logger.info("   - Integrate with the auto loans explainer system")
        logger.info("=" * 80)
        
        return model, feature_names, history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 