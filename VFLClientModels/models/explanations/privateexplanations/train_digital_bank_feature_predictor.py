#!/usr/bin/env python3
"""
Digital Bank Feature Predictor Training Script

This script trains a neural network to predict the top 3 features, their direction,
and impact from the intermediate representations of the digital bank model.

The model predicts:
- Top 3 feature indices (from feature list)
- Feature directions (positive/negative)
- Feature impacts (Very High/High/Medium/Low)

Usage:
    python train_digital_bank_feature_predictor.py
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
            logging.FileHandler(f'VFLClientModels/logs/digital_bank_feature_predictor_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset():
    """Load the digital bank feature predictor dataset"""
    logger = setup_logging()
    logger.info("Loading digital bank feature predictor dataset...")
    
    try:
        # Load dataset
        with open('VFLClientModels/models/explanations/data/digital_bank_feature_predictor_dataset_sample.json', 'r') as f:
            dataset = json.load(f)
        
        # Load feature names
        with open('VFLClientModels/models/explanations/data/digital_bank_feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        # Log dataset statistics
        logger.info("📊 DATASET STATISTICS:")
        logger.info(f"   - Total samples: {len(dataset)}")
        logger.info(f"   - Total features: {len(feature_names)}")
        
        # Analyze intermediate representation dimensions
        if dataset:
            sample_ir = dataset[0]['intermediate_representation']
            logger.info(f"   - Intermediate representation dimension: {len(sample_ir)}")
            logger.info(f"   - IR data type: {type(sample_ir)}")
            logger.info(f"   - IR value range: [{min(sample_ir):.6f}, {max(sample_ir):.6f}]")
            logger.info(f"   - IR mean: {np.mean(sample_ir):.6f}")
            logger.info(f"   - IR std: {np.std(sample_ir):.6f}")
        
        # Analyze feature distribution
        feature_counts = {}
        direction_counts = {'Positive': 0, 'Negative': 0}
        impact_counts = {'Very High': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for sample in dataset:
            for feature in sample['top_features']:
                # Count features
                feature_name = feature['feature_name']
                feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
                
                # Count directions
                direction_counts[feature['direction']] += 1
                
                # Count impacts
                impact_counts[feature['impact']] += 1
        
        logger.info("   - Feature frequency (top 10):")
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, count) in enumerate(sorted_features[:10]):
            logger.info(f"     {i+1:2d}. {feature}: {count} times")
        
        logger.info("   - Direction distribution:")
        for direction, count in direction_counts.items():
            percentage = (count / sum(direction_counts.values())) * 100
            logger.info(f"     {direction}: {count} ({percentage:.1f}%)")
        
        logger.info("   - Impact distribution:")
        for impact, count in impact_counts.items():
            percentage = (count / sum(impact_counts.values())) * 100
            logger.info(f"     {impact}: {count} ({percentage:.1f}%)")
        
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
    
    logger.info("📋 DATA PREPARATION DETAILS:")
    logger.info(f"   - Impact mapping: {impact_mapping}")
    logger.info(f"   - Direction mapping: {direction_mapping}")
    
    # Track statistics
    total_features_processed = 0
    features_with_less_than_3 = 0
    
    for sample_idx, sample in enumerate(dataset):
        # Input: intermediate representation
        X.append(sample['intermediate_representation'])
        
        # Extract top 3 features
        top_features = sample['top_features']
        total_features_processed += len(top_features)
        
        if len(top_features) < 3:
            features_with_less_than_3 += 1
        
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
    
    logger.info("📊 TRAINING DATA STATISTICS:")
    logger.info(f"   - X shape: {X.shape}")
    logger.info(f"   - Feature indices: {y_feature1_idx.shape}, {y_feature2_idx.shape}, {y_feature3_idx.shape}")
    logger.info(f"   - Feature directions: {y_feature1_direction.shape}, {y_feature2_direction.shape}, {y_feature3_direction.shape}")
    logger.info(f"   - Feature impacts: {y_feature1_impact.shape}, {y_feature2_impact.shape}, {y_feature3_impact.shape}")
    logger.info(f"   - Total features processed: {total_features_processed}")
    logger.info(f"   - Samples with <3 features: {features_with_less_than_3} ({features_with_less_than_3/len(dataset)*100:.1f}%)")
    logger.info(f"   - Average features per sample: {total_features_processed/len(dataset):.2f}")
    
    # Log data statistics
    logger.info("   - Input data statistics:")
    logger.info(f"     X range: [{X.min():.6f}, {X.max():.6f}]")
    logger.info(f"     X mean: {X.mean():.6f}")
    logger.info(f"     X std: {X.std():.6f}")
    logger.info(f"     X has NaN: {np.isnan(X).any()}")
    logger.info(f"     X has Inf: {np.isinf(X).any()}")
    
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
    model = keras.Model(inputs=input_layer, outputs=outputs, name='digital_bank_feature_predictor')
    
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
    
    logger.info("🏗️ MODEL ARCHITECTURE:")
    logger.info(f"   - Input dimension: {input_dim}")
    logger.info(f"   - Number of features: {num_features}")
    logger.info(f"   - Number of directions: {num_directions}")
    logger.info(f"   - Number of impacts: {num_impacts}")
    logger.info(f"   - Model parameters: {model.count_params():,}")
    logger.info(f"   - Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    logger.info(f"   - Non-trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    # Log layer information
    logger.info("   - Layer structure:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'output_shape'):
            logger.info(f"     Layer {i+1:2d}: {layer.name:25s} - Output: {layer.output_shape}")
        else:
            logger.info(f"     Layer {i+1:2d}: {layer.name:25s}")
    
    logger.info("   - Output layers:")
    for output_name, output_layer in outputs.items():
        logger.info(f"     {output_name}: {output_layer.shape}")
    
    logger.info("Model created successfully")
    
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
    
    logger.info("📊 TRAINING SETUP:")
    logger.info(f"   - Training samples: {len(X_train)}")
    logger.info(f"   - Validation samples: {len(X_val)}")
    logger.info(f"   - Training/validation split: {len(X_train)/(len(X_train)+len(X_val))*100:.1f}% / {len(X_val)/(len(X_train)+len(X_val))*100:.1f}%")
    
    # Log training data statistics
    logger.info("   - Training data statistics:")
    logger.info(f"     X_train range: [{X_train.min():.6f}, {X_train.max():.6f}]")
    logger.info(f"     X_train mean: {X_train.mean():.6f}")
    logger.info(f"     X_train std: {X_train.std():.6f}")
    logger.info(f"     X_val range: [{X_val.min():.6f}, {X_val.max():.6f}]")
    logger.info(f"     X_val mean: {X_val.mean():.6f}")
    logger.info(f"     X_val std: {X_val.std():.6f}")
    
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
    
    logger.info("🎯 TRAINING CONFIGURATION:")
    logger.info("   - Optimizer: Adam (lr=0.001)")
    logger.info("   - Loss: Categorical Crossentropy")
    logger.info("   - Metrics: Accuracy")
    logger.info("   - Batch size: 32")
    logger.info("   - Max epochs: 100")
    logger.info("   - Early stopping: patience=10")
    logger.info("   - Learning rate reduction: patience=5, factor=0.5")
    
    # Train model
    logger.info("🚀 STARTING TRAINING...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("✅ Training completed successfully")
    
    # Log final training results
    logger.info("📈 FINAL TRAINING RESULTS:")
    final_epoch = len(history.history['loss'])
    logger.info(f"   - Total epochs trained: {final_epoch}")
    logger.info(f"   - Final training loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"   - Final validation loss: {history.history['val_loss'][-1]:.6f}")
    
    # Log final accuracies for each output
    logger.info("   - Final accuracies:")
    for feature_num in [1, 2, 3]:
        idx_acc = history.history[f'feature_{feature_num}_idx_accuracy'][-1]
        dir_acc = history.history[f'feature_{feature_num}_direction_accuracy'][-1]
        impact_acc = history.history[f'feature_{feature_num}_impact_accuracy'][-1]
        
        logger.info(f"     Feature {feature_num}:")
        logger.info(f"       Index accuracy: {idx_acc:.4f}")
        logger.info(f"       Direction accuracy: {dir_acc:.4f}")
        logger.info(f"       Impact accuracy: {impact_acc:.4f}")
    
    # Log best accuracies
    logger.info("   - Best accuracies achieved:")
    for feature_num in [1, 2, 3]:
        idx_acc = max(history.history[f'feature_{feature_num}_idx_accuracy'])
        dir_acc = max(history.history[f'feature_{feature_num}_direction_accuracy'])
        impact_acc = max(history.history[f'feature_{feature_num}_impact_accuracy'])
        
        logger.info(f"     Feature {feature_num}:")
        logger.info(f"       Best index accuracy: {idx_acc:.4f}")
        logger.info(f"       Best direction accuracy: {dir_acc:.4f}")
        logger.info(f"       Best impact accuracy: {impact_acc:.4f}")
    
    return history

def save_model_and_metadata(model, feature_names, history):
    """Save the trained model and metadata"""
    logger = setup_logging()
    logger.info("Saving model and metadata...")
    
    # Create directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Save model
    model_path = os.path.join('VFLClientModels','saved_models', 'digital_bank_feature_predictor.keras')
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Save feature names
    feature_names_path = os.path.join('VFLClientModels','saved_models', 'digital_bank_feature_predictor_feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Feature names saved: {feature_names_path}")
    
    # Save model info
    model_info = {
        'input_dim': int(model.input_shape[1]),
        'num_features': len(feature_names),
        'num_directions': 2,
        'num_impacts': 4,
        'model_parameters': int(model.count_params()),
        'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        'non_trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])),
        'training_date': datetime.now().isoformat(),
        'architecture': 'Multi-output neural network for feature prediction',
        'training_history': {
            'final_epoch': len(history.history['loss']),
            'final_training_loss': float(history.history['loss'][-1]),
            'final_validation_loss': float(history.history['val_loss'][-1]),
            'best_accuracies': {}
        }
    }
    
    # Add best accuracies to model info
    for feature_num in [1, 2, 3]:
        model_info['training_history']['best_accuracies'][f'feature_{feature_num}'] = {
            'idx_accuracy': float(max(history.history[f'feature_{feature_num}_idx_accuracy'])),
            'direction_accuracy': float(max(history.history[f'feature_{feature_num}_direction_accuracy'])),
            'impact_accuracy': float(max(history.history[f'feature_{feature_num}_impact_accuracy']))
        }
    
    info_path = os.path.join('VFLClientModels','saved_models', 'digital_bank_feature_predictor_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model info saved: {info_path}")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Digital Bank Feature Predictor Training History', fontsize=16)
    
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
    
    plot_path = os.path.join('VFLClientModels','plots', 'digital_bank_feature_predictor_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved: {plot_path}")
    
    plt.close()

def main():
    """Main function to train the digital bank feature predictor"""
    logger = setup_logging()
    
    logger.info("🚀 DIGITAL BANK FEATURE PREDICTOR TRAINING")
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
        
        # Get actual input dimension from the data
        input_dim = X.shape[1]
        
        # Step 3: Create model
        logger.info("Step 3: Creating model...")
        model = create_model(
            input_dim=input_dim,
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
        logger.info(f"Input: {input_dim}D intermediate representations")
        logger.info(f"Output: Top 3 features with indices, directions, and impacts")
        logger.info(f"Model parameters: {model.count_params():,}")
        logger.info(f"Final training loss: {history.history['loss'][-1]:.6f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        logger.info("")
        logger.info("📁 Generated Files:")
        logger.info("   - Model: saved_models/digital_bank_feature_predictor.keras")
        logger.info("   - Feature names: saved_models/digital_bank_feature_predictor_feature_names.json")
        logger.info("   - Model info: saved_models/digital_bank_feature_predictor_info.json")
        logger.info("   - Training plots: plots/digital_bank_feature_predictor_training_history.png")
        logger.info("")
        logger.info("🔧 Next Steps:")
        logger.info("   - Use the trained model to predict features from new intermediate representations")
        logger.info("   - Integrate with the digital bank explainer system")
        logger.info("=" * 80)
        
        return model, feature_names, history
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 