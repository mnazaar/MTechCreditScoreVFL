#!/usr/bin/env python3
"""
Credit Card Feature Predictor Usage Script

This script demonstrates how to use the trained feature predictor model
to predict the top 3 features and their impacts/directions from 
12D intermediate representations.

Usage:
    python use_credit_card_feature_predictor.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging for the feature predictor usage"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/credit_card_feature_predictor_usage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_feature_predictor_model():
    """Load the trained feature predictor model and metadata"""
    logger = setup_logging()
    logger.info("Loading feature predictor model...")
    
    try:
        # Load model
        model_path = os.path.join('saved_models', 'credit_card_feature_predictor.keras')
        model = keras.models.load_model(model_path, compile=False)
        logger.info("✅ Feature predictor model loaded successfully")
        
        # Load feature names
        feature_names_path = os.path.join('saved_models', 'credit_card_feature_predictor_feature_names.json')
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
        logger.info(f"✅ Feature names loaded: {len(feature_names)} features")
        
        # Load model info
        info_path = os.path.join('saved_models', 'credit_card_feature_predictor_info.json')
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        logger.info("✅ Model info loaded")
        
        return model, feature_names, model_info
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise

def predict_top3_features(model, intermediate_repr, feature_names):
    """
    Predict top 3 features from intermediate representation
    
    Args:
        model: Trained feature predictor model
        intermediate_repr: 12D intermediate representation (numpy array)
        feature_names: List of feature names
    
    Returns:
        dict: Top 3 features with their directions, and impacts
    """
    logger = setup_logging()
    
    # Ensure input is correct shape
    if intermediate_repr.shape != (1, 12):
        intermediate_repr = intermediate_repr.reshape(1, 12)
    
    # Make prediction
    predictions = model.predict(intermediate_repr, verbose=0)
    
    # Process predictions
    top3_features = []
    
    for i in range(1, 4):
        # Get predictions
        pred_idx = np.argmax(predictions[f'feature_{i}_idx'][0])
        pred_direction = np.argmax(predictions[f'feature_{i}_direction'][0])
        pred_impact = np.argmax(predictions[f'feature_{i}_impact'][0])
        
        # Feature name
        pred_feature = feature_names[pred_idx] if pred_idx < len(feature_names) else f"Unknown_{pred_idx}"
        
        # Direction and impact labels
        direction_labels = ['Positive', 'Negative']
        impact_labels = ['Very High', 'High', 'Medium', 'Low']
        
        feature_info = {
            'rank': i,
            'name': pred_feature,
            'direction': direction_labels[pred_direction],
            'impact': impact_labels[pred_impact],
            'confidence': {
                'idx_confidence': float(np.max(predictions[f'feature_{i}_idx'][0])),
                'direction_confidence': float(np.max(predictions[f'feature_{i}_direction'][0])),
                'impact_confidence': float(np.max(predictions[f'feature_{i}_impact'][0]))
            }
        }
        
        top3_features.append(feature_info)
    
    return top3_features

def print_feature_prediction(top3_features):
    """Print the feature prediction results"""
    logger = setup_logging()
    
    logger.info("📊 TOP 3 FEATURES PREDICTION")
    logger.info("=" * 80)
    
    for i, feature in enumerate(top3_features, 1):
        logger.info(f"Feature {i}: {feature['name']}")
        logger.info(f"   - Direction: {feature['direction']}")
        logger.info(f"   - Impact: {feature['impact']}")
        logger.info(f"   - Confidence: {feature['confidence']['idx_confidence']:.3f}")
        logger.info("")

def create_sample_intermediate_representations():
    """Create sample intermediate representations for testing"""
    logger = setup_logging()
    logger.info("Creating sample intermediate representations...")
    
    # Create 5 sample representations
    samples = []
    
    for i in range(5):
        # Create a realistic intermediate representation
        # These would normally come from the credit card model's penultimate layer
        sample_repr = np.random.normal(0, 1, (1, 12))
        
        # Normalize to [0, 1] range (as done in VFL)
        sample_repr = (sample_repr - sample_repr.min()) / (sample_repr.max() - sample_repr.min())
        
        samples.append(sample_repr)
        logger.info(f"Sample {i+1}: shape {sample_repr.shape}, range [{sample_repr.min():.3f}, {sample_repr.max():.3f}]")
    
    return samples

def demonstrate_feature_prediction():
    """Demonstrate feature prediction on sample data"""
    logger = setup_logging()
    
    logger.info("🚀 CREDIT CARD FEATURE PREDICTOR DEMONSTRATION")
    logger.info("=" * 80)
    
    try:
        # Load model
        model, feature_names, model_info = load_feature_predictor_model()
        
        # Create sample intermediate representations
        logger.info("Creating sample intermediate representations...")
        sample_reprs = create_sample_intermediate_representations()
        
        # Make predictions for each sample
        logger.info("Making feature predictions...")
        logger.info("")
        
        for i, sample_repr in enumerate(sample_reprs, 1):
            logger.info(f"Sample {i} Intermediate Representation:")
            logger.info("-" * 60)
            
            # Print original intermediate representation values
            logger.info("Original 12D Intermediate Representation:")
            for j, val in enumerate(sample_repr[0]):
                logger.info(f"   Dim {j+1:2d}: {val:.6f}")
            logger.info("")
            
            # Predict top 3 features
            top3_features = predict_top3_features(model, sample_repr, feature_names)
            
            # Print results
            print_feature_prediction(top3_features)
            
            logger.info("")
        
        logger.info("✅ Feature prediction demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {str(e)}")
        raise

def predict_from_dataset_samples():
    """Predict features from actual dataset samples"""
    logger = setup_logging()
    
    logger.info("🔄 PREDICTING FROM DATASET SAMPLES")
    logger.info("=" * 80)
    
    try:
        # Load feature predictor model
        feature_predictor_model, feature_names, model_info = load_feature_predictor_model()
        
        # Load dataset samples
        with open('data/credit_card_feature_predictor_dataset_sample.json', 'r') as f:
            dataset = json.load(f)
        
        # Select 5 random samples
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(dataset)), 100)
        
        logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
        logger.info(f"Selected {len(sample_indices)} random samples for prediction")
        
        # Predict top 3 features for each sample
        logger.info("Predicting top 3 features for each sample...")
        logger.info("")
        
        for i, idx in enumerate(sample_indices, 1):
            sample = dataset[idx]
            customer_id = sample['customer_id']
            intermediate_repr = np.array(sample['intermediate_representation']).reshape(1, 12)
            
            logger.info(f"Sample {i}: Customer {customer_id}")
            logger.info("-" * 60)
            
            # Predict top 3 features
            top3_features = predict_top3_features(feature_predictor_model, intermediate_repr, feature_names)
            
            # Print results
            print_feature_prediction(top3_features)
            
            # Compare with actual features
            actual_features = sample['top_features']
            logger.info("Actual top features from dataset:")
            for j, feature in enumerate(actual_features, 1):
                logger.info(f"   Feature {j}: {feature['feature_name']} ({feature['direction']}, {feature['impact']})")
            
            logger.info("")
        
        logger.info("✅ Dataset sample prediction completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Dataset sample prediction failed: {str(e)}")
        raise

def main():
    """Main function to demonstrate feature predictor usage"""
    logger = setup_logging()
    
    logger.info("🚀 CREDIT CARD FEATURE PREDICTOR USAGE DEMONSTRATION")
    logger.info("=" * 80)
    
    try:
        # Demonstration 1: Sample data
        logger.info("DEMONSTRATION 1: Sample Intermediate Representations")
        logger.info("=" * 60)
        demonstrate_feature_prediction()
        
        logger.info("")
        
        # Demonstration 2: Real dataset samples
        logger.info("DEMONSTRATION 2: Real Dataset Samples")
        logger.info("=" * 60)
        predict_from_dataset_samples()
        
        logger.info("")
        logger.info("🎉 FEATURE PREDICTOR USAGE DEMONSTRATION COMPLETED!")
        logger.info("=" * 80)
        logger.info("The feature predictor model can now be used to:")
        logger.info("   - Predict top 3 features from any 12D intermediate representation")
        logger.info("   - Get feature directions (Positive/Negative) and impacts (Very High/High/Medium/Low)")
        logger.info("   - Integrate with the credit card explainer system")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Usage demonstration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 