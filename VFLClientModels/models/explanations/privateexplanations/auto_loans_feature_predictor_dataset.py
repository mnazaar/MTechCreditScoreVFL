import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import logging
from datetime import datetime
import warnings
import shap
import pickle
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def setup_logging():
    """Setup logging for the dataset generator"""
    os.makedirs('logs', exist_ok=True)
    
    # Create a custom formatter that handles Unicode
    class UnicodeFormatter(logging.Formatter):
        def format(self, record):
            # Replace problematic Unicode characters with ASCII equivalents
            record.msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
            return super().format(record)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'VFLClientModels/logs/auto_loans_feature_predictor_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Apply custom formatter to console handler
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(UnicodeFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    return logger

def get_penultimate_layer_model(model):
    """Get the penultimate layer model for feature extraction"""
    # Get the layer before the final output layer
    penultimate_layer = model.layers[-2]  # Second to last layer
    
    # Create a new model that outputs the penultimate layer
    penultimate_model = keras.Model(
        inputs=model.inputs,
        outputs=penultimate_layer.output
    )
    
    return penultimate_model

def load_auto_loans_model_and_data():
    """Load the auto loans model and prepare data"""
    logger = setup_logging()
    logger.info("Loading auto loans model and data.")
    
    try:
        # Load the trained model
        model = keras.models.load_model(os.path.join('VFLClientModels', 'saved_models', 'auto_loans_model.keras'), compile=False)
        logger.info("Auto loans model loaded successfully")
        
        # Load feature names
        feature_names = np.load(os.path.join('VFLClientModels', 'saved_models', 'auto_loans_feature_names.npy'), allow_pickle=True).tolist()
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        # Try to load scaler, create fallback if not available
        scaler_path = os.path.join('VFLClientModels','saved_models', 'auto_loans_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            # Create a fallback scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            logger.info("Scaler file not found, using fallback StandardScaler")
        
        # Load original data for generating explanations
        df = pd.read_csv(os.path.join('VFLClientModels', 'dataset', 'data', 'banks', 'auto_loans_bank.csv'))
        logger.info(f"Original data loaded: {len(df)} records")
        
        return model, feature_names, scaler, df
        
    except Exception as e:
        logger.error(f"Error loading model/data: {str(e)}")
        raise

def compute_shap_for_sample(i, X_scaled, feature_names, model):
    # Use a small background set for speed
    background_data = X_scaled[max(0, i-50):i+1] if i > 0 else X_scaled[:50]
    explainer = shap.KernelExplainer(lambda x: model.predict(x), background_data)
    shap_values = explainer.shap_values(X_scaled[i:i+1])
    if isinstance(shap_values, list):
        shap_val = shap_values[0].ravel() if hasattr(shap_values[0], 'ravel') else np.array(shap_values[0])
    else:
        shap_val = shap_values.ravel() if hasattr(shap_values, 'ravel') else np.array(shap_values)
    # Top 3 features
    feature_importance = np.abs(shap_val)
    top_indices = np.argsort(feature_importance)[-3:][::-1]
    top_features = []
    for idx in top_indices:
        if int(idx) < len(feature_names) and int(idx) < len(shap_val):
            feature_name = feature_names[int(idx)]
            shap_value = shap_val[int(idx)]
            impact_level = get_impact_level(shap_value)
            direction = "Positive" if shap_value > 0 else "Negative"
            top_features.append({
                'feature_name': feature_name,
                'impact': impact_level,
                'direction': direction
            })
    return i, top_features

def get_impact_level(shap_value):
    """Get impact level based on SHAP value magnitude"""
    abs_weight = abs(shap_value)
    if abs_weight >= 0.1:
        return "Very High"
    elif abs_weight >= 0.05:
        return "High"
    elif abs_weight >= 0.02:
        return "Medium"
    else:
        return "Low"

def main():
    logger = setup_logging()
    logger.info("Starting Auto Loans Feature Prediction Dataset Generation (Multithreaded)")
    logger.info("=" * 80)
    logger.info("Step 1: Loading auto loans model and data.")
    try:
        model, feature_names, scaler, df = load_auto_loans_model_and_data()
        logger.info("=" * 80)
        logger.info("Step 2: Extracting intermediate representations.")
        penultimate_model = get_penultimate_layer_model(model)
        X = df[feature_names].values
        if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = scaler.fit_transform(X)
            logger.info("Scaler fitted to data")
        intermediate_reps = penultimate_model.predict(X_scaled, verbose=0)
        logger.info(f"Intermediate representations extracted: {intermediate_reps.shape}")
        logger.info("=" * 80)
        logger.info("Step 3: Generating SHAP explanations (Multithreaded).")
        num_samples = 1000
        customer_ids = df["tax_id"].values if "tax_id" in df.columns else [None]*num_samples
        results = [None] * num_samples
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_shap_for_sample, i, X_scaled, feature_names, model) for i in range(num_samples)]
            for future in as_completed(futures):
                i, top_features = future.result()
                results[i] = top_features
        logger.info("=" * 80)
        logger.info("Step 4: Creating training dataset.")
        dataset = []
        for i, (rep, top_features) in enumerate(zip(intermediate_reps[:num_samples], results)):
            rep_flat = rep.flatten()
            dataset.append({
                'customer_id': customer_ids[i],
                'intermediate_representation': rep_flat.tolist(),
                'top_features': top_features
            })
        logger.info(f"Dataset created with {len(dataset)} samples")
        logger.info("=" * 80)
        logger.info("Step 5: Saving dataset.")
        os.makedirs('data', exist_ok=True)
        with open('VFLClientModels/models/explanations/data/auto_loans_feature_predictor_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        with open('VFLClientModels/models/explanations/data/auto_loans_feature_predictor_dataset_sample.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        with open('VFLClientModels/models/explanations/data/auto_loans_feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        logger.info("Dataset saved successfully!")
        logger.info(f"- Pickle file: VFLClientModels/models/explanations/data/auto_loans_feature_predictor_dataset.pkl")
        logger.info(f"- JSON sample: VFLClientModels/models/explanations/data/auto_loans_feature_predictor_dataset_sample.json")
        logger.info(f"- Feature names: VFLClientModels/models/explanations/data/auto_loans_feature_names.txt")
        logger.info("=" * 80)
        logger.info("Dataset Statistics:")
        logger.info(f"- Total samples: {len(dataset)}")
        logger.info(f"- Intermediate representation dimension: {len(dataset[0]['intermediate_representation'])}")
        logger.info(f"- Top features per sample: 3")
        logger.info("Sample data:")
        sample = dataset[0]
        logger.info(f"Customer ID: {sample['customer_id']}")
        logger.info(f"Intermediate representation shape: {len(sample['intermediate_representation'])}")
        for idx, f in enumerate(sample['top_features'], 1):
            logger.info(f"Top feature {idx}: {f['feature_name']} ({f['impact']}, {f['direction']})")
        logger.info("=" * 80)
        logger.info("Dataset generation completed successfully!")
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 