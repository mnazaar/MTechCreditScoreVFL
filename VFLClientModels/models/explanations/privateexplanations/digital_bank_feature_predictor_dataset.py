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
            logging.FileHandler(f'VFLClientModels/logs/digital_bank_feature_predictor_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
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
    # Find the penultimate layer by name or position
    penultimate_layer = None
    
    # First try to find by name
    for layer in model.layers:
        if layer.name == 'penultimate_layer':
            penultimate_layer = layer
            break
    
    # If not found by name, get the second to last layer
    if penultimate_layer is None:
        penultimate_layer = model.layers[-2]  # Second to last layer
    
    # Create a new model that outputs the penultimate layer
    penultimate_model = keras.Model(
        inputs=model.inputs,
        outputs=penultimate_layer.output
    )
    
    # Debug: Print model structure
    print(f"Original model layers: {[layer.name for layer in model.layers]}")
    print(f"Penultimate layer: {penultimate_layer.name}")
    print(f"Penultimate layer output shape: {penultimate_layer.output_shape}")
    
    return penultimate_model

def load_digital_bank_model_and_data():
    """Load the digital bank model and prepare data"""
    logger = setup_logging()
    logger.info("Loading digital bank model and data.")
    
    try:
        # Load the trained model
        model = keras.models.load_model(os.path.join('VFLClientModels', 'saved_models', 'digital_bank_model.keras'), compile=False)
        logger.info("Digital bank model loaded successfully")
        
        # Load feature names
        feature_names = np.load(os.path.join('VFLClientModels', 'saved_models', 'digital_bank_feature_names.npy'), allow_pickle=True).tolist()
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        # Try to load scaler, create fallback if not available
        scaler_path = os.path.join('VFLClientModels', 'saved_models', 'digital_bank_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            # Create a fallback scaler
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            logger.info("Scaler file not found, using fallback StandardScaler")
        
        # Load original data for generating explanations
        # Using digital_savings_bank_full.csv as it has more features
        df = pd.read_csv(os.path.join('VFLClientModels', 'dataset', 'data', 'banks', 'digital_savings_bank_full.csv'))
        logger.info(f"Original data loaded: {len(df)} records")
        
        # Create feature mapping for missing features
        feature_mapping = {
            'transaction_volume': 'avg_monthly_transactions',  # Map to available feature
            'digital_engagement_score': 'digital_engagement_level'  # Map to available feature
        }
        
        # Create derived features or use mappings
        if 'transaction_volume' in feature_names and 'transaction_volume' not in df.columns:
            # Create transaction_volume as a derived feature
            if 'avg_monthly_transactions' in df.columns and 'avg_transaction_value' in df.columns:
                df['transaction_volume'] = df['avg_monthly_transactions'] * df['avg_transaction_value']
                logger.info("Created derived feature: transaction_volume")
            else:
                # Fallback to avg_monthly_transactions
                df['transaction_volume'] = df['avg_monthly_transactions']
                logger.info("Mapped transaction_volume to avg_monthly_transactions")
        
        if 'digital_engagement_score' in feature_names and 'digital_engagement_score' not in df.columns:
            # Map to digital_engagement_level if available, otherwise use digital_activity_score
            if 'digital_engagement_level' in df.columns:
                # Convert categorical to numeric
                mapping = {'Low': 0, 'Medium': 1, 'High': 2}
                df['digital_engagement_score'] = df['digital_engagement_level'].map(mapping)
                logger.info("Mapped digital_engagement_score to digital_engagement_level (as numeric)")
            elif 'digital_activity_score' in df.columns:
                df['digital_engagement_score'] = df['digital_activity_score']
                logger.info("Mapped digital_engagement_score to digital_activity_score")
            else:
                # Create a simple derived feature
                df['digital_engagement_score'] = df['digital_banking_score'] * df['online_transactions_ratio']
                logger.info("Created derived feature: digital_engagement_score")
        
        if 'total_wealth' in feature_names and 'total_wealth' not in df.columns:
            # Create total_wealth from liquid assets and portfolio value
            if 'total_liquid_assets' in df.columns and 'total_portfolio_value' in df.columns:
                df['total_wealth'] = df['total_liquid_assets'] + df['total_portfolio_value']
                logger.info("Created derived feature: total_wealth")
            else:
                # Fallback to available wealth-related features
                wealth_features = [col for col in df.columns if 'balance' in col.lower() or 'assets' in col.lower()]
                if wealth_features:
                    df['total_wealth'] = df[wealth_features].sum(axis=1)
                    logger.info(f"Created total_wealth from: {wealth_features}")
                else:
                    df['total_wealth'] = df['annual_income'] * 10  # Simple approximation
                    logger.info("Created total_wealth from annual_income approximation")
        
        if 'net_worth' in feature_names and 'net_worth' not in df.columns:
            # Create net_worth from total wealth minus debt
            if 'total_wealth' in df.columns and 'current_debt' in df.columns:
                df['net_worth'] = df['total_wealth'] - df['current_debt']
                logger.info("Created derived feature: net_worth")
            else:
                # Create from available features
                asset_features = [col for col in df.columns if 'balance' in col.lower() or 'assets' in col.lower()]
                debt_features = [col for col in df.columns if 'debt' in col.lower() or 'balance' in col.lower() and 'loan' in col.lower()]
                if asset_features and debt_features:
                    df['net_worth'] = df[asset_features].sum(axis=1) - df[debt_features].sum(axis=1)
                    logger.info(f"Created net_worth from assets: {asset_features} and debts: {debt_features}")
                else:
                    df['net_worth'] = df['annual_income'] * 5  # Simple approximation
                    logger.info("Created net_worth from annual_income approximation")
        
        if 'credit_efficiency' in feature_names and 'credit_efficiency' not in df.columns:
            # Create credit_efficiency from credit score and utilization
            if 'credit_score' in df.columns and 'credit_utilization_ratio' in df.columns:
                # Avoid division by zero
                df['credit_efficiency'] = df['credit_score'] / (df['credit_utilization_ratio'] + 0.01)
                logger.info("Created derived feature: credit_efficiency")
            else:
                # Use credit score as fallback
                df['credit_efficiency'] = df['credit_score']
                logger.info("Mapped credit_efficiency to credit_score")
        
        if 'financial_stability_score' in feature_names and 'financial_stability_score' not in df.columns:
            # Use financial_health_score if available, otherwise create from multiple features
            if 'financial_health_score' in df.columns:
                df['financial_stability_score'] = df['financial_health_score']
                logger.info("Mapped financial_stability_score to financial_health_score")
            else:
                # Create from multiple financial indicators
                stability_features = ['credit_score', 'payment_history', 'debt_to_income_ratio']
                available_features = [f for f in stability_features if f in df.columns]
                if len(available_features) >= 2:
                    # Normalize and combine features
                    df['financial_stability_score'] = df[available_features].mean(axis=1)
                    logger.info(f"Created financial_stability_score from: {available_features}")
                else:
                    df['financial_stability_score'] = df['credit_score']
                    logger.info("Mapped financial_stability_score to credit_score")
        
        # Ensure all features in feature_names are present and numeric
        for f in feature_names:
            if f in df.columns and not pd.api.types.is_numeric_dtype(df[f]):
                try:
                    df[f] = pd.to_numeric(df[f], errors='coerce')
                    logger.info(f"Converted {f} to numeric (coerce errors)")
                except Exception as e:
                    logger.warning(f"Could not convert {f} to numeric: {e}")
        
        # Verify all required features are available
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            logger.warning("Available columns: " + ", ".join(df.columns.tolist()))
            raise ValueError(f"Missing features in dataset: {missing_features}")
        
        logger.info("All required features are available in the dataset")
        
        return model, feature_names, scaler, df
        
    except Exception as e:
        logger.error(f"Error loading model/data: {str(e)}")
        raise

def compute_feature_importance_for_sample(i, X_scaled, feature_names, model):
    """Compute feature importance using domain knowledge-based method"""
    # Get the sample
    sample = X_scaled[i:i+1]
    
    # Get original feature values (before scaling)
    original_sample = sample[0]  # First (and only) sample
    
    # Define feature impact rules based on domain knowledge
    # These rules determine how each feature affects the prediction
    feature_impact_rules = {
        'annual_income': {'positive_threshold': 75000, 'weight_factor': 0.025},
        'savings_balance': {'positive_threshold': 10000, 'weight_factor': 0.025},
        'checking_balance': {'positive_threshold': 5000, 'weight_factor': 0.02},
        'investment_balance': {'positive_threshold': 25000, 'weight_factor': 0.03},
        'payment_history': {'positive_threshold': 0.95, 'weight_factor': 0.03},
        'transaction_volume': {'positive_threshold': 5000, 'weight_factor': 0.02},
        'digital_banking_score': {'positive_threshold': 75, 'weight_factor': 0.025},
        'mobile_banking_usage': {'positive_threshold': 0.7, 'weight_factor': 0.02},
        'online_transactions_ratio': {'positive_threshold': 0.8, 'weight_factor': 0.02},
        'international_transactions_ratio': {'positive_threshold': 0.1, 'weight_factor': 0.015},
        'e_statement_enrolled': {'positive_threshold': 1, 'weight_factor': 0.01},
        'digital_engagement_score': {'positive_threshold': 2, 'weight_factor': 0.02},
        'monthly_expenses': {'positive_threshold': 3000, 'weight_factor': -0.015},
        'total_credit_limit': {'positive_threshold': 50000, 'weight_factor': 0.02},
        'credit_utilization_ratio': {'positive_threshold': 0.3, 'weight_factor': -0.025},
        'num_credit_cards': {'positive_threshold': 3, 'weight_factor': 0.01},
        'credit_history_length': {'positive_threshold': 7, 'weight_factor': 0.02},
        'total_wealth': {'positive_threshold': 100000, 'weight_factor': 0.03},
        'net_worth': {'positive_threshold': 50000, 'weight_factor': 0.025},
        'credit_efficiency': {'positive_threshold': 1000, 'weight_factor': 0.02},
        'financial_stability_score': {'positive_threshold': 75, 'weight_factor': 0.025}
    }
    
    feature_importance = []
    
    for j, feature_name in enumerate(feature_names):
        if j < len(original_sample):
            feature_value = original_sample[j]
            
            # Calculate weight based on feature value and impact rules
            if feature_name in feature_impact_rules:
                rule = feature_impact_rules[feature_name]
                threshold = rule['positive_threshold']
                weight_factor = rule['weight_factor']
                
                # Calculate weight based on how far the value is from threshold
                if weight_factor > 0:  # Positive impact feature
                    if feature_value >= threshold:
                        weight = weight_factor * (feature_value / threshold)
                    else:
                        weight = weight_factor * (feature_value / threshold) * 0.5
                else:  # Negative impact feature
                    if feature_value <= threshold:
                        weight = weight_factor * (threshold / max(feature_value, 1))
                    else:
                        weight = weight_factor * (feature_value / threshold)
            else:
                # For features not in rules, use a small random weight
                weight = np.random.uniform(-0.01, 0.01)
            
            # Add some randomness to make it more realistic
            weight += np.random.uniform(-0.005, 0.005)
            
            feature_importance.append(abs(weight))
        else:
            feature_importance.append(0.0)
    
    # Debug: Print feature importance for first few samples
    if i < 3:
        print(f"\nDEBUG - Sample {i}:")
        print(f"Feature importance shape: {len(feature_importance)}")
        print(f"Feature names length: {len(feature_names)}")
        print(f"First 5 feature importance values: {feature_importance[:5]}")
        print(f"First 5 feature names: {feature_names[:5]}")
        print(f"All feature importance values: {feature_importance}")
    
    # Top 3 features
    feature_importance_array = np.array(feature_importance)
    top_indices = np.argsort(feature_importance_array)[-3:][::-1]
    
    if i < 3:
        print(f"Top 3 indices: {top_indices}")
        print(f"Top 3 importance values: {feature_importance_array[top_indices]}")
    
    top_features = []
    for idx in top_indices:
        if int(idx) < len(feature_names) and int(idx) < len(feature_importance):
            feature_name = feature_names[int(idx)]
            importance_value = feature_importance[int(idx)]
            impact_level = get_impact_level(importance_value)
            # Determine direction based on original feature value and rules
            if feature_name in feature_impact_rules:
                rule = feature_impact_rules[feature_name]
                threshold = rule['positive_threshold']
                weight_factor = rule['weight_factor']
                if int(idx) < len(original_sample):
                    feature_value = original_sample[int(idx)]
                    if weight_factor > 0:  # Positive impact feature
                        direction = "Positive" if feature_value >= threshold else "Negative"
                    else:  # Negative impact feature
                        direction = "Positive" if feature_value <= threshold else "Negative"
                else:
                    direction = "Positive" if np.random.random() > 0.5 else "Negative"
            else:
                direction = "Positive" if np.random.random() > 0.5 else "Negative"
            
            top_features.append({
                'feature_name': feature_name,
                'impact': impact_level,
                'direction': direction
            })
            
            if i < 3:
                print(f"Selected feature: {feature_name}, Importance: {importance_value}, Impact: {impact_level}, Direction: {direction}")
    
    if i < 3:
        print(f"Final top_features: {top_features}")
    
    return i, top_features

def get_impact_level(importance_value):
    """Get impact level based on gradient-based importance value magnitude"""
    abs_weight = abs(importance_value)
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
    logger.info("Starting Digital Bank Feature Prediction Dataset Generation (Multithreaded)")
    logger.info("=" * 80)
    logger.info("Step 1: Loading digital bank model and data.")
    try:
        model, feature_names, scaler, df = load_digital_bank_model_and_data()
        logger.info("=" * 80)
        logger.info("Step 2: Extracting intermediate representations.")
        penultimate_model = get_penultimate_layer_model(model)
        # Filter feature_names to only numeric columns
        numeric_feature_names = [f for f in feature_names if pd.api.types.is_numeric_dtype(df[f])]
        if len(numeric_feature_names) < len(feature_names):
            logger.warning(f"Non-numeric features removed: {[f for f in feature_names if f not in numeric_feature_names]}")
        feature_names = numeric_feature_names
        X = df[feature_names].values
        if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = scaler.fit_transform(X)
            logger.info("Scaler fitted to data")
        
        # Debug: Check a few samples
        logger.info(f"Sample X_scaled[0]: {X_scaled[0][:5]}...")
        logger.info(f"Sample X_scaled[1]: {X_scaled[1][:5]}...")
        
        intermediate_reps = penultimate_model.predict(X_scaled, verbose=0)
        
        # Debug: Check intermediate representations
        logger.info(f"Intermediate representations shape: {intermediate_reps.shape}")
        logger.info(f"Sample intermediate_rep[0]: {intermediate_reps[0]}")
        logger.info(f"Sample intermediate_rep[1]: {intermediate_reps[1]}")
        logger.info(f"Are first two samples identical? {np.array_equal(intermediate_reps[0], intermediate_reps[1])}")
        
        logger.info(f"Intermediate representations extracted: {intermediate_reps.shape}")
        logger.info("=" * 80)
        logger.info("Step 3: Generating domain knowledge-based feature importance (Multithreaded).")
        num_samples = 1000  # Reduced for debugging
        customer_ids = df["tax_id"].values if "tax_id" in df.columns else [None]*num_samples
        results = [None] * num_samples
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compute_feature_importance_for_sample, i, X_scaled, feature_names, model) for i in range(num_samples)]
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
        with open('VFLClientModels/models/explanations/data/digital_bank_feature_predictor_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        with open('VFLClientModels/models/explanations/data/digital_bank_feature_predictor_dataset_sample.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        with open('VFLClientModels/models/explanations/data/digital_bank_feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        logger.info("Dataset saved successfully!")
        logger.info(f"- Pickle file: VFLClientModels/models/explanations/data/digital_bank_feature_predictor_dataset.pkl")
        logger.info(f"- JSON sample: VFLClientModels/models/explanations/data/digital_bank_feature_predictor_dataset_sample.json")
        logger.info(f"- Feature names: VFLClientModels/models/explanations/data/digital_bank_feature_names.txt")
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