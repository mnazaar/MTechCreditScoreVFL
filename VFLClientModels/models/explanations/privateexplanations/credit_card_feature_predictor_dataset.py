import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
from sklearn.decomposition import PCA

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the main explainer to ensure consistency
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from credit_card_feature_explainer import CreditCardFeatureExplainer

# XGBoost Credit Card Configuration (same as VFL AutoML model)
XGBOOST_OUTPUT_DIM = 12             # Target output dimension for XGBoost representations
XGBOOST_PCA_RANDOM_STATE = 42       # Random state for PCA dimensionality reduction

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
            logging.FileHandler(f'VFLClientModels/logs/credit_card_feature_predictor_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Apply custom formatter to console handler
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(UnicodeFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    return logger

def load_credit_card_model_and_data():
    """Load the credit card model and prepare data"""
    logger = setup_logging()
    logger.info("Loading credit card model and data...")
    
    try:
        # Import the model class
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from credit_card_xgboost_model import IndependentXGBoostModel, load_and_preprocess_data
        
        # Load the trained model
        model = IndependentXGBoostModel.load_model(os.path.join('VFLClientModels', 'saved_models', 'credit_card_xgboost_independent.pkl'))
        logger.info("Credit card model loaded successfully")
        
        # Load feature names
        feature_names = np.load(os.path.join('VFLClientModels', 'saved_models', 'credit_card_xgboost_feature_names.npy'), allow_pickle=True)
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        # Load label encoder for class names
        label_encoder = joblib.load(os.path.join('VFLClientModels', 'saved_models', 'credit_card_xgboost_label_encoder.pkl'))
        class_names = label_encoder.classes_.tolist()
        logger.info(f"Class names loaded: {class_names}")
        
        # Load original data for generating explanations
        df = pd.read_csv(os.path.join('VFLClientModels', 'dataset', 'data', 'banks', 'credit_card_bank.csv'))
        logger.info(f"Original data loaded: {len(df)} records")
        
        return model, feature_names, class_names, df
        
    except Exception as e:
        logger.error(f"Error loading model/data: {str(e)}")
        raise

def preprocess_features(df, feature_names):
    """Preprocess features for credit card model"""
    # Enhanced feature engineering for credit card classification
    df_proc = df.copy()
    
    # Create derived features exactly as in credit_card_model.py
    df_proc['credit_capacity_ratio'] = df_proc['credit_card_limit'] / df_proc['total_credit_limit'].replace(0, 1)
    df_proc['income_to_limit_ratio'] = df_proc['annual_income'] / df_proc['credit_card_limit'].replace(0, 1)
    df_proc['debt_service_ratio'] = (df_proc['current_debt'] * 0.03) / (df_proc['annual_income'] / 12)  # Assuming 3% monthly payment
    df_proc['risk_adjusted_income'] = df_proc['annual_income'] * (df_proc['risk_score'] / 100)

    # Use only the features that the model was trained with (25 features)
    # Add derived features to the feature_names if not already present
    derived_features = ['credit_capacity_ratio', 'income_to_limit_ratio', 'debt_service_ratio', 'risk_adjusted_income']
    all_features = list(feature_names) + [f for f in derived_features if f not in feature_names]

    # Ensure all required feature columns are present
    for feat in all_features:
        if feat not in df_proc.columns:
            df_proc[feat] = 0
    
    # Reorder columns to match all_features
    df_proc = df_proc[all_features]
    return df_proc

def extract_xgboost_representations(model, customer_data, target_dim=XGBOOST_OUTPUT_DIM):
    """
    Extract fixed-dimension representations from XGBoost model using leaf indices
    Same logic as in vfl_automl_xgboost_model.py
    
    Args:
        model: XGBoost model with classifier and scaler
        customer_data: DataFrame with customer features (already preprocessed)
        target_dim: Target dimension for output representations (default: 12)
    
    Returns:
        numpy array of shape (n_customers, target_dim)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"🌳 Extracting XGBoost representations (target: {target_dim}D).")
    
    # Use the exact features that the model was trained with (25 features + 4 derived = 29 features)
    # Remove the 5 XGBoost-specific features that weren't in training data
    credit_card_features = [
        # Core financial features
        'annual_income', 'credit_score', 'payment_history', 'employment_length', 
        'debt_to_income_ratio', 'age',
        # Credit behavior and history
        'credit_history_length', 'num_credit_cards', 'num_loan_accounts', 
        'total_credit_limit', 'credit_utilization_ratio', 'late_payments', 
        'credit_inquiries', 'last_late_payment_days',
        # Financial position
        'current_debt', 'monthly_expenses', 'savings_balance', 
        'checking_balance', 'investment_balance', 'auto_loan_balance', 'mortgage_balance',
        # Derived features (already calculated in preprocess_features)
        'credit_capacity_ratio', 'income_to_limit_ratio', 'debt_service_ratio', 'risk_adjusted_income'
    ]
    
    logger.info(f"   📊 Processing {len(customer_data)} customers with {len(credit_card_features)} features")
    
    # Select only the required features for this model (derived features already calculated)
    feature_data = customer_data[credit_card_features].copy()
    
    # Handle infinite and missing values
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
    feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
    
    # Scale features
    X_scaled = model.scaler.transform(feature_data)
    logger.info(f"   ✅ Scaled features: {X_scaled.shape}")
    
    # Get leaf indices from XGBoost (this gives us n_estimators features)
    leaf_indices = model.classifier.apply(X_scaled)
    logger.info(f"   🌳 XGBoost leaf indices shape: {leaf_indices.shape}")
    
    # Convert leaf indices to target dimensions using same logic as VFL AutoML model
    if leaf_indices.shape[1] == target_dim:
        # Perfect match, just normalize
        representations = leaf_indices.astype(np.float32)
        logger.info(f"   ✅ Perfect dimension match: {representations.shape}")
    elif leaf_indices.shape[1] > target_dim:
        # Use PCA to reduce dimensions (same as VFL AutoML model)
        logger.info(f"   🔄 Reducing {leaf_indices.shape[1]}D → {target_dim}D using PCA.")
        pca = PCA(n_components=target_dim, random_state=XGBOOST_PCA_RANDOM_STATE)
        representations = pca.fit_transform(leaf_indices.astype(np.float32))
        explained_variance = pca.explained_variance_ratio_.sum()
        logger.info(f"   ✅ PCA completed: {representations.shape}, explained variance: {explained_variance:.3f}")
    else:
        # Pad with zeros or use feature engineering to expand
        logger.info(f"   🔄 Expanding {leaf_indices.shape[1]}D → {target_dim}D using padding.")
        representations = np.zeros((leaf_indices.shape[0], target_dim), dtype=np.float32)
        representations[:, :leaf_indices.shape[1]] = leaf_indices.astype(np.float32)
        logger.info(f"   ✅ Zero-padded to target dimension: {representations.shape}")
    
    # Normalize representations to [0, 1] range for consistency with neural networks
    if representations.max() > representations.min():
        representations = (representations - representations.min()) / (representations.max() - representations.min())
    
    logger.info(f"   🎯 Final XGBoost representations: {representations.shape}")
    logger.info(f"   📊 Stats: min={representations.min():.3f}, max={representations.max():.3f}, mean={representations.mean():.3f}")
    
    return representations

def get_top_features_from_explainer(explainer, customer_data, customer_id, num_features=3):
    """Get top features using sophisticated feature contribution method"""
    try:
        # Use sophisticated method instead of explainer's fallback
        feature_contributions = get_sophisticated_feature_contributions(
            explainer.model, customer_data, explainer.feature_names, 0
        )
        
        if feature_contributions:
            # Extract top features from the sophisticated analysis
            top_features = []
            for feature_info in feature_contributions[:num_features]:
                top_features.append({
                    'feature_name': feature_info['feature_name'],
                    'impact': get_impact_level(feature_info['weight']),
                    'direction': 'Positive' if feature_info['weight'] > 0 else 'Negative'
                })
            
            # Ensure we always return exactly num_features
            if len(top_features) < num_features:
                # Fill remaining features with random selection from remaining features
                remaining_features = num_features - len(top_features)
                remaining_contributions = feature_contributions[num_features:num_features + remaining_features]
                for feature_info in remaining_contributions:
                    top_features.append({
                        'feature_name': feature_info['feature_name'],
                        'impact': get_impact_level(feature_info['weight']),
                        'direction': 'Positive' if feature_info['weight'] > 0 else 'Negative'
                    })
            
            return top_features[:num_features]  # Ensure exactly num_features
        else:
            # Fallback to simple random selection if sophisticated method fails
            logger.warning(f"Sophisticated method failed for customer {customer_id}, using fallback")
            fallback_features = []
            for i in range(num_features):
                fallback_features.append({
                    'feature_name': f"feature_{i+1}",
                    'impact': 'Medium',
                    'direction': 'Positive' if np.random.random() > 0.5 else 'Negative'
                })
            return fallback_features
            
    except Exception as e:
        logger.warning(f"Feature analysis failed for customer {customer_id}: {e}")
        # Use fallback method to ensure we always get features
        fallback_features = []
        for i in range(num_features):
            fallback_features.append({
                'feature_name': f"feature_{i+1}",
                'impact': 'Medium',
                'direction': 'Positive' if np.random.random() > 0.5 else 'Negative'
            })
        return fallback_features

def get_sophisticated_feature_contributions(model, customer_data, feature_names, predicted_class):
    """Sophisticated method for determining feature contributions using relative values and thresholds"""
    try:
        # Get feature importance from the model
        feature_importance = model.get_feature_importance()
        
        # Create feature contributions based on importance and actual customer data
        feature_contributions = []
        customer_values = customer_data.values.flatten()  # Get actual customer feature values
        
        # Features to completely exclude from top selection
        excluded_features = ['risk_adjusted_income', 'total_credit_limit', 'credit_score']
        
        # Define feature-specific thresholds and ranges for more sophisticated analysis
        feature_thresholds = {
            # Positive impact features (higher is better)
            'credit_score': {'good_threshold': 700, 'excellent_threshold': 800, 'max_value': 850},
            'payment_history': {'good_threshold': 0.8, 'excellent_threshold': 0.95, 'max_value': 1.0},
            'annual_income': {'good_threshold': 50000, 'excellent_threshold': 100000, 'max_value': 200000},
            'employment_length': {'good_threshold': 5, 'excellent_threshold': 10, 'max_value': 20},
            'savings_balance': {'good_threshold': 10000, 'excellent_threshold': 50000, 'max_value': 100000},
            'checking_balance': {'good_threshold': 5000, 'excellent_threshold': 20000, 'max_value': 50000},
            'credit_history_length': {'good_threshold': 5, 'excellent_threshold': 10, 'max_value': 20},
            'investment_balance': {'good_threshold': 10000, 'excellent_threshold': 50000, 'max_value': 100000},
            
            # Negative impact features (lower is better)
            'debt_to_income_ratio': {'good_threshold': 0.6, 'excellent_threshold': 0.3, 'max_value': 1},
            'late_payments': {'good_threshold': 2, 'excellent_threshold': 0, 'max_value': 10},
            'credit_inquiries': {'good_threshold': 3, 'excellent_threshold': 1, 'max_value': 10},
            'current_debt': {'good_threshold': 20000, 'excellent_threshold': 10000, 'max_value': 100000},
            'credit_utilization_ratio': {'good_threshold': 0.3, 'excellent_threshold': 0.1, 'max_value': 0.8},
            'last_late_payment_days': {'good_threshold': 30, 'excellent_threshold': 0, 'max_value': 365},
            
            # Neutral features (context-dependent)
            'age': {'good_threshold': 30, 'excellent_threshold': 45, 'max_value': 80},
            'num_credit_cards': {'good_threshold': 3, 'excellent_threshold': 5, 'max_value': 10},
            'num_loan_accounts': {'good_threshold': 2, 'excellent_threshold': 4, 'max_value': 8},
            'monthly_expenses': {'good_threshold': 3000, 'excellent_threshold': 2000, 'max_value': 10000},
            'auto_loan_balance': {'good_threshold': 15000, 'excellent_threshold': 5000, 'max_value': 50000},
            'mortgage_balance': {'good_threshold': 200000, 'excellent_threshold': 100000, 'max_value': 500000},
            
            # Derived features
            'credit_capacity_ratio': {'good_threshold': 0.3, 'excellent_threshold': 0.5, 'max_value': 1.0},
            'income_to_limit_ratio': {'good_threshold': 0.2, 'excellent_threshold': 0.4, 'max_value': 1.0},
            'debt_service_ratio': {'good_threshold': 0.3, 'excellent_threshold': 0.2, 'max_value': 0.8},
        }
        
        for i, (feature_name, importance) in enumerate(zip(feature_names, feature_importance)):
            # Skip excluded features entirely
            if feature_name in excluded_features:
                continue
                
            if i < len(customer_values):
                customer_value = customer_values[i]
                
                # Get thresholds for this feature
                thresholds = feature_thresholds.get(feature_name, {
                    'good_threshold': 0.5, 
                    'excellent_threshold': 0.8, 
                    'max_value': 1.0
                })
                
                good_threshold = thresholds['good_threshold']
                excellent_threshold = thresholds['excellent_threshold']
                max_value = thresholds['max_value']
                
                # Calculate sophisticated weight based on relative position
                if feature_name in ['credit_score', 'payment_history', 'annual_income', 'employment_length', 
                                  'savings_balance', 'checking_balance', 'credit_history_length', 
                                  'investment_balance', 'credit_capacity_ratio', 'income_to_limit_ratio']:
                    # Positive impact features (higher is better)
                    if customer_value >= excellent_threshold:
                        # Excellent range: strong positive contribution
                        relative_position = 1.0
                    elif customer_value >= good_threshold:
                        # Good range: moderate positive contribution
                        relative_position = 0.5 + 0.5 * (customer_value - good_threshold) / (excellent_threshold - good_threshold)
                    else:
                        # Below good: negative contribution
                        relative_position = -0.5 * (good_threshold - customer_value) / good_threshold
                        
                elif feature_name in ['debt_to_income_ratio', 'late_payments', 'credit_inquiries', 
                                    'current_debt', 'credit_utilization_ratio', 'last_late_payment_days',
                                    'debt_service_ratio']:
                    # Negative impact features (lower is better)
                    if customer_value <= excellent_threshold:
                        # Excellent range: strong positive contribution
                        relative_position = 1.0
                    elif customer_value <= good_threshold:
                        # Good range: moderate positive contribution
                        relative_position = 0.5 + 0.5 * (good_threshold - customer_value) / (good_threshold - excellent_threshold)
                    else:
                        # Above good: negative contribution
                        relative_position = -0.5 * (customer_value - good_threshold) / (max_value - good_threshold)
                        
                else:
                    # Neutral features: use normalized value with some randomness
                    normalized_value = (customer_value - good_threshold) / (max_value - good_threshold)
                    relative_position = normalized_value * 0.5 + np.random.uniform(-0.1, 0.1)
                
                # Calculate final weight
                weight = importance * 0.1 * relative_position
                
            else:
                # Fallback for missing values
                weight = importance * 0.1 * np.random.choice([-1, 1])
            
            feature_contributions.append({
                'feature_name': feature_name,
                'weight': weight,
                'contribution': 'positive' if weight > 0 else 'negative'
            })
        
        # Sort by absolute importance
        feature_contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
        return feature_contributions
        
    except Exception as e:
        logger.warning(f"   ⚠️  Sophisticated feature contribution method failed: {str(e)}")
        # Return empty list if everything fails
        return []

def get_impact_level(weight):
    """Get impact level based on weight magnitude"""
    abs_weight = abs(weight)
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
    logger.info("Starting Credit Card Feature Prediction Dataset Generation (Using Consistent Explainer)")
    logger.info("=" * 80)
    logger.info("Step 1: Loading credit card model and data.")
    try:
        model, feature_names, class_names, df = load_credit_card_model_and_data()
        logger.info("=" * 80)
        logger.info("Step 2: Preprocessing features.")
        df_proc = preprocess_features(df, feature_names)
        logger.info(f"Preprocessing complete. Data shape: {df_proc.shape}")
        logger.info("Step 3: Extracting intermediate representations using XGBoost leaf indices.")
        # Use the same XGBoost representation extraction as VFL AutoML model
        intermediate_reps = extract_xgboost_representations(model, df_proc, target_dim=XGBOOST_OUTPUT_DIM)
        logger.info(f"Intermediate representations extracted: {intermediate_reps.shape}")
        logger.info("=" * 80)
        logger.info("Step 4: Initializing consistent explainer.")
        # Initialize the explainer using the same code as the main explainer
        explainer = CreditCardFeatureExplainer(
            model=model,
            feature_names=feature_names,
            class_names=class_names,
            random_state=42
        )
        
        # Initialize with training data sample
        X_scaled = model.scaler.transform(df_proc.values)
        # Use random 1000 samples for initialization instead of first 1000
        np.random.seed(42)  # For reproducibility
        random_indices_init = np.random.choice(len(X_scaled), size=1000, replace=False)
        X_train_sample = X_scaled[random_indices_init]
        explainer.initialize_explainer(X_train_sample)
        
        logger.info("Consistent explainer initialized successfully.")
        logger.info("Step 5: Generating feature explanations using consistent explainer.")
        # Instead of 5 specific customers, select 1000 random unique indices
        num_samples = 1000  # Generate 1000 samples for full dataset
        np.random.seed(42)
        all_indices = np.arange(len(df))
        random_indices = np.random.choice(all_indices, size=num_samples, replace=False)
        customer_ids = df.iloc[random_indices]['tax_id'].tolist()
        customer_data_list = [df_proc.iloc[idx][feature_names] for idx in random_indices]
        
        results = [None] * num_samples
        for i, (customer_id, customer_data) in enumerate(zip(customer_ids, customer_data_list)):
            logger.info(f"Processing customer {i+1}/{num_samples}: {customer_id}")
            # Get top features using the same logic as credit card explainer
            top_features = get_top_features_from_explainer(explainer, customer_data, customer_id, num_features=3)
            results[i] = top_features
            # Print detailed sample for this customer
            logger.info(f"Customer {customer_id} - Top 3 Features:")
            for j, feature in enumerate(top_features, 1):
                logger.info(f"  {j}. {feature['feature_name']} ({feature['impact']}, {feature['direction']})")
            logger.info("")
        logger.info("=" * 80)
        logger.info("Step 6: Creating training dataset.")
        dataset = []
        for i, (customer_id, customer_data) in enumerate(zip(customer_ids, customer_data_list)):
            rep = intermediate_reps[i]
            top_features = results[i]
            rep_flat = rep.flatten()
            dataset.append({
                'customer_id': customer_id,
                'intermediate_representation': rep_flat.tolist(),
                'top_features': top_features
            })
        logger.info(f"Dataset created with {len(dataset)} samples")
        logger.info("=" * 80)
        logger.info("Step 7: Saving dataset.")
        os.makedirs('data', exist_ok=True)
        with open('VFLClientModels/models/explanations/data/credit_card_feature_predictor_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        with open('VFLClientModels/models/explanations/data/credit_card_feature_predictor_dataset_sample.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        with open('VFLClientModels/models/explanations/data/credit_card_feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        logger.info("Dataset saved successfully!")
        logger.info(f"- Pickle file: VFLClientModels/models/explanations/data/credit_card_feature_predictor_dataset.pkl")
        logger.info(f"- JSON sample: VFLClientModels/models/explanations/data/credit_card_feature_predictor_dataset_sample.json")
        logger.info(f"- Feature names: VFLClientModels/models/explanations/data/credit_card_feature_names.txt")
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
        logger.info("Note: Using sophisticated feature contribution method with relative thresholds")
        logger.info(f"XGBoost intermediate representations: {XGBOOST_OUTPUT_DIM}D using PCA (same as VFL AutoML model)")
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 