#!/usr/bin/env python3
"""
Auto Loans Feature Prediction API

This Flask API provides endpoints to:
1. Get intermediate representations and predicted features for a customer ID
2. Health check endpoint

Usage:
    python app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import logging
from datetime import datetime
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.decomposition import PCA
import requests  # Add this import for internal API calls

# Set base directory to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the credit score predictor (now uses vfl_automl_xgboost_simple.py via nlg/credit_score_predictor.py)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nlg'))
from credit_score_predictor import CreditScorePredictor

# Instantiate the credit score predictor once (singleton)
credit_score_predictor_instance = CreditScorePredictor()

# Global variables to store loaded models and data
auto_loans_model = None
auto_loans_feature_predictor = None
feature_names = None
scaler = None
df = None
penultimate_model = None
X_scaled_data = None  # Store pre-scaled data for consistent lookups

# Home loans global variables
home_loans_model = None
home_loans_feature_predictor = None
home_loans_feature_names = None
home_loans_scaler = None
home_loans_df = None
home_loans_penultimate_model = None
home_loans_X_scaled_data = None  # Store pre-scaled data for consistent lookups

# Credit card global variables
credit_card_model = None
credit_card_feature_predictor = None
credit_card_feature_names = None
credit_card_scaler = None
credit_card_df = None
credit_card_penultimate_model = None
credit_card_X_scaled_data = None  # Store pre-scaled data for consistent lookups
credit_card_df_proc = None  # Store preprocessed data

# XGBoost Credit Card Configuration (same as dataset file)
XGBOOST_OUTPUT_DIM = 12             # Target output dimension for XGBoost representations
XGBOOST_PCA_RANDOM_STATE = 42       # Random state for PCA dimensionality reduction

# Add these global variables at the top with other credit card globals
credit_card_pca = None  # Store the fitted PCA for reuse
credit_card_leaf_indices_all = None  # Store all leaf indices for PCA fitting

# Digital savings global variables (using digital_bank model and feature predictor)
digital_savings_model = None
digital_savings_feature_predictor = None
digital_savings_feature_names = None
digital_savings_scaler = None
digital_savings_df = None
digital_savings_penultimate_model = None
digital_savings_X_scaled_data = None  # Store pre-scaled data for consistent lookups

def setup_logging():
    """Setup logging for the API"""
    logs_dir = os.path.join(BASE_DIR, 'VFLClientModels', 'models', 'apis', 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, f'api_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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

def load_models_and_data():
    """Load all required models and data"""
    logger = setup_logging()
    logger.info("Loading models and data.")
    
    global auto_loans_model, auto_loans_feature_predictor, feature_names, scaler, df, penultimate_model
    global home_loans_model, home_loans_feature_predictor, home_loans_feature_names, home_loans_scaler, home_loans_df, home_loans_penultimate_model, home_loans_X_scaled_data
    global credit_card_model, credit_card_feature_predictor, credit_card_feature_names, credit_card_scaler, credit_card_df, credit_card_penultimate_model, credit_card_X_scaled_data
    global credit_card_pca, credit_card_leaf_indices_all
    global digital_savings_model, digital_savings_feature_predictor, digital_savings_feature_names, digital_savings_scaler, digital_savings_df, digital_savings_penultimate_model, digital_savings_X_scaled_data
    
    try:
        # Load the auto loans model
        auto_loans_model = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'auto_loans_model.keras'), 
            compile=False
        )
        logger.info("Auto loans model loaded successfully")
        
        # Load the feature predictor model
        auto_loans_feature_predictor = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models',  'auto_loans_feature_predictor.keras'),
            compile=False
        )
        logger.info("Auto loans feature predictor model loaded successfully")
        
        # Load feature names
        feature_names = np.load(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'auto_loans_feature_names.npy'), 
            allow_pickle=True
        ).tolist()
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
        # Try to load scaler
        scaler_path = os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'auto_loans_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            logger.info("Scaler file not found, using fallback StandardScaler")
        
        # Load data
        df = pd.read_csv(os.path.join(BASE_DIR, 'VFLClientModels', 'dataset', 'data', 'banks', 'auto_loans_bank.csv'))
        logger.info(f"Data loaded: {len(df)} records")
        
        # Create penultimate model
        penultimate_model = get_penultimate_layer_model(auto_loans_model)
        logger.info("Penultimate model created successfully")
        
        # Fit scaler on the entire dataset (exactly like dataset producer)
        X = df[feature_names].values
        if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
            # Scaler already fitted, just transform
            X_scaled = scaler.transform(X)
            logger.info("Scaler already fitted, using existing parameters")
        else:
            # Fit scaler on entire dataset
            X_scaled = scaler.fit_transform(X)
            logger.info("Scaler fitted to entire dataset")
        
        # Store the scaled data for consistent lookups
        global X_scaled_data
        X_scaled_data = X_scaled
        
        # Load Home Loans models and data
        logger.info("Loading Home Loans models and data.")
        
        # Load the home loans model
        home_loans_model = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'home_loans_model.keras'), 
            compile=False
        )
        logger.info("Home loans model loaded successfully")
        
        # Load the home loans feature predictor model
        home_loans_feature_predictor = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models',  'home_loans_feature_predictor.keras'),
            compile=False
        )
        logger.info("Home loans feature predictor model loaded successfully")
        
        # Load home loans feature names
        home_loans_feature_names = np.load(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'home_loans_feature_names.npy'), 
            allow_pickle=True
        ).tolist()
        logger.info(f"Home loans feature names loaded: {len(home_loans_feature_names)} features")
        
        # Try to load home loans scaler
        home_loans_scaler_path = os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'home_loans_scaler.pkl')
        if os.path.exists(home_loans_scaler_path):
            home_loans_scaler = joblib.load(home_loans_scaler_path)
            logger.info("Home loans scaler loaded successfully")
        else:
            from sklearn.preprocessing import StandardScaler
            home_loans_scaler = StandardScaler()
            logger.info("Home loans scaler file not found, using fallback StandardScaler")
        
        # Load home loans data
        home_loans_df = pd.read_csv(os.path.join(BASE_DIR, 'VFLClientModels', 'dataset', 'data', 'banks', 'home_loans_bank.csv'))
        logger.info(f"Home loans data loaded: {len(home_loans_df)} records")
        
        # Create home loans penultimate model
        home_loans_penultimate_model = get_penultimate_layer_model(home_loans_model)
        logger.info("Home loans penultimate model created successfully")
        
        # Fit home loans scaler on the entire dataset (exactly like dataset producer)
        home_loans_X = home_loans_df[home_loans_feature_names].values
        if hasattr(home_loans_scaler, 'mean_') and home_loans_scaler.mean_ is not None:
            # Scaler already fitted, just transform
            home_loans_X_scaled = home_loans_scaler.transform(home_loans_X)
            logger.info("Home loans scaler already fitted, using existing parameters")
        else:
            # Fit scaler on entire dataset
            home_loans_X_scaled = home_loans_scaler.fit_transform(home_loans_X)
            logger.info("Home loans scaler fitted to entire dataset")
        
        # Store the home loans scaled data for consistent lookups
        home_loans_X_scaled_data = home_loans_X_scaled
        
        # Load Credit Card models and data
        logger.info("Loading Credit Card models and data.")
        
        # Load the credit card XGBoost model
        try:
            # Import the XGBoost model class
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from credit_card_xgboost_model import IndependentXGBoostModel
            
            # Load the XGBoost model
            credit_card_model = IndependentXGBoostModel.load_model(
                os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'credit_card_xgboost_independent.pkl')
            )
            logger.info("Credit card XGBoost model loaded successfully")
            
            # For XGBoost models, we use PCA for intermediate representation
            credit_card_penultimate_model = None  # XGBoost uses PCA instead
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {str(e)}")
            raise Exception(f"Could not load credit card XGBoost model: {str(e)}")
        
        # Load the credit card feature predictor model
        credit_card_feature_predictor = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models',  'credit_card_feature_predictor.keras'),
            compile=False
        )
        logger.info("Credit card feature predictor model loaded successfully")
        
        # Load credit card XGBoost feature names
        credit_card_feature_names = np.load(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'credit_card_xgboost_feature_names.npy'), 
            allow_pickle=True
        ).tolist()
        logger.info(f"Credit card XGBoost feature names loaded: {len(credit_card_feature_names)} features")
        
        # Load credit card data
        credit_card_df = pd.read_csv(os.path.join(BASE_DIR, 'VFLClientModels', 'dataset', 'data', 'banks', 'credit_card_bank.csv'))
        logger.info(f"Credit card data loaded: {len(credit_card_df)} records")
        
        # Preprocess features exactly like the dataset file
        def preprocess_credit_card_features(df, feature_names):
            """Preprocess features for credit card model (same as dataset file)"""
            df_proc = df.copy()
            
            # Create derived features exactly as in credit_card_model.py
            df_proc['credit_capacity_ratio'] = df_proc['credit_card_limit'] / df_proc['total_credit_limit'].replace(0, 1)
            df_proc['income_to_limit_ratio'] = df_proc['annual_income'] / df_proc['credit_card_limit'].replace(0, 1)
            df_proc['debt_service_ratio'] = (df_proc['current_debt'] * 0.03) / (df_proc['annual_income'] / 12)
            df_proc['risk_adjusted_income'] = df_proc['annual_income'] * (df_proc['risk_score'] / 100)

            # Use the exact features that the model was trained with (25 features + 4 derived = 29 features)
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
                # Derived features (already calculated above)
                'credit_capacity_ratio', 'income_to_limit_ratio', 'debt_service_ratio', 'risk_adjusted_income'
            ]
            
            # Ensure all required feature columns are present
            for feat in credit_card_features:
                if feat not in df_proc.columns:
                    df_proc[feat] = 0
            
            return df_proc[credit_card_features]

        # Preprocess the entire dataset
        credit_card_df_proc = preprocess_credit_card_features(credit_card_df, credit_card_feature_names)
        logger.info(f"Credit card features preprocessed: {credit_card_df_proc.shape}")

        # Handle infinite and missing values
        credit_card_df_proc = credit_card_df_proc.replace([np.inf, -np.inf], np.nan)
        numeric_cols = credit_card_df_proc.select_dtypes(include=[np.number]).columns
        credit_card_df_proc[numeric_cols] = credit_card_df_proc[numeric_cols].fillna(credit_card_df_proc[numeric_cols].median())

        # Scale features using the model's scaler
        credit_card_X_scaled = credit_card_model.scaler.transform(credit_card_df_proc.values)
        logger.info(f"Credit card features scaled: {credit_card_X_scaled.shape}")

        # Store the scaled data for consistent lookups
        credit_card_X_scaled_data = credit_card_X_scaled

        # EXACTLY LIKE DATASET FILE: Extract XGBoost representations for entire dataset
        logger.info("Extracting XGBoost representations for entire dataset (same as dataset file).")
        
        # Get leaf indices from XGBoost for entire dataset
        credit_card_leaf_indices_all = credit_card_model.classifier.apply(credit_card_X_scaled)
        logger.info(f"XGBoost leaf indices shape: {credit_card_leaf_indices_all.shape}")
        
        # Fit PCA on entire dataset (same as dataset file)
        XGBOOST_OUTPUT_DIM = 12
        XGBOOST_PCA_RANDOM_STATE = 42
        
        if credit_card_leaf_indices_all.shape[1] == XGBOOST_OUTPUT_DIM:
            # Perfect match, no PCA needed
            credit_card_pca = None
            logger.info("Perfect dimension match, no PCA needed")
        elif credit_card_leaf_indices_all.shape[1] > XGBOOST_OUTPUT_DIM:
            # Use PCA to reduce dimensions (same as dataset file)
            logger.info(f"Fitting PCA: {credit_card_leaf_indices_all.shape[1]}D → {XGBOOST_OUTPUT_DIM}D")
            credit_card_pca = PCA(n_components=XGBOOST_OUTPUT_DIM, random_state=XGBOOST_PCA_RANDOM_STATE)
            credit_card_pca.fit(credit_card_leaf_indices_all.astype(np.float32))
            explained_variance = credit_card_pca.explained_variance_ratio_.sum()
            logger.info(f"PCA fitted: explained variance: {explained_variance:.3f}")
        else:
            # Pad with zeros, no PCA needed
            credit_card_pca = None
            logger.info(f"Leaf indices smaller than target, will pad with zeros")
        
        # Load Digital Savings models and data (using digital_bank model and feature predictor)
        logger.info("Loading Digital Savings models and data.")

        # Load the digital bank model (used for digital savings)
        digital_savings_model = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'digital_bank_model.keras'), 
            compile=False
        )
        logger.info("Digital savings model (digital_bank) loaded successfully")

        # Load the digital bank feature predictor
        digital_savings_feature_predictor = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models',  'digital_bank_feature_predictor.keras'), 
            compile=False
        )
        logger.info("Digital savings feature predictor loaded successfully")

        # Load feature names
        digital_savings_feature_names = np.load(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'digital_bank_feature_names.npy'), 
            allow_pickle=True
        ).tolist()
        logger.info(f"Digital savings feature names loaded: {len(digital_savings_feature_names)} features")

        # Try to load scaler, create fallback if not available (same as auto loans)
        scaler_path = os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'digital_bank_scaler.pkl')
        if os.path.exists(scaler_path):
            digital_savings_scaler = joblib.load(scaler_path)
            logger.info("Digital savings scaler loaded successfully")
        else:
            # Create a fallback scaler
            from sklearn.preprocessing import StandardScaler
            digital_savings_scaler = StandardScaler()
            logger.info("Digital savings scaler file not found, using fallback StandardScaler")

        # Load data using the regular digital_savings_bank.csv (not the full version)
        digital_savings_df = pd.read_csv(
            os.path.join(BASE_DIR, 'VFLClientModels', 'dataset', 'data', 'banks', 'digital_savings_bank.csv')
        )
        logger.info(f"Digital savings data loaded: {len(digital_savings_df)} records")

        # Create derived features for missing features (same logic as dataset file but adapted for regular CSV)
        if 'transaction_volume' in digital_savings_feature_names and 'transaction_volume' not in digital_savings_df.columns:
            if 'avg_monthly_transactions' in digital_savings_df.columns and 'avg_transaction_value' in digital_savings_df.columns:
                digital_savings_df['transaction_volume'] = digital_savings_df['avg_monthly_transactions'] * digital_savings_df['avg_transaction_value']
                logger.info("Created derived feature: transaction_volume")
            else:
                digital_savings_df['transaction_volume'] = digital_savings_df['avg_monthly_transactions']
                logger.info("Mapped transaction_volume to avg_monthly_transactions")

        if 'digital_engagement_score' in digital_savings_feature_names and 'digital_engagement_score' not in digital_savings_df.columns:
            if 'digital_engagement_level' in digital_savings_df.columns:
                mapping = {'Low': 0, 'Medium': 1, 'High': 2}
                digital_savings_df['digital_engagement_score'] = digital_savings_df['digital_engagement_level'].map(mapping)
                logger.info("Mapped digital_engagement_score to digital_engagement_level (as numeric)")
            elif 'digital_activity_score' in digital_savings_df.columns:
                digital_savings_df['digital_engagement_score'] = digital_savings_df['digital_activity_score']
                logger.info("Mapped digital_engagement_score to digital_activity_score")
            else:
                digital_savings_df['digital_engagement_score'] = digital_savings_df['digital_banking_score'] * digital_savings_df['online_transactions_ratio']
                logger.info("Created derived feature: digital_engagement_score")

        if 'total_wealth' in digital_savings_feature_names and 'total_wealth' not in digital_savings_df.columns:
            if 'total_liquid_assets' in digital_savings_df.columns and 'total_portfolio_value' in digital_savings_df.columns:
                digital_savings_df['total_wealth'] = digital_savings_df['total_liquid_assets'] + digital_savings_df['total_portfolio_value']
                logger.info("Created derived feature: total_wealth")
            else:
                wealth_features = [col for col in digital_savings_df.columns if 'balance' in col.lower() or 'assets' in col.lower()]
                if wealth_features:
                    digital_savings_df['total_wealth'] = digital_savings_df[wealth_features].sum(axis=1)
                    logger.info(f"Created total_wealth from: {wealth_features}")
                else:
                    digital_savings_df['total_wealth'] = digital_savings_df['annual_income'] * 10
                    logger.info("Created total_wealth from annual_income approximation")

        if 'net_worth' in digital_savings_feature_names and 'net_worth' not in digital_savings_df.columns:
            if 'total_wealth' in digital_savings_df.columns and 'current_debt' in digital_savings_df.columns:
                digital_savings_df['net_worth'] = digital_savings_df['total_wealth'] - digital_savings_df['current_debt']
                logger.info("Created derived feature: net_worth")
            else:
                asset_features = [col for col in digital_savings_df.columns if 'balance' in col.lower() or 'assets' in col.lower()]
                debt_features = [col for col in digital_savings_df.columns if 'debt' in col.lower() or 'balance' in col.lower() and 'loan' in col.lower()]
                if asset_features and debt_features:
                    digital_savings_df['net_worth'] = digital_savings_df[asset_features].sum(axis=1) - digital_savings_df[debt_features].sum(axis=1)
                    logger.info(f"Created net_worth from assets: {asset_features} and debts: {debt_features}")
                else:
                    digital_savings_df['net_worth'] = digital_savings_df['annual_income'] * 5
                    logger.info("Created net_worth from annual_income approximation")

        if 'credit_efficiency' in digital_savings_feature_names and 'credit_efficiency' not in digital_savings_df.columns:
            if 'credit_score' in digital_savings_df.columns and 'credit_utilization_ratio' in digital_savings_df.columns:
                digital_savings_df['credit_efficiency'] = digital_savings_df['credit_score'] / (digital_savings_df['credit_utilization_ratio'] + 0.01)
                logger.info("Created derived feature: credit_efficiency")
            else:
                digital_savings_df['credit_efficiency'] = digital_savings_df['credit_score']
                logger.info("Mapped credit_efficiency to credit_score")

        if 'financial_stability_score' in digital_savings_feature_names and 'financial_stability_score' not in digital_savings_df.columns:
            if 'financial_health_score' in digital_savings_df.columns:
                digital_savings_df['financial_stability_score'] = digital_savings_df['financial_health_score']
                logger.info("Mapped financial_stability_score to financial_health_score")
            else:
                stability_features = ['credit_score', 'payment_history', 'debt_to_income_ratio']
                available_features = [f for f in stability_features if f in digital_savings_df.columns]
                if len(available_features) >= 2:
                    digital_savings_df['financial_stability_score'] = digital_savings_df[available_features].mean(axis=1)
                    logger.info(f"Created financial_stability_score from: {available_features}")
                else:
                    digital_savings_df['financial_stability_score'] = digital_savings_df['credit_score']
                    logger.info("Mapped financial_stability_score to credit_score")

        # Ensure all features are numeric
        for f in digital_savings_feature_names:
            if f in digital_savings_df.columns and not pd.api.types.is_numeric_dtype(digital_savings_df[f]):
                try:
                    digital_savings_df[f] = pd.to_numeric(digital_savings_df[f], errors='coerce')
                    logger.info(f"Converted {f} to numeric (coerce errors)")
                except Exception as e:
                    logger.warning(f"Could not convert {f} to numeric: {e}")

        # Verify all required features are available
        missing_features = [f for f in digital_savings_feature_names if f not in digital_savings_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            logger.warning("Available columns: " + ", ".join(digital_savings_df.columns.tolist()))
            raise ValueError(f"Missing features in dataset: {missing_features}")

        logger.info("All required features are available in the dataset")

        # Create penultimate model
        digital_savings_penultimate_model = get_penultimate_layer_model(digital_savings_model)
        logger.info("Digital savings penultimate model created")

        # Process the entire dataset to get scaled data (same approach as dataset file)
        numeric_feature_names = [f for f in digital_savings_feature_names if pd.api.types.is_numeric_dtype(digital_savings_df[f])]
        if len(numeric_feature_names) < len(digital_savings_feature_names):
            logger.warning(f"Non-numeric features removed: {[f for f in digital_savings_feature_names if f not in numeric_feature_names]}")
        digital_savings_feature_names = numeric_feature_names

        X = digital_savings_df[digital_savings_feature_names].values
        if hasattr(digital_savings_scaler, 'mean_') and digital_savings_scaler.mean_ is not None:
            digital_savings_X_scaled_data = digital_savings_scaler.transform(X)
            logger.info("Digital savings scaler already fitted, using existing parameters")
        else:
            digital_savings_X_scaled_data = digital_savings_scaler.fit_transform(X)
            logger.info("Digital savings scaler fitted to data")

        logger.info(f"Digital savings scaled data shape: {digital_savings_X_scaled_data.shape}")

        logger.info("All models and data loaded successfully!")
        logger.info(f"Auto loans: {len(feature_names)} features, {len(df)} customers")
        logger.info(f"Home loans: {len(home_loans_feature_names)} features, {len(home_loans_df)} customers")
        logger.info(f"Credit card: {len(credit_card_feature_names)} features, {len(credit_card_df)} customers")
        logger.info(f"Digital savings: {len(digital_savings_feature_names)} features, {len(digital_savings_df)} customers")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models/data: {str(e)}")
        return False

def get_customer_data(customer_id):
    """Get customer data by ID and return both data and index"""
    global df
    
    if df is None:
        return None, None
    
    # Try to find customer by tax_id
    if 'tax_id' in df.columns:
        customer_data = df[df['tax_id'] == customer_id]
        if not customer_data.empty:
            customer_idx = customer_data.index[0]
            return customer_data.iloc[0], customer_idx
    
    # If not found by tax_id, try by index (assuming customer_id is numeric index)
    try:
        customer_idx = int(customer_id)
        if 0 <= customer_idx < len(df):
            return df.iloc[customer_idx], customer_idx
    except (ValueError, IndexError):
        pass
    
    return None, None

def get_intermediate_representation(customer_idx):
    """Get intermediate representation for customer by index (exactly like dataset producer)"""
    global X_scaled_data, penultimate_model
    
    if customer_idx is None or X_scaled_data is None or penultimate_model is None:
        return None
    
    try:
        # Get the pre-scaled data for this customer (exactly like dataset producer)
        X_scaled = X_scaled_data[customer_idx:customer_idx+1]
        
        # Get intermediate representation
        intermediate_rep = penultimate_model.predict(X_scaled, verbose=0)
        
        return intermediate_rep.flatten().tolist()
        
    except Exception as e:
        logging.error(f"Error getting intermediate representation: {str(e)}")
        return None

def get_home_loans_customer_data(customer_id):
    """Get home loans customer data by ID and return both data and index"""
    global home_loans_df
    
    if home_loans_df is None:
        return None, None
    
    # Try to find customer by tax_id
    if 'tax_id' in home_loans_df.columns:
        customer_data = home_loans_df[home_loans_df['tax_id'] == customer_id]
        if not customer_data.empty:
            customer_idx = customer_data.index[0]
            return customer_data.iloc[0], customer_idx
    
    # If not found by tax_id, try by index (assuming customer_id is numeric index)
    try:
        customer_idx = int(customer_id)
        if 0 <= customer_idx < len(home_loans_df):
            return home_loans_df.iloc[customer_idx], customer_idx
    except (ValueError, IndexError):
        pass
    
    return None, None

def get_home_loans_intermediate_representation(customer_idx):
    """Get home loans intermediate representation for customer by index (exactly like dataset producer)"""
    global home_loans_X_scaled_data, home_loans_penultimate_model
    
    if customer_idx is None or home_loans_X_scaled_data is None or home_loans_penultimate_model is None:
        return None
    
    try:
        # Get the pre-scaled data for this customer (exactly like dataset producer)
        X_scaled = home_loans_X_scaled_data[customer_idx:customer_idx+1]
        
        # Get intermediate representation
        intermediate_rep = home_loans_penultimate_model.predict(X_scaled, verbose=0)
        
        return intermediate_rep.flatten().tolist()
        
    except Exception as e:
        logging.error(f"Error getting home loans intermediate representation: {str(e)}")
        return None

def predict_home_loans_features(intermediate_rep):
    """Predict home loans features from intermediate representation"""
    global home_loans_feature_predictor, home_loans_feature_names
    
    if intermediate_rep is None or home_loans_feature_predictor is None:
        return None
    
    try:
        # Convert to numpy array
        X = np.array([intermediate_rep], dtype=np.float32)
        
        # Predict
        predictions = home_loans_feature_predictor.predict(X, verbose=0)
        
        # Process predictions
        predicted_features = []
        
        for feature_num in [1, 2, 3]:
            # Get feature index
            idx_pred = predictions[f'feature_{feature_num}_idx'][0]
            feature_idx = np.argmax(idx_pred)
            feature_name = home_loans_feature_names[feature_idx] if feature_idx < len(home_loans_feature_names) else "Unknown"
            
            # Get direction
            dir_pred = predictions[f'feature_{feature_num}_direction'][0]
            direction_idx = np.argmax(dir_pred)
            direction = "Positive" if direction_idx == 0 else "Negative"
            
            # Get impact
            impact_pred = predictions[f'feature_{feature_num}_impact'][0]
            impact_idx = np.argmax(impact_pred)
            impact_mapping = {0: "Very High", 1: "High", 2: "Medium", 3: "Low"}
            impact = impact_mapping.get(impact_idx, "Medium")
            
            predicted_features.append({
                'feature_name': feature_name,
                'direction': direction,
                'impact': impact,
                'confidence': {
                    'index': float(np.max(idx_pred)),
                    'direction': float(np.max(dir_pred)),
                    'impact': float(np.max(impact_pred))
                }
            })
        
        return predicted_features
        
    except Exception as e:
        logging.error(f"Error predicting home loans features: {str(e)}")
        return None

def get_credit_card_customer_data(customer_id):
    """Get credit card customer data by ID and return both data and index"""
    global credit_card_df
    
    if credit_card_df is None:
        return None, None
    
    # Try to find customer by tax_id
    if 'tax_id' in credit_card_df.columns:
        customer_data = credit_card_df[credit_card_df['tax_id'] == customer_id]
        if not customer_data.empty:
            customer_idx = customer_data.index[0]
            return customer_data.iloc[0], customer_idx
    
    # If not found by tax_id, try by index (assuming customer_id is numeric index)
    try:
        customer_idx = int(customer_id)
        if 0 <= customer_idx < len(credit_card_df):
            return credit_card_df.iloc[customer_idx], customer_idx
    except (ValueError, IndexError):
        pass
    
    return None, None

def get_credit_card_intermediate_representation(customer_idx):
    """Get credit card intermediate representation using XGBoost leaf indices + PCA (exactly like dataset file)"""
    global credit_card_X_scaled_data, credit_card_model, credit_card_pca, credit_card_leaf_indices_all
    
    if customer_idx is None or credit_card_X_scaled_data is None or credit_card_model is None:
        logging.error(f"Missing required data: customer_idx={customer_idx}, scaled_data={credit_card_X_scaled_data is not None}, model={credit_card_model is not None}")
        return None
    
    try:
        # Get the scaled features for this customer
        X_scaled = credit_card_X_scaled_data[customer_idx:customer_idx+1]
        
        # Get leaf indices from XGBoost for this customer
        leaf_indices = credit_card_model.classifier.apply(X_scaled)
        
        # Convert leaf indices to target dimensions using fitted PCA (same as dataset file)
        XGBOOST_OUTPUT_DIM = 12
        
        if credit_card_pca is None:
            # No PCA needed, either perfect match or padding
            if leaf_indices.shape[1] == XGBOOST_OUTPUT_DIM:
                representations = leaf_indices.astype(np.float32)
            else:
                # Pad with zeros
                representations = np.zeros((leaf_indices.shape[0], XGBOOST_OUTPUT_DIM), dtype=np.float32)
                representations[:, :leaf_indices.shape[1]] = leaf_indices.astype(np.float32)
        else:
            # Use fitted PCA to transform
            representations = credit_card_pca.transform(leaf_indices.astype(np.float32))
        
        # Normalize representations to [0, 1] range for consistency (same as dataset file)
        if representations.max() > representations.min():
            representations = (representations - representations.min()) / (representations.max() - representations.min())
        
        result = representations.flatten().tolist()
        return result
        
    except Exception as e:
        logging.error(f"Error getting credit card intermediate representation: {str(e)}")
        return None

def predict_credit_card_features(intermediate_rep):
    """Predict credit card features from intermediate representation"""
    global credit_card_feature_predictor, credit_card_feature_names
    
    if intermediate_rep is None or credit_card_feature_predictor is None:
        return None
    
    try:
        # Convert to numpy array
        X = np.array([intermediate_rep], dtype=np.float32)
        
        # Predict
        predictions = credit_card_feature_predictor.predict(X, verbose=0)
        
        # Process predictions
        predicted_features = []
        
        for feature_num in [1, 2, 3]:
            # Get feature index
            idx_pred = predictions[f'feature_{feature_num}_idx'][0]
            feature_idx = np.argmax(idx_pred)
            feature_name = credit_card_feature_names[feature_idx] if feature_idx < len(credit_card_feature_names) else "Unknown"
            
            # Get direction
            dir_pred = predictions[f'feature_{feature_num}_direction'][0]
            direction_idx = np.argmax(dir_pred)
            direction = "Positive" if direction_idx == 0 else "Negative"
            
            # Get impact
            impact_pred = predictions[f'feature_{feature_num}_impact'][0]
            impact_idx = np.argmax(impact_pred)
            impact_mapping = {0: "Very High", 1: "High", 2: "Medium", 3: "Low"}
            impact = impact_mapping.get(impact_idx, "Medium")
            
            predicted_features.append({
                'feature_name': feature_name,
                'direction': direction,
                'impact': impact,
                'confidence': {
                    'index': float(np.max(idx_pred)),
                    'direction': float(np.max(dir_pred)),
                    'impact': float(np.max(impact_pred))
                }
            })
        
        return predicted_features
        
    except Exception as e:
        logging.error(f"Error predicting credit card features: {str(e)}")
        return None

def predict_features(intermediate_rep):
    """Predict features from intermediate representation"""
    global auto_loans_feature_predictor, feature_names
    
    if intermediate_rep is None or auto_loans_feature_predictor is None:
        return None
    
    try:
        # Convert to numpy array
        X = np.array([intermediate_rep], dtype=np.float32)
        
        # Predict
        predictions = auto_loans_feature_predictor.predict(X, verbose=0)
        
        # Process predictions
        predicted_features = []
        
        for feature_num in [1, 2, 3]:
            # Get feature index
            idx_pred = predictions[f'feature_{feature_num}_idx'][0]
            feature_idx = np.argmax(idx_pred)
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else "Unknown"
            
            # Get direction
            dir_pred = predictions[f'feature_{feature_num}_direction'][0]
            direction_idx = np.argmax(dir_pred)
            direction = "Positive" if direction_idx == 0 else "Negative"
            
            # Get impact
            impact_pred = predictions[f'feature_{feature_num}_impact'][0]
            impact_idx = np.argmax(impact_pred)
            impact_mapping = {0: "Very High", 1: "High", 2: "Medium", 3: "Low"}
            impact = impact_mapping.get(impact_idx, "Medium")
            
            predicted_features.append({
                'feature_name': feature_name,
                'direction': direction,
                'impact': impact,
                'confidence': {
                    'index': float(np.max(idx_pred)),
                    'direction': float(np.max(dir_pred)),
                    'impact': float(np.max(impact_pred))
                }
            })
        
        return predicted_features
        
    except Exception as e:
        logging.error(f"Error predicting features: {str(e)}")
        return None

def get_digital_savings_customer_data(customer_id):
    """Get digital savings customer data by ID and return both data and index"""
    global digital_savings_df
    
    if digital_savings_df is None:
        return None, None
    
    # Try to find customer by tax_id
    if 'tax_id' in digital_savings_df.columns:
        customer_data = digital_savings_df[digital_savings_df['tax_id'] == customer_id]
        if not customer_data.empty:
            customer_idx = customer_data.index[0]
            return customer_data.iloc[0], customer_idx
    
    # If not found by tax_id, try by index (assuming customer_id is numeric index)
    try:
        customer_idx = int(customer_id)
        if 0 <= customer_idx < len(digital_savings_df):
            return digital_savings_df.iloc[customer_idx], customer_idx
    except (ValueError, IndexError):
        pass
    
    return None, None

def get_digital_savings_intermediate_representation(customer_idx):
    """Get digital savings intermediate representation (EXACTLY as in dataset file)"""
    global digital_savings_X_scaled_data, digital_savings_penultimate_model
    
    if customer_idx is None or digital_savings_X_scaled_data is None or digital_savings_penultimate_model is None:
        logging.error("Digital savings data or model not loaded")
        return None
    
    try:
        # Get the scaled data for this customer (EXACTLY as in dataset file)
        customer_scaled_data = digital_savings_X_scaled_data[customer_idx:customer_idx+1]
        
        # Get intermediate representation using penultimate model (EXACTLY as in dataset file)
        intermediate_rep = digital_savings_penultimate_model.predict(customer_scaled_data, verbose=0)
        
        # Flatten the representation (EXACTLY as in dataset file)
        rep_flat = intermediate_rep.flatten()
        
        return rep_flat.tolist()
        
    except Exception as e:
        logging.error(f"Error getting digital savings intermediate representation: {str(e)}")
        return None

def predict_digital_savings_features(intermediate_rep):
    """Predict digital savings features from intermediate representation"""
    global digital_savings_feature_predictor, digital_savings_feature_names
    
    if intermediate_rep is None or digital_savings_feature_predictor is None:
        return None
    
    try:
        # Convert to numpy array
        X = np.array([intermediate_rep], dtype=np.float32)
        
        # Predict
        predictions = digital_savings_feature_predictor.predict(X, verbose=0)
        
        # Process predictions
        predicted_features = []
        
        for feature_num in [1, 2, 3]:
            # Get feature index
            idx_pred = predictions[f'feature_{feature_num}_idx'][0]
            feature_idx = np.argmax(idx_pred)
            feature_name = digital_savings_feature_names[feature_idx] if feature_idx < len(digital_savings_feature_names) else "Unknown"
            
            # Get direction
            dir_pred = predictions[f'feature_{feature_num}_direction'][0]
            direction_idx = np.argmax(dir_pred)
            direction = "Positive" if direction_idx == 0 else "Negative"
            
            # Get impact
            impact_pred = predictions[f'feature_{feature_num}_impact'][0]
            impact_idx = np.argmax(impact_pred)
            impact_mapping = {0: "Very High", 1: "High", 2: "Medium", 3: "Low"}
            impact = impact_mapping.get(impact_idx, "Medium")
            
            predicted_features.append({
                'feature_name': feature_name,
                'direction': direction,
                'impact': impact,
                'confidence': {
                    'index': float(np.max(idx_pred)),
                    'direction': float(np.max(dir_pred)),
                    'impact': float(np.max(impact_pred))
                }
            })
        
        return predicted_features
        
    except Exception as e:
        logging.error(f"Error predicting digital savings features: {str(e)}")
        return None

def call_internal_api(endpoint, payload):
    """Helper to call internal Flask endpoints via HTTP POST."""
    try:
        url = f"http://127.0.0.1:5001{endpoint}"
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            # Try to extract error from response body
            try:
                err_json = response.json()
                err_msg = err_json.get('error', str(err_json))
            except Exception:
                err_msg = response.text
            return {'error': f"{endpoint} failed: {err_msg}", 'status_code': response.status_code}
    except Exception as e:
        return {'error': f'Exception calling {endpoint}: {str(e)}'}

def filter_out_credit_score(features):
    if isinstance(features, list):
        return [f for f in features if f.get('feature_name') != 'credit_score']
    return features


def aggregate_feature_explanations(customer_id):
    """Call all feature explanation endpoints and aggregate their results."""
    endpoints = [
        ('auto-loan', '/auto-loan/predict'),
        ('credit-card', '/credit-card/predict'),
        ('home-loan', '/home-loan/predict'),
        ('digital-savings', '/digital-savings/predict'),
    ]
    explanations = {}
    for key, endpoint in endpoints:
        result = call_internal_api(endpoint, {'customer_id': customer_id})
        if 'predicted_features' in result:
            explanations[key] = filter_out_credit_score(result['predicted_features'])
        else:
            explanations[key] = {'error': result.get('error', 'No features found')}
    return explanations


def format_slm_prompt(customer_id, explanations):
    """Format a prompt for the SLM based on feature explanations."""
    prompt = f"""
Customer ID: {customer_id}

For this customer, the following features were identified as most important for their credit score prediction in each product:
"""
    for product, feats in explanations.items():
        prompt += f"\n- {product.replace('-', ' ').title()} important features:\n"
        if isinstance(feats, list):
            for feat in feats:
                fname = feat.get('feature_name', 'Unknown')
                direction = feat.get('direction', 'Unknown')
                impact = feat.get('impact', 'Unknown')
                prompt += f"    • {fname}: {direction}, impact: {impact}\n"
        else:
            prompt += f"    • {feats.get('error', 'No data')}\n"
    prompt += "\nIn two sentences, briefly describe the impact of this on the customer credit score."
    return prompt

def format_slm_prompt_single(product, feats, customer_id, credit_score=None):
    """Format a prompt for the SLM for a single product's features with score-based guidance."""
    
    # Determine score category and tone
    score_category = "unknown"
    tone_guidance = ""
    
    if credit_score is not None:
        if credit_score > 750:
            score_category = "excellent"
            tone_guidance = "This customer has an excellent credit score (>750). Focus on the positive impact of their strong features."
        elif credit_score >= 700:
            score_category = "good"
            tone_guidance = "This customer has a good credit score (700-750). Highlight their solid features and note areas for enhancement."
        elif credit_score >= 600:
            score_category = "above_average"
            tone_guidance = "This customer has an above-average credit score (600-700). Provide balanced analysis of their features."
        elif credit_score >= 400:
            score_category = "below_average"
            tone_guidance = "This customer has a below-average credit score (400-600). Focus on critical features requiring improvement."
        else:
            score_category = "poor"
            tone_guidance = "This customer has a very poor credit score (<400). Emphasize critical features significantly impacting their score."
    
    # If feats is a dict with an 'error' key, handle the case where the customer does not have this product
    if isinstance(feats, dict) and 'error' in feats:
        prompt = f"""
Customer ID: {customer_id}
Credit Score: {credit_score if credit_score is not None else 'Unknown'}
Score Category: {score_category.replace('_', ' ').title()}

The customer does not have a {product.replace('-', ' ')}.

{tone_guidance}

Explain how the absence of a {product.replace('-', ' ')} impacts the credit score based on the analysis.
"""
        return prompt

    prompt = f"""
Customer ID: {customer_id}
Credit Score: {credit_score if credit_score is not None else 'Unknown'}
Score Category: {score_category.replace('_', ' ').title()}

For this customer, the following features were identified as most important for their credit score prediction for {product.replace('-', ' ').title()}:
"""
    if isinstance(feats, list) and feats:
        for feat in feats:
            fname = feat.get('feature_name', 'Unknown')
            direction = feat.get('direction', 'Unknown')
            impact = feat.get('impact', 'Unknown')
            prompt += f"    • {fname}: {direction}, impact: {impact}\n"
    elif isinstance(feats, list) and not feats:
        prompt += "    • No important features were identified.\n"
    else:
        prompt += f"    • {feats.get('error', 'No data')}\n"
    
    prompt += f"""

{tone_guidance}

Explain which factors most positively and negatively affected the customer's credit score based on the feature analysis.
"""
    return prompt


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'auto_loans_models_loaded': all([auto_loans_model, auto_loans_feature_predictor, feature_names, df]),
        'home_loans_models_loaded': all([home_loans_model, home_loans_feature_predictor, home_loans_feature_names, home_loans_df]),
        'credit_card_models_loaded': all([credit_card_model, credit_card_feature_predictor, credit_card_feature_names, credit_card_df]),
        'digital_savings_models_loaded': all([digital_savings_model, digital_savings_feature_predictor, digital_savings_feature_names, digital_savings_df])
    })

@app.route('/auto-loan/predict', methods=['POST'])
def predict_customer_features():
    """Predict features for a customer ID"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'customer_id' not in data:
            return jsonify({
                'error': 'Missing customer_id in request body'
            }), 400
        
        customer_id = data['customer_id']
        
        # Check if models are loaded
        if any([auto_loans_model is None, auto_loans_feature_predictor is None, 
                feature_names is None, df is None]):
            return jsonify({
                'error': 'Models not loaded. Please check server logs.'
            }), 500
        
        # Get customer data and index
        customer_data, customer_idx = get_customer_data(customer_id)
        if customer_data is None or customer_idx is None:
            return jsonify({
                'error': f'Customer with ID {customer_id} does not have an auto loan'
            }), 404
        
        # Get intermediate representation
        intermediate_rep = get_intermediate_representation(customer_idx)
        if intermediate_rep is None:
            return jsonify({
                'error': 'Failed to generate intermediate representation'
            }), 500
        
        # Predict features
        predicted_features = predict_features(intermediate_rep)
        if predicted_features is None:
            return jsonify({
                'error': 'Failed to predict features'
            }), 500
        predicted_features = filter_out_credit_score(predicted_features)
        
        # Get credit score for feature modification
        try:
            logging.info(f"🔍 Getting credit score for customer {customer_id} to modify features")
            credit_score_result = credit_score_predictor_instance.predict_credit_score(customer_id)
            credit_score = credit_score_result.get('predicted') if credit_score_result else None
            logging.info(f"📊 Credit score for customer {customer_id}: {credit_score}")
            # Modify features based on credit score thresholds
            predicted_features = modify_features_based_on_credit_score(predicted_features, credit_score)
        except Exception as e:
            logging.warning(f"⚠️ Could not get credit score for feature modification: {str(e)}")
            # Continue without modification if credit score prediction fails
        
        # Prepare response
        response = {
            'customer_id': customer_id,
            'description': 'Auto loan explanation',
            'intermediate_representation': convert_numpy_types(intermediate_rep),
            'predicted_features': convert_numpy_types(predicted_features),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/home-loan/predict', methods=['POST'])
def predict_home_loans_customer_features():
    """Predict home loans features for a customer ID"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'customer_id' not in data:
            return jsonify({
                'error': 'Missing customer_id in request body'
            }), 400
        
        customer_id = data['customer_id']
        
        # Check if models are loaded
        if any([home_loans_model is None, home_loans_feature_predictor is None, 
                home_loans_feature_names is None, home_loans_df is None]):
            return jsonify({
                'error': 'Home loans models not loaded. Please check server logs.'
            }), 500
        
        # Get customer data and index
        customer_data, customer_idx = get_home_loans_customer_data(customer_id)
        if customer_data is None or customer_idx is None:
            return jsonify({
                'error': f'Customer with ID {customer_id} does not have a home loan'
            }), 404
        
        # Get intermediate representation
        intermediate_rep = get_home_loans_intermediate_representation(customer_idx)
        if intermediate_rep is None:
            return jsonify({
                'error': 'Failed to generate home loans intermediate representation'
            }), 500
        
        # Predict features
        predicted_features = predict_home_loans_features(intermediate_rep)
        if predicted_features is None:
            return jsonify({
                'error': 'Failed to predict home loans features'
            }), 500
        predicted_features = filter_out_credit_score(predicted_features)
        
        # Get credit score for feature modification
        try:
            logging.info(f"🔍 Getting credit score for customer {customer_id} to modify features")
            credit_score_result = credit_score_predictor_instance.predict_credit_score(customer_id)
            credit_score = credit_score_result.get('predicted') if credit_score_result else None
            logging.info(f"📊 Credit score for customer {customer_id}: {credit_score}")
            # Modify features based on credit score thresholds
            predicted_features = modify_features_based_on_credit_score(predicted_features, credit_score)
        except Exception as e:
            logging.warning(f"⚠️ Could not get credit score for feature modification: {str(e)}")
            # Continue without modification if credit score prediction fails
        
        # Prepare response
        response = {
            'customer_id': customer_id,
            'description': 'Home loan explanation',
            'intermediate_representation': convert_numpy_types(intermediate_rep),
            'predicted_features': convert_numpy_types(predicted_features),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in home loans predict endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/credit-card/predict', methods=['POST'])
def predict_credit_card_customer_features():
    """Predict credit card features for a customer ID"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'customer_id' not in data:
            return jsonify({
                'error': 'Missing customer_id in request body'
            }), 400
        
        customer_id = data['customer_id']
        
        # Check if models are loaded
        if any([credit_card_model is None, credit_card_feature_predictor is None, 
                credit_card_feature_names is None, credit_card_df is None]):
            return jsonify({
                'error': 'Credit card models not loaded. Please check server logs.'
            }), 500
        
        # Get customer data and index
        customer_data, customer_idx = get_credit_card_customer_data(customer_id)
        if customer_data is None or customer_idx is None:
            return jsonify({
                'error': f'Customer with ID {customer_id} does not have a credit card'
            }), 404
        
        # Get intermediate representation
        intermediate_rep = get_credit_card_intermediate_representation(customer_idx)
        if intermediate_rep is None:
            return jsonify({
                'error': 'Failed to generate credit card intermediate representation'
            }), 500
        
        # Predict features
        predicted_features = predict_credit_card_features(intermediate_rep)
        if predicted_features is None:
            return jsonify({
                'error': 'Failed to predict credit card features'
            }), 500
        predicted_features = filter_out_credit_score(predicted_features)
        
        # Get credit score for feature modification
        try:
            logging.info(f"🔍 Getting credit score for customer {customer_id} to modify features")
            credit_score_result = credit_score_predictor_instance.predict_credit_score(customer_id)
            credit_score = credit_score_result.get('predicted') if credit_score_result else None
            logging.info(f"📊 Credit score for customer {customer_id}: {credit_score}")
            # Modify features based on credit score thresholds
            predicted_features = modify_features_based_on_credit_score(predicted_features, credit_score)
        except Exception as e:
            logging.warning(f"⚠️ Could not get credit score for feature modification: {str(e)}")
            # Continue without modification if credit score prediction fails
        
        # Prepare response
        response = {
            'customer_id': customer_id,
            'description': 'Credit card explanation',
            'intermediate_representation': convert_numpy_types(intermediate_rep),
            'predicted_features': convert_numpy_types(predicted_features),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in credit card predict endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/customers', methods=['GET'])
def list_customers():
    """List available auto loans customer IDs"""
    global df
    
    if df is None:
        return jsonify({
            'error': 'Auto loans data not loaded'
        }), 500
    
    try:
        # Get first 100 customer IDs
        if 'tax_id' in df.columns:
            customer_ids = df['tax_id'].head(100).tolist()
        else:
            customer_ids = list(range(min(100, len(df))))
        
        return jsonify({
            'customer_ids': customer_ids,
            'total_customers': len(df),
            'sample_size': len(customer_ids)
        })
        
    except Exception as e:
        logging.error(f"Error listing auto loans customers: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/home-loan/customers', methods=['GET'])
def list_home_loans_customers():
    """List available home loans customer IDs"""
    global home_loans_df
    
    if home_loans_df is None:
        return jsonify({
            'error': 'Home loans data not loaded'
        }), 500
    
    try:
        # Get first 100 customer IDs
        if 'tax_id' in home_loans_df.columns:
            customer_ids = home_loans_df['tax_id'].head(100).tolist()
        else:
            customer_ids = list(range(min(100, len(home_loans_df))))
        
        return jsonify({
            'customer_ids': customer_ids,
            'total_customers': len(home_loans_df),
            'sample_size': len(customer_ids)
        })
        
    except Exception as e:
        logging.error(f"Error listing home loans customers: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/credit-card/customers', methods=['GET'])
def list_credit_card_customers():
    """List available credit card customer IDs"""
    global credit_card_df
    
    if credit_card_df is None:
        return jsonify({
            'error': 'Credit card data not loaded'
        }), 500
    
    try:
        # Get first 100 customer IDs
        if 'tax_id' in credit_card_df.columns:
            customer_ids = credit_card_df['tax_id'].head(100).tolist()
        else:
            customer_ids = list(range(min(100, len(credit_card_df))))
        
        return jsonify({
            'customer_ids': customer_ids,
            'total_customers': len(credit_card_df),
            'sample_size': len(customer_ids)
        })
        
    except Exception as e:
        logging.error(f"Error listing credit card customers: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/test_intermediate/<customer_id>', methods=['GET'])
def test_intermediate_representation(customer_id):
    """Test endpoint to compare intermediate representations with dataset"""
    global df, X_scaled_data
    
    if df is None or X_scaled_data is None:
        return jsonify({
            'error': 'Data not loaded'
        }), 500
    
    try:
        # Get customer data and index
        customer_data, customer_idx = get_customer_data(customer_id)
        if customer_data is None or customer_idx is None:
            return jsonify({
                'error': f'Customer with ID {customer_id} does not have an auto loan'
            }), 404
        
        # Get intermediate representation from API
        api_intermediate_rep = get_intermediate_representation(customer_idx)
        
        # Get intermediate representation from dataset (if available)
        dataset_intermediate_rep = None
        try:
            with open(os.path.join(BASE_DIR, 'VFLClientModels', 'models', 'explanations', 'privateexplanations', 'data', 'auto_loans_feature_predictor_dataset_sample.json'), 'r') as f:
                dataset = json.load(f)
            
            # Find customer in dataset
            for sample in dataset:
                if sample['customer_id'] == customer_id:
                    dataset_intermediate_rep = sample['intermediate_representation']
                    break
        except Exception as e:
            logging.warning(f"Could not load dataset for comparison: {str(e)}")
        
        # Compare if both are available
        comparison = None
        if dataset_intermediate_rep and api_intermediate_rep:
            if len(api_intermediate_rep) == len(dataset_intermediate_rep):
                # Check if they're identical (within small tolerance)
                api_array = np.array(api_intermediate_rep)
                dataset_array = np.array(dataset_intermediate_rep)
                diff = np.abs(api_array - dataset_array)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                comparison = {
                    'identical': np.allclose(api_array, dataset_array, rtol=1e-5),
                    'max_difference': float(max_diff),
                    'mean_difference': float(mean_diff),
                    'dimensions_match': len(api_intermediate_rep) == len(dataset_intermediate_rep)
                }
        
        return jsonify({
            'customer_id': customer_id,
            'customer_index': customer_idx,
            'api_intermediate_representation': convert_numpy_types(api_intermediate_rep),
            'dataset_intermediate_representation': convert_numpy_types(dataset_intermediate_rep),
            'comparison': convert_numpy_types(comparison),
            'api_dimension': len(api_intermediate_rep) if api_intermediate_rep else None,
            'dataset_dimension': len(dataset_intermediate_rep) if dataset_intermediate_rep else None
        })
        
    except Exception as e:
        logging.error(f"Error in test endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/digital-savings/predict', methods=['POST'])
def predict_digital_savings_customer_features():
    """Predict digital savings features for a customer ID"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'customer_id' not in data:
            return jsonify({
                'error': 'Missing customer_id in request body'
            }), 400
        
        customer_id = data['customer_id']
        
        # Check if models are loaded
        if any([digital_savings_model is None, digital_savings_feature_predictor is None, 
                digital_savings_feature_names is None, digital_savings_df is None]):
            return jsonify({
                'error': 'Digital savings models not loaded. Please check server logs.'
            }), 500
        
        # Get customer data and index
        customer_data, customer_idx = get_digital_savings_customer_data(customer_id)
        if customer_data is None or customer_idx is None:
            return jsonify({
                'error': f'Customer with ID {customer_id} does not have a digital savings account'
            }), 404
        
        # Get intermediate representation
        intermediate_rep = get_digital_savings_intermediate_representation(customer_idx)
        if intermediate_rep is None:
            return jsonify({
                'error': 'Failed to generate digital savings intermediate representation'
            }), 500
        
        # Predict features
        predicted_features = predict_digital_savings_features(intermediate_rep)
        if predicted_features is None:
            return jsonify({
                'error': 'Failed to predict digital savings features'
            }), 500
        predicted_features = filter_out_credit_score(predicted_features)
        
        # Get credit score for feature modification
        try:
            logging.info(f"🔍 Getting credit score for customer {customer_id} to modify features")
            credit_score_result = credit_score_predictor_instance.predict_credit_score(customer_id)
            credit_score = credit_score_result.get('predicted') if credit_score_result else None
            logging.info(f"📊 Credit score for customer {customer_id}: {credit_score}")
            # Modify features based on credit score thresholds
            predicted_features = modify_features_based_on_credit_score(predicted_features, credit_score)
        except Exception as e:
            logging.warning(f"⚠️ Could not get credit score for feature modification: {str(e)}")
            # Continue without modification if credit score prediction fails
        
        # Prepare response
        response = {
            'customer_id': customer_id,
            'description': 'Digital savings explanation',
            'intermediate_representation': convert_numpy_types(intermediate_rep),
            'predicted_features': convert_numpy_types(predicted_features),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in digital savings predict endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/digital-savings/customers', methods=['GET'])
def list_digital_savings_customers():
    """List available digital savings customer IDs"""
    global digital_savings_df
    
    if digital_savings_df is None:
        return jsonify({
            'error': 'Digital savings data not loaded'
        }), 500
    
    try:
        # Get first 100 customer IDs
        if 'tax_id' in digital_savings_df.columns:
            customer_ids = digital_savings_df['tax_id'].head(100).tolist()
        else:
            customer_ids = list(range(min(100, len(digital_savings_df))))
        
        return jsonify({
            'customer_ids': customer_ids,
            'total_customers': len(digital_savings_df),
            'sample_size': len(customer_ids)
        })
        
    except Exception as e:
        logging.error(f"Error listing digital savings customers: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/test_digital_savings_intermediate', methods=['GET'])
def test_digital_savings_intermediate_representation():
    """Test function to debug intermediate representation mismatch"""
    try:
        global digital_savings_model, digital_savings_feature_names, digital_savings_scaler, digital_savings_df
        
        logger = setup_logging()
        logger.info("Testing digital savings intermediate representation.")
        
        # Find customer 100-10-1003
        customer_data = digital_savings_df[digital_savings_df['tax_id'] == '100-10-1003']
        if customer_data.empty:
            logger.error("Customer 100-10-1003 not found")
            return jsonify({'error': 'Customer 100-10-1003 not found'}), 404
        
        customer_idx = customer_data.index[0]
        logger.info(f"Found customer 100-10-1003 at index {customer_idx}")
        
        # Get the raw features for this customer
        raw_features = digital_savings_df.iloc[customer_idx]
        logger.info(f"Raw features shape: {len(raw_features)}")
        
        # Preprocess exactly like dataset file
        from sklearn.preprocessing import StandardScaler
        
        # Create a fresh scaler and model for testing
        test_scaler = StandardScaler()
        test_model = keras.models.load_model(
            os.path.join(BASE_DIR, 'VFLClientModels', 'saved_models', 'digital_bank_model.keras'), 
            compile=False
        )
        test_penultimate_model = get_penultimate_layer_model(test_model)
        
        # Preprocess features exactly like dataset file
        def preprocess_features_exact(df, feature_names):
            df_proc = df.copy()
            
            # Create derived features exactly as in dataset file
            if 'transaction_volume' in feature_names and 'transaction_volume' not in df_proc.columns:
                if 'avg_monthly_transactions' in df_proc.columns and 'avg_transaction_value' in df_proc.columns:
                    df_proc['transaction_volume'] = df_proc['avg_monthly_transactions'] * df_proc['avg_transaction_value']
                else:
                    df_proc['transaction_volume'] = df_proc['avg_monthly_transactions']
            
            if 'digital_engagement_score' in feature_names and 'digital_engagement_score' not in df_proc.columns:
                if 'digital_engagement_level' in df_proc.columns:
                    mapping = {'Low': 0, 'Medium': 1, 'High': 2}
                    df_proc['digital_engagement_score'] = df_proc['digital_engagement_level'].map(mapping)
                elif 'digital_activity_score' in df_proc.columns:
                    df_proc['digital_engagement_score'] = df_proc['digital_activity_score']
                else:
                    df_proc['digital_engagement_score'] = df_proc['digital_banking_score'] * df_proc['online_transactions_ratio']
            
            # Add other derived features as in dataset file.
            # (simplified for brevity)
            
            return df_proc[feature_names]
        
        # Get the customer's features
        customer_features = digital_savings_df.iloc[customer_idx:customer_idx+1]
        processed_features = preprocess_features_exact(customer_features, digital_savings_feature_names)
        
        # Scale features
        X_scaled = test_scaler.fit_transform(processed_features.values)
        
        # Get intermediate representation
        intermediate_rep = test_penultimate_model.predict(X_scaled, verbose=0)
        result = intermediate_rep.flatten().tolist()
        
        logger.info(f"Test intermediate representation: {result}")
        logger.info(f"Expected: [0.4582552909851074, -0.8135838508605957, -0.5271198153495789, 2.123650312423706, -1.7490688562393188, 2.7965002059936523, -0.4255126118659973, -0.0996234342455864]")
        
        return jsonify({
            'customer_id': '100-10-1003',
            'intermediate_representation': result,
            'expected': [0.4582552909851074, -0.8135838508605957, -0.5271198153495789, 2.123650312423706, -1.7490688562393188, 2.7965002059936523, -0.4255126118659973, -0.0996234342455864]
        })
        
    except Exception as e:
        logger.error(f"Error in test_digital_savings_intermediate: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

def round_insights_numericals(obj):
    """Recursively round numericals in dicts/lists: scores to int, confidence values to 2 decimals, others to 2 decimals."""
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k in ('predicted_credit_score', 'actual_credit_score') and isinstance(v, (int, float)):
                new_obj[k] = int(round(v))
            elif k in ('index', 'direction', 'impact') and isinstance(v, (int, float)):
                new_obj[k] = round(v, 2)
            elif isinstance(v, (int, float)):
                new_obj[k] = round(v, 2)
            else:
                new_obj[k] = round_insights_numericals(v)
        return new_obj
    elif isinstance(obj, list):
        return [round_insights_numericals(x) for x in obj]
    else:
        return obj

@app.route('/credit-score/customer-insights', methods=['POST'])
def get_customer_credit_insights():
    """
    Get customer credit insights based on their ID.
    This endpoint uses the CreditScorePredictor to predict credit scores and related metrics.
    Now also aggregates feature explanations and gets an SLM-generated explanation.
    Input JSON can include:
        - customer_id: str (required)
        - debug: bool (optional)
        - nl_explanation: bool (optional, default True) -- if False, do not call OpenAI for natural language explanations
    """
    logger = setup_logging()
    import time
    t0 = time.time()
    logger.info("[API] /credit-score/customer-insights called")
    try:
        data = request.get_json()
        logger.info(f"[API] Received request: {data}")
        if not data or 'customer_id' not in data:
            logger.warning("[API] Missing customer_id in request body")
            return jsonify({
                'error': 'Missing customer_id in request body'
            }), 400

        customer_id = data['customer_id']
        debug_flag = data.get('debug', False)  # Default to False if not provided
        nl_explanation = data.get('nl_explanation', True)  # Default to True

        logger.info(f"[API] Calling predictor for customer_id: {customer_id}")
        insights = credit_score_predictor_instance.predict_customer_insights(customer_id)
        logger.info(f"[API] Predictor returned for {customer_id}")
        
        # Check if prediction was successful
        if insights.get('status') == 'failed':
            logger.error(f"[API] Prediction failed for {customer_id}: {insights.get('error')}")
            return jsonify({
                'error': insights.get('error', 'Prediction failed'),
                'customer_id': customer_id
            }), 500

        # --- New: Aggregate feature explanations and get SLM explanation ---
        logger.info(f"[API] Aggregating feature explanations for {customer_id}")
        explanations = aggregate_feature_explanations(customer_id)
        explanations_dict = {}
        
        # Get the predicted credit score for tone-based explanations
        predicted_score = insights.get('predicted_credit_score')
        
        if nl_explanation:
            from nlg.slm_phi_client import get_phi_explanation
            for product, feats in explanations.items():
                prompt = format_slm_prompt_single(product, feats, customer_id, predicted_score)
                explanations_dict[product] = get_phi_explanation(prompt, credit_score=predicted_score)
            logger.info(f"[API] SLM explanations generated for {customer_id} with score {predicted_score}")
        else:
            logger.info(f"[API] Skipping SLM explanations for {customer_id} (nl_explanation=False)")
        # --- End New ---

        # Remove debug information based on flag
        if not debug_flag:
            # Remove actual score and error information for production
            insights.pop('actual_credit_score', None)
            insights.pop('prediction_error', None)
            insights.pop('score_adjusted', None)

        # Convert numpy types to native Python types for JSON serialization
        insights = convert_numpy_types(insights)

        # Round numericals as requested
        insights = round_insights_numericals(insights)

        # Add SLM explanations and feature explanations to the response
        if nl_explanation:
            insights['explanations'] = explanations_dict
        insights['feature_explanations'] = explanations

        logger.info(f"[API] Returning response for {customer_id} in {time.time()-t0:.2f}s")
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"[API] Error in credit-score/customer-insights endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

def modify_features_based_on_credit_score(features, credit_score):
    """
    Modify predicted features based on credit score thresholds:
    - If score < 400: Make all features negative even if positive
    - If score > 700: Make all features positive even if negative
    - For scores 400-700: Keep original values (no modification)
    """
    if not isinstance(features, list) or credit_score is None:
        return features
    
    # Log the modification
    if credit_score < 400:
        logging.info(f"🎯 Credit score {credit_score} < 400: Making all features negative")
    elif credit_score > 700:
        logging.info(f"🎯 Credit score {credit_score} > 700: Making all features positive")
    else:
        logging.info(f"🎯 Credit score {credit_score} between 400-700: No feature modification needed")
    
    modified_features = []
    for feature in features:
        modified_feature = feature.copy()
        
        if credit_score < 400:
            # Make all features negative
            if 'direction' in modified_feature:
                modified_feature['direction'] = 'Negative'
            if 'impact' in modified_feature and isinstance(modified_feature['impact'], (int, float)):
                modified_feature['impact'] = abs(modified_feature['impact']) * -1
        elif credit_score > 700:
            # Make all features positive
            if 'direction' in modified_feature:
                modified_feature['direction'] = 'Positive'
            if 'impact' in modified_feature and isinstance(modified_feature['impact'], (int, float)):
                modified_feature['impact'] = abs(modified_feature['impact'])
        # For scores 400-700: Keep original values (no modification)
        
        modified_features.append(modified_feature)
    
    return modified_features

if __name__ == '__main__':
    logger = setup_logging()
    
    # Load models and data
    if not load_models_and_data():
        logger.error("Failed to load models and data. Exiting.")
        sys.exit(1)
    
    logger.info("Starting Auto Loans Feature Prediction API.")
    logger.info("Available endpoints:")
    logger.info("  - GET  /health     - Health check")
    logger.info("  - POST /auto-loan/predict    - Predict features for customer ID")
    logger.info("  - GET  /customers  - List available customer IDs")
    logger.info("  - GET  /test_intermediate/<customer_id> - Test intermediate representation comparison")
    logger.info("  - POST /credit-score/customer-insights - Get customer credit insights")
    
    # Run the app
    app.run(host='0.0.0.0', port=5001, debug=False) 