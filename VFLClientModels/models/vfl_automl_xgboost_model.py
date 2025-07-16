import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import keras
from keras import layers, models
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import keras_tuner as kt
from datetime import datetime
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
import glob
import joblib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
from keras import initializers

# ============================================================================
# VFL AutoML Model - Four Bank System Integration with XGBoost Credit Card
# 
# This system now integrates FOUR banks:
# 1. Auto Loans Bank (neural network regression model) - 16D
# 2. Digital Banking Bank (neural network classification model) - 8D
# 3. Home Loans Bank (neural network regression model) - 16D
# 4. Credit Card Bank (XGBoost classification model - NEW!) - 8D
#
# Each bank contributes representations from their models:
# - Neural networks: penultimate layer features
# - XGBoost: leaf indices converted to 8D embeddings
# 
# NEW: XGBoost Credit Card Integration
# - Uses actual trained XGBoost model for credit card predictions
# - Converts tree leaf indices to 8D representations using PCA
# - Maintains compatibility with existing VFL architecture
# ============================================================================

def setup_logging():
    """Setup comprehensive logging for VFL AutoML training"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger
    logger = logging.getLogger('VFL_AutoML_XGBoost')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Comprehensive file handler for all logs (training, debugging, performance)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = RotatingFileHandler(
        f'VFLClientModels/logs/vfl_automl_xgboost_{timestamp}.log',
        maxBytes=15*1024*1024,  # 15MB per file (increased for comprehensive logging)
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # TensorFlow logging configuration
    tf.get_logger().setLevel('ERROR')  # Reduce TF verbosity
    
    logger.info("=" * 80)
    logger.info("VFL AutoML XGBoost Credit Card Comprehensive Logging System Initialized")
    logger.info(f"Log file: logs/vfl_automl_xgboost_{timestamp}.log")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return logger

# Initialize logging
logger = setup_logging()

# ============================================================================
# AUTOML CONFIGURATION - MODIFY THESE VALUES FOR YOUR SEARCH SPACE
# ============================================================================

# Training Phase Configuration
ENABLE_PHASE_2 = True           # Set to True for full training, False for AutoML search only
AUTOML_SAMPLE_SIZE = 1000        # Number of customers for AutoML search (increased for stable stratification)
FINAL_SAMPLE_SIZE = 200000      # Number of customers for final model training (larger for accuracy)
RANDOM_SEED = 42                 # For reproducible results

# AutoML Search Configuration
MAX_TRIALS = 1                  # Number of different architectures to try (increased from 3)
EXECUTIONS_PER_TRIAL = 1         # Number of times to train each architecture
EPOCHS_PER_TRIAL = 1           # Max epochs for each trial (increased from 20)
FINAL_EPOCHS = 10                   # Epochs for final model training with full data (increased from 30)
SEARCH_OBJECTIVE = 'val_mae'     # Objective to optimize ('val_loss', 'val_mae', etc.)

# Search Space Ranges
MIN_LAYERS = 2      # Minimum number of hidden layers
MAX_LAYERS = 32     # Maximum number of hidden layers (reduced for faster search)
MIN_UNITS = 8      # Minimum units per layer
MAX_UNITS = 4096     # Maximum units per layer (reduced for faster search)

# Confidence Score Configuration
ENABLE_CONFIDENCE_SCORES = True      # Enable confidence score calculation
MC_DROPOUT_SAMPLES = 30              # Number of Monte Carlo dropout samples for uncertainty
CONFIDENCE_INTERVALS = [68, 95]      # Confidence intervals to calculate (68% = 1σ, 95% = 2σ)
MIN_CONFIDENCE_THRESHOLD = 0.7       # Minimum confidence threshold for reliable predictions

# XGBoost Credit Card Configuration
XGBOOST_OUTPUT_DIM = 12             # Target output dimension for XGBoost representations
XGBOOST_PCA_RANDOM_STATE = 42       # Random state for PCA dimensionality reduction

# Parallel Processing Configuration
ENABLE_PARALLEL_PROCESSING = True    # Enable multithreaded representation extraction
MAX_WORKERS = 4                      # Number of parallel workers for representation extraction
CHUNK_SIZE_MULTIPLIER = 2            # Multiplier for chunk size calculation

# Anti-Overfitting Configuration
L2_REGULARIZATION = 1e-5            # L2 regularization strength (reduced from 1e-4)
GRADIENT_CLIP_NORM = 1.0            # Gradient clipping norm
EARLY_STOPPING_PATIENCE = 10         # Increased patience for early stopping
REDUCE_LR_PATIENCE = 8              # Patience for learning rate reduction
MIN_LEARNING_RATE = 1e-6            # Minimum learning rate

# ============================================================================ 

def get_penultimate_layer_model(model):
    """Create a model that outputs the penultimate layer activations"""
    # Find the penultimate layer (the one before the final output layer)
    for i, layer in enumerate(reversed(model.layers)):
        if hasattr(layer, 'activation') and layer.activation is not None:
            if i > 0:  # Skip the output layer
                penultimate_layer = model.layers[-(i+1)]
                break
    else:
        # Fallback to second-to-last layer
        penultimate_layer = model.layers[-2]
    
    # Create new model that outputs the penultimate layer
    feature_extractor = models.Model(
        inputs=model.inputs,
        outputs=penultimate_layer.output,
        name=f"{model.name}_feature_extractor"
    )
    
    print(f"Feature extractor input shape: {feature_extractor.input_shape}")
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")
    
    return feature_extractor

def load_xgboost_credit_card_model():
    """Load the trained XGBoost credit card model"""
    logger.info("🔄 Loading XGBoost credit card model...")
    
    try:
        # Load XGBoost model components
        model_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_independent.pkl'
        scaler_path = 'VFLClientModels/models/saved_models/credit_card_scaler.pkl'
        feature_names_path = 'VFLClientModels/models/saved_models/credit_card_feature_names.npy'
        
        logger.debug(f"Loading XGBoost model from: {model_path}")
        model_data = joblib.load(model_path)
        
        # Handle different model file formats
        if isinstance(model_data, dict):
            # Model is stored as dictionary with components
            logger.info(f"   📦 Model file contains dictionary with keys: {list(model_data.keys())}")
            
            # Extract components from dictionary
            if 'classifier' in model_data and 'scaler' in model_data:
                classifier = model_data['classifier']
                scaler = model_data['scaler']
                feature_names = model_data.get('feature_names', None)
                label_encoder = model_data.get('label_encoder', None)
                
                logger.info(f"   ✅ Loaded XGBoost components from dictionary")
                logger.info(f"   🌳 Model type: {type(classifier).__name__}")
                logger.info(f"   📊 Scaler type: {type(scaler).__name__}")
                logger.info(f"   📝 Feature names available: {feature_names is not None}")
                
            else:
                logger.warning("   ⚠️  Dictionary format not recognized, using direct model")
                classifier = model_data
                scaler = None
                feature_names = None
                label_encoder = None
        else:
            # Direct model object
            classifier = model_data
            scaler = None
            feature_names = None
            label_encoder = None
            logger.info(f"   ✅ Loaded direct XGBoost model: {type(classifier).__name__}")
        
        # Try to load separate scaler if not included in model
        if scaler is None:
            try:
                scaler = joblib.load(scaler_path)
                logger.info(f"   ✅ Loaded separate scaler: {type(scaler).__name__}")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not load separate scaler: {e}")
                scaler = StandardScaler()  # Create dummy scaler
        
        # Try to load feature names
        if feature_names is None:
            try:
                feature_names = np.load(feature_names_path, allow_pickle=True)
                logger.info(f"   ✅ Loaded feature names: {len(feature_names)} features")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not load feature names: {e}")
                feature_names = None
        
        # Wrap in a consistent interface
        class XGBoostModelWrapper:
            def __init__(self, classifier, scaler, feature_names=None, label_encoder=None):
                self.classifier = classifier
                self.scaler = scaler
                self.feature_names = feature_names
                self.label_encoder = label_encoder
                
            def get_model_info(self):
                return {
                    'model_type': 'XGBoost',
                    'n_estimators': getattr(self.classifier, 'n_estimators', 'Unknown'),
                    'max_depth': getattr(self.classifier, 'max_depth', 'Unknown'),
                    'learning_rate': getattr(self.classifier, 'learning_rate', 'Unknown')
                }
        
        wrapped_model = XGBoostModelWrapper(classifier, scaler, feature_names, label_encoder)
        
        logger.info("✅ XGBoost credit card model loaded successfully")
        logger.info(f"   Model info: {wrapped_model.get_model_info()}")
        
        return wrapped_model
        
    except Exception as e:
        logger.error(f"❌ Error loading XGBoost credit card model: {str(e)}")
        logger.error("Please ensure the model exists at:")
        logger.error("- VFLClientModels/models/saved_models/credit_card_xgboost_independent.pkl")
        logger.error("- VFLClientModels/models/saved_models/credit_card_scaler.pkl")
        logger.error("- VFLClientModels/models/saved_models/credit_card_feature_names.npy")
        raise

def extract_xgboost_representations(model, customer_data, target_dim=XGBOOST_OUTPUT_DIM):
    """
    Extract fixed-dimension representations from XGBoost model using leaf indices.
    During inference, always loads and uses the saved PCA for dimensionality reduction.
    Args:
        model: XGBoostModelWrapper with classifier and scaler
        customer_data: DataFrame with customer features
        target_dim: Target dimension for output representations (default: 8)
    Returns:
        numpy array of shape (n_customers, target_dim)
    """
    logger.info(f"🌳 Extracting XGBoost representations (target: {target_dim}D)...")

    # Load the exact feature names used during XGBoost training
    feature_names_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_feature_names.npy'
    features_to_use = list(np.load(feature_names_path, allow_pickle=True))

    logger.info(f"   📊 Processing {len(customer_data)} customers with {len(features_to_use)} features")

    # Only create derived features if they are in the expected feature list
    if 'credit_capacity_ratio' in features_to_use:
        customer_data['credit_capacity_ratio'] = customer_data['credit_card_limit'] / customer_data['total_credit_limit'].replace(0, 1)
    if 'income_to_limit_ratio' in features_to_use:
        customer_data['income_to_limit_ratio'] = customer_data['annual_income'] / customer_data['credit_card_limit'].replace(0, 1)
    if 'debt_service_ratio' in features_to_use:
        customer_data['debt_service_ratio'] = (customer_data['current_debt'] * 0.03) / (customer_data['annual_income'] / 12)
    if 'risk_adjusted_income' in features_to_use:
        customer_data['risk_adjusted_income'] = customer_data['annual_income'] * (customer_data['risk_score'] / 100)
    if 'credit_to_income_ratio' in features_to_use:
        customer_data['credit_to_income_ratio'] = customer_data['credit_card_limit'] / customer_data['annual_income'].replace(0, 1)

    # Select only the required features for this model, in the correct order
    feature_data = customer_data[features_to_use].copy()

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

    # Convert leaf indices to target dimensions
    if leaf_indices.shape[1] == target_dim:
        # Perfect match, just normalize
        representations = leaf_indices.astype(np.float32)
        logger.info(f"   ✅ Perfect dimension match: {representations.shape}")
    elif leaf_indices.shape[1] > target_dim:
        # Always use the saved PCA for inference
        pca_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_pca.pkl'
        import joblib
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
            representations = pca.transform(leaf_indices.astype(np.float32))
            logger.info(f"   ✅ Used saved PCA for inference: {representations.shape}")
        else:
            logger.warning(f"   ⚠️  PCA file not found at {pca_path}, falling back to truncation.")
            representations = leaf_indices[:, :target_dim].astype(np.float32)
    else:
        # Pad with zeros or use feature engineering to expand
        logger.info(f"   🔄 Expanding {leaf_indices.shape[1]}D → {target_dim}D using padding...")
        representations = np.zeros((leaf_indices.shape[0], target_dim), dtype=np.float32)
        representations[:, :leaf_indices.shape[1]] = leaf_indices.astype(np.float32)
        logger.info(f"   ✅ Zero-padded to target dimension: {representations.shape}")

    # Normalize representations to [0, 1] range for consistency with neural networks
    if representations.max() > representations.min():
        representations = (representations - representations.min()) / (representations.max() - representations.min())

    logger.info(f"   🎯 Final XGBoost representations: {representations.shape}")
    logger.info(f"   📊 Stats: min={representations.min():.3f}, max={representations.max():.3f}, mean={representations.mean():.3f}")

    return representations

def extract_bank_representations_parallel(bank_df, features, scaler, extractor, customer_df, service_column, bank_name, max_workers=4):
    """Extract representations for customers with service at this bank using parallel processing"""
    output_size = extractor.output_shape[-1]
    all_representations = np.zeros((len(customer_df), output_size))
    
    if len(bank_df) == 0:
        logger.info(f"   ⚠️  {bank_name} Bank: No data available")
        service_mask = customer_df[service_column].values.astype(np.float32).reshape(-
                                                                                     1)
        return all_representations, service_mask, scaler
    
    logger.info(f"🔄 Processing {bank_name} Bank (Parallel):")
    logger.info(f"   Dataset size: {len(bank_df):,}")
    logger.info(f"   Feature count: {len(features)}")
    logger.info(f"   Output representation size: {output_size}")
    logger.info(f"   Max workers: {max_workers}")
    
    # Select only the required features for this model
    feature_data = bank_df[features].copy()
    
    # Handle infinite and missing values in feature data only
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values with median for numeric columns only
    numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
    feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
    
    # For any remaining non-numeric columns, forward fill or use mode
    non_numeric_cols = feature_data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        logger.debug(f"   Handling {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
        for col in non_numeric_cols:
            feature_data[col] = feature_data[col].fillna(feature_data[col].mode()[0] if not feature_data[col].mode().empty else 0)
    
    # Scale features
    X_scaled = scaler.fit_transform(feature_data)
    
    # Split data into chunks for parallel processing
    chunk_size = max(1, len(X_scaled) // (max_workers * 4))  # Consistent with neural network banks
    chunks = [X_scaled[i:i + chunk_size] for i in range(0, len(X_scaled), chunk_size)]
    
    logger.info(f"   Split into {len(chunks)} chunks of ~{chunk_size} samples each")
    
    # Progress tracking variables
    total_records = len(X_scaled)
    processed_records = 0
    progress_lock = threading.Lock()
    last_reported_percentage = 0
    
    def update_progress(records_in_chunk):
        nonlocal processed_records, last_reported_percentage
        with progress_lock:
            processed_records += records_in_chunk
            percentage = min(100, (processed_records * 100) / total_records)
            
            # Report progress every 1% or when a chunk completes
            if percentage - last_reported_percentage >= 1 or percentage >= 100:
                logger.info(f"   Progress: {processed_records:,}/{total_records:,} records ({percentage:.1f}%)")
                last_reported_percentage = percentage
    
    # Function to process a single chunk
    def process_chunk(chunk_data, chunk_idx):
        try:
            start_time = time.time()
            # Log NaN/infinite values in chunk_data
            if np.isnan(chunk_data).any() or np.isinf(chunk_data).any():
                logger.error(f"   Chunk {chunk_idx + 1}: NaN or infinite values detected in input features!")
                nan_rows = np.where(np.isnan(chunk_data).any(axis=1))[0]
                inf_rows = np.where(np.isinf(chunk_data).any(axis=1))[0]
                logger.error(f"   Chunk {chunk_idx + 1}: Rows with NaNs: {nan_rows}")
                logger.error(f"   Chunk {chunk_idx + 1}: Rows with infs: {inf_rows}")
            representations = extractor.predict(chunk_data, verbose=2)
            end_time = time.time()
            update_progress(len(chunk_data))  # Pass actual chunk size
            logger.debug(f"   Chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_data)} samples in {end_time - start_time:.2f}s")
            return chunk_idx, representations
        except Exception as e:
            # Log the indices and customer IDs of the failed chunk
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(bank_df))
            failed_customer_ids = bank_df.iloc[chunk_start:chunk_end]['tax_id'].tolist()
            logger.error(f"   Error processing chunk {chunk_idx + 1}: {e}")
            logger.error(f"   Failed chunk indices: {chunk_start} to {chunk_end - 1}")
            logger.error(f"   Failed customer IDs: {failed_customer_ids}")
            return chunk_idx, None
    
    # Process chunks in parallel
    start_time = time.time()
    all_representations_list = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(process_chunk, chunk, i): i for i, chunk in enumerate(chunks)}
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx, representations = future.result()
            if representations is not None:
                all_representations_list[chunk_idx] = representations
    
    # Combine all chunk results (preserve order)
    all_representations_combined = np.vstack([repr for repr in all_representations_list if repr is not None])
    end_time = time.time()
    
    logger.info(f"   ✅ Parallel processing completed in {end_time - start_time:.2f}s")
    logger.info(f"   Combined representations shape: {all_representations_combined.shape}")
    
    # Map representations to correct customer positions (optimized with lookup)
    logger.info(f"   🔄 Mapping representations to customer positions...")
    mapping_start = time.time()
    
    # Create lookup dictionary for faster customer ID to index mapping
    customer_id_to_index = {tax_id: idx for idx, tax_id in enumerate(customer_df['tax_id'])}
    bank_customer_ids = bank_df['tax_id'].values
    
    # Verify data consistency
    if len(bank_customer_ids) != len(all_representations_combined):
        logger.error(f"   ❌ Data mismatch: {len(bank_customer_ids)} customer IDs vs {len(all_representations_combined)} representations")
        raise ValueError("Customer ID count doesn't match representation count")
    
    # Progress tracking for mapping
    total_mappings = len(bank_customer_ids)
    mapping_progress_step = max(1, total_mappings // 100)  # Show progress every 1%
    
    for i, customer_id in enumerate(bank_customer_ids):
        customer_idx = customer_id_to_index[customer_id]
        all_representations[customer_idx] = all_representations_combined[i]
        
        # Show mapping progress
        if (i + 1) % mapping_progress_step == 0 or i == total_mappings - 1:
            mapping_percentage = ((i + 1) * 100) / total_mappings
            logger.info(f"   Mapping Progress: {i + 1:,}/{total_mappings:,} customers ({mapping_percentage:.1f}%)")
    
    mapping_end = time.time()
    logger.info(f"   ✅ Customer mapping completed in {mapping_end - mapping_start:.2f}s")
    
    # Verify final representation count
    non_zero_repr = np.sum(np.any(all_representations != 0, axis=1))
    logger.info(f"   📊 Final verification: {non_zero_repr:,} customers have representations (expected: {len(bank_df):,})")
    
    # Create service availability mask
    service_mask = customer_df[service_column].values.astype(np.float32).reshape(-1, 1)
    
    return all_representations, service_mask, scaler

def extract_xgboost_representations_parallel(model, customer_data, target_dim=XGBOOST_OUTPUT_DIM, max_workers=4):
    """
    Extract fixed-dimension representations from XGBoost model using leaf indices with parallel processing
    
    Args:
        model: XGBoostModelWrapper with classifier and scaler
        customer_data: DataFrame with customer features
        target_dim: Target dimension for output representations (default: 8)
        max_workers: Number of parallel workers
    
    Returns:
        numpy array of shape (n_customers, target_dim)
    """
    logger.info(f"🌳 Extracting XGBoost representations (Parallel, target: {target_dim}D)...")
    logger.info(f"   Dataset size: {len(customer_data):,}")
    logger.info(f"   Max workers: {max_workers}")
    
    # Load the exact feature names used during XGBoost training
    feature_names_path = os.path.join('VFLClientModels', 'models','saved_models', 'credit_card_xgboost_feature_names.npy')
    if os.path.exists(feature_names_path):
        features_to_use = list(np.load(feature_names_path, allow_pickle=True))
        logger.info(f"   📋 Loaded {len(features_to_use)} features from saved file")
    else:
        logger.error(f"   ❌ Feature names file not found: {feature_names_path}")
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    logger.info(f"   📊 Processing {len(customer_data)} customers with {len(features_to_use)} features")
    
    # Create derived features exactly as in credit_card_xgboost_model.py
    customer_data = customer_data.copy()
    
    # Only create derived features if they are in the expected feature list
    if 'credit_capacity_ratio' in features_to_use:
        customer_data['credit_capacity_ratio'] = customer_data['total_credit_limit'] / customer_data['total_credit_limit'].replace(0, 1)
    if 'income_to_limit_ratio' in features_to_use:
        customer_data['income_to_limit_ratio'] = customer_data['annual_income'] / customer_data['total_credit_limit'].replace(0, 1)
    if 'debt_service_ratio' in features_to_use:
        customer_data['debt_service_ratio'] = (customer_data['current_debt'] * 0.03) / (customer_data['annual_income'] / 12)
    if 'risk_adjusted_income' in features_to_use:
        customer_data['risk_adjusted_income'] = customer_data['annual_income'] * (1 - customer_data['debt_to_income_ratio'])
    
    # Check which features are available in the DataFrame
    available_features = [f for f in features_to_use if f in customer_data.columns]
    missing_features = [f for f in features_to_use if f not in customer_data.columns]
    extra_features = [f for f in customer_data.columns if f not in features_to_use]
    
    if missing_features:
        logger.error(f"   ❌ Missing required features in DataFrame: {missing_features}")
        raise ValueError(f"Missing required features in DataFrame: {missing_features}")
    if extra_features:
        logger.info(f"   ℹ️ Extra features in DataFrame (not used by model): {extra_features}")
    
    logger.info(f"   📝 Using features (in order): {available_features}")
    feature_data = customer_data[available_features].copy()
    
    # Handle infinite and missing values
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
    feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
    
    # Scale features
    X_scaled = model.scaler.transform(feature_data)
    logger.info(f"   ✅ Scaled features: {X_scaled.shape}")
    
    # Split data into chunks for parallel processing
    chunk_size = max(1, len(X_scaled) // (max_workers * 4))  # Consistent with neural network banks
    chunks = [X_scaled[i:i + chunk_size] for i in range(0, len(X_scaled), chunk_size)]
    
    logger.info(f"   Split into {len(chunks)} chunks of ~{chunk_size} samples each")
    
    # Progress tracking variables
    total_records = len(X_scaled)
    processed_records = 0
    progress_lock = threading.Lock()
    last_reported_percentage = 0
    
    def update_progress(records_in_chunk):
        nonlocal processed_records, last_reported_percentage
        with progress_lock:
            processed_records += records_in_chunk
            percentage = min(100, (processed_records * 100) / total_records)
            
            # Report progress every 1% or when a chunk completes
            if percentage - last_reported_percentage >= 1 or percentage >= 100:
                logger.info(f"   Progress: {processed_records:,}/{total_records:,} records ({percentage:.1f}%)")
                last_reported_percentage = percentage
    
    # Function to process a single chunk
    def process_xgboost_chunk(chunk_data, chunk_idx):
        try:
            start_time = time.time()
            # Get leaf indices from XGBoost for this chunk
            leaf_indices = model.classifier.apply(chunk_data)
            end_time = time.time()
            update_progress(len(chunk_data))  # Pass actual chunk size
            logger.debug(f"   Chunk {chunk_idx + 1}/{len(chunks)}: {len(chunk_data)} samples in {end_time - start_time:.2f}s")
            return chunk_idx, leaf_indices
        except Exception as e:
            logger.error(f"   Error processing XGBoost chunk {chunk_idx + 1}: {e}")
            return chunk_idx, None
    
    # Process chunks in parallel
    start_time = time.time()
    all_leaf_indices_list = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {executor.submit(process_xgboost_chunk, chunk, i): i for i, chunk in enumerate(chunks)}
        
        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx, leaf_indices = future.result()
            if leaf_indices is not None:
                all_leaf_indices_list[chunk_idx] = leaf_indices
    
    # Combine all chunk results (preserve order)
    leaf_indices = np.vstack([indices for indices in all_leaf_indices_list if indices is not None])
    end_time = time.time()
    
    logger.info(f"   ✅ Parallel XGBoost processing completed in {end_time - start_time:.2f}s")
    logger.info(f"   🌳 XGBoost leaf indices shape: {leaf_indices.shape}")
    
    # Convert leaf indices to target dimensions
    if leaf_indices.shape[1] == target_dim:
        # Perfect match, just normalize
        representations = leaf_indices.astype(np.float32)
        logger.info(f"   ✅ Perfect dimension match: {representations.shape}")
    elif leaf_indices.shape[1] > target_dim:
        # Always use the saved PCA for inference
        pca_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_pca.pkl'
        import joblib
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
            representations = pca.transform(leaf_indices.astype(np.float32))
            logger.info(f"   ✅ Used saved PCA for inference: {representations.shape}")
        else:
            logger.warning(f"   ⚠️  PCA file not found at {pca_path}, falling back to truncation.")
            representations = leaf_indices[:, :target_dim].astype(np.float32)
    else:
        # Pad with zeros or use feature engineering to expand
        logger.info(f"   🔄 Expanding {leaf_indices.shape[1]}D → {target_dim}D using padding...")
        representations = np.zeros((leaf_indices.shape[0], target_dim), dtype=np.float32)
        representations[:, :leaf_indices.shape[1]] = leaf_indices.astype(np.float32)
        logger.info(f"   ✅ Zero-padded to target dimension: {representations.shape}")
    
    # Normalize representations to [0, 1] range for consistency with neural networks
    if representations.max() > representations.min():
        representations = (representations - representations.min()) / (representations.max() - representations.min())
    
    logger.info(f"   🎯 Final XGBoost representations: {representations.shape}")
    logger.info(f"   📊 Stats: min={representations.min():.3f}, max={representations.max():.3f}, mean={representations.mean():.3f}")
    
    return representations

def load_client_models():
    """Load the trained client models - MODIFIED FOR XGBOOST CREDIT CARD"""
    logger.info("🔄 Loading client models (3 Neural Networks + 1 XGBoost)...")
    
    try:
        logger.debug("Loading auto loans model...")
        auto_loans_model = load_model('VFLClientModels/models/saved_models/auto_loans_model.keras', compile=False)
        logger.debug(f"Auto loans model loaded: input shape {auto_loans_model.input_shape}")
        
        logger.debug("Loading digital bank model...")
        digital_bank_model = load_model('VFLClientModels/models/saved_models/digital_bank_model.keras', compile=False)
        logger.debug(f"Digital bank model loaded: input shape {digital_bank_model.input_shape}")
        
        logger.debug("Loading home loans model...")
        home_loans_model = load_model('VFLClientModels/models/saved_models/home_loans_model.keras', compile=False)
        logger.debug(f"Home loans model loaded: input shape {home_loans_model.input_shape}")
        
        logger.debug("Loading XGBoost credit card model...")
        credit_card_model = load_xgboost_credit_card_model()
        logger.debug(f"XGBoost credit card model loaded: {credit_card_model.get_model_info()}")
        
        logger.info("✅ All client models loaded successfully")
        logger.info(f"Auto model input shape: {auto_loans_model.input_shape}")
        logger.info(f"Digital model input shape: {digital_bank_model.input_shape}")
        logger.info(f"Home loans model input shape: {home_loans_model.input_shape}")
        logger.info(f"Credit card model type: XGBoost ({credit_card_model.get_model_info()['model_type']})")
        
        return auto_loans_model, digital_bank_model, home_loans_model, credit_card_model
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error("Please ensure all models exist at:")
        logger.error("- saved_models/auto_loans_model.keras")
        logger.error("- saved_models/digital_bank_model.keras") 
        logger.error("- saved_models/home_loans_model.keras")
        logger.error("- saved_models/credit_card_xgboost_independent.pkl")
        raise

def load_and_preprocess_data(sample_size=None):
    """Load and preprocess data from all sources, handling XGBoost for credit card bank with optional sampling"""
    logger.info(f"🔄 Loading and preprocessing data (XGBoost Credit Card Version)...")
    logger.info(f"Sample size: {sample_size:,} customers" if sample_size else "Using full dataset")
    
    use_sampling = sample_size is not None
    
    # Set random seed for reproducible sampling
    np.random.seed(RANDOM_SEED)
    logger.debug(f"Random seed set to: {RANDOM_SEED}")
    
    # Load datasets - THREE NEURAL NETWORKS + ONE XGBOOST
    logger.debug("Loading bank datasets...")
    auto_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/auto_loans_bank.csv')
    digital_bank_df = pd.read_csv('VFLClientModels/dataset/data/banks/digital_savings_bank.csv')
    home_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/home_loans_bank.csv')
    credit_card_df = pd.read_csv('VFLClientModels/dataset/data/banks/credit_card_bank.csv')
    master_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    
    logger.info("📊 Original Dataset Statistics:")
    logger.info(f"Auto Loans customers: {len(auto_loans_df):,}")
    logger.info(f"Digital Bank customers: {len(digital_bank_df):,}")
    logger.info(f"Home Loans customers: {len(home_loans_df):,}")
    logger.info(f"Credit Card customers: {len(credit_card_df):,}")
    logger.info(f"Master dataset customers: {len(master_df):,}")
    
    # Apply sampling if configured (same logic as original)
    if use_sampling and sample_size < len(master_df):
        logger.info(f"🔀 Applying stratified sampling:")
        logger.info(f"  Target sample size: {sample_size:,} customers")
        logger.info(f"  Random seed: {RANDOM_SEED}")
        
        # Calculate original service combination ratios and maintain them - WITH FOUR BANKS
        all_customers = set(master_df['tax_id'])
        auto_customers = set(auto_loans_df['tax_id'])
        digital_customers = set(digital_bank_df['tax_id'])
        home_customers = set(home_loans_df['tax_id'])
        credit_card_customers = set(credit_card_df['tax_id'])
        
        # Create temporary alignment for ratio calculation
        temp_df = pd.DataFrame({'tax_id': list(all_customers)})
        temp_df['has_auto'] = temp_df['tax_id'].isin(auto_customers)
        temp_df['has_digital'] = temp_df['tax_id'].isin(digital_customers)
        temp_df['has_home'] = temp_df['tax_id'].isin(home_customers)
        temp_df['has_credit_card'] = temp_df['tax_id'].isin(credit_card_customers)
        temp_df['combo'] = (temp_df['has_auto'].astype(str) + 
                           temp_df['has_digital'].astype(str) + 
                           temp_df['has_home'].astype(str) +
                           temp_df['has_credit_card'].astype(str))
        
        # Calculate target counts for each combination
        combo_ratios = temp_df['combo'].value_counts(normalize=True)
        target_counts = (combo_ratios * sample_size).round().astype(int)
        
        logger.debug("Service combination sampling targets:")
        for combo, count in target_counts.items():
            auto_flag = "Auto" if combo[0] == '1' else ""
            digital_flag = "Digital" if combo[1] == '1' else ""
            home_flag = "Home" if combo[2] == '1' else ""
            credit_flag = "Credit" if combo[3] == '1' else ""
            services = "+".join(filter(None, [auto_flag, digital_flag, home_flag, credit_flag])) or "None"
            logger.debug(f"  {services}: {count:,} customers ({combo_ratios[combo]*100:.1f}%)")
        
        # Ensure we don't exceed sample_size due to rounding
        if target_counts.sum() > sample_size:
            largest_combo = target_counts.idxmax()
            target_counts[largest_combo] -= (target_counts.sum() - sample_size)
            logger.debug(f"Adjusted {largest_combo} count to maintain sample size")
        elif target_counts.sum() < sample_size:
            largest_combo = target_counts.idxmax()
            target_counts[largest_combo] += (sample_size - target_counts.sum())
            logger.debug(f"Increased {largest_combo} count to reach sample size")
        
        # Sample from each combination
        logger.debug("Sampling customers from each service combination...")
        sampled_customers = []
        merged_temp = master_df.merge(temp_df, on='tax_id', how='left')
        
        for combo, target_count in target_counts.items():
            combo_customers = merged_temp[merged_temp['combo'] == combo]
            if len(combo_customers) >= target_count:
                sampled = combo_customers.sample(n=target_count, random_state=RANDOM_SEED)
            else:
                sampled = combo_customers
                logger.warning(f"Service combo {combo}: only {len(combo_customers)} available, need {target_count}")
            sampled_customers.append(sampled)
        
        # Combine all sampled customers
        sampled_master = pd.concat(sampled_customers, ignore_index=True)
        
        # Filter bank datasets to match sampled customers - INCLUDING CREDIT CARD
        sampled_customer_ids = set(sampled_master['tax_id'])
        auto_loans_df = auto_loans_df[auto_loans_df['tax_id'].isin(sampled_customer_ids)]
        digital_bank_df = digital_bank_df[digital_bank_df['tax_id'].isin(sampled_customer_ids)]
        home_loans_df = home_loans_df[home_loans_df['tax_id'].isin(sampled_customer_ids)]
        credit_card_df = credit_card_df[credit_card_df['tax_id'].isin(sampled_customer_ids)]
        master_df = sampled_master
        
        logger.info(f"📊 Sampled Dataset Statistics:")
        logger.info(f"Auto Loans customers: {len(auto_loans_df):,}")
        logger.info(f"Digital Bank customers: {len(digital_bank_df):,}")
        logger.info(f"Home Loans customers: {len(home_loans_df):,}")
        logger.info(f"Credit Card customers: {len(credit_card_df):,}")
        logger.info(f"Master dataset customers: {len(master_df):,}")
    
    else:
        if use_sampling:
            logger.info(f"Sampling disabled: sample_size ({sample_size:,}) >= dataset size ({len(master_df):,})")
        else:
            logger.info(f"Using full dataset (sampling disabled)")
    
    # Get all unique customers from master dataset - WITH FOUR BANKS
    all_customers = set(master_df['tax_id'])
    auto_customers = set(auto_loans_df['tax_id'])
    digital_customers = set(digital_bank_df['tax_id'])
    home_customers = set(home_loans_df['tax_id'])
    credit_card_customers = set(credit_card_df['tax_id'])
    
    logger.info(f"📈 Final Alignment Statistics (3 NN + 1 XGBoost):")
    logger.info(f"Total customers in master: {len(all_customers):,}")
    logger.info(f"Customers with auto loans: {len(auto_customers):,}")
    logger.info(f"Customers with digital banking: {len(digital_customers):,}")
    logger.info(f"Customers with home loans: {len(home_customers):,}")
    logger.info(f"Customers with credit cards: {len(credit_card_customers):,}")
    logger.info(f"Auto & Digital: {len(auto_customers.intersection(digital_customers)):,}")
    logger.info(f"Auto & Home: {len(auto_customers.intersection(home_customers)):,}")
    logger.info(f"Auto & Credit: {len(auto_customers.intersection(credit_card_customers)):,}")
    logger.info(f"Digital & Home: {len(digital_customers.intersection(home_customers)):,}")
    logger.info(f"Digital & Credit: {len(digital_customers.intersection(credit_card_customers)):,}")
    logger.info(f"Home & Credit: {len(home_customers.intersection(credit_card_customers)):,}")
    logger.info(f"All four services: {len(auto_customers.intersection(digital_customers).intersection(home_customers).intersection(credit_card_customers)):,}")
    
    # Create alignment matrix for all customers - WITH FOUR BANKS
    customer_df = pd.DataFrame({'tax_id': sorted(list(all_customers))})
    customer_df['has_auto'] = customer_df['tax_id'].isin(auto_customers)
    customer_df['has_digital'] = customer_df['tax_id'].isin(digital_customers)
    customer_df['has_home'] = customer_df['tax_id'].isin(home_customers)
    customer_df['has_credit_card'] = customer_df['tax_id'].isin(credit_card_customers)
    
    # Sort datasets by tax_id for consistent indexing - INCLUDING CREDIT CARD
    auto_loans_df = auto_loans_df.sort_values('tax_id').reset_index(drop=True)
    digital_bank_df = digital_bank_df.sort_values('tax_id').reset_index(drop=True)
    home_loans_df = home_loans_df.sort_values('tax_id').reset_index(drop=True)
    credit_card_df = credit_card_df.sort_values('tax_id').reset_index(drop=True)
    master_df = master_df.sort_values('tax_id').reset_index(drop=True)
    
    # Load client models and create feature extractors - 3 NN + 1 XGBOOST
    logger.info("🔄 Creating feature extractors from client models...")
    auto_loans_model, digital_bank_model, home_loans_model, credit_card_model = load_client_models()
    auto_loans_extractor = get_penultimate_layer_model(auto_loans_model)
    digital_bank_extractor = get_penultimate_layer_model(digital_bank_model)
    home_loans_extractor = get_penultimate_layer_model(home_loans_model)
    # Note: credit_card_model is XGBoost, no extractor needed
    
    # Define feature sets used by each model (matching the actual models)
    auto_features = [
        # Core financial features
        'annual_income', 'credit_score', 'payment_history', 'employment_length', 
        'debt_to_income_ratio', 'age',
        # Credit history and behavior  
        'credit_history_length', 'num_credit_cards', 'num_loan_accounts', 
        'total_credit_limit', 'credit_utilization_ratio', 'late_payments', 
        'credit_inquiries', 'last_late_payment_days',
        # Financial position
        'current_debt', 'monthly_expenses', 'savings_balance', 
        'checking_balance', 'investment_balance',
        # Existing loans
        'auto_loan_balance', 'mortgage_balance'
    ]
    
    digital_features = [
        # Core financial features
        'annual_income', 'savings_balance', 'checking_balance', 'investment_balance',
        'payment_history', 'credit_score', 'age', 'employment_length',
        # Transaction and banking behavior  
        'avg_monthly_transactions', 'avg_transaction_value', 'digital_banking_score',
        'mobile_banking_usage', 'online_transactions_ratio', 'international_transactions_ratio',
        'e_statement_enrolled',
        # Financial behavior and credit
        'monthly_expenses', 'total_credit_limit', 'credit_utilization_ratio',
        'num_credit_cards', 'credit_history_length',
        # Additional loan and debt information
        'current_debt', 'mortgage_balance',
        # Additional calculated metrics (from the dataset)
        'total_wealth', 'net_worth', 'credit_efficiency', 'financial_stability_score'
    ]
    
    home_features = [
        # Core financial features
        'annual_income', 'credit_score', 'payment_history', 'employment_length',
        'debt_to_income_ratio', 'age',
        # Credit history and behavior
        'credit_history_length', 'num_credit_cards', 'num_loan_accounts',
        'total_credit_limit', 'credit_utilization_ratio', 'late_payments',
        'credit_inquiries', 'last_late_payment_days',
        # Financial position and assets
        'current_debt', 'monthly_expenses', 'savings_balance',
        'checking_balance', 'investment_balance', 'mortgage_balance', 'auto_loan_balance',
        # Home loan specific calculated features
        'estimated_property_value', 'required_down_payment',
        'available_down_payment_funds', 'mortgage_risk_score',
        'loan_to_value_ratio', 'min_down_payment_pct',
        'interest_rate', 'dti_after_mortgage'
    ]
    
    # Credit card features will be handled directly in extract_xgboost_representations
    
    print(f"\nFeature Verification (3 NN + 1 XGBoost):")
    print(f"Auto model expects {len(auto_features)} features")
    print(f"Digital model expects {len(digital_features)} features")
    print(f"Home model expects {len(home_features)} features")
    print(f"Credit card model: XGBoost with {XGBOOST_OUTPUT_DIM}D output")
    
    # Initialize scalers - 3 NN SCALERS
    auto_scaler = StandardScaler()
    digital_scaler = StandardScaler()
    home_scaler = StandardScaler()
    # Note: XGBoost credit card model has its own scaler built-in
    
    def extract_bank_representations(bank_df, features, scaler, extractor, customer_df, service_column, bank_name):
        """Extract representations for customers with service at this bank (NEURAL NETWORKS ONLY)"""
        output_size = extractor.output_shape[-1]
        all_representations = np.zeros((len(customer_df), output_size))
        
        if len(bank_df) > 0:
            print(f"\nProcessing {bank_name} Bank:")
            print(f"  Dataset size: {len(bank_df)}")
            print(f"  Feature count: {len(features)}")
            print(f"  Output representation size: {output_size}")
            
            # Select only the required features for this model
            feature_data = bank_df[features].copy()
            
            # Handle infinite and missing values in feature data only
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values with median for numeric columns only
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
            
            # For any remaining non-numeric columns, forward fill or use mode
            non_numeric_cols = feature_data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                logger.debug(f"Handling {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
                for col in non_numeric_cols:
                    feature_data[col] = feature_data[col].fillna(feature_data[col].mode()[0] if not feature_data[col].mode().empty else 0)
            
            # Scale features
            X_scaled = scaler.fit_transform(feature_data)
            
            # Get representations from model
            representations = extractor.predict(X_scaled, verbose=2)
            
            # Map representations to correct customer positions
            bank_customer_ids = bank_df['tax_id'].values
            for i, customer_id in enumerate(bank_customer_ids):
                customer_idx = customer_df[customer_df['tax_id'] == customer_id].index[0]
                all_representations[customer_idx] = representations[i]
        
        # Create service availability mask
        service_mask = customer_df[service_column].values.astype(np.float32).reshape(-1, 1)
        
        return all_representations, service_mask, scaler
    
    # Extract representations from three neural network banks
    if ENABLE_PARALLEL_PROCESSING:
        logger.info("🚀 Using PARALLEL processing for representation extraction...")
        logger.info(f"   Max workers: {MAX_WORKERS}")
        logger.info(f"   Chunk size multiplier: {CHUNK_SIZE_MULTIPLIER}")
        
        # Start overall timing
        overall_start_time = time.time()
        
        # Calculate optimal chunk sizes
        auto_chunk_size = max(1, len(auto_loans_df) // (MAX_WORKERS * CHUNK_SIZE_MULTIPLIER))
        digital_chunk_size = max(1, len(digital_bank_df) // (MAX_WORKERS * CHUNK_SIZE_MULTIPLIER))
        home_chunk_size = max(1, len(home_loans_df) // (MAX_WORKERS * CHUNK_SIZE_MULTIPLIER))
        credit_chunk_size = max(1, len(credit_card_df) // (MAX_WORKERS * CHUNK_SIZE_MULTIPLIER))
        
        logger.info(f"   Chunk sizes: Auto={auto_chunk_size}, Digital={digital_chunk_size}, Home={home_chunk_size}, Credit={credit_chunk_size}")
        
        # Extract representations in parallel with timing
        bank_start_time = time.time()
        auto_repr, auto_mask, fitted_auto_scaler = extract_bank_representations_parallel(
            auto_loans_df, auto_features, auto_scaler, auto_loans_extractor, 
            customer_df, 'has_auto', 'Auto Loans', max_workers=MAX_WORKERS
        )
        auto_time = time.time() - bank_start_time
        
        bank_start_time = time.time()
        digital_repr, digital_mask, fitted_digital_scaler = extract_bank_representations_parallel(
            digital_bank_df, digital_features, digital_scaler, digital_bank_extractor,
            customer_df, 'has_digital', 'Digital Banking', max_workers=MAX_WORKERS
        )
        digital_time = time.time() - bank_start_time
        
        bank_start_time = time.time()
        home_repr, home_mask, fitted_home_scaler = extract_bank_representations_parallel(
            home_loans_df, home_features, home_scaler, home_loans_extractor,
            customer_df, 'has_home', 'Home Loans', max_workers=MAX_WORKERS
        )
        home_time = time.time() - bank_start_time
        
        # SPECIAL HANDLING FOR XGBOOST CREDIT CARD BANK (Parallel)
        logger.info("\nProcessing Credit Card Bank (XGBoost - Parallel):")
        logger.info(f"  Dataset size: {len(credit_card_df)}")
        logger.info(f"  Output representation size: {XGBOOST_OUTPUT_DIM}")
        
        # Initialize credit card representations array
        credit_card_repr = np.zeros((len(customer_df), XGBOOST_OUTPUT_DIM))
        
        bank_start_time = time.time()
        if len(credit_card_df) > 0:
            # Extract XGBoost representations using parallel processing
            xgb_representations = extract_xgboost_representations_parallel(credit_card_model, credit_card_df, XGBOOST_OUTPUT_DIM, max_workers=MAX_WORKERS)
            # --- Ensure derived features are created before PCA fitting ---
            features_to_use = list(credit_card_model.feature_names)
            if 'credit_capacity_ratio' in features_to_use:
                credit_card_df['credit_capacity_ratio'] = credit_card_df['credit_card_limit'] / credit_card_df['total_credit_limit'].replace(0, 1)
            if 'income_to_limit_ratio' in features_to_use:
                credit_card_df['income_to_limit_ratio'] = credit_card_df['annual_income'] / credit_card_df['credit_card_limit'].replace(0, 1)
            if 'debt_service_ratio' in features_to_use:
                credit_card_df['debt_service_ratio'] = (credit_card_df['current_debt'] * 0.03) / (credit_card_df['annual_income'] / 12)
            if 'risk_adjusted_income' in features_to_use:
                credit_card_df['risk_adjusted_income'] = credit_card_df['annual_income'] * (credit_card_df['risk_score'] / 100)
            if 'credit_to_income_ratio' in features_to_use:
                credit_card_df['credit_to_income_ratio'] = credit_card_df['credit_card_limit'] / credit_card_df['annual_income'].replace(0, 1)
            # --- Select only the required features for scaler/model ---
            expected_features = list(credit_card_model.scaler.feature_names_in_)
            logger.info(f"Scaler expects features: {expected_features}")
            logger.info(f"features_to_use: {features_to_use}")
            logger.info(f"credit_card_df columns: {list(credit_card_df.columns)}")
            # Check for missing features
            missing = [f for f in expected_features if f not in credit_card_df.columns]
            if missing:
                logger.error(f"Missing features required by scaler: {missing}")
                raise ValueError(f"Missing features required by scaler: {missing}")
            # Select only expected features, in correct order
            feature_data = credit_card_df[expected_features].copy()
            leaf_indices = credit_card_model.classifier.apply(credit_card_model.scaler.transform(feature_data))
            fit_and_save_xgboost_pca(leaf_indices, XGBOOST_OUTPUT_DIM)
            
            # Map representations to correct customer positions (optimized with lookup)
            logger.info(f"   🔄 Mapping XGBoost representations to customer positions...")
            mapping_start = time.time()
            
            # Create lookup dictionary for faster customer ID to index mapping
            customer_id_to_index = {tax_id: idx for idx, tax_id in enumerate(customer_df['tax_id'])}
            credit_card_customer_ids = credit_card_df['tax_id'].values
            
            # Verify data consistency
            if len(credit_card_customer_ids) != len(xgb_representations):
                logger.error(f"   ❌ XGBoost data mismatch: {len(credit_card_customer_ids)} customer IDs vs {len(xgb_representations)} representations")
                raise ValueError("XGBoost customer ID count doesn't match representation count")
            
            # Progress tracking for mapping
            total_mappings = len(credit_card_customer_ids)
            mapping_progress_step = max(1, total_mappings // 100)  # Show progress every 1%
            
            for i, customer_id in enumerate(credit_card_customer_ids):
                customer_idx = customer_id_to_index[customer_id]
                credit_card_repr[customer_idx] = xgb_representations[i]
                
                # Show mapping progress
                if (i + 1) % mapping_progress_step == 0 or i == total_mappings - 1:
                    mapping_percentage = ((i + 1) * 100) / total_mappings
                    logger.info(f"   XGBoost Mapping Progress: {i + 1:,}/{total_mappings:,} customers ({mapping_percentage:.1f}%)")
            
            mapping_end = time.time()
            logger.info(f"   ✅ XGBoost customer mapping completed in {mapping_end - mapping_start:.2f}s")
            
            # Verify final representation count
            non_zero_repr = np.sum(np.any(credit_card_repr != 0, axis=1))
            logger.info(f"   📊 XGBoost final verification: {non_zero_repr:,} customers have representations (expected: {len(credit_card_df):,})")
        
        # Create service availability mask for credit cards
        credit_card_mask = customer_df['has_credit_card'].values.astype(np.float32).reshape(-1, 1)
        credit_time = time.time() - bank_start_time
        
        overall_time = time.time() - overall_start_time
        
        logger.info(f"✅ Parallel XGBoost Credit Card representations: {credit_card_repr.shape}")
        logger.info(f"⏱️  Parallel Processing Performance:")
        logger.info(f"   Auto Loans: {auto_time:.2f}s")
        logger.info(f"   Digital Banking: {digital_time:.2f}s")
        logger.info(f"   Home Loans: {home_time:.2f}s")
        logger.info(f"   Credit Card (XGBoost): {credit_time:.2f}s")
        logger.info(f"   Total Parallel Time: {overall_time:.2f}s")
        
    else:
        logger.info("🐌 Using SEQUENTIAL processing for representation extraction...")
        
        # Start overall timing
        overall_start_time = time.time()
        
        # Extract representations from three neural network banks (sequential)
        bank_start_time = time.time()
        auto_repr, auto_mask, fitted_auto_scaler = extract_bank_representations(
            auto_loans_df, auto_features, auto_scaler, auto_loans_extractor, 
            customer_df, 'has_auto', 'Auto Loans'
        )
        auto_time = time.time() - bank_start_time
        
        bank_start_time = time.time()
        digital_repr, digital_mask, fitted_digital_scaler = extract_bank_representations(
            digital_bank_df, digital_features, digital_scaler, digital_bank_extractor,
            customer_df, 'has_digital', 'Digital Banking'
        )
        digital_time = time.time() - bank_start_time
        
        bank_start_time = time.time()
        home_repr, home_mask, fitted_home_scaler = extract_bank_representations(
            home_loans_df, home_features, home_scaler, home_loans_extractor,
            customer_df, 'has_home', 'Home Loans'
        )
        home_time = time.time() - bank_start_time
        
        # SPECIAL HANDLING FOR XGBOOST CREDIT CARD BANK (Sequential)
        logger.info("\nProcessing Credit Card Bank (XGBoost):")
        logger.info(f"  Dataset size: {len(credit_card_df)}")
        logger.info(f"  Output representation size: {XGBOOST_OUTPUT_DIM}")
        
        # Initialize credit card representations array
        credit_card_repr = np.zeros((len(customer_df), XGBOOST_OUTPUT_DIM))
        
        bank_start_time = time.time()
        if len(credit_card_df) > 0:
            # Extract XGBoost representations
            xgb_representations = extract_xgboost_representations(credit_card_model, credit_card_df, XGBOOST_OUTPUT_DIM)
            # --- Ensure derived features are created before PCA fitting ---
            features_to_use = list(credit_card_model.feature_names)
            if 'credit_capacity_ratio' in features_to_use:
                credit_card_df['credit_capacity_ratio'] = credit_card_df['credit_card_limit'] / credit_card_df['total_credit_limit'].replace(0, 1)
            if 'income_to_limit_ratio' in features_to_use:
                credit_card_df['income_to_limit_ratio'] = credit_card_df['annual_income'] / credit_card_df['credit_card_limit'].replace(0, 1)
            if 'debt_service_ratio' in features_to_use:
                credit_card_df['debt_service_ratio'] = (credit_card_df['current_debt'] * 0.03) / (credit_card_df['annual_income'] / 12)
            if 'risk_adjusted_income' in features_to_use:
                credit_card_df['risk_adjusted_income'] = credit_card_df['annual_income'] * (credit_card_df['risk_score'] / 100)
            if 'credit_to_income_ratio' in features_to_use:
                credit_card_df['credit_to_income_ratio'] = credit_card_df['credit_card_limit'] / credit_card_df['annual_income'].replace(0, 1)
            # --- Select only the required features for scaler/model ---
            expected_features = list(credit_card_model.scaler.feature_names_in_)
            logger.info(f"Scaler expects features: {expected_features}")
            logger.info(f"features_to_use: {features_to_use}")
            logger.info(f"credit_card_df columns: {list(credit_card_df.columns)}")
            # Check for missing features
            missing = [f for f in expected_features if f not in credit_card_df.columns]
            if missing:
                logger.error(f"Missing features required by scaler: {missing}")
                raise ValueError(f"Missing features required by scaler: {missing}")
            # Select only expected features, in correct order
            feature_data = credit_card_df[expected_features].copy()
            leaf_indices = credit_card_model.classifier.apply(credit_card_model.scaler.transform(feature_data))
            fit_and_save_xgboost_pca(leaf_indices, XGBOOST_OUTPUT_DIM)
            
            # Map representations to correct customer positions (optimized with lookup)
            logger.info(f"   🔄 Mapping XGBoost representations to customer positions...")
            mapping_start = time.time()
            
            # Create lookup dictionary for faster customer ID to index mapping
            customer_id_to_index = {tax_id: idx for idx, tax_id in enumerate(customer_df['tax_id'])}
            credit_card_customer_ids = credit_card_df['tax_id'].values
            
            # Progress tracking for mapping
            total_mappings = len(credit_card_customer_ids)
            mapping_progress_step = max(1, total_mappings // 20)  # Show progress every 5%
            
            for i, customer_id in enumerate(credit_card_customer_ids):
                customer_idx = customer_id_to_index[customer_id]
                credit_card_repr[customer_idx] = xgb_representations[i]
                
                # Show mapping progress
                if (i + 1) % mapping_progress_step == 0 or i == total_mappings - 1:
                    mapping_percentage = ((i + 1) * 100) / total_mappings
                    logger.info(f"   XGBoost Mapping Progress: {i + 1:,}/{total_mappings:,} customers ({mapping_percentage:.1f}%)")
            
            mapping_end = time.time()
            logger.info(f"   ✅ XGBoost customer mapping completed in {mapping_end - mapping_start:.2f}s")
        
        # Create service availability mask for credit cards
        credit_card_mask = customer_df['has_credit_card'].values.astype(np.float32).reshape(-1, 1)
        credit_time = time.time() - bank_start_time
        
        overall_time = time.time() - overall_start_time
        
        logger.info(f"✅ XGBoost Credit Card representations: {credit_card_repr.shape}")
        logger.info(f"⏱️  Sequential Processing Performance:")
        logger.info(f"   Auto Loans: {auto_time:.2f}s")
        logger.info(f"   Digital Banking: {digital_time:.2f}s")
        logger.info(f"   Home Loans: {home_time:.2f}s")
        logger.info(f"   Credit Card (XGBoost): {credit_time:.2f}s")
        logger.info(f"   Total Sequential Time: {overall_time:.2f}s")
    
    # Complete the rest of the preprocessing function
    # Save intermediate representations to CSV for review (SAMPLE ONLY) 
    logger.info("💾 Saving sample intermediate representations to CSV for review...")
    
    # --- DATA AUGMENTATION FOR MISSING DATA HANDLING ---
    # Randomly drop some bank representations to help model learn missing data patterns
    if ENABLE_PHASE_2:  # Only during final training, not during AutoML search
        logger.info("🔄 Applying data augmentation for missing data handling...")
        
        # Set random seed for reproducible augmentation
        np.random.seed(RANDOM_SEED)
        
        # Augmentation parameters
        drop_prob = 0.08  # 8% chance to drop each bank per customer (reduced from 15%)
        logger.info(f"   Drop probability per bank: {drop_prob:.1%}")
        
        # Create augmented masks
        auto_mask_aug = auto_mask.copy()
        digital_mask_aug = digital_mask.copy()
        home_mask_aug = home_mask.copy()
        credit_card_mask_aug = credit_card_mask.copy()
        
        # Randomly drop some banks (but ensure at least one bank remains)
        for i in range(len(customer_df)):
            # Get current service availability
            has_auto = auto_mask_aug[i, 0] > 0
            has_digital = digital_mask_aug[i, 0] > 0
            has_home = home_mask_aug[i, 0] > 0
            has_credit = credit_card_mask_aug[i, 0] > 0
            
            available_banks = sum([has_auto, has_digital, has_home, has_credit])
            
            # Only augment if customer has multiple banks
            if available_banks > 1:
                # Randomly drop some banks
                if has_auto and np.random.random() < drop_prob:
                    auto_mask_aug[i, 0] = 0.0
                if has_digital and np.random.random() < drop_prob:
                    digital_mask_aug[i, 0] = 0.0
                if has_home and np.random.random() < drop_prob:
                    home_mask_aug[i, 0] = 0.0
                if has_credit and np.random.random() < drop_prob:
                    credit_card_mask_aug[i, 0] = 0.0
                
                # Ensure at least one bank remains
                remaining_banks = sum([
                    auto_mask_aug[i, 0] > 0,
                    digital_mask_aug[i, 0] > 0,
                    home_mask_aug[i, 0] > 0,
                    credit_card_mask_aug[i, 0] > 0
                ])
                
                if remaining_banks == 0:
                    # Restore one random bank if all were dropped
                    banks = ['auto', 'digital', 'home', 'credit']
                    available = [has_auto, has_digital, has_home, has_credit]
                    available_banks = [b for b, a in zip(banks, available) if a]
                    if available_banks:
                        restore_bank = np.random.choice(available_banks)
                        if restore_bank == 'auto':
                            auto_mask_aug[i, 0] = 1.0
                        elif restore_bank == 'digital':
                            digital_mask_aug[i, 0] = 1.0
                        elif restore_bank == 'home':
                            home_mask_aug[i, 0] = 1.0
                        elif restore_bank == 'credit':
                            credit_card_mask_aug[i, 0] = 1.0
        
        # Update masks with augmented versions
        auto_mask = auto_mask_aug
        digital_mask = digital_mask_aug
        home_mask = home_mask_aug
        credit_card_mask = credit_card_mask_aug
        
        # Log augmentation statistics
        original_auto = np.sum(auto_mask_aug == 0)
        original_digital = np.sum(digital_mask_aug == 0)
        original_home = np.sum(home_mask_aug == 0)
        original_credit = np.sum(credit_card_mask_aug == 0)
        
        logger.info(f"   Augmentation results:")
        logger.info(f"     Auto loans dropped: {original_auto:,} customers")
        logger.info(f"     Digital banking dropped: {original_digital:,} customers")
        logger.info(f"     Home loans dropped: {original_home:,} customers")
        logger.info(f"     Credit cards dropped: {original_credit:,} customers")
    else:
        logger.info("   Data augmentation skipped (AutoML search phase)")
    
    # Create comprehensive dataframe with all representations
    representations_df = pd.DataFrame({
        'tax_id': customer_df['tax_id'],
        'credit_score': master_df.set_index('tax_id').loc[customer_df['tax_id'], 'credit_score'].values,
        'has_auto': customer_df['has_auto'].astype(int),
        'has_digital': customer_df['has_digital'].astype(int), 
        'has_home': customer_df['has_home'].astype(int),
        'has_credit_card': customer_df['has_credit_card'].astype(int)
    })
    
    # Add auto loan representations (16 dimensions)
    for i in range(auto_repr.shape[1]):
        representations_df[f'auto_repr_{i+1:02d}'] = auto_repr[:, i]
    
    # Add digital banking representations (8 dimensions)
    for i in range(digital_repr.shape[1]):
        representations_df[f'digital_repr_{i+1:02d}'] = digital_repr[:, i]
    
    # Add home loan representations (16 dimensions)  
    for i in range(home_repr.shape[1]):
        representations_df[f'home_repr_{i+1:02d}'] = home_repr[:, i]
    
    # Add XGBoost credit card representations (8 dimensions)
    for i in range(credit_card_repr.shape[1]):
        representations_df[f'credit_card_repr_{i+1:02d}'] = credit_card_repr[:, i]
    
    # Add service availability masks
    representations_df['auto_mask'] = auto_mask.flatten()
    representations_df['digital_mask'] = digital_mask.flatten()
    representations_df['home_mask'] = home_mask.flatten()
    representations_df['credit_card_mask'] = credit_card_mask.flatten()
    
    # Calculate service combinations for analysis
    representations_df['service_count'] = (representations_df[['has_auto', 'has_digital', 'has_home', 'has_credit_card']].sum(axis=1))
    representations_df['service_combination'] = (
        representations_df['has_auto'].astype(str) + 
        representations_df['has_digital'].astype(str) + 
        representations_df['has_home'].astype(str) + 
        representations_df['has_credit_card'].astype(str)
    )
    
    # Add human-readable service labels
    service_labels = {
        '0000': 'No Services',
        '1000': 'Auto Only', 
        '0100': 'Digital Only',
        '0010': 'Home Only',
        '0001': 'Credit Only',
        '1100': 'Auto+Digital',
        '1010': 'Auto+Home', 
        '1001': 'Auto+Credit',
        '0110': 'Digital+Home',
        '0101': 'Digital+Credit',
        '0011': 'Home+Credit',
        '1110': 'Auto+Digital+Home',
        '1101': 'Auto+Digital+Credit',
        '1011': 'Auto+Home+Credit',
        '0111': 'Digital+Home+Credit',
        '1111': 'All Four Services'
    }
    representations_df['service_label'] = representations_df['service_combination'].map(service_labels)
    
    # Select representative sample for visualization (max 15 records)
    logger.info("🎯 Selecting representative sample for visualization...")
    
    # Try to get 1-2 examples from each service combination type
    sample_records = []
    service_dist = representations_df['service_label'].value_counts()
    
    for service_type, count in service_dist.items():
        # Get 1-2 examples from each service type (based on availability)
        subset = representations_df[representations_df['service_label'] == service_type]
        n_samples = min(2, count, max(1, 15 // len(service_dist)))  # Distribute ~15 samples across types
        
        if len(subset) > 0:
            sampled = subset.sample(n=n_samples, random_state=42)
            sample_records.append(sampled)
            logger.debug(f"   - {service_type}: selected {len(sampled)} of {count} customers")
    
    # Combine samples and limit to 15 total
    if sample_records:
        sample_df = pd.concat(sample_records, ignore_index=True)
        if len(sample_df) > 15:
            sample_df = sample_df.sample(n=15, random_state=42)
    else:
        # Fallback: just take first 15 records
        sample_df = representations_df.head(15)
    
    # Sort by service combination for better readability
    sample_df = sample_df.sort_values(['service_combination', 'credit_score'], ascending=[True, False])
    sample_df = sample_df.reset_index(drop=True)
    
    # Create output directory
    os.makedirs('data/intermediate_representations', exist_ok=True)
    
    # Save sample to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/intermediate_representations/vfl_xgboost_representations_sample_{timestamp}.csv'
    sample_df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Sample intermediate representations saved to: {output_file}")
    logger.info(f"📊 Sample DataFrame Info:")
    logger.info(f"   - Sample size: {len(sample_df)} customers (from {len(representations_df)} total)")
    logger.info(f"   - Total columns: {len(sample_df.columns)}")
    logger.info(f"   - Auto representations: {auto_repr.shape[1]} columns (auto_repr_01 to auto_repr_{auto_repr.shape[1]:02d})")
    logger.info(f"   - Digital representations: {digital_repr.shape[1]} columns (digital_repr_01 to digital_repr_{digital_repr.shape[1]:02d})")
    logger.info(f"   - Home representations: {home_repr.shape[1]} columns (home_repr_01 to home_repr_{home_repr.shape[1]:02d})")
    logger.info(f"   - XGBoost Credit Card representations: {credit_card_repr.shape[1]} columns (credit_card_repr_01 to credit_card_repr_{credit_card_repr.shape[1]:02d})")
    
    # Log sample distribution
    logger.info(f"📈 Sample Service Distribution:")
    sample_service_dist = sample_df['service_label'].value_counts()
    for service, count in sample_service_dist.items():
        percentage = (count / len(sample_df)) * 100
        total_of_type = service_dist[service] if service in service_dist else 0
        logger.info(f"   - {service}: {count} samples ({percentage:.1f}%) from {total_of_type} total")
    
    # ============================================================================
    # CREATE TRAINING DATA FROM BANK REPRESENTATIONS - MODIFIED FOR XGBOOST
    # ============================================================================
    logger.info("🔗 Creating VFL training data from bank representations (3 NN + 1 XGBoost)...")
    
    # Combine all bank representations into single feature matrix
    # Format: [auto_repr | auto_mask | digital_repr | digital_mask | home_repr | home_mask | credit_card_repr | credit_card_mask]
    X_combined = np.concatenate([
        auto_repr,           # 16 dimensions (auto loan representations)
        auto_mask,           # 1 dimension (auto service availability)
        digital_repr,        # 8 dimensions (digital banking representations) 
        digital_mask,        # 1 dimension (digital service availability)
        home_repr,           # 16 dimensions (home loan representations)
        home_mask,           # 1 dimension (home service availability)
        credit_card_repr,    # 8 dimensions (XGBoost credit card representations)
        credit_card_mask     # 1 dimension (credit card service availability)
    ], axis=1)
    
    # Get target variable (credit scores) from master dataset
    y_combined = master_df['credit_score'].values
    ids_combined = master_df['tax_id'].values
    
    logger.info(f"✅ VFL training data created (XGBoost version):")
    logger.info(f"   - Combined feature matrix: {X_combined.shape}")
    logger.info(f"   - Feature breakdown: Auto({auto_repr.shape[1]}+1) + Digital({digital_repr.shape[1]}+1) + Home({home_repr.shape[1]}+1) + XGBoost Credit({credit_card_repr.shape[1]}+1) = {X_combined.shape[1]} total")
    logger.info(f"   - Target variable shape: {y_combined.shape}")
    logger.info(f"   - Customer ID shape: {ids_combined.shape}")
    logger.info(f"   - Credit score range: {y_combined.min():.0f} - {y_combined.max():.0f}")
    
    # Split data into training and test sets
    logger.info("🔀 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_combined, y_combined, ids_combined,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=pd.cut(y_combined, bins=5, labels=False)  # Stratify by credit score ranges
    )
    
    logger.info(f"📊 Train/Test Split Results:")
    logger.info(f"   - Training samples: {len(X_train):,}")
    logger.info(f"   - Test samples: {len(X_test):,}")
    logger.info(f"   - Training credit score range: {y_train.min():.0f} - {y_train.max():.0f}")
    logger.info(f"   - Test credit score range: {y_test.min():.0f} - {y_test.max():.0f}")
    logger.info(f"   - Random seed: {RANDOM_SEED}")

    return (X_train, X_test, y_train, y_test, ids_train, ids_test, 
            fitted_auto_scaler, fitted_digital_scaler, fitted_home_scaler, None,  # No scaler for XGBoost
            auto_repr.shape[1], digital_repr.shape[1], home_repr.shape[1], credit_card_repr.shape[1],
            X_combined, y_combined, ids_combined)

class VFLHyperModel(kt.HyperModel):
    """Hypermodel for VFL architecture search - SIMPLIFIED: No Dropout, L2, BatchNorm, or missing data embeddings"""
    
    def __init__(self, input_shape, auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size):
        self.input_shape = input_shape
        self.auto_repr_size = auto_repr_size
        self.digital_repr_size = digital_repr_size
        self.home_repr_size = home_repr_size
        self.credit_card_repr_size = credit_card_repr_size  # Now XGBoost 8D representations
    
    def get_config(self):
        """Return config for serialization"""
        return {
            'input_shape': self.input_shape,
            'auto_repr_size': self.auto_repr_size,
            'digital_repr_size': self.digital_repr_size,
            'home_repr_size': self.home_repr_size,
            'credit_card_repr_size': self.credit_card_repr_size
        }
    
    @classmethod
    def from_config(cls, config):
        """Create instance from config"""
        return cls(**config)
    
    def build(self, hp):
        """Build model with hyperparameters - Dropout after each hidden dense layer for MC Dropout"""
        from keras import layers, models
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='vfl_input')
        
        # Split combined input into components using Lambda layers
        auto_repr_size = self.auto_repr_size
        digital_repr_size = self.digital_repr_size
        home_repr_size = self.home_repr_size
        credit_card_repr_size = self.credit_card_repr_size
        
        auto_repr = layers.Lambda(lambda x, s=auto_repr_size: x[:, :s], name='auto_representations')(inputs)
        auto_mask = layers.Lambda(lambda x, s=auto_repr_size: x[:, s:s+1], name='auto_mask')(inputs)
        
        digital_start = auto_repr_size + 1
        digital_end = digital_start + digital_repr_size
        digital_repr = layers.Lambda(lambda x, start=digital_start, end=digital_end: x[:, start:end], name='digital_representations')(inputs)
        digital_mask = layers.Lambda(lambda x, end=digital_end: x[:, end:end+1], name='digital_mask')(inputs)
        
        home_start = digital_end + 1
        home_end = home_start + home_repr_size
        home_repr = layers.Lambda(lambda x, start=home_start, end=home_end: x[:, start:end], name='home_representations')(inputs)
        home_mask = layers.Lambda(lambda x, end=home_end: x[:, end:end+1], name='home_mask')(inputs)
        
        credit_card_start = home_end + 1
        credit_card_end = credit_card_start + credit_card_repr_size
        credit_card_repr = layers.Lambda(lambda x, start=credit_card_start, end=credit_card_end: x[:, start:end], name='xgboost_credit_card_representations')(inputs)
        credit_card_mask = layers.Lambda(lambda x, end=credit_card_end: x[:, end:end+1], name='credit_card_mask')(inputs)
        
        # Just concatenate all representations and masks
        concat = layers.Concatenate(name='all_features_concat')([
            auto_repr, auto_mask, digital_repr, digital_mask, home_repr, home_mask, credit_card_repr, credit_card_mask
        ])
        
        # Tunable main architecture (same as original AutoML search)
        x = concat
        num_layers = hp.Int('num_layers', min_value=MIN_LAYERS, max_value=MAX_LAYERS)
        for i in range(num_layers):
            if i == 0:
                units = hp.Int(f'layer_{i}_units', min_value=MIN_UNITS, max_value=MAX_UNITS, step=64)
            else:
                prev_units = hp.get(f'layer_{i-1}_units')
                max_units_this_layer = min(MAX_UNITS, prev_units)
                units = hp.Int(f'layer_{i}_units', min_value=MIN_UNITS, max_value=max_units_this_layer, step=64)
            activation = hp.Choice(f'layer_{i}_activation', values=['relu', 'swish', 'gelu'])
            x = layers.Dense(units, activation=activation, name=f'hidden_dense_{i+1}')(x)
            # Add Dropout after each hidden dense layer
            dropout_rate = hp.Float(f'layer_{i}_dropout', min_value=0.0, max_value=0.5, step=0.1)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name=f'hidden_dropout_{i+1}')(x)
        final_units = hp.Int('final_units', min_value=16, max_value=64, step=16)
        x = layers.Dense(final_units, activation='relu', name='pre_output_dense')(x)
        raw_output = layers.Dense(1, activation='sigmoid', name='raw_output')(x)
        outputs = layers.Lambda(lambda x: x * 550 + 300, name='credit_score_output')(raw_output)
        model = models.Model(inputs=inputs, outputs=outputs, name='vfl_automl_xgboost_model')
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        clipnorm = hp.Float('clipnorm', min_value=0.5, max_value=2.0, step=0.5)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
            beta_1=0.9,
            beta_2=0.999
        )
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']
        )
        return model

def run_automl_search():
    """Run AutoML architecture search with two-phase training - MODIFIED FOR XGBOOST CREDIT CARD"""
    
    start_time = datetime.now()
    logger.info("🚀 VFL AutoML Architecture Search - XGBoost Credit Card System")
    logger.info("=" * 80)
    logger.info(f"Training Configuration:")
    logger.info(f"  Phase 1 (AutoML Search): {'✅ Enabled' if True else '❌ Disabled'}")
    logger.info(f"  Phase 2 (Full Training): {'✅ Enabled' if ENABLE_PHASE_2 else '❌ Disabled'}")
    logger.info(f"  Confidence Scoring: {'✅ Enabled' if ENABLE_CONFIDENCE_SCORES else '❌ Disabled'}")
    logger.info("")
    logger.info(f"🏦 Bank Configuration (MODIFIED FOR XGBOOST):")
    logger.info(f"  Auto Loans: Neural Network (16D representations)")
    logger.info(f"  Digital Banking: Neural Network (8D representations)")
    logger.info(f"  Home Loans: Neural Network (16D representations)")
    logger.info(f"  Credit Cards: XGBoost ({XGBOOST_OUTPUT_DIM}D representations)")
    logger.info("")
    logger.info(f"Phase 1 - AutoML Search Configuration:")
    logger.info(f"  Max Trials: {MAX_TRIALS}")
    logger.info(f"  Executions per Trial: {EXECUTIONS_PER_TRIAL}")
    logger.info(f"  Epochs per Trial: {EPOCHS_PER_TRIAL}")
    logger.info(f"  Objective: {SEARCH_OBJECTIVE}")
    logger.info(f"  AutoML Sample Size: {AUTOML_SAMPLE_SIZE:,} customers")
    logger.info(f"  Random Seed: {RANDOM_SEED}")
    logger.info(f"  Search Space: {MIN_LAYERS}-{MAX_LAYERS} layers, {MIN_UNITS}-{MAX_UNITS} units")
    
    if ENABLE_CONFIDENCE_SCORES:
        logger.info("")
        logger.info(f"Confidence Scoring Configuration:")
        logger.info(f"  Monte Carlo Dropout Samples: {MC_DROPOUT_SAMPLES}")
        logger.info(f"  Confidence Intervals: {CONFIDENCE_INTERVALS}%")
        logger.info(f"  High Confidence Threshold: {MIN_CONFIDENCE_THRESHOLD}")
    
    if ENABLE_PHASE_2:
        logger.info("")
        logger.info(f"Phase 2 - Final Training Configuration:")
        logger.info(f"  Final Sample Size: {FINAL_SAMPLE_SIZE:,} customers")
        logger.info(f"  Final Training Epochs: {FINAL_EPOCHS}")
    else:
        logger.info("")
        logger.info(f"💡 Phase 2 is disabled - only AutoML search will be performed")
        logger.info(f"   To enable full training, set ENABLE_PHASE_2 = True")
    
    # Create necessary directories
    logger.debug("Creating output directories...")
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('automl_results', exist_ok=True)
    
    # Phase 1: AutoML Search with smaller sample
    logger.info("")
    logger.info("🚀 PHASE 1: AutoML Architecture Search (XGBoost Credit Card)")
    logger.info("=" * 60)
    
    phase1_start = datetime.now()
    
    # Load and preprocess data for AutoML search
    logger.debug("Loading data for AutoML search...")
    (X_train, X_test, y_train, y_test, ids_train, ids_test, 
     auto_scaler, digital_scaler, home_scaler, credit_card_scaler,  # Note: credit_card_scaler is None for XGBoost
     auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size,
     X_combined, y_combined, ids_combined) = load_and_preprocess_data(AUTOML_SAMPLE_SIZE)
    
    logger.info(f"🔍 AutoML Search Data Summary (3 NN + 1 XGBoost):")
    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Feature vector size: {X_train.shape[1]}")
    logger.info(f"  Auto loans representation size: {auto_repr_size}")
    logger.info(f"  Digital bank representation size: {digital_repr_size}")
    logger.info(f"  Home loans representation size: {home_repr_size}")
    logger.info(f"  XGBoost credit card representation size: {credit_card_repr_size}")
    
    # Create hypermodel
    logger.debug("Creating hypermodel for architecture search...")
    hypermodel = VFLHyperModel(
        input_shape=(X_train.shape[1],),
        auto_repr_size=auto_repr_size,
        digital_repr_size=digital_repr_size,
        home_repr_size=home_repr_size,
        credit_card_repr_size=credit_card_repr_size  # XGBoost 8D representations
    )
    
    # ============================================================================
    # DETAILED HYPERPARAMETER SEARCH LOGGING
    # ============================================================================
    
    # Log search space details before starting
    logger.info("")
    logger.info("🔍 HYPERPARAMETER SEARCH SPACE CONFIGURATION (XGBoost Version):")
    logger.info("=" * 70)
    logger.info(f"🎯 Search Strategy: Random Search")
    logger.info(f"📊 Objective to Optimize: {SEARCH_OBJECTIVE}")
    logger.info(f"🔢 Maximum Trials: {MAX_TRIALS}")
    logger.info(f"🔄 Executions per Trial: {EXECUTIONS_PER_TRIAL}")
    logger.info(f"⏱️  Epochs per Trial: {EPOCHS_PER_TRIAL}")
    logger.info("")
    
    logger.info("🏗️  Architecture Search Space:")
    logger.info(f"   - Hidden Layers: {MIN_LAYERS} to {MAX_LAYERS}")
    logger.info(f"   - Units per Layer: {MIN_UNITS} to {MAX_UNITS}")
    logger.info(f"   - Learning Rate: 1e-4 to 1e-2 (log scale)")
    logger.info(f"   - Gradient Clipping: 0.5 to 2.0")
    logger.info("")
    
    logger.info("🏦 Bank-Specific Processing Units (MODIFIED):")
    logger.info(f"   - Auto Bank (NN): 64 to 256 units (step: 64)")
    logger.info(f"   - Digital Bank (NN): 32 to 128 units (step: 32)")
    logger.info(f"   - Home Bank (NN): 64 to 256 units (step: 64)")
    logger.info(f"   - Credit Card Bank (XGBoost): 64 to 256 units (step: 64) [Processing XGBoost 8D representations]")
    logger.info(f"   - Service Info: 8 to 32 units (step: 8)")
    logger.info("")
    
    logger.info("🎛️  Tunable Options:")
    logger.info(f"   - Activation Functions: relu, swish, gelu")
    logger.info(f"   - Batch Normalization: True/False per layer")
    logger.info(f"   - Dropout Rates: 0.0 to 0.5 (step: 0.1)")
    logger.info(f"   - Final Dropout: 0.0 to 0.3 (step: 0.05)")
    logger.info(f"   - Final Units: 16 to 64 (step: 16)")
    
    # Configure tuner with simpler directory structure
    logger.debug("Configuring Keras Tuner...")
    tuner = kt.RandomSearch(
        hypermodel,
        objective=SEARCH_OBJECTIVE,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        directory='automl_results',
        project_name=f'vfl_xgboost_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        overwrite=True,
        tune_new_entries=True,
        allow_new_entries=True
    )
    
    estimated_time_min = MAX_TRIALS * EXECUTIONS_PER_TRIAL * EPOCHS_PER_TRIAL / 60
    estimated_time_max = MAX_TRIALS * EXECUTIONS_PER_TRIAL * EPOCHS_PER_TRIAL / 30
    
    logger.info("=" * 70)
    logger.info(f"⏱️  Starting AutoML Search (XGBoost Credit Card Version)...")
    logger.info(f"  Testing {MAX_TRIALS} different architectures")
    logger.info(f"  Estimated time: {estimated_time_min:.0f}-{estimated_time_max:.0f} minutes")
    logger.info(f"  Search objective: minimize {SEARCH_OBJECTIVE}")
    logger.info("=" * 70)
    
    # Define callbacks for each trial
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=SEARCH_OBJECTIVE,
            patience=EARLY_STOPPING_PATIENCE,  # Increased patience
            restore_best_weights=True,
            verbose=2
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.3,
            patience=REDUCE_LR_PATIENCE,  # Increased patience
            min_lr=MIN_LEARNING_RATE,  # Lower minimum learning rate
            verbose=2  # Set to 0 to eliminate callback progress printing
        )
    ]
    
    # Start AutoML search
    logger.info("")
    logger.info("🔍 Starting Keras Tuner search...")
    search_start = datetime.now()
    
    try:
        # Perform the search with detailed logging
        logger.info("🎯 Starting hyperparameter search trials...")
        logger.info("=" * 80)
        
        # Perform the actual search
        tuner.search(
            X_train, y_train,
            epochs=EPOCHS_PER_TRIAL,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=2  # Set to 0 to eliminate per-batch progress printing
        )
        
        search_end = datetime.now()
        search_duration = search_end - search_start
        phase1_end = datetime.now()
        phase1_duration = phase1_end - phase1_start
        
        logger.info(f"✅ AutoML search completed in {search_duration}")
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]
        
        logger.info("")
        logger.info("🏆 PHASE 1 COMPLETE - AutoML Search Results (XGBoost Credit Card)")
        logger.info("=" * 60)
        logger.info(f"⏱️  Search Duration: {search_duration}")
        logger.info(f"🎯 Search Objective: {SEARCH_OBJECTIVE}")
        logger.info(f"📊 Trials Completed: {len(tuner.oracle.trials)}")
        logger.info("")
        
        # Evaluate best model on test set
        test_loss, test_mae, test_mse = best_model.evaluate(X_test, y_test, verbose=2)
        y_pred_phase1 = best_model.predict(X_test, verbose=2)
        
        logger.info(f"🎯 Best Architecture Performance:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")
        import numpy as np
        logger.info(f"  Test RMSE: {np.sqrt(test_mse):.2f}")
        logger.info(f"  Parameters: {best_model.count_params():,}")
        
        # Log best hyperparameters
        logger.info("")
        logger.info("🏗️  Best Architecture Configuration (XGBoost Version):")
        logger.info(f"  Hidden Layers: {best_hp.get('num_layers')}")
        logger.info(f"  Learning Rate: {best_hp.get('learning_rate'):.6f}")
        logger.info(f"  Gradient Clipping: {best_hp.get('clipnorm')}")
        
        # Sample predictions for Phase 1 with confidence scores
        print_detailed_sample_predictions_with_confidence(X_test, y_test, ids_test, best_model, 
                                        auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size, phase="Phase 1")
        
        # Save Phase 1 results
        phase1_results = {
            'search_config': {
                'sample_size': AUTOML_SAMPLE_SIZE,
                'max_trials': MAX_TRIALS,
                'executions_per_trial': EXECUTIONS_PER_TRIAL,
                'epochs_per_trial': EPOCHS_PER_TRIAL,
                'objective': SEARCH_OBJECTIVE,
                'search_space': {
                    'min_layers': MIN_LAYERS,
                    'max_layers': MAX_LAYERS,
                    'min_units': MIN_UNITS,
                    'max_units': MAX_UNITS
                },
                'xgboost_credit_card': {
                    'enabled': True,
                    'output_dimension': XGBOOST_OUTPUT_DIM,
                    'pca_random_state': XGBOOST_PCA_RANDOM_STATE
                }
            },
            'best_hyperparameters': best_hp.values,
            'best_performance': {
                'test_loss': test_loss,
                'test_mae': test_mae,
                'test_rmse': np.sqrt(test_mse),
                'parameters': best_model.count_params()
            },
            'timing': {
                'search_start': search_start.isoformat(),
                'search_end': search_end.isoformat(),
                'search_duration_seconds': search_duration.total_seconds(),
                'phase1_duration_seconds': phase1_duration.total_seconds()
            }
        }
        
        with open('automl_results/phase1_xgboost_results.json', 'w') as f:
            json.dump(phase1_results, f, indent=2, default=str)
        
        logger.info("")
        logger.info(f"📁 Phase 1 results saved to: automl_results/phase1_xgboost_results.json")
        
        # Check if Phase 2 is enabled
        if not ENABLE_PHASE_2:
            total_duration = datetime.now() - start_time
            logger.info("")
            logger.info("🎉 VFL AutoML Search Completed (Phase 1 Only, XGBoost Credit Card)")
            logger.info("=" * 70)
            logger.info(f"⏱️  Total Duration: {total_duration}")
            logger.info(f"💡 Phase 2 is disabled. To enable full training, set ENABLE_PHASE_2 = True")
            logger.info(f"🏆 Best MAE: {test_mae:.2f} points")
            logger.info(f"📊 Architecture: {best_hp.get('num_layers')} layers, {best_model.count_params():,} parameters")
            logger.info(f"🏦 Credit Card Model: XGBoost with {XGBOOST_OUTPUT_DIM}D representations")
            return best_model, best_hp, phase1_results
        
        # Continue with Phase 2 (same logic as original, but with XGBoost modifications)
        logger.info("")
        logger.info("🚀 PHASE 2: Final Model Training with Larger Dataset (XGBoost Credit Card)")
        logger.info("=" * 60)
        
        phase2_start = datetime.now()
        
        logger.info(f"🎯 Phase 2 Configuration:")
        logger.info(f"  Dataset size: {FINAL_SAMPLE_SIZE:,} customers (vs {AUTOML_SAMPLE_SIZE:,} in Phase 1)")
        logger.info(f"  Training epochs: {FINAL_EPOCHS} (vs {EPOCHS_PER_TRIAL} in Phase 1)")
        logger.info(f"  Using best architecture from Phase 1")
        logger.info(f"  XGBoost Credit Card: {XGBOOST_OUTPUT_DIM}D representations")
        logger.info(f"  Random seed: {RANDOM_SEED}")
        
        # Load larger dataset for final training
        logger.info("🔄 Loading larger dataset for Phase 2 training...")
        (X_train_final, X_test_final, y_train_final, y_test_final, ids_train_final, ids_test_final,
         auto_scaler_final, digital_scaler_final, home_scaler_final, credit_card_scaler_final,  # credit_card_scaler_final is None
         auto_repr_size_final, digital_repr_size_final, home_repr_size_final, credit_card_repr_size_final,
         X_combined_final, y_combined_final, ids_combined_final) = load_and_preprocess_data(FINAL_SAMPLE_SIZE)
        # Save fitted scalers for inference
        import joblib
        joblib.dump(auto_scaler_final, 'VFLClientModels/models/saved_models/auto_scaler.pkl')
        joblib.dump(digital_scaler_final, 'VFLClientModels/models/saved_models/digital_scaler.pkl')
        joblib.dump(home_scaler_final, 'VFLClientModels/models/saved_models/home_scaler.pkl')
        logger.info('✅ Saved fitted scalers to saved_models/')
        
        
        logger.info(f"📊 Phase 2 Dataset Statistics:")
        logger.info(f"  Training samples: {len(X_train_final):,}")
        logger.info(f"  Test samples: {len(X_test_final):,}")
        logger.info(f"  Feature vector size: {X_train_final.shape[1]}")
        logger.info(f"  Credit score range: {y_train_final.min():.0f} - {y_train_final.max():.0f}")
        
        # === DEBUG: Inspect X_train_final and y_train_final before training ===
        logger.info("\n=== DEBUG: X_train_final[:5] ===\n" + str(X_train_final[:5]))
        logger.info("\n=== DEBUG: y_train_final[:5] ===\n" + str(y_train_final[:5]))
        # Print summary stats for each column in X_train_final
        import numpy as np
        col_stats = []
        for i in range(X_train_final.shape[1]):
            col = X_train_final[:, i]
            stats = {
                'col': i,
                'min': np.min(col),
                'max': np.max(col),
                'mean': np.mean(col),
                'std': np.std(col),
                'n_nan': np.sum(np.isnan(col)),
                'n_unique': len(np.unique(col))
            }
            col_stats.append(stats)
        logger.info("\n=== DEBUG: X_train_final column stats ===\n" + str(col_stats))
        # Print summary stats for y_train_final
        logger.info(f"\n=== DEBUG: y_train_final stats === min: {np.min(y_train_final)}, max: {np.max(y_train_final)}, mean: {np.mean(y_train_final)}, std: {np.std(y_train_final)}, n_nan: {np.sum(np.isnan(y_train_final))}, n_unique: {len(np.unique(y_train_final))}")
        # === END DEBUG ===
        
        # Build final model with best hyperparameters
        logger.info("🏗️  Building final model with best hyperparameters...")
        final_hypermodel = VFLHyperModel(
            input_shape=(X_train_final.shape[1],),
            auto_repr_size=auto_repr_size_final,
            digital_repr_size=digital_repr_size_final,
            home_repr_size=home_repr_size_final,
            credit_card_repr_size=credit_card_repr_size_final  # XGBoost 8D
        )
        
        final_model = final_hypermodel.build(best_hp)
        
        logger.info(f"✅ Final model built:")
        logger.info(f"  Architecture: {best_hp.get('num_layers')} hidden layers")
        logger.info(f"  Parameters: {final_model.count_params():,}")
        logger.info(f"  Learning rate: {best_hp.get('learning_rate'):.6f}")
        logger.info(f"  Gradient clipping: {best_hp.get('clipnorm')}")
        
        # Enhanced callbacks for final training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=EARLY_STOPPING_PATIENCE,  # Increased patience
            restore_best_weights=True,
            verbose=2  # Set to 0 to eliminate callback progress printing
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,  # Increased patience
            min_lr=MIN_LEARNING_RATE,  # Lower minimum learning rate
            verbose=2  # Set to 0 to eliminate callback progress printing
        )
        
        # Custom progress callback for epoch-level progress
        class EpochProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                logger.info(f"🔄 Epoch {epoch+1}/{FINAL_EPOCHS} starting...")
            
            def on_epoch_end(self, epoch, logs=None):
                loss = logs.get('loss', 0)
                mae = logs.get('mae', 0)
                val_loss = logs.get('val_loss', 0)
                val_mae = logs.get('val_mae', 0)
                
                # Add prediction variance check to detect if model is learning
                if hasattr(self.model, 'predict'):
                    try:
                        # Get a small sample of predictions to check variance
                        sample_predictions = self.model.predict(X_train_final[:100], verbose=0).flatten()
                        pred_variance = np.var(sample_predictions)
                        pred_range = np.max(sample_predictions) - np.min(sample_predictions)
                        
                        logger.info(f"✅ Epoch {epoch+1}/{FINAL_EPOCHS} completed - Loss: {loss:.4f}, MAE: {mae:.2f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")
                        logger.info(f"   📊 Prediction variance: {pred_variance:.2f}, Range: {pred_range:.2f}")
                        
                        # Warn if predictions are too similar
                        if pred_variance < 1.0:
                            logger.warning(f"   ⚠️  Low prediction variance ({pred_variance:.2f}) - model may not be learning!")
                        if pred_range < 10.0:
                            logger.warning(f"   ⚠️  Small prediction range ({pred_range:.2f}) - model may be stuck!")
                            
                    except Exception as e:
                        logger.warning(f"   ⚠️  Could not check prediction variance: {e}")
                else:
                    logger.info(f"✅ Epoch {epoch+1}/{FINAL_EPOCHS} completed - Loss: {loss:.4f}, MAE: {mae:.2f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}")
        
        final_callbacks = [early_stopping, reduce_lr, EpochProgressCallback()]
        
        # Train final model
        logger.info("")
        logger.info(f"🎯 Starting Phase 2 training ({FINAL_EPOCHS} epochs)...")
        logger.info("=" * 80)
        
        training_start = datetime.now()
        
        history = final_model.fit(
            X_train_final, y_train_final,
            epochs=FINAL_EPOCHS,
            validation_split=0.2,
            batch_size=256,
            callbacks=final_callbacks,
            verbose=2  # Set to 0 to eliminate per-batch progress printing
        )
        
        training_end = datetime.now()
        training_duration = training_end - training_start
        phase2_end = datetime.now()
        phase2_duration = phase2_end - phase2_start
        
        logger.info("=" * 80)
        logger.info(f"✅ Phase 2 training completed in {training_duration}")
        
        # Final evaluation
        test_loss_final, test_mae_final, test_mse_final = final_model.evaluate(X_test_final, y_test_final, verbose=2)
        test_rmse_final = np.sqrt(test_mse_final)
        y_pred_final = final_model.predict(X_test_final, verbose=2)
        
        logger.info("")
        logger.info("📊 FINAL MODEL EVALUATION (XGBoost Credit Card)")
        logger.info("=" * 80)
        logger.info(f"🎯 Final Model Performance:")
        logger.info(f"  Test Loss (Huber): {test_loss_final:.4f}")
        logger.info(f"  Test MAE: {test_mae_final:.2f} points")
        logger.info(f"  Test RMSE: {test_rmse_final:.2f} points")
        
        # Sample predictions for Phase 2
        print_detailed_sample_predictions_with_confidence(X_test_final, y_test_final, ids_test_final, final_model,
                                        auto_repr_size_final, digital_repr_size_final, home_repr_size_final, credit_card_repr_size_final, 
                                        phase="Phase 2 Final")
        
        # Save final model
        logger.info("")
        logger.info("💾 Saving final model and results...")
        
        final_model_path = 'saved_models/vfl_automl_xgboost_final_model.keras'
        final_model.save(final_model_path)
        logger.info(f"✅ Final model saved: {final_model_path}")
        
        # Compare Phase 1 vs Phase 2 performance
        test_mae_phase1 = test_mae
        mae_improvement = test_mae_phase1 - test_mae_final
        mae_improvement_pct = (mae_improvement / test_mae_phase1) * 100
        
        logger.info("")
        logger.info("⚖️  PHASE 1 vs PHASE 2 COMPARISON (XGBoost Credit Card):")
        logger.info("=" * 50)
        logger.info(f"📊 Model Performance Comparison:")
        logger.info(f"  Phase 1 (AutoML): {test_mae_phase1:.2f} MAE on {len(X_test):,} samples")
        logger.info(f"  Phase 2 (Final):  {test_mae_final:.2f} MAE on {len(X_test_final):,} samples")
        logger.info(f"  Improvement: {mae_improvement:+.2f} points ({mae_improvement_pct:+.1f}%)")
        
        # Save comprehensive results
        final_results = {
            'phase1_results': phase1_results,
            'phase2_performance': {
                'test_loss': test_loss_final,
                'test_mae': test_mae_final,
                'test_rmse': test_rmse_final,
                'parameters': final_model.count_params()
            },
            'comparison': {
                'phase1_mae': test_mae_phase1,
                'phase2_mae': test_mae_final,
                'mae_improvement': mae_improvement,
                'mae_improvement_percent': mae_improvement_pct
            },
            'xgboost_configuration': {
                'credit_card_output_dim': XGBOOST_OUTPUT_DIM,
                'pca_random_state': XGBOOST_PCA_RANDOM_STATE,
                'model_type': 'XGBoost Credit Card + 3 Neural Networks'
            },
            'timing': {
                'phase1_duration_seconds': phase1_duration.total_seconds(),
                'phase2_training_duration_seconds': training_duration.total_seconds(),
                'phase2_total_duration_seconds': phase2_duration.total_seconds(),
                'total_duration_seconds': (phase1_duration + phase2_duration).total_seconds()
            },
            'final_architecture': best_hp.values
        }
        
        results_path = 'automl_results/final_xgboost_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"📁 Complete results saved: {results_path}")
        
        total_duration = datetime.now() - start_time
        logger.info("")
        logger.info("🎉 VFL AutoML COMPLETE - XGBoost Credit Card Version")
        logger.info("=" * 80)
        logger.info(f"🏆 FINAL RESULTS SUMMARY:")
        logger.info(f"  Best Architecture MAE: {test_mae_final:.2f} points")
        logger.info(f"  Model Parameters: {final_model.count_params():,}")
        logger.info(f"  Total Duration: {total_duration}")
        logger.info(f"  Training Samples: {len(X_train_final):,}")
        logger.info(f"  Architecture: {best_hp.get('num_layers')} layers")
        logger.info(f"  Credit Card Model: XGBoost ({XGBOOST_OUTPUT_DIM}D representations)")
        logger.info("")
        logger.info(f"📁 Saved Files:")
        logger.info(f"  Final Model: {final_model_path}")
        logger.info(f"  Complete Results: {results_path}")
        logger.info("=" * 80)
        
        return final_model, best_hp, final_results
        
    except Exception as e:
        logger.error(f"❌ Error during AutoML search: {str(e)}")
        raise

# Additional helper functions (same as original but with XGBoost modifications)
def print_detailed_sample_predictions_with_confidence(X_test, y_test, ids_test, model, auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size, phase="Final"):
    """Log detailed sample predictions with confidence scores for the model - MODIFIED FOR XGBOOST"""
    
    logger.info("\n" + "=" * 180)
    if phase == "Phase 1":
        logger.info(f"📊 DETAILED SAMPLE PREDICTIONS WITH CONFIDENCE - {phase} MODEL (AutoML Search Results, XGBoost Credit Card)")
        logger.info(f"📋 Note: These predictions are from the best architecture found during AutoML search")
        logger.info(f"📊 Dataset: {len(y_test):,} test samples from AutoML search dataset")
    else:
        logger.info(f"📊 DETAILED SAMPLE PREDICTIONS WITH CONFIDENCE - FINAL VFL MODEL (3 NN + 1 XGBoost)")
        logger.info(f"📋 Note: These predictions are from the final model trained on larger dataset")
        logger.info(f"📊 Dataset: {len(y_test):,} test samples from final training dataset")
    logger.info("=" * 180)
    logger.info(f"Showing 10 randomly selected customers from test set with actual vs predicted credit scores")
    logger.info(f"Credit score range: 300-850 points")
    logger.info("")
    
    # Select 10 random samples for detailed analysis
    np.random.seed(42)  # For reproducible sample selection
    sample_indices = np.random.choice(len(X_test), size=min(10, len(X_test)), replace=False)
    sample_indices = sorted(sample_indices)
    
    # Get sample data
    y_sample = y_test[sample_indices]
    ids_sample = ids_test[sample_indices]

    # Use the new predict_credit_score_by_tax_id method for each sample
    sample_results = []
    for i, tax_id in enumerate(ids_sample):
        try:
            logger.info("🎯 Example: Single Customer Credit Score Prediction line 2051")
            result = predict_credit_score_by_tax_id(
                tax_id=tax_id,
                model=model,
                auto_repr_size=auto_repr_size,
                digital_repr_size=digital_repr_size,
                home_repr_size=home_repr_size,
                credit_card_repr_size=credit_card_repr_size,
                phase_name=phase
            )
            sample_results.append(result)
        except Exception as e:
            logger.warning(f"⚠️  Could not predict for {tax_id}: {e}")
            # Create a fallback result
            fallback_result = {
                'tax_id': tax_id,
                'predicted_credit_score': 0.0,
                'confidence_score': 0.0,
                'confidence_level': 'Error',
                'prediction_uncertainty': 0.0,
                'confidence_intervals': {
                    '68_percent': {'lower': 0, 'upper': 0, 'width': 0},
                    '95_percent': {'lower': 0, 'upper': 0, 'width': 0}
                },
                'services_available': {'auto_loans': False, 'digital_banking': False, 'home_loans': False, 'credit_card': False},
                'actual_credit_score': y_sample[i],
                'prediction_error': {'absolute_error': 0, 'percentage_error': 0}
            }
            sample_results.append(fallback_result)
    
    # Print header
    logger.info("🎯 SAMPLE PREDICTIONS WITH CONFIDENCE ANALYSIS (XGBoost Credit Card):")
    logger.info("=" * 180)
    logger.info("#   Index   Tax ID          Actual   Predicted  Confidence Uncertainty CI (68%)        CI (95%)        Error    Error%   Services             Conf. Level ")
    logger.info("-" * 180)
    
    # Print each prediction
    for i, result in enumerate(sample_results):
        idx = sample_indices[i]
        tax_id = result['tax_id']
        actual = result['actual_credit_score']
        predicted = result['predicted_credit_score']
        confidence = result['confidence_score']
        uncertainty = result['prediction_uncertainty']
        
        # Format confidence intervals with proper spacing
        ci_68 = f"{result['confidence_intervals']['68_percent']['lower']:.0f}-{result['confidence_intervals']['68_percent']['upper']:.0f}"
        ci_95 = f"{result['confidence_intervals']['95_percent']['lower']:.0f}-{result['confidence_intervals']['95_percent']['upper']:.0f}"
        
        error = result['prediction_error']['absolute_error']
        error_pct = result['prediction_error']['percentage_error']
        
        # Create service label
        services = []
        if result['services_available']['auto_loans']:
            services.append("Auto")
        if result['services_available']['digital_banking']:
            services.append("Digital")
        if result['services_available']['home_loans']:
            services.append("Home")
        if result['services_available']['credit_card']:
            services.append("Credit")
        
        if len(services) == 4:
            service_label = "All Four"
        elif len(services) == 0:
            service_label = "None"
        else:
            service_label = "+".join(services)
        
        confidence_level = result['confidence_level']
        
        # Format and log the prediction with proper spacing
        logger.info(f"{i+1:<4} {idx:<8} {tax_id:<15} {actual:<8.0f} {predicted:<10.1f} {confidence:<10.3f} {uncertainty:<10.1f} {ci_68:<15} {ci_95:<15} {error:<8.1f} {error_pct:<8.1f} {service_label:<20} {confidence_level:<12}")
    
    # Print summary statistics
    logger.info("-" * 180)
    
    # Calculate statistics from successful predictions
    successful_results = [r for r in sample_results if r['predicted_credit_score'] > 0]
    if successful_results:
        predictions = [r['predicted_credit_score'] for r in successful_results]
        actuals = [r['actual_credit_score'] for r in successful_results]
        confidences = [r['confidence_score'] for r in successful_results]
        uncertainties = [r['prediction_uncertainty'] for r in successful_results]
        
        overall_mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        overall_rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        avg_confidence = np.mean(confidences)
        avg_uncertainty = np.mean(uncertainties)
        high_confidence_count = np.sum(np.array(confidences) >= MIN_CONFIDENCE_THRESHOLD)
        
        logger.info(f"📊 SAMPLE STATISTICS ({len(successful_results)} successful predictions, XGBoost Credit Card):")
        logger.info(f"   MAE: {overall_mae:.2f} points  |  RMSE: {overall_rmse:.2f} points  |  Avg Confidence: {avg_confidence:.3f}")
        logger.info(f"   Avg Uncertainty: {avg_uncertainty:.1f} points  |  High Confidence (≥{MIN_CONFIDENCE_THRESHOLD}): {high_confidence_count}/{len(successful_results)} customers")
        
        # Service distribution in sample
        service_counts = {'Auto': 0, 'Digital': 0, 'Home': 0, 'Credit': 0, 'All Four': 0, 'None': 0}
        for result in successful_results:
            services = []
            if result['services_available']['auto_loans']:
                services.append("Auto")
            if result['services_available']['digital_banking']:
                services.append("Digital")
            if result['services_available']['home_loans']:
                services.append("Home")
            if result['services_available']['credit_card']:
                services.append("Credit")
            
            if len(services) == 4:
                service_counts['All Four'] += 1
            elif len(services) == 0:
                service_counts['None'] += 1
            else:
                for service in services:
                    service_counts[service] += 1
        
        service_dist_str = " | ".join([f"{k}: {v}" for k, v in service_counts.items() if v > 0])
        logger.info(f"   Service Distribution: {service_dist_str}")
    else:
        logger.warning("⚠️  No successful predictions to calculate statistics")
    
    logger.info("=" * 180)
    logger.info(f"🏦 Bank Model Types: Auto(NN-16D), Digital(NN-8D), Home(NN-16D), Credit(XGBoost-{credit_card_repr_size}D)")
    logger.info("=" * 180)
    
    # Log confidence distribution
    if successful_results and ENABLE_CONFIDENCE_SCORES:
        confidences = [r['confidence_score'] for r in successful_results]
        very_high = np.sum(np.array(confidences) >= 0.9)
        high = np.sum((np.array(confidences) >= 0.8) & (np.array(confidences) < 0.9))
        medium = np.sum((np.array(confidences) >= 0.7) & (np.array(confidences) < 0.8))
        low = np.sum((np.array(confidences) >= 0.6) & (np.array(confidences) < 0.7))
        very_low = np.sum(np.array(confidences) < 0.6)
        
        logger.info(f"📈 CONFIDENCE DISTRIBUTION:")
        logger.info(f"   Very High (≥0.9): {very_high}  |  High (0.8-0.9): {high}  |  Medium (0.7-0.8): {medium}  |  Low (0.6-0.7): {low}  |  Very Low (<0.6): {very_low}")
        logger.info("=" * 180)

def evaluate_model_with_confidence(model, X_test, y_test, ids_test, phase_name="Final"):
    """
    Evaluate model performance including confidence-based metrics
    """
    logger.info(f"🎯 Evaluating {phase_name} model with confidence scores...")
    
    # Calculate confidence scores if enabled
    if ENABLE_CONFIDENCE_SCORES:
        conf_results = calculate_confidence_scores(model, X_test)
        predictions = conf_results['predictions']
        confidence_scores = conf_results['confidence_scores']
        prediction_std = conf_results['prediction_std']
        confidence_intervals = conf_results['confidence_intervals']
        confidence_categories = conf_results['confidence_categories']
    else:
        predictions = model.predict(X_test, verbose=2).flatten()
        confidence_scores = np.ones_like(predictions) * 0.5  # Neutral confidence
        prediction_std = np.zeros_like(predictions)
        confidence_intervals = {}
        confidence_categories = np.array(['N/A'] * len(predictions))
    
    # Standard metrics
    mae = np.mean(np.abs(y_test - predictions))
    rmse = np.sqrt(np.mean((y_test - predictions)**2))
    
    # Confidence-based metrics
    high_conf_mask = confidence_scores >= MIN_CONFIDENCE_THRESHOLD
    if np.sum(high_conf_mask) > 0:
        high_conf_mae = np.mean(np.abs(y_test[high_conf_mask] - predictions[high_conf_mask]))
        high_conf_coverage = np.sum(high_conf_mask) / len(predictions)
    else:
        high_conf_mae = float('inf')
        high_conf_coverage = 0.0
    
    # Calibration metrics (for confidence intervals)
    calibration_scores = {}
    if confidence_intervals:
        for conf_level, intervals in confidence_intervals.items():
            # Check if actual values fall within predicted intervals
            in_interval = (y_test >= intervals['lower']) & (y_test <= intervals['upper'])
            actual_coverage = np.mean(in_interval)
            expected_coverage = int(conf_level.replace('%', '')) / 100
            calibration_scores[conf_level] = {
                'actual_coverage': actual_coverage,
                'expected_coverage': expected_coverage,
                'calibration_error': abs(actual_coverage - expected_coverage)
            }
    
    results = {
        'predictions': predictions,
        'confidence_scores': confidence_scores,
        'prediction_std': prediction_std,
        'confidence_intervals': confidence_intervals,
        'confidence_categories': confidence_categories,
        'mae': mae,
        'rmse': rmse,
        'high_confidence_mae': high_conf_mae,
        'high_confidence_coverage': high_conf_coverage,
        'calibration_scores': calibration_scores
    }
    
    # Log performance summary
    logger.info(f"📊 {phase_name} Model Performance with Confidence:")
    logger.info(f"   Overall MAE: {mae:.2f} points")
    logger.info(f"   Overall RMSE: {rmse:.2f} points")
    logger.info(f"   High Confidence MAE: {high_conf_mae:.2f} points")
    logger.info(f"   High Confidence Coverage: {high_conf_coverage:.1%}")
    
    if calibration_scores:
        logger.info(f"📏 Confidence Interval Calibration:")
        for conf_level, scores in calibration_scores.items():
            logger.info(f"   {conf_level}: {scores['actual_coverage']:.1%} actual vs {scores['expected_coverage']:.1%} expected (error: {scores['calibration_error']:.3f})")
    
    return results

def calculate_confidence_scores(model, X_data, enable_mc_dropout=True, mc_samples=MC_DROPOUT_SAMPLES):
    """
    Calculate confidence scores for predictions using multiple uncertainty estimation methods
    
    Args:
        model: Trained Keras model
        X_data: Input data for prediction
        enable_mc_dropout: Whether to use Monte Carlo Dropout
        mc_samples: Number of MC dropout samples
    
    Returns:
        dict containing predictions and various confidence metrics
    """
    
    logger.info(f"🔮 Calculating confidence scores for {len(X_data):,} samples...")
    logger.info(f"   Monte Carlo Dropout: {'✅ Enabled' if enable_mc_dropout else '❌ Disabled'}")
    logger.info(f"   MC Samples: {mc_samples}")
    
    results = {
        'predictions': [],
        'confidence_scores': [],
        'prediction_std': [],
        'confidence_intervals': {},
        'epistemic_uncertainty': [],
        'prediction_entropy': [],
        'confidence_categories': []
    }
    
    if enable_mc_dropout:
        # Monte Carlo Dropout for uncertainty estimation
        logger.debug("Running Monte Carlo Dropout inference...")
        
        # Enable dropout during inference by creating a function that keeps training mode
        mc_model = tf.keras.Model(model.inputs, model.outputs)
        
        # Store original training mode
        original_training = []
        for layer in mc_model.layers:
            if hasattr(layer, 'training'):
                original_training.append(getattr(layer, 'training', None))
        
        # Collect predictions from multiple forward passes
        mc_predictions = []
        
        for i in range(mc_samples):
            if i % 10 == 0:
                logger.debug(f"   MC sample {i+1}/{mc_samples}")
            
            # Force dropout to be active (training mode)
            for layer in mc_model.layers:
                if 'dropout' in layer.name.lower():
                    layer.training = True
            
            # Get prediction with dropout active
            pred = mc_model(X_data, training=True)
            mc_predictions.append(pred.numpy().flatten())
        
        # Restore original training mode
        for i, layer in enumerate(mc_model.layers):
            if i < len(original_training) and hasattr(layer, 'training'):
                layer.training = original_training[i]
        
        mc_predictions = np.array(mc_predictions)  # Shape: (mc_samples, n_samples)
        
        # Calculate statistics
        mean_predictions = np.mean(mc_predictions, axis=0)
        prediction_std = np.std(mc_predictions, axis=0)
        
        # Debug logging for confidence calculation
        logger.debug(f"   MC Predictions shape: {mc_predictions.shape}")
        logger.debug(f"   Mean predictions range: [{mean_predictions.min():.2f}, {mean_predictions.max():.2f}]")
        logger.debug(f"   Prediction std range: [{prediction_std.min():.2f}, {prediction_std.max():.2f}]")
        
        # FIXED CONFIDENCE INTERVAL CALCULATION
        # Calculate intervals relative to the mean prediction using standard deviations
        # This ensures the prediction (mean) is always inside the confidence intervals
        for confidence_level in CONFIDENCE_INTERVALS:
            if confidence_level == 68:
                # 68% CI = mean ± 1σ
                margin = prediction_std
            elif confidence_level == 95:
                # 95% CI = mean ± 2σ
                margin = 2.0 * prediction_std
            else:
                # For other confidence levels, use z-score approximation
                z_score = 1.96 if confidence_level == 95 else 1.0  # Default to 68%
                margin = z_score * prediction_std
            
            lower_bound = mean_predictions - margin
            upper_bound = mean_predictions + margin
            
            results['confidence_intervals'][f'{confidence_level}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        # Debug logging to verify the calculation
        logger.debug(f"   Confidence interval calculation verification:")
        logger.debug(f"   Mean predictions: {mean_predictions[:5]}")  # First 5 samples
        logger.debug(f"   Prediction std: {prediction_std[:5]}")     # First 5 samples
        for conf_level in CONFIDENCE_INTERVALS:
            interval = results['confidence_intervals'][f'{conf_level}%']
            logger.debug(f"   {conf_level}% CI bounds: {interval['lower'][:5]} to {interval['upper'][:5]}")
            # Verify that mean is inside the interval
            is_inside = np.all((mean_predictions >= interval['lower']) & (mean_predictions <= interval['upper']))
            logger.debug(f"   {conf_level}% CI: Mean inside interval = {is_inside}")
        
        # Debug logging for confidence intervals
        logger.debug(f"   Confidence intervals calculated:")
        for conf_level in CONFIDENCE_INTERVALS:
            interval = results['confidence_intervals'][f'{conf_level}%']
            logger.debug(f"     {conf_level}% CI: [{interval['lower'].min():.1f}, {interval['upper'].max():.1f}] (width: {interval['width'].mean():.1f})")
        
        # IMPROVED CONFIDENCE SCORE CALCULATION
        # Use sigmoid-based confidence that maps uncertainty to [0.1, 0.95] range
        # This prevents 0.0 confidence scores and provides more reasonable values
        
        # Normalize uncertainty to [0, 1] range, but cap it to prevent extreme values
        max_std = np.max(prediction_std) if np.max(prediction_std) > 0 else 1.0
        min_std = np.min(prediction_std) if np.min(prediction_std) > 0 else 0.0
        
        # Use a more robust normalization
        if max_std > min_std:
            normalized_uncertainty = (prediction_std - min_std) / (max_std - min_std)
            # Cap to prevent extreme values
            normalized_uncertainty = np.clip(normalized_uncertainty, 0.0, 1.0)
        else:
            normalized_uncertainty = np.ones_like(prediction_std) * 0.5
        
        # Use sigmoid to map uncertainty to confidence [0.1, 0.95]
        # Lower uncertainty = higher confidence
        confidence_scores = 0.1 + 0.85 * (1.0 / (1.0 + np.exp(3.0 * (normalized_uncertainty - 0.5))))
        
        # Alternative: Use coefficient of variation for more interpretable confidence
        # cv = prediction_std / (np.abs(mean_predictions) + 1e-8)
        # confidence_scores = np.clip(1.0 / (1.0 + cv), 0.1, 0.95)
        
        logger.debug(f"   Normalized uncertainty range: [{normalized_uncertainty.min():.3f}, {normalized_uncertainty.max():.3f}]")
        logger.debug(f"   Confidence scores range: [{confidence_scores.min():.3f}, {confidence_scores.max():.3f}]")
        
        # Calculate epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = prediction_std
        
        # Calculate prediction entropy (for regression, we use coefficient of variation)
        prediction_entropy = prediction_std / (np.abs(mean_predictions) + 1e-8)
        
        results.update({
            'predictions': mean_predictions,
            'confidence_scores': confidence_scores,
            'prediction_std': prediction_std,
            'epistemic_uncertainty': epistemic_uncertainty,
            'prediction_entropy': prediction_entropy
        })
        
    else:
        # Standard single prediction
        logger.debug("Running standard inference...")
        predictions = model.predict(X_data, verbose=2).flatten()
        
        # For standard inference, use a simple heuristic for confidence
        # Based on distance from mean prediction
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Confidence based on how close prediction is to the mean
        if std_pred > 0:
            normalized_distance = np.abs(predictions - mean_pred) / std_pred
            confidence_scores = np.exp(-normalized_distance / 2)  # Gaussian-like confidence
            # Ensure confidence is in reasonable range
            confidence_scores = np.clip(confidence_scores, 0.1, 0.95)
        else:
            confidence_scores = np.ones_like(predictions) * 0.5  # Neutral confidence
        
        results.update({
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'prediction_std': np.full_like(predictions, std_pred),
            'epistemic_uncertainty': np.full_like(predictions, std_pred),
            'prediction_entropy': np.full_like(predictions, 0.0)
        })
    
    # Categorize confidence levels
    confidence_categories = []
    for conf in results['confidence_scores']:
        if conf >= 0.9:
            confidence_categories.append('Very High')
        elif conf >= 0.8:
            confidence_categories.append('High')
        elif conf >= 0.7:
            confidence_categories.append('Medium')
        elif conf >= 0.6:
            confidence_categories.append('Low')
        else:
            confidence_categories.append('Very Low')
    
    results['confidence_categories'] = np.array(confidence_categories)
    
    # Log summary statistics
    logger.info(f"✅ Confidence scores calculated:")
    logger.info(f"   Mean Confidence: {np.mean(results['confidence_scores']):.3f}")
    logger.info(f"   Std Confidence: {np.std(results['confidence_scores']):.3f}")
    logger.info(f"   Min Confidence: {np.min(results['confidence_scores']):.3f}")
    logger.info(f"   Max Confidence: {np.max(results['confidence_scores']):.3f}")
    
    # Confidence distribution
    conf_dist = pd.Series(confidence_categories).value_counts()
    logger.info(f"📊 Confidence Distribution:")
    for category, count in conf_dist.items():
        percentage = (count / len(confidence_categories)) * 100
        logger.info(f"   {category}: {count} ({percentage:.1f}%)")
    
    return results

def predict_credit_score_by_tax_id(tax_id, model=None, auto_scaler=None, digital_scaler=None, home_scaler=None, 
                                  auto_repr_size=None, digital_repr_size=None, home_repr_size=None, 
                                  credit_card_repr_size=None, phase_name="Model"):
    """
    Predict credit score for a single customer by tax ID with confidence intervals.
    
    Args:
        tax_id (str): Customer tax ID
        model: Trained VFL model (if None, will load the saved model)
        auto_scaler, digital_scaler, home_scaler: Fitted scalers for each bank
        auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size: Representation sizes
        phase_name (str): Name for logging (e.g., "Phase 1", "Final Model")
    
    Returns:
        dict: Prediction results with confidence intervals
    """
    logger.info(f"🎯 Predicting credit score for tax ID: {tax_id} using {phase_name}")
    
    try:
        # Load model if not provided
        if model is None:
            model_path = 'saved_models/vfl_automl_xgboost_final_model.keras'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")
            
            # Try to load with custom objects for Lambda layers
            try:
                model = load_model(model_path, compile=False)
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Could not load model directly: {e}")
                logger.info("🔄 Trying to load with custom objects...")
                
                def custom_lambda(x):
                    return x
                
                custom_objects = {
                    'custom_lambda': custom_lambda,
                    'lambda': custom_lambda
                }
                
                try:
                    model = load_model(model_path, compile=False, custom_objects=custom_objects)
                    logger.info("✅ Model loaded with custom objects")
                except Exception as e2:
                    raise Exception(f"Failed to load model: {e2}")
        
        # Load client models for representation extraction
        auto_loans_model, digital_bank_model, home_loans_model, credit_card_model = load_client_models()
        
        # Get representation sizes if not provided
        if auto_repr_size is None:
            auto_extractor = get_penultimate_layer_model(auto_loans_model)
            auto_repr_size = auto_extractor.output_shape[1]
        if digital_repr_size is None:
            digital_extractor = get_penultimate_layer_model(digital_bank_model)
            digital_repr_size = digital_extractor.output_shape[1]
        if home_repr_size is None:
            home_extractor = get_penultimate_layer_model(home_loans_model)
            home_repr_size = home_extractor.output_shape[1]
        if credit_card_repr_size is None:
            credit_card_repr_size = XGBOOST_OUTPUT_DIM
        
        # Load customer data
        logger.info(f"🔄 Loading customer data for tax ID: {tax_id}")
        
        # Load datasets
        auto_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/auto_loans_bank.csv')
        digital_bank_df = pd.read_csv('VFLClientModels/dataset/data/banks/digital_savings_bank.csv')
        home_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/home_loans_bank.csv')
        credit_card_df = pd.read_csv('VFLClientModels/dataset/data/banks/credit_card_bank.csv')
        master_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
        
        # Find customer in each dataset
        customer_data = {}
        customer_data['auto_loans'] = auto_loans_df[auto_loans_df['tax_id'] == tax_id].iloc[0].to_dict() if not auto_loans_df[auto_loans_df['tax_id'] == tax_id].empty else None
        customer_data['digital_banking'] = digital_bank_df[digital_bank_df['tax_id'] == tax_id].iloc[0].to_dict() if not digital_bank_df[digital_bank_df['tax_id'] == tax_id].empty else None
        customer_data['home_loans'] = home_loans_df[home_loans_df['tax_id'] == tax_id].iloc[0].to_dict() if not home_loans_df[home_loans_df['tax_id'] == tax_id].empty else None
        customer_data['credit_card'] = credit_card_df[credit_card_df['tax_id'] == tax_id].iloc[0].to_dict() if not credit_card_df[credit_card_df['tax_id'] == tax_id].empty else None
        customer_data['master'] = master_df[master_df['tax_id'] == tax_id].iloc[0].to_dict() if not master_df[master_df['tax_id'] == tax_id].empty else None
        
        # Check if customer exists
        services_found = sum(1 for service, data in customer_data.items() if data is not None and service != 'master')
        if services_found == 0:
            raise ValueError(f"Customer {tax_id} not found in any dataset")
        
        logger.info(f"✅ Customer found in {services_found} bank services")
        
        # Extract representations
        logger.info("🔄 Extracting bank representations...")
        
        # Initialize representations
        auto_repr = np.zeros((1, auto_repr_size))
        digital_repr = np.zeros((1, digital_repr_size))
        home_repr = np.zeros((1, home_repr_size))
        credit_card_repr = np.zeros((1, credit_card_repr_size))
        
        # Initialize masks
        auto_mask = np.array([[0.0]])
        digital_mask = np.array([[0.0]])
        home_mask = np.array([[0.0]])
        credit_card_mask = np.array([[0.0]])
        
        # Extract auto loans representations
        if customer_data['auto_loans'] is not None:
            if auto_scaler is None:
                raise ValueError("auto_scaler must be provided for inference for customers with auto loans.")
            auto_extractor = get_penultimate_layer_model(auto_loans_model)
            auto_df = pd.DataFrame([customer_data['auto_loans']])
            auto_features = [
                'annual_income', 'credit_score', 'payment_history', 'employment_length', 
                'debt_to_income_ratio', 'age', 'credit_history_length', 'num_credit_cards', 
                'num_loan_accounts', 'total_credit_limit', 'credit_utilization_ratio', 
                'late_payments', 'credit_inquiries', 'last_late_payment_days',
                'current_debt', 'monthly_expenses', 'savings_balance', 
                'checking_balance', 'investment_balance', 'auto_loan_balance', 'mortgage_balance'
            ]
            feature_data = auto_df[auto_features].copy()
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
            X_scaled = auto_scaler.transform(feature_data)
            auto_repr = auto_extractor.predict(X_scaled, verbose=2)
            auto_mask = np.array([[1.0]])
            logger.info("   ✅ Auto loans representations extracted")
        
        # Extract digital banking representations
        if customer_data['digital_banking'] is not None:
            if digital_scaler is None:
                raise ValueError("digital_scaler must be provided for inference for customers with digital banking.")
            digital_extractor = get_penultimate_layer_model(digital_bank_model)
            digital_df = pd.DataFrame([customer_data['digital_banking']])
            digital_features = [
                'annual_income', 'savings_balance', 'checking_balance', 'investment_balance',
                'payment_history', 'credit_score', 'age', 'employment_length',
                'avg_monthly_transactions', 'avg_transaction_value', 'digital_banking_score',
                'mobile_banking_usage', 'online_transactions_ratio', 'international_transactions_ratio',
                'e_statement_enrolled', 'monthly_expenses', 'total_credit_limit', 
                'credit_utilization_ratio', 'num_credit_cards', 'credit_history_length',
                'current_debt', 'mortgage_balance', 'total_wealth', 'net_worth', 
                'credit_efficiency', 'financial_stability_score'
            ]
            feature_data = digital_df[digital_features].copy()
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
            X_scaled = digital_scaler.transform(feature_data)
            digital_repr = digital_extractor.predict(X_scaled, verbose=2)
            digital_mask = np.array([[1.0]])
            logger.info("   ✅ Digital banking representations extracted")
        
        # Extract home loans representations
        if customer_data['home_loans'] is not None:
            if home_scaler is None:
                raise ValueError("home_scaler must be provided for inference for customers with home loans.")
            home_extractor = get_penultimate_layer_model(home_loans_model)
            home_df = pd.DataFrame([customer_data['home_loans']])
            home_features = [
                'annual_income', 'credit_score', 'payment_history', 'employment_length',
                'debt_to_income_ratio', 'age', 'credit_history_length', 'num_credit_cards',
                'num_loan_accounts', 'total_credit_limit', 'credit_utilization_ratio',
                'late_payments', 'credit_inquiries', 'last_late_payment_days',
                'current_debt', 'monthly_expenses', 'savings_balance', 'checking_balance',
                'investment_balance', 'mortgage_balance', 'auto_loan_balance',
                'estimated_property_value', 'required_down_payment', 'available_down_payment_funds',
                'mortgage_risk_score', 'loan_to_value_ratio', 'min_down_payment_pct',
                'interest_rate', 'dti_after_mortgage'
            ]
            feature_data = home_df[home_features].copy()
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
            X_scaled = home_scaler.transform(feature_data)
            home_repr = home_extractor.predict(X_scaled, verbose=2)
            home_mask = np.array([[1.0]])
            logger.info("   ✅ Home loans representations extracted")
        
        # Extract XGBoost credit card representations
        if customer_data['credit_card'] is not None:
            credit_card_df = pd.DataFrame([customer_data['credit_card']])
            credit_card_repr = extract_xgboost_representations(
                credit_card_model, credit_card_df, XGBOOST_OUTPUT_DIM
            )
            credit_card_mask = np.array([[1.0]])
            logger.info("   ✅ Credit card (XGBoost) representations extracted")
        
        # Combine representations
        X_combined = np.concatenate([
            auto_repr, auto_mask, digital_repr, digital_mask,
            home_repr, home_mask, credit_card_repr, credit_card_mask
        ], axis=1)
        
        logger.info(f"✅ Combined feature vector shape: {X_combined.shape}")
        
        # Make prediction
        logger.info("🔄 Making credit score prediction...")
        
        # Calculate confidence intervals and get prediction
        if ENABLE_CONFIDENCE_SCORES:
            logger.info("🔄 Calculating confidence intervals with MC Dropout...")
            conf_results = calculate_confidence_scores(
                model, X_combined, 
                enable_mc_dropout=True, 
                mc_samples=MC_DROPOUT_SAMPLES
            )
            
            # Use MC Dropout mean as the final prediction
            prediction = conf_results['predictions'][0]
            confidence_score = conf_results['confidence_scores'][0]
            prediction_std = conf_results['prediction_std'][0]
            confidence_intervals = conf_results['confidence_intervals']
            
            # Debug logging for confidence intervals
            logger.debug(f"   MC Dropout mean prediction: {prediction:.1f}")
            logger.debug(f"   Prediction std: {prediction_std:.1f}")
            logger.debug(f"   Confidence intervals keys: {list(confidence_intervals.keys())}")
            logger.debug(f"   68% CI from calculate_confidence_scores: {confidence_intervals.get('68%', 'Not found')}")
            logger.debug(f"   95% CI from calculate_confidence_scores: {confidence_intervals.get('95%', 'Not found')}")
            
            ci_68 = confidence_intervals.get('68%', {})
            ci_95 = confidence_intervals.get('95%', {})
            
            # Extract confidence intervals correctly
            if 'lower' in ci_68 and 'upper' in ci_68:
                ci_68_lower = ci_68['lower'][0] if hasattr(ci_68['lower'], '__getitem__') else ci_68['lower']
                ci_68_upper = ci_68['upper'][0] if hasattr(ci_68['upper'], '__getitem__') else ci_68['upper']
            else:
                # Fallback to standard deviation-based intervals
                ci_68_lower = prediction - prediction_std
                ci_68_upper = prediction + prediction_std
            
            if 'lower' in ci_95 and 'upper' in ci_95:
                ci_95_lower = ci_95['lower'][0] if hasattr(ci_95['lower'], '__getitem__') else ci_95['lower']
                ci_95_upper = ci_95['upper'][0] if hasattr(ci_95['upper'], '__getitem__') else ci_95['upper']
            else:
                # Fallback to standard deviation-based intervals
                ci_95_lower = prediction - 2*prediction_std
                ci_95_upper = prediction + 2*prediction_std
        else:
            # Standard single prediction when confidence scores are disabled
            prediction = model.predict(X_combined, verbose=2).flatten()[0]
            confidence_score = 0.8
            prediction_std = 15.0
            ci_68_lower = prediction - 15
            ci_68_upper = prediction + 15
            ci_95_lower = prediction - 30
            ci_95_upper = prediction + 30
        
        # Get actual credit score if available
        actual_score = None
        if customer_data['master'] is not None:
            actual_score = customer_data['master'].get('credit_score')
        
        # Determine confidence level
        if confidence_score >= 0.9:
            confidence_level = "Very High"
        elif confidence_score >= 0.8:
            confidence_level = "High"
        elif confidence_score >= 0.7:
            confidence_level = "Medium"
        elif confidence_score >= 0.6:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        # Create service availability summary
        services_available = {
            'auto_loans': bool(auto_mask[0, 0]),
            'digital_banking': bool(digital_mask[0, 0]),
            'home_loans': bool(home_mask[0, 0]),
            'credit_card': bool(credit_card_mask[0, 0])
        }
        
        # Prepare results
        results = {
            'tax_id': tax_id,
            'predicted_credit_score': round(prediction, 1),
            'confidence_score': round(confidence_score, 3),
            'confidence_level': confidence_level,
            'prediction_uncertainty': round(prediction_std, 1),
            'confidence_intervals': {
                '68_percent': {
                    'lower': round(ci_68_lower, 1),
                    'upper': round(ci_68_upper, 1),
                    'width': round(ci_68_upper - ci_68_lower, 1)
                },
                '95_percent': {
                    'lower': round(ci_95_lower, 1),
                    'upper': round(ci_95_upper, 1),
                    'width': round(ci_95_upper - ci_95_lower, 1)
                }
            },
            'services_available': services_available,
            'actual_credit_score': actual_score,
            'prediction_timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_type': 'VFL AutoML XGBoost (3 NN + 1 XGBoost)',
                'representation_sizes': {
                    'auto': auto_repr_size,
                    'digital': digital_repr_size,
                    'home': home_repr_size,
                    'credit_card': credit_card_repr_size
                }
            }
        }
        
        # Calculate error if actual score is available
        if actual_score is not None:
            error = abs(prediction - actual_score)
            error_percentage = (error / actual_score) * 100
            results['prediction_error'] = {
                'absolute_error': round(error, 1),
                'percentage_error': round(error_percentage, 1)
            }
        
        logger.info("✅ Credit score prediction completed")
        logger.info(f"   Predicted Score: {results['predicted_credit_score']}")
        logger.info(f"   Confidence: {results['confidence_level']} ({results['confidence_score']})")
        logger.info(f"   68% CI: {results['confidence_intervals']['68_percent']['lower']} - {results['confidence_intervals']['68_percent']['upper']}")
        logger.info(f"   95% CI: {results['confidence_intervals']['95_percent']['lower']} - {results['confidence_intervals']['95_percent']['upper']}")
        
        # --- DIAGNOSTIC: Compare inference features to training features for this tax_id ---
        # Reconstruct the feature vector as in training
        # [auto_repr | auto_mask | digital_repr | digital_mask | home_repr | home_mask | credit_card_repr | credit_card_mask]
        feature_vector = np.concatenate([
            auto_repr, auto_mask, digital_repr, digital_mask, home_repr, home_mask, credit_card_repr, credit_card_mask
        ], axis=1)
        print(f"[DIAGNOSTIC] Inference feature vector for {tax_id}:\n", feature_vector)
        # Try to find this tax_id in the training set (X_train, ids_train must be available in scope or loaded)
        try:
            # Load training data (small sample for speed)
            _, _, _, _, ids_train, _, _, _, _, _, _, _, _, _, X_combined, _, ids_combined = load_and_preprocess_data(1000)
            idx = np.where(ids_combined == tax_id)[0]
            if len(idx) > 0:
                train_feat = X_combined[idx[0]]
                print(f"[DIAGNOSTIC] Training feature vector for {tax_id}:\n", train_feat)
                print(f"[DIAGNOSTIC] Difference (inference - training):\n", feature_vector.flatten() - train_feat)
            else:
                print(f"[DIAGNOSTIC] Tax ID {tax_id} not found in training data sample.")
        except Exception as e:
            print(f"[DIAGNOSTIC] Could not compare to training features: {e}")
        # --- END DIAGNOSTIC ---
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error predicting credit score for {tax_id}: {str(e)}")
        raise

def example_single_customer_prediction():
    """
    Example function showing how to use predict_credit_score_by_tax_id for a single customer.
    This can be called from other files to get predictions for any tax ID.
    Loads fitted scalers from disk for correct inference.
    """
    logger.info("🎯 Example: Single Customer Credit Score Prediction")
    logger.info("=" * 60)
    
    # Example tax ID (replace with actual tax ID from your dataset)
    example_tax_id = "TAX001"
    
    try:
        # Load fitted scalers for inference
        import joblib
        auto_scaler = joblib.load('VFLClientModels/models/saved_models/auto_scaler.pkl')
        digital_scaler = joblib.load('VFLClientModels/models/saved_models/digital_scaler.pkl')
        home_scaler = joblib.load('VFLClientModels/models/saved_models/home_scaler.pkl')
        
        logger.info("🎯 Example: Single Customer Credit Score Prediction= line 2850")
        # Make prediction using the new method
        result = predict_credit_score_by_tax_id(
            tax_id=example_tax_id,
            auto_scaler=auto_scaler,
            digital_scaler=digital_scaler,
            home_scaler=home_scaler,
            phase_name="Example Prediction"
        )
        
        # Display results
        logger.info(f"✅ Prediction completed for {example_tax_id}")
        logger.info(f"📊 Results:")
        logger.info(f"   Predicted Credit Score: {result['predicted_credit_score']} points")
        logger.info(f"   Confidence Level: {result['confidence_level']} ({result['confidence_score']})")
        logger.info(f"   Prediction Uncertainty: ±{result['prediction_uncertainty']} points")
        logger.info(f"   68% Confidence Interval: {result['confidence_intervals']['68_percent']['lower']} - {result['confidence_intervals']['68_percent']['upper']} points")
        logger.info(f"   95% Confidence Interval: {result['confidence_intervals']['95_percent']['lower']} - {result['confidence_intervals']['95_percent']['upper']} points")
        
        # Show available services
        services = [k for k, v in result['services_available'].items() if v]
        logger.info(f"   Available Services: {services}")
        
        # Show actual vs predicted if available
        if result['actual_credit_score'] is not None:
            logger.info(f"   Actual Credit Score: {result['actual_credit_score']} points")
            logger.info(f"   Prediction Error: {result['prediction_error']['absolute_error']} points ({result['prediction_error']['percentage_error']}%)")
        
        logger.info("=" * 60)
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in example prediction: {str(e)}")
        logger.info("💡 Make sure to replace 'TAX001' with an actual tax ID from your dataset")
        return None

# Add this helper function near the top of the file:
def fit_and_save_xgboost_pca(leaf_indices, target_dim):
    """
    Fit PCA on XGBoost leaf indices and save it for inference.
    Args:
        leaf_indices: numpy array of shape (n_samples, n_leaves)
        target_dim: int, number of PCA components
    """
    from sklearn.decomposition import PCA
    import joblib
    pca = PCA(n_components=target_dim, random_state=XGBOOST_PCA_RANDOM_STATE)
    pca.fit(leaf_indices.astype(np.float32))
    pca_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_pca.pkl'
    joblib.dump(pca, pca_path)
    logger.info(f"   💾 Saved PCA to {pca_path}")
    return pca

if __name__ == "__main__":
    run_automl_search()