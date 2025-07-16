import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import keras_tuner as kt
from datetime import datetime
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
import glob

# ============================================================================
# VFL AutoML Model - Four Bank System Integration
# 
# This system now integrates FOUR banks:
# 1. Auto Loans Bank (regression model)
# 2. Digital Banking Bank (classification model) 
# 3. Home Loans Bank (regression model)
# 4. Credit Card Bank (classification model - NEW!)
#
# Each bank contributes 8-dimensional representations from their penultimate
# layers, plus service availability masks, for federated credit scoring.
# 
# NEW: Confidence Score Implementation
# - Monte Carlo Dropout for uncertainty estimation
# - Prediction intervals for confidence bounds
# - Multiple confidence metrics for robust uncertainty quantification
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
    logger = logging.getLogger('VFL_AutoML_FourBanks')
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
        f'logs/vfl_automl_four_banks_{timestamp}.log',
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
    logger.info("VFL AutoML Four Banks Comprehensive Logging System Initialized")
    logger.info(f"Log file: logs/vfl_automl_four_banks_{timestamp}.log")
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
AUTOML_SAMPLE_SIZE = 10000        # Number of customers for AutoML search (increased for stable stratification)
FINAL_SAMPLE_SIZE = 200000       # Number of customers for final model training (larger for accuracy)
RANDOM_SEED = 42                 # For reproducible results

# AutoML Search Configuration
MAX_TRIALS = 2                  # Number of different architectures to try
EXECUTIONS_PER_TRIAL = 1         # Number of times to train each architecture
EPOCHS_PER_TRIAL = 25            # Max epochs for each trial (reduced for faster search)
FINAL_EPOCHS = 100               # Epochs for final model training with full data
SEARCH_OBJECTIVE = 'val_mae'     # Objective to optimize ('val_loss', 'val_mae', etc.)

# Search Space Ranges
MIN_LAYERS = 2      # Minimum number of hidden layers
MAX_LAYERS = 16     # Maximum number of hidden layers (reduced for faster search)
MIN_UNITS = 8      # Minimum units per layer
MAX_UNITS = 4096     # Maximum units per layer (reduced for faster search)

# Confidence Score Configuration
ENABLE_CONFIDENCE_SCORES = True      # Enable confidence score calculation
MC_DROPOUT_SAMPLES = 30              # Number of Monte Carlo dropout samples for uncertainty
CONFIDENCE_INTERVALS = [68, 95]      # Confidence intervals to calculate (68% = 1σ, 95% = 2σ)
MIN_CONFIDENCE_THRESHOLD = 0.7       # Minimum confidence threshold for reliable predictions

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

def load_client_models():
    """Load the trained client models"""
    logger.info("🔄 Loading client models...")
    
    try:
        logger.debug("Loading auto loans model...")
        auto_loans_model = load_model('saved_models/auto_loans_model.keras', compile=False)
        logger.debug(f"Auto loans model loaded: input shape {auto_loans_model.input_shape}")
        
        logger.debug("Loading digital bank model...")
        digital_bank_model = load_model('saved_models/digital_bank_model.keras', compile=False)
        logger.debug(f"Digital bank model loaded: input shape {digital_bank_model.input_shape}")
        
        logger.debug("Loading home loans model...")
        home_loans_model = load_model('saved_models/home_loans_model.keras', compile=False)
        logger.debug(f"Home loans model loaded: input shape {home_loans_model.input_shape}")
        
        logger.debug("Loading credit card model...")
        credit_card_model = load_model('saved_models/credit_card_model.keras', compile=False)
        logger.debug(f"Credit card model loaded: input shape {credit_card_model.input_shape}")
        
        logger.info("✅ All client models loaded successfully")
        logger.info(f"Auto model input shape: {auto_loans_model.input_shape}")
        logger.info(f"Digital model input shape: {digital_bank_model.input_shape}")
        logger.info(f"Home loans model input shape: {home_loans_model.input_shape}")
        logger.info(f"Credit card model input shape: {credit_card_model.input_shape}")
        
        return auto_loans_model, digital_bank_model, home_loans_model, credit_card_model
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error("Please ensure all models exist at:")
        logger.error("- saved_models/auto_loans_model.keras")
        logger.error("- saved_models/digital_bank_model.keras") 
        logger.error("- saved_models/home_loans_model.keras")
        logger.error("- saved_models/credit_card_model.keras")
        raise

def load_and_preprocess_data(sample_size=None):
    """Load and preprocess data from all sources, handling limited alignment with optional sampling"""
    logger.info(f"🔄 Loading and preprocessing data...")
    logger.info(f"Sample size: {sample_size:,} customers" if sample_size else "Using full dataset")
    
    use_sampling = sample_size is not None
    
    # Set random seed for reproducible sampling
    np.random.seed(RANDOM_SEED)
    logger.debug(f"Random seed set to: {RANDOM_SEED}")
    
    # Load datasets - NOW INCLUDING CREDIT CARD BANK
    logger.debug("Loading bank datasets...")
    auto_loans_df = pd.read_csv('../dataset/data/banks/auto_loans_bank.csv')
    digital_bank_df = pd.read_csv('../dataset/data/banks/digital_savings_bank.csv')
    home_loans_df = pd.read_csv('../dataset/data/banks/home_loans_bank.csv')
    credit_card_df = pd.read_csv('../dataset/data/banks/credit_card_bank.csv')
    master_df = pd.read_csv('../dataset/data/credit_scoring_dataset.csv')
    
    logger.info("📊 Original Dataset Statistics:")
    logger.info(f"Auto Loans customers: {len(auto_loans_df):,}")
    logger.info(f"Digital Bank customers: {len(digital_bank_df):,}")
    logger.info(f"Home Loans customers: {len(home_loans_df):,}")
    logger.info(f"Credit Card customers: {len(credit_card_df):,}")
    logger.info(f"Master dataset customers: {len(master_df):,}")
    
    # Apply sampling if configured
    if use_sampling and sample_size < len(master_df):
        logger.info(f"🔀 Applying stratified sampling:")
        logger.info(f"  Target sample size: {sample_size:,} customers")
        logger.info(f"  Random seed: {RANDOM_SEED}")
        
        # Calculate original service combination ratios and maintain them - NOW WITH FOUR BANKS
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
        
        # Filter bank datasets to match sampled customers - NOW INCLUDING CREDIT CARD
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
    
    # Get all unique customers from master dataset (this is our universe) - NOW WITH FOUR BANKS
    all_customers = set(master_df['tax_id'])
    auto_customers = set(auto_loans_df['tax_id'])
    digital_customers = set(digital_bank_df['tax_id'])
    home_customers = set(home_loans_df['tax_id'])
    credit_card_customers = set(credit_card_df['tax_id'])
    
    logger.info(f"📈 Final Alignment Statistics (Four Banks):")
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
    
    # Create alignment matrix for all customers - NOW WITH FOUR BANKS
    customer_df = pd.DataFrame({'tax_id': sorted(list(all_customers))})
    customer_df['has_auto'] = customer_df['tax_id'].isin(auto_customers)
    customer_df['has_digital'] = customer_df['tax_id'].isin(digital_customers)
    customer_df['has_home'] = customer_df['tax_id'].isin(home_customers)
    customer_df['has_credit_card'] = customer_df['tax_id'].isin(credit_card_customers)
    
    # Sort datasets by tax_id for consistent indexing - NOW INCLUDING CREDIT CARD
    auto_loans_df = auto_loans_df.sort_values('tax_id').reset_index(drop=True)
    digital_bank_df = digital_bank_df.sort_values('tax_id').reset_index(drop=True)
    home_loans_df = home_loans_df.sort_values('tax_id').reset_index(drop=True)
    credit_card_df = credit_card_df.sort_values('tax_id').reset_index(drop=True)
    master_df = master_df.sort_values('tax_id').reset_index(drop=True)
    
    # Load client models and create feature extractors - NOW WITH FOUR MODELS
    logger.info("🔄 Creating feature extractors from client models...")
    auto_loans_model, digital_bank_model, home_loans_model, credit_card_model = load_client_models()
    auto_loans_extractor = get_penultimate_layer_model(auto_loans_model)
    digital_bank_extractor = get_penultimate_layer_model(digital_bank_model)
    home_loans_extractor = get_penultimate_layer_model(home_loans_model)
    credit_card_extractor = get_penultimate_layer_model(credit_card_model)
    
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
    
    # NEW: Credit card features (matching credit_card_model.py exactly - including derived features)
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
        # Credit card specific metrics (available in dataset)
        'apr', 'risk_score', 'total_available_credit', 'credit_to_income_ratio', 'cash_advance_limit',
        # Derived features (need to be calculated like in credit_card_model.py)
        'credit_capacity_ratio', 'income_to_limit_ratio', 'debt_service_ratio', 'risk_adjusted_income'
    ]
    
    print(f"\nFeature Verification (Four Banks):")
    print(f"Auto model expects {len(auto_features)} features")
    print(f"Digital model expects {len(digital_features)} features")
    print(f"Home model expects {len(home_features)} features")
    print(f"Credit card model expects {len(credit_card_features)} features")
    
    # Initialize scalers - NOW WITH FOUR SCALERS
    auto_scaler = StandardScaler()
    digital_scaler = StandardScaler()
    home_scaler = StandardScaler()
    credit_card_scaler = StandardScaler()
    
    def extract_bank_representations(bank_df, features, scaler, extractor, customers_with_service, bank_name):
        """Extract representations for customers with service at this bank"""
        output_size = extractor.output_shape[-1]
        all_representations = np.zeros((len(customer_df), output_size))
        
        if len(bank_df) > 0:
            print(f"\nProcessing {bank_name} Bank:")
            print(f"  Dataset size: {len(bank_df)}")
            print(f"  Feature count: {len(features)}")
            print(f"  Output representation size: {output_size}")
            
            # Special preprocessing for credit card features (matching credit_card_model.py)
            if bank_name == 'Credit Cards':
                logger.debug("Creating derived features for credit card model compatibility...")
                
                # Create derived features exactly as in credit_card_model.py
                bank_df = bank_df.copy()
                bank_df['credit_capacity_ratio'] = bank_df['credit_card_limit'] / bank_df['total_credit_limit'].replace(0, 1)
                bank_df['income_to_limit_ratio'] = bank_df['annual_income'] / bank_df['credit_card_limit'].replace(0, 1)
                bank_df['debt_service_ratio'] = (bank_df['current_debt'] * 0.03) / (bank_df['annual_income'] / 12)
                bank_df['risk_adjusted_income'] = bank_df['annual_income'] * (bank_df['risk_score'] / 100)
                
                logger.debug("✅ Derived features created for credit card compatibility")
            
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
            representations = extractor.predict(X_scaled, verbose=0)
            
            # Map representations to correct customer positions
            bank_customer_ids = bank_df['tax_id'].values
            for i, customer_id in enumerate(bank_customer_ids):
                customer_idx = customer_df[customer_df['tax_id'] == customer_id].index[0]
                all_representations[customer_idx] = representations[i]
        
        # Create service availability mask
        service_mask = customers_with_service.values.astype(np.float32).reshape(-1, 1)
        
        return all_representations, service_mask, scaler
    
    # Extract representations from all four banks
    auto_repr, auto_mask, fitted_auto_scaler = extract_bank_representations(
        auto_loans_df, auto_features, auto_scaler, auto_loans_extractor, 
        customer_df['has_auto'], 'Auto Loans'
    )
    
    digital_repr, digital_mask, fitted_digital_scaler = extract_bank_representations(
        digital_bank_df, digital_features, digital_scaler, digital_bank_extractor,
        customer_df['has_digital'], 'Digital Banking'
    )
    
    home_repr, home_mask, fitted_home_scaler = extract_bank_representations(
        home_loans_df, home_features, home_scaler, home_loans_extractor,
        customer_df['has_home'], 'Home Loans'
    )
    
    # NEW: Extract credit card representations
    credit_card_repr, credit_card_mask, fitted_credit_card_scaler = extract_bank_representations(
        credit_card_df, credit_card_features, credit_card_scaler, credit_card_extractor,
        customer_df['has_credit_card'], 'Credit Cards'
    )
    
    # Save intermediate representations to CSV for review (SAMPLE ONLY)
    logger.info("💾 Saving sample intermediate representations to CSV for review...")
    
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
    
    # Add credit card representations (8 dimensions)
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
    output_file = f'data/intermediate_representations/vfl_representations_sample_{timestamp}.csv'
    sample_df.to_csv(output_file, index=False)
    
    logger.info(f"✅ Sample intermediate representations saved to: {output_file}")
    logger.info(f"📊 Sample DataFrame Info:")
    logger.info(f"   - Sample size: {len(sample_df)} customers (from {len(representations_df)} total)")
    logger.info(f"   - Total columns: {len(sample_df.columns)}")
    logger.info(f"   - Auto representations: {auto_repr.shape[1]} columns (auto_repr_01 to auto_repr_{auto_repr.shape[1]:02d})")
    logger.info(f"   - Digital representations: {digital_repr.shape[1]} columns (digital_repr_01 to digital_repr_{digital_repr.shape[1]:02d})")
    logger.info(f"   - Home representations: {home_repr.shape[1]} columns (home_repr_01 to home_repr_{home_repr.shape[1]:02d})")
    logger.info(f"   - Credit card representations: {credit_card_repr.shape[1]} columns (credit_card_repr_01 to credit_card_repr_{credit_card_repr.shape[1]:02d})")
    
    # Log sample distribution
    logger.info(f"📈 Sample Service Distribution:")
    sample_service_dist = sample_df['service_label'].value_counts()
    for service, count in sample_service_dist.items():
        percentage = (count / len(sample_df)) * 100
        total_of_type = service_dist[service] if service in service_dist else 0
        logger.info(f"   - {service}: {count} samples ({percentage:.1f}%) from {total_of_type} total")
    
    # ============================================================================
    # CREATE TRAINING DATA FROM BANK REPRESENTATIONS - THE MISSING PIECE!
    # ============================================================================
    logger.info("🔗 Creating VFL training data from bank representations...")
    
    # Combine all bank representations into single feature matrix
    # Format: [auto_repr | auto_mask | digital_repr | digital_mask | home_repr | home_mask | credit_card_repr | credit_card_mask]
    X_combined = np.concatenate([
        auto_repr,           # 16 dimensions (auto loan representations)
        auto_mask,           # 1 dimension (auto service availability)
        digital_repr,        # 8 dimensions (digital banking representations) 
        digital_mask,        # 1 dimension (digital service availability)
        home_repr,           # 16 dimensions (home loan representations)
        home_mask,           # 1 dimension (home service availability)
        credit_card_repr,    # 8 dimensions (credit card representations)
        credit_card_mask     # 1 dimension (credit card service availability)
    ], axis=1)
    
    # Get target variable (credit scores) from master dataset
    y_combined = master_df['credit_score'].values
    ids_combined = master_df['tax_id'].values
    
    logger.info(f"✅ VFL training data created:")
    logger.info(f"   - Combined feature matrix: {X_combined.shape}")
    logger.info(f"   - Feature breakdown: Auto({auto_repr.shape[1]}+1) + Digital({digital_repr.shape[1]}+1) + Home({home_repr.shape[1]}+1) + Credit({credit_card_repr.shape[1]}+1) = {X_combined.shape[1]} total")
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
            fitted_auto_scaler, fitted_digital_scaler, fitted_home_scaler, fitted_credit_card_scaler,
            auto_repr.shape[1], digital_repr.shape[1], home_repr.shape[1], credit_card_repr.shape[1],
            X_combined, y_combined, ids_combined)

class VFLHyperModel(kt.HyperModel):
    """Hypermodel for VFL architecture search"""
    
    def __init__(self, input_shape, auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size):
        self.input_shape = input_shape
        self.auto_repr_size = auto_repr_size
        self.digital_repr_size = digital_repr_size
        self.home_repr_size = home_repr_size
        self.credit_card_repr_size = credit_card_repr_size
    
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
        """Build model with hyperparameters"""
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='vfl_input')
        
        # Split combined input into components using Lambda layers
        auto_repr = layers.Lambda(lambda x: x[:, :self.auto_repr_size], name='auto_representations')(inputs)
        auto_mask = layers.Lambda(lambda x: x[:, self.auto_repr_size:self.auto_repr_size+1], name='auto_mask')(inputs)
        
        digital_start = self.auto_repr_size + 1
        digital_end = digital_start + self.digital_repr_size
        digital_repr = layers.Lambda(lambda x: x[:, digital_start:digital_end], name='digital_representations')(inputs)
        digital_mask = layers.Lambda(lambda x: x[:, digital_end:digital_end+1], name='digital_mask')(inputs)
        
        home_start = digital_end + 1
        home_end = home_start + self.home_repr_size
        home_repr = layers.Lambda(lambda x: x[:, home_start:home_end], name='home_representations')(inputs)
        home_mask = layers.Lambda(lambda x: x[:, home_end:home_end+1], name='home_mask')(inputs)
        
        # NEW: Credit card representations
        credit_card_start = home_end + 1
        credit_card_end = credit_card_start + self.credit_card_repr_size
        credit_card_repr = layers.Lambda(lambda x: x[:, credit_card_start:credit_card_end], name='credit_card_representations')(inputs)
        credit_card_mask = layers.Lambda(lambda x: x[:, credit_card_end:credit_card_end+1], name='credit_card_mask')(inputs)
        
        # Expand masks to match representation dimensions
        auto_mask_expanded = layers.RepeatVector(self.auto_repr_size)(auto_mask)
        auto_mask_expanded = layers.Reshape((self.auto_repr_size,), name='auto_mask_expanded')(auto_mask_expanded)
        
        digital_mask_expanded = layers.RepeatVector(self.digital_repr_size)(digital_mask)
        digital_mask_expanded = layers.Reshape((self.digital_repr_size,), name='digital_mask_expanded')(digital_mask_expanded)
        
        home_mask_expanded = layers.RepeatVector(self.home_repr_size)(home_mask)
        home_mask_expanded = layers.Reshape((self.home_repr_size,), name='home_mask_expanded')(home_mask_expanded)
        
        # NEW: Credit card mask expansion
        credit_card_mask_expanded = layers.RepeatVector(self.credit_card_repr_size)(credit_card_mask)
        credit_card_mask_expanded = layers.Reshape((self.credit_card_repr_size,), name='credit_card_mask_expanded')(credit_card_mask_expanded)
        
        # Apply masks to representations
        auto_masked = layers.Multiply(name='auto_masked')([auto_repr, auto_mask_expanded])
        digital_masked = layers.Multiply(name='digital_masked')([digital_repr, digital_mask_expanded])
        home_masked = layers.Multiply(name='home_masked')([home_repr, home_mask_expanded])
        credit_card_masked = layers.Multiply(name='credit_card_masked')([credit_card_repr, credit_card_mask_expanded])
        
        # Bank-specific processing with tunable units
        auto_units = hp.Int('auto_units', min_value=64, max_value=256, step=64)
        digital_units = hp.Int('digital_units', min_value=32, max_value=128, step=32)
        home_units = hp.Int('home_units', min_value=64, max_value=256, step=64)
        credit_card_units = hp.Int('credit_card_units', min_value=64, max_value=256, step=64)
        
        auto_processed = layers.Dense(auto_units, activation='relu', name='auto_dense')(auto_masked)
        auto_processed = layers.BatchNormalization(name='auto_bn')(auto_processed)
        auto_dropout = hp.Float('auto_dropout', min_value=0.1, max_value=0.4, step=0.1)
        auto_processed = layers.Dropout(auto_dropout, name='auto_dropout')(auto_processed)
        
        digital_processed = layers.Dense(digital_units, activation='relu', name='digital_dense')(digital_masked)
        digital_processed = layers.BatchNormalization(name='digital_bn')(digital_processed)
        digital_dropout = hp.Float('digital_dropout', min_value=0.1, max_value=0.4, step=0.1)
        digital_processed = layers.Dropout(digital_dropout, name='digital_dropout')(digital_processed)
        
        home_processed = layers.Dense(home_units, activation='relu', name='home_dense')(home_masked)
        home_processed = layers.BatchNormalization(name='home_bn')(home_processed)
        home_dropout = hp.Float('home_dropout', min_value=0.1, max_value=0.4, step=0.1)
        home_processed = layers.Dropout(home_dropout, name='home_dropout')(home_processed)
        
        # NEW: Credit card processing
        credit_card_processed = layers.Dense(credit_card_units, activation='relu', name='credit_card_dense')(credit_card_masked)
        credit_card_processed = layers.BatchNormalization(name='credit_card_bn')(credit_card_processed)
        credit_card_dropout = hp.Float('credit_card_dropout', min_value=0.1, max_value=0.4, step=0.1)
        credit_card_processed = layers.Dropout(credit_card_dropout, name='credit_card_dropout')(credit_card_processed)
        
        # Combine bank features
        combined_features = layers.Concatenate(name='combined_bank_features')([
            auto_processed, digital_processed, home_processed, credit_card_processed
        ])
        
        # Add service availability information
        service_info = layers.Concatenate(name='service_availability')([
            auto_mask, digital_mask, home_mask, credit_card_mask
        ])
        service_units = hp.Int('service_units', min_value=8, max_value=32, step=8)
        service_processed = layers.Dense(service_units, activation='relu', name='service_dense')(service_info)
        service_processed = layers.BatchNormalization(name='service_bn')(service_processed)
        
        # Final feature combination
        all_features = layers.Concatenate(name='all_features')([combined_features, service_processed])
        
        # Tunable main architecture
        x = all_features
        
        # Number of hidden layers
        num_layers = hp.Int('num_layers', min_value=MIN_LAYERS, max_value=MAX_LAYERS)
        
        for i in range(num_layers):
            # Units per layer (decreasing pattern encouraged)
            if i == 0:
                units = hp.Int(f'layer_{i}_units', min_value=MIN_UNITS, max_value=MAX_UNITS, step=64)
            else:
                # Encourage decreasing pattern but allow flexibility
                prev_units = hp.get(f'layer_{i-1}_units')
                max_units_this_layer = min(MAX_UNITS, prev_units)
                units = hp.Int(f'layer_{i}_units', min_value=MIN_UNITS, max_value=max_units_this_layer, step=64)
            
            # Activation function
            activation = hp.Choice(f'layer_{i}_activation', values=['relu', 'swish', 'gelu'])
            
            x = layers.Dense(units, activation=activation, name=f'hidden_dense_{i+1}')(x)
            
            # Batch normalization choice
            use_bn = hp.Boolean(f'layer_{i}_batch_norm')
            if use_bn:
                x = layers.BatchNormalization(name=f'hidden_bn_{i+1}')(x)
            
            # Dropout rate
            dropout_rate = hp.Float(f'layer_{i}_dropout', min_value=0.0, max_value=0.5, step=0.1)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name=f'hidden_dropout_{i+1}')(x)
        
        # Final prediction layers
        final_units = hp.Int('final_units', min_value=16, max_value=64, step=16)
        x = layers.Dense(final_units, activation='relu', name='pre_output_dense')(x)
        x = layers.BatchNormalization(name='pre_output_bn')(x)
        final_dropout = hp.Float('final_dropout', min_value=0.0, max_value=0.3, step=0.05)
        if final_dropout > 0:
            x = layers.Dropout(final_dropout, name='pre_output_dropout')(x)
        
        # Output layer with credit score scaling
        raw_output = layers.Dense(1, activation='sigmoid', name='raw_output')(x)
        outputs = layers.Lambda(lambda x: x * 550 + 300, name='credit_score_output')(raw_output)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='vfl_automl_four_banks_model')
        
        # Tunable optimizer settings
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        clipnorm = hp.Float('clipnorm', min_value=0.5, max_value=2.0, step=0.5)
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=clipnorm,
            beta_1=0.9,
            beta_2=0.999
        )
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model

def run_automl_search():
    """Run AutoML architecture search with two-phase training"""
    
    start_time = datetime.now()
    logger.info("🚀 VFL AutoML Architecture Search - Four Bank System")
    logger.info("=" * 80)
    logger.info(f"Training Configuration:")
    logger.info(f"  Phase 1 (AutoML Search): {'✅ Enabled' if True else '❌ Disabled'}")
    logger.info(f"  Phase 2 (Full Training): {'✅ Enabled' if ENABLE_PHASE_2 else '❌ Disabled'}")
    logger.info(f"  Confidence Scoring: {'✅ Enabled' if ENABLE_CONFIDENCE_SCORES else '❌ Disabled'}")
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
    logger.info("🚀 PHASE 1: AutoML Architecture Search")
    logger.info("=" * 60)
    
    phase1_start = datetime.now()
    
    # Load and preprocess data for AutoML search
    logger.debug("Loading data for AutoML search...")
    (X_train, X_test, y_train, y_test, ids_train, ids_test, 
     auto_scaler, digital_scaler, home_scaler, credit_card_scaler,
     auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size,
     X_combined, y_combined, ids_combined) = load_and_preprocess_data(AUTOML_SAMPLE_SIZE)
    
    logger.info(f"🔍 AutoML Search Data Summary (Four Banks):")
    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Feature vector size: {X_train.shape[1]}")
    logger.info(f"  Auto loans representation size: {auto_repr_size}")
    logger.info(f"  Digital bank representation size: {digital_repr_size}")
    logger.info(f"  Home loans representation size: {home_repr_size}")
    logger.info(f"  Credit card representation size: {credit_card_repr_size}")
    
    # Create hypermodel
    logger.debug("Creating hypermodel for architecture search...")
    hypermodel = VFLHyperModel(
        input_shape=(X_train.shape[1],),
        auto_repr_size=auto_repr_size,
        digital_repr_size=digital_repr_size,
        home_repr_size=home_repr_size,
        credit_card_repr_size=credit_card_repr_size
    )
    
    # ============================================================================
    # DETAILED HYPERPARAMETER SEARCH LOGGING
    # ============================================================================
    
    # Log search space details before starting
    logger.info("")
    logger.info("🔍 HYPERPARAMETER SEARCH SPACE CONFIGURATION:")
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
    
    logger.info("🏦 Bank-Specific Processing Units:")
    logger.info(f"   - Auto Bank: 64 to 256 units (step: 64)")
    logger.info(f"   - Digital Bank: 32 to 128 units (step: 32)")
    logger.info(f"   - Home Bank: 64 to 256 units (step: 64)")
    logger.info(f"   - Credit Card Bank: 64 to 256 units (step: 64)")
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
        project_name=f'vfl_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        overwrite=True,
        tune_new_entries=True,
        allow_new_entries=True
    )
    
    estimated_time_min = MAX_TRIALS * EXECUTIONS_PER_TRIAL * EPOCHS_PER_TRIAL / 60
    estimated_time_max = MAX_TRIALS * EXECUTIONS_PER_TRIAL * EPOCHS_PER_TRIAL / 30
    
    logger.info("=" * 70)
    logger.info(f"⏱️  Starting AutoML Search...")
    logger.info(f"  Testing {MAX_TRIALS} different architectures")
    logger.info(f"  Estimated time: {estimated_time_min:.0f}-{estimated_time_max:.0f} minutes")
    logger.info(f"  Search objective: minimize {SEARCH_OBJECTIVE}")
    logger.info("=" * 70)
    
    # Custom class to track trial progress
    class TrialLogger:
        def __init__(self):
            self.trial_results = []
            self.current_trial = 0
        
        def log_trial_start(self, trial_id, hyperparameters):
            self.current_trial += 1
            logger.info(f"")
            logger.info(f"🚀 TRIAL {self.current_trial}/{MAX_TRIALS} - ID: {trial_id}")
            logger.info("─" * 60)
            
            # Log key hyperparameters for this trial
            logger.info(f"🏗️  Architecture for Trial {self.current_trial}:")
            logger.info(f"   - Hidden Layers: {hyperparameters.get('num_layers', 'N/A')}")
            logger.info(f"   - Learning Rate: {hyperparameters.get('learning_rate', 'N/A'):.6f}")
            logger.info(f"   - Gradient Clipping: {hyperparameters.get('clipnorm', 'N/A')}")
            
            logger.info(f"🏦 Bank Processing Units:")
            logger.info(f"   - Auto: {hyperparameters.get('auto_units', 'N/A')} units (dropout: {hyperparameters.get('auto_dropout', 'N/A'):.2f})")
            logger.info(f"   - Digital: {hyperparameters.get('digital_units', 'N/A')} units (dropout: {hyperparameters.get('digital_dropout', 'N/A'):.2f})")
            logger.info(f"   - Home: {hyperparameters.get('home_units', 'N/A')} units (dropout: {hyperparameters.get('home_dropout', 'N/A'):.2f})")
            logger.info(f"   - Credit: {hyperparameters.get('credit_card_units', 'N/A')} units (dropout: {hyperparameters.get('credit_card_dropout', 'N/A'):.2f})")
            logger.info(f"   - Service: {hyperparameters.get('service_units', 'N/A')} units")
            
            # Log layer-specific details
            num_layers = hyperparameters.get('num_layers', 0)
            if num_layers > 0:
                logger.info(f"🔧 Hidden Layer Details:")
                for i in range(num_layers):
                    units = hyperparameters.get(f'layer_{i}_units', 'N/A')
                    activation = hyperparameters.get(f'layer_{i}_activation', 'N/A')
                    dropout = hyperparameters.get(f'layer_{i}_dropout', 'N/A')
                    batch_norm = hyperparameters.get(f'layer_{i}_batch_norm', 'N/A')
                    logger.info(f"   - Layer {i+1}: {units} units, {activation}, dropout={dropout:.2f}, bn={batch_norm}")
            
            logger.info(f"🎯 Final Processing:")
            logger.info(f"   - Final Units: {hyperparameters.get('final_units', 'N/A')}")
            logger.info(f"   - Final Dropout: {hyperparameters.get('final_dropout', 'N/A'):.3f}")
            
            trial_start_time = datetime.now()
            logger.info(f"⏱️  Trial {self.current_trial} started at: {trial_start_time.strftime('%H:%M:%S')}")
            
            return trial_start_time
        
        def log_trial_end(self, trial_id, trial_start_time, score, status="completed"):
            trial_end_time = datetime.now()
            trial_duration = trial_end_time - trial_start_time
            
            logger.info(f"")
            logger.info(f"✅ TRIAL {self.current_trial} {status.upper()}")
            logger.info(f"   - Duration: {trial_duration}")
            logger.info(f"   - Final {SEARCH_OBJECTIVE}: {score:.6f}")
            logger.info(f"   - Status: {status}")
            logger.info(f"   - End time: {trial_end_time.strftime('%H:%M:%S')}")
            
            # Store trial results
            trial_result = {
                'trial_number': self.current_trial,
                'trial_id': trial_id,
                'score': score,
                'duration_seconds': trial_duration.total_seconds(),
                'status': status,
                'start_time': trial_start_time.isoformat(),
                'end_time': trial_end_time.isoformat()
            }
            self.trial_results.append(trial_result)
            
            # Show progress
            logger.info(f"📊 Progress: {self.current_trial}/{MAX_TRIALS} trials completed ({self.current_trial/MAX_TRIALS*100:.1f}%)")
            if len(self.trial_results) > 1:
                best_so_far = min(r['score'] for r in self.trial_results if r['status'] == 'completed')
                logger.info(f"🏆 Best {SEARCH_OBJECTIVE} so far: {best_so_far:.6f}")
            
            logger.info("─" * 60)
    
    # Initialize trial logger
    trial_logger = TrialLogger()
    
    # Define callbacks for each trial
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=SEARCH_OBJECTIVE,
            patience=10,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.3,
            patience=10,
            min_lr=1e-8,
            verbose=1
        )
    ]
    
    # Start AutoML search
    logger.info("")
    logger.info("🔍 Starting Keras Tuner search...")
    search_start = datetime.now()
    
    # Create a custom Keras Tuner logging callback
    class TunerLoggingCallback:
        def __init__(self, trial_logger):
            self.trial_logger = trial_logger
            self.current_trial_start = None
            
        def on_search_begin(self, tuner):
            """Called at the beginning of the search"""
            logger.info("🔍 Keras Tuner search session starting...")
            logger.info(f"   - Tuner type: {type(tuner).__name__}")
            
        def on_search_end(self, tuner):
            """Called at the end of the search"""
            logger.info("✅ Keras Tuner search session completed")
            logger.info(f"   - Total trials completed: {len(tuner.oracle.trials)}")
            logger.info(f"   - Best trial ID: {tuner.oracle.get_best_trials(1)[0].trial_id if tuner.oracle.trials else 'None'}")
    
    # Initialize the custom logging callback
    tuner_callback = TunerLoggingCallback(trial_logger)
    
    try:
        # Log detailed pre-search information
        logger.info("🚀 DETAILED KERAS TUNER SEARCH STARTING")
        logger.info("=" * 80)
        logger.info(f"🔧 Tuner Configuration:")
        logger.info(f"   - Tuner Type: {type(tuner).__name__}")
        logger.info(f"   - Max Trials: {MAX_TRIALS}")
        logger.info(f"   - Executions per Trial: {EXECUTIONS_PER_TRIAL}")
        logger.info(f"   - Epochs per Trial: {EPOCHS_PER_TRIAL}")
        logger.info(f"   - Objective: minimize {SEARCH_OBJECTIVE}")
        logger.info(f"   - Training samples: {len(X_train):,}")
        logger.info(f"   - Validation split: 20%")
        logger.info(f"   - Validation samples: ~{int(len(X_train) * 0.2):,}")
        logger.info("")
        
        # Trigger on_search_begin
        tuner_callback.on_search_begin(tuner)
        
        # Perform the search with detailed logging
        logger.info("🎯 Starting hyperparameter search trials...")
        logger.info("=" * 80)
        
        # Perform the actual search without overriding internal methods
        tuner.search(
            X_train, y_train,
            epochs=EPOCHS_PER_TRIAL,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Trigger on_search_end
        # tuner_callback.on_search_end(tuner)  # Commented out due to API issues
        logger.info("✅ Keras Tuner search session completed")
        logger.info(f"   - Total trials completed: {len(tuner.oracle.trials)}")
        logger.info(f"   - Best trial ID: {tuner.oracle.get_best_trials(1)[0].trial_id if tuner.oracle.trials else 'None'}")
        
        search_end = datetime.now()
        search_duration = search_end - search_start
        phase1_end = datetime.now()
        phase1_duration = phase1_end - phase1_start
        
        logger.info(f"✅ AutoML search completed in {search_duration}")
        
        # Log comprehensive trial summary
        logger.info("")
        logger.info("📊 COMPREHENSIVE TRIAL SUMMARY")
        logger.info("=" * 80)
        
        all_trials = tuner.oracle.trials
        if all_trials:
            # Sort trials by score
            completed_trials = [t for t in all_trials.values() if hasattr(t, 'score') and t.score is not None]
            if completed_trials:
                sorted_trials = sorted(completed_trials, key=lambda x: x.score)
                
                logger.info(f"📈 Trial Performance Ranking (by {SEARCH_OBJECTIVE}):")
                logger.info(f"{'Rank':<6} {'Trial ID':<12} {'Score':<12} {'Status':<12} {'Duration':<10}")
                logger.info("-" * 70)
                
                for rank, trial in enumerate(sorted_trials[:10], 1):  # Show top 10
                    trial_duration = "N/A"  # Simplified since duration tracking is complex
                    
                    status_str = trial.status.name if hasattr(trial, 'status') and hasattr(trial.status, 'name') else str(trial.status) if hasattr(trial, 'status') else 'N/A'
                    
                    logger.info(f"{rank:<6} {trial.trial_id:<12} {trial.score:<12.6f} {status_str:<12} {trial_duration:<10}")
                
                # Log hyperparameters of best trials
                logger.info("")
                logger.info("🏆 TOP 3 TRIAL HYPERPARAMETERS:")
                logger.info("=" * 80)
                
                for rank, trial in enumerate(sorted_trials[:3], 1):
                    logger.info(f"🥇 RANK {rank} - Trial {trial.trial_id} (Score: {trial.score:.6f})")
                    logger.info("-" * 60)
                    
                    hp_values = trial.hyperparameters.values
                    
                    # Core architecture
                    logger.info(f"🏗️  Architecture:")
                    logger.info(f"   - Hidden Layers: {hp_values.get('num_layers', 'N/A')}")
                    logger.info(f"   - Learning Rate: {hp_values.get('learning_rate', 'N/A'):.6f}")
                    logger.info(f"   - Gradient Clipping: {hp_values.get('clipnorm', 'N/A')}")
                    
                    # Bank-specific processing
                    logger.info(f"🏦 Bank Processing:")
                    logger.info(f"   - Auto: {hp_values.get('auto_units', 'N/A')} units (dropout: {hp_values.get('auto_dropout', 'N/A'):.2f})")
                    logger.info(f"   - Digital: {hp_values.get('digital_units', 'N/A')} units (dropout: {hp_values.get('digital_dropout', 'N/A'):.2f})")
                    logger.info(f"   - Home: {hp_values.get('home_units', 'N/A')} units (dropout: {hp_values.get('home_dropout', 'N/A'):.2f})")
                    logger.info(f"   - Credit: {hp_values.get('credit_card_units', 'N/A')} units (dropout: {hp_values.get('credit_card_dropout', 'N/A'):.2f})")
                    logger.info(f"   - Service: {hp_values.get('service_units', 'N/A')} units")
                    
                    # Final processing
                    logger.info(f"🎯 Final Layer:")
                    logger.info(f"   - Final Units: {hp_values.get('final_units', 'N/A')}")
                    logger.info(f"   - Final Dropout: {hp_values.get('final_dropout', 'N/A'):.3f}")
                    
                    # Layer details for best trial
                    if rank == 1:
                        num_layers = hp_values.get('num_layers', 0)
                        if num_layers and num_layers > 0:
                            logger.info(f"🔧 Hidden Layer Details (Best Trial):")
                            for i in range(num_layers):
                                units = hp_values.get(f'layer_{i}_units', 'N/A')
                                activation = hp_values.get(f'layer_{i}_activation', 'N/A')
                                dropout = hp_values.get(f'layer_{i}_dropout', 'N/A')
                                batch_norm = hp_values.get(f'layer_{i}_batch_norm', 'N/A')
                                logger.info(f"   - Layer {i+1}: {units} units, {activation}, dropout={dropout:.2f}, bn={batch_norm}")
                    
                    logger.info("")
            else:
                logger.warning("⚠️  No completed trials found with valid scores")
        else:
            logger.warning("⚠️  No trials found in oracle")
        
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models()[0]
        
        logger.info("")
        logger.info("🏆 PHASE 1 COMPLETE - AutoML Search Results")
        logger.info("=" * 60)
        logger.info(f"⏱️  Search Duration: {search_duration}")
        logger.info(f"🎯 Search Objective: {SEARCH_OBJECTIVE}")
        logger.info(f"📊 Trials Completed: {len(tuner.oracle.trials)}")
        logger.info("")
        
        # Evaluate best model on test set
        test_loss, test_mae, test_mse = best_model.evaluate(X_test, y_test, verbose=0)
        y_pred_phase1 = best_model.predict(X_test, verbose=0)
        
        logger.info(f"🎯 Best Architecture Performance:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test MAE: {test_mae:.2f}")
        logger.info(f"  Test RMSE: {np.sqrt(test_mse):.2f}")
        logger.info(f"  Parameters: {best_model.count_params():,}")
        
        # Log best hyperparameters
        logger.info("")
        logger.info("🏗️  Best Architecture Configuration:")
        logger.info(f"  Hidden Layers: {best_hp.get('num_layers')}")
        logger.info(f"  Learning Rate: {best_hp.get('learning_rate'):.6f}")
        logger.info(f"  Gradient Clipping: {best_hp.get('clipnorm')}")
        logger.info(f"  Auto Bank Units: {best_hp.get('auto_units')} (dropout: {best_hp.get('auto_dropout'):.2f})")
        logger.info(f"  Digital Bank Units: {best_hp.get('digital_units')} (dropout: {best_hp.get('digital_dropout'):.2f})")
        logger.info(f"  Home Bank Units: {best_hp.get('home_units')} (dropout: {best_hp.get('home_dropout'):.2f})")
        logger.info(f"  Credit Bank Units: {best_hp.get('credit_card_units')} (dropout: {best_hp.get('credit_card_dropout'):.2f})")
        logger.info(f"  Service Units: {best_hp.get('service_units')}")
        logger.info(f"  Final Units: {best_hp.get('final_units')} (dropout: {best_hp.get('final_dropout'):.3f})")
        
        # Sample predictions for Phase 1 with confidence scores
        print_detailed_sample_predictions_with_confidence(X_test, y_test, ids_test, best_model, 
                                        auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size, phase="Phase 1")
        
        # UPDATE INTERMEDIATE REPRESENTATIONS CSV WITH PREDICTIONS
        logger.info("🔄 Updating intermediate representations CSV with best model predictions...")
        add_predictions_to_representations_csv(best_model, X_combined, y_combined, ids_combined)
        
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
        
        with open('automl_results/phase1_results.json', 'w') as f:
            json.dump(phase1_results, f, indent=2, default=str)
        
        logger.info("")
        logger.info(f"📁 Phase 1 results saved to: automl_results/phase1_results.json")
        
        # Phase 2 check
        if not ENABLE_PHASE_2:
            total_duration = datetime.now() - start_time
            logger.info("")
            logger.info("🎉 VFL AutoML Search Completed (Phase 1 Only)")
            logger.info("=" * 70)
            logger.info(f"⏱️  Total Duration: {total_duration}")
            logger.info(f"💡 Phase 2 is disabled. To enable full training, set ENABLE_PHASE_2 = True")
            logger.info(f"🏆 Best MAE: {test_mae:.2f} points")
            logger.info(f"📊 Architecture: {best_hp.get('num_layers')} layers, {best_model.count_params():,} parameters")
            return best_model, best_hp, phase1_results
        
        logger.info("")
        logger.info("🚀 PHASE 2: Final Model Training with Larger Dataset")
        logger.info("=" * 60)
        
        phase2_start = datetime.now()
        
        logger.info(f"🎯 Phase 2 Configuration:")
        logger.info(f"  Dataset size: {FINAL_SAMPLE_SIZE:,} customers (vs {AUTOML_SAMPLE_SIZE:,} in Phase 1)")
        logger.info(f"  Training epochs: {FINAL_EPOCHS} (vs {EPOCHS_PER_TRIAL} in Phase 1)")
        logger.info(f"  Using best architecture from Phase 1")
        logger.info(f"  Random seed: {RANDOM_SEED}")
        
        # Load larger dataset for final training
        logger.info("🔄 Loading larger dataset for Phase 2 training...")
        (X_train_final, X_test_final, y_train_final, y_test_final, ids_train_final, ids_test_final,
         auto_scaler_final, digital_scaler_final, home_scaler_final, credit_card_scaler_final,
         auto_repr_size_final, digital_repr_size_final, home_repr_size_final, credit_card_repr_size_final,
         X_combined_final, y_combined_final, ids_combined_final) = load_and_preprocess_data(FINAL_SAMPLE_SIZE)
        
        logger.info(f"📊 Phase 2 Dataset Statistics:")
        logger.info(f"  Training samples: {len(X_train_final):,}")
        logger.info(f"  Test samples: {len(X_test_final):,}")
        logger.info(f"  Feature vector size: {X_train_final.shape[1]}")
        logger.info(f"  Credit score range: {y_train_final.min():.0f} - {y_train_final.max():.0f}")
        
        # Build final model with best hyperparameters
        logger.info("🏗️  Building final model with best hyperparameters...")
        final_hypermodel = VFLHyperModel(
            input_shape=(X_train_final.shape[1],),
            auto_repr_size=auto_repr_size_final,
            digital_repr_size=digital_repr_size_final,
            home_repr_size=home_repr_size_final,
            credit_card_repr_size=credit_card_repr_size_final
        )
        
        final_model = final_hypermodel.build(best_hp)
        
        logger.info(f"✅ Final model built:")
        logger.info(f"  Architecture: {best_hp.get('num_layers')} hidden layers")
        logger.info(f"  Parameters: {final_model.count_params():,}")
        logger.info(f"  Learning rate: {best_hp.get('learning_rate'):.6f}")
        logger.info(f"  Gradient clipping: {best_hp.get('clipnorm')}")
        
        # Log detailed architecture for final model
        logger.info(f"🏦 Final Model Bank Processing Configuration:")
        logger.info(f"  Auto Bank: {best_hp.get('auto_units')} units (dropout: {best_hp.get('auto_dropout'):.2f})")
        logger.info(f"  Digital Bank: {best_hp.get('digital_units')} units (dropout: {best_hp.get('digital_dropout'):.2f})")
        logger.info(f"  Home Bank: {best_hp.get('home_units')} units (dropout: {best_hp.get('home_dropout'):.2f})")
        logger.info(f"  Credit Bank: {best_hp.get('credit_card_units')} units (dropout: {best_hp.get('credit_card_dropout'):.2f})")
        logger.info(f"  Service Processing: {best_hp.get('service_units')} units")
        logger.info(f"  Final Layer: {best_hp.get('final_units')} units (dropout: {best_hp.get('final_dropout'):.3f})")
        
        # Enhanced callbacks for final training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-8,
            verbose=1
        )
        
        # Custom training logger
        class Phase2TrainingLogger(tf.keras.callbacks.Callback):
            def __init__(self):
                self.epoch_start_time = None
                
            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = datetime.now()
                logger.info(f"🔄 Epoch {epoch + 1}/{FINAL_EPOCHS} started at {self.epoch_start_time.strftime('%H:%M:%S')}")
                
            def on_epoch_end(self, epoch, logs=None):
                epoch_end_time = datetime.now()
                epoch_duration = epoch_end_time - self.epoch_start_time
                
                train_loss = logs.get('loss', 0)
                train_mae = logs.get('mae', 0)
                val_loss = logs.get('val_loss', 0)
                val_mae = logs.get('val_mae', 0)
                lr = logs.get('lr', self.model.optimizer.learning_rate.numpy())
                
                logger.info(f"✅ Epoch {epoch + 1}/{FINAL_EPOCHS} completed in {epoch_duration}")
                logger.info(f"   📊 Training   - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}")
                logger.info(f"   📊 Validation - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}")
                logger.info(f"   📈 Learning Rate: {lr:.8f}")
                
                # Progress indicator
                progress = (epoch + 1) / FINAL_EPOCHS * 100
                logger.info(f"   🎯 Progress: {progress:.1f}% complete")
        
        training_logger = Phase2TrainingLogger()
        
        final_callbacks = [early_stopping, reduce_lr, training_logger]
        
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
            verbose=1
        )
        
        training_end = datetime.now()
        training_duration = training_end - training_start
        phase2_end = datetime.now()
        phase2_duration = phase2_end - phase2_start
        
        logger.info("=" * 80)
        logger.info(f"✅ Phase 2 training completed in {training_duration}")
        
        # Comprehensive final evaluation
        logger.info("")
        logger.info("📊 COMPREHENSIVE FINAL MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Evaluate with confidence scores
        final_evaluation = evaluate_model_with_confidence(
            final_model, X_test_final, y_test_final, ids_test_final, "Phase 2 Final"
        )
        
        # Extract results
        test_mae_final = final_evaluation['mae']
        test_rmse_final = final_evaluation['rmse']
        y_pred_final = final_evaluation['predictions']
        confidence_scores_final = final_evaluation['confidence_scores']
        prediction_std_final = final_evaluation['prediction_std']
        confidence_intervals_final = final_evaluation['confidence_intervals']
        
        # Calculate additional traditional metrics for compatibility
        test_loss_final, _, test_mse_final = final_model.evaluate(X_test_final, y_test_final, verbose=0)
        
        # Calculate additional metrics
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        
        r2_final = r2_score(y_test_final, y_pred_final)
        mape_final = mean_absolute_percentage_error(y_test_final, y_pred_final) * 100
        
        # Calculate accuracy within different error ranges
        errors = np.abs(y_test_final - y_pred_final.flatten())
        accuracy_10 = np.mean(errors <= 10) * 100
        accuracy_20 = np.mean(errors <= 20) * 100
        accuracy_30 = np.mean(errors <= 30) * 100
        accuracy_50 = np.mean(errors <= 50) * 100
        
        logger.info(f"🎯 Final Model Performance Metrics (with Confidence):")
        logger.info(f"  Test Loss (Huber): {test_loss_final:.4f}")
        logger.info(f"  Test MAE: {test_mae_final:.2f} points")
        logger.info(f"  Test RMSE: {test_rmse_final:.2f} points")
        logger.info(f"  Test R²: {r2_final:.4f}")
        logger.info(f"  Test MAPE: {mape_final:.2f}%")
        logger.info("")
        logger.info(f"📈 Prediction Accuracy (within error ranges):")
        logger.info(f"  Within 10 points: {accuracy_10:.1f}%")
        logger.info(f"  Within 20 points: {accuracy_20:.1f}%")
        logger.info(f"  Within 30 points: {accuracy_30:.1f}%")
        logger.info(f"  Within 50 points: {accuracy_50:.1f}%")
        
        # Enhanced confidence metrics logging
        if ENABLE_CONFIDENCE_SCORES:
            logger.info("")
            logger.info(f"🔮 Confidence Score Analysis:")
            logger.info(f"  Mean Confidence: {np.mean(confidence_scores_final):.3f}")
            logger.info(f"  High Confidence Predictions (≥{MIN_CONFIDENCE_THRESHOLD}): {final_evaluation['high_confidence_coverage']:.1%}")
            logger.info(f"  High Confidence MAE: {final_evaluation['high_confidence_mae']:.2f} points")
            logger.info(f"  Mean Prediction Uncertainty: {np.mean(prediction_std_final):.2f} points")
            
            if confidence_intervals_final:
                logger.info(f"📏 Confidence Intervals:")
                for conf_level, intervals in confidence_intervals_final.items():
                    mean_width = np.mean(intervals['width'])
                    logger.info(f"  {conf_level} Average Width: ±{mean_width:.1f} points")
        
        # Compare Phase 1 vs Phase 2 performance (ensure test_mae is defined)
        logger.info("")
        logger.info("⚖️  PHASE 1 vs PHASE 2 COMPARISON:")
        logger.info("=" * 50)
        
        # Get Phase 1 results for comparison
        test_loss_phase1, test_mae_phase1, test_mse_phase1 = best_model.evaluate(X_test, y_test, verbose=0)
        
        mae_improvement = test_mae_phase1 - test_mae_final
        mae_improvement_pct = (mae_improvement / test_mae_phase1) * 100
        
        logger.info(f"📊 Model Performance Comparison:")
        logger.info(f"  Phase 1 (AutoML): {test_mae_phase1:.2f} MAE on {len(X_test):,} samples")
        logger.info(f"  Phase 2 (Final):  {test_mae_final:.2f} MAE on {len(X_test_final):,} samples")
        logger.info(f"  Improvement: {mae_improvement:+.2f} points ({mae_improvement_pct:+.1f}%)")
        
        logger.info(f"📈 Dataset Size Comparison:")
        logger.info(f"  Phase 1 Training: {len(X_train):,} samples")
        logger.info(f"  Phase 2 Training: {len(X_train_final):,} samples")
        logger.info(f"  Size Increase: {len(X_train_final) - len(X_train):+,} samples ({(len(X_train_final)/len(X_train) - 1)*100:+.0f}%)")
        
        logger.info(f"⏱️  Training Time Comparison:")
        logger.info(f"  Phase 1 (Search): {search_duration}")
        logger.info(f"  Phase 2 (Training): {training_duration}")
        logger.info(f"  Total Time: {phase1_duration + phase2_duration}")
        
        # Save final model
        logger.info("")
        logger.info("💾 Saving final model and results...")
        
        final_model_path = 'saved_models/vfl_automl_final_model.keras'
        final_model.save(final_model_path)
        logger.info(f"✅ Final model saved: {final_model_path}")
        
        # Save comprehensive results (enhanced with confidence scores)
        final_results = {
            'phase1_results': phase1_results,
            'phase2_config': {
                'final_sample_size': FINAL_SAMPLE_SIZE,
                'final_epochs': FINAL_EPOCHS,
                'training_samples': len(X_train_final),
                'test_samples': len(X_test_final),
                'early_stopping_patience': 10,
                'confidence_scoring_enabled': ENABLE_CONFIDENCE_SCORES,
                'mc_dropout_samples': MC_DROPOUT_SAMPLES if ENABLE_CONFIDENCE_SCORES else None
            },
            'phase2_performance': {
                'test_loss': test_loss_final,
                'test_mae': test_mae_final,
                'test_rmse': test_rmse_final,
                'test_r2': r2_final,
                'test_mape': mape_final,
                'accuracy_within_10': accuracy_10,
                'accuracy_within_20': accuracy_20,
                'accuracy_within_30': accuracy_30,
                'accuracy_within_50': accuracy_50,
                'mean_confidence': float(np.mean(confidence_scores_final)) if ENABLE_CONFIDENCE_SCORES else None,
                'high_confidence_coverage': final_evaluation['high_confidence_coverage'] if ENABLE_CONFIDENCE_SCORES else None,
                'high_confidence_mae': final_evaluation['high_confidence_mae'] if ENABLE_CONFIDENCE_SCORES else None,
                'mean_prediction_uncertainty': float(np.mean(prediction_std_final)) if ENABLE_CONFIDENCE_SCORES else None,
                'calibration_scores': final_evaluation['calibration_scores'] if ENABLE_CONFIDENCE_SCORES else None
            },
            'comparison': {
                'phase1_mae': test_mae_phase1,
                'phase2_mae': test_mae_final,
                'mae_improvement': mae_improvement,
                'mae_improvement_percent': mae_improvement_pct,
                'phase1_samples': len(X_train),
                'phase2_samples': len(X_train_final)
            },
            'timing': {
                'phase1_duration_seconds': phase1_duration.total_seconds(),
                'phase2_training_duration_seconds': training_duration.total_seconds(),
                'phase2_total_duration_seconds': phase2_duration.total_seconds(),
                'total_duration_seconds': (phase1_duration + phase2_duration).total_seconds()
            },
            'final_architecture': best_hp.values,
            'training_history': {
                'epochs_completed': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_train_mae': float(history.history['mae'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_val_mae': float(history.history['val_mae'][-1])
            }
        }
        
        results_path = 'automl_results/final_results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"📁 Complete results saved: {results_path}")
        
        # Detailed sample predictions for Phase 2 with confidence scores
        print_detailed_sample_predictions_with_confidence(X_test_final, y_test_final, ids_test_final, final_model,
                                        auto_repr_size_final, digital_repr_size_final, home_repr_size_final, credit_card_repr_size_final, 
                                        phase="Phase 2 Final")
        
        # Update intermediate representations CSV with final model predictions
        logger.info("🔄 Updating intermediate representations CSV with final model predictions...")
        add_predictions_to_representations_csv(final_model, X_combined_final, y_combined_final, ids_combined_final)
        
        total_duration = datetime.now() - start_time
        logger.info("")
        logger.info("🎉 VFL AutoML COMPLETE - Two-Phase Training Finished")
        logger.info("=" * 80)
        logger.info(f"🏆 FINAL RESULTS SUMMARY:")
        logger.info(f"  Best Architecture MAE: {test_mae_final:.2f} points")
        logger.info(f"  Prediction Accuracy (±20 pts): {accuracy_20:.1f}%")
        logger.info(f"  Model Parameters: {final_model.count_params():,}")
        logger.info(f"  Total Duration: {total_duration}")
        logger.info(f"  Training Samples: {len(X_train_final):,}")
        logger.info(f"  Architecture: {best_hp.get('num_layers')} layers")
        logger.info("")
        logger.info(f"📁 Saved Files:")
        logger.info(f"  Final Model: {final_model_path}")
        logger.info(f"  Complete Results: {results_path}")
        logger.info(f"  Intermediate Data: Updated CSV with predictions")
        logger.info("=" * 80)
        
        return final_model, best_hp, final_results
        
    except Exception as e:
        logger.error(f"❌ Error during AutoML search: {str(e)}")
        raise

def print_detailed_sample_predictions(X_test, y_test, ids_test, y_pred, auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size, phase="Final"):
    """Log detailed sample predictions for the model - NOW WITH FOUR BANKS"""
    
    logger.info("\n" + "=" * 130)
    if phase == "Phase 1":
        logger.info(f"📊 DETAILED SAMPLE PREDICTIONS - {phase} MODEL (AutoML Search Results)")
        logger.info(f"📋 Note: These predictions are from the best architecture found during AutoML search")
        logger.info(f"📊 Dataset: {len(y_test):,} test samples from AutoML search dataset")
    else:
        logger.info(f"📊 DETAILED SAMPLE PREDICTIONS - FINAL VFL MODEL (Four Banks)")
        logger.info(f"📋 Note: These predictions are from the final model trained on larger dataset")
        logger.info(f"📊 Dataset: {len(y_test):,} test samples from final training dataset")
    logger.info("=" * 130)
    logger.info(f"Showing 10 randomly selected customers from test set with actual vs predicted credit scores")
    logger.info(f"Credit score range: 300-850 points")
    logger.info("")
    
    # Randomly sample 10 customers for detailed analysis
    n_samples = min(10, len(ids_test))
    np.random.seed(RANDOM_SEED)  # For reproducible results
    random_indices = np.random.choice(len(ids_test), size=n_samples, replace=False)
    random_indices = sorted(random_indices)  # Sort for consistent ordering
    
    logger.info(f"🎯 Random sample indices (seed={RANDOM_SEED}): {random_indices}")
    logger.info("")
    
    # Enhanced table header for four banks
    logger.info(f"{'#':<3} {'Index':<7} {'Tax ID':<15} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Error%':<8} {'Auto':<6} {'Digital':<8} {'Home':<6} {'Credit':<8} {'Services':<20} {'Score Range':<12}")
    logger.info("-" * 140)
    
    total_error = 0
    for sample_num, i in enumerate(random_indices):
        tax_id = ids_test[i]
        actual = y_test[i]
        predicted = y_pred.flatten()[i]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100 if actual > 0 else 0
        total_error += error
        
        # Service information for four banks - Calculate positions correctly
        auto_mask_pos = auto_repr_size
        digital_mask_pos = auto_repr_size + 1 + digital_repr_size
        home_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size
        credit_card_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size + 1 + credit_card_repr_size
        
        has_auto = "Yes" if X_test[i, auto_mask_pos] == 1 else "No"
        has_digital = "Yes" if X_test[i, digital_mask_pos] == 1 else "No"
        has_home = "Yes" if X_test[i, home_mask_pos] == 1 else "No"
        has_credit_card = "Yes" if X_test[i, credit_card_mask_pos] == 1 else "No"
        
        # Determine service combination
        service_count = sum([has_auto == "Yes", has_digital == "Yes", has_home == "Yes", has_credit_card == "Yes"])
        if service_count == 4:
            services = "All Four"
        elif service_count == 3:
            if has_auto == "Yes" and has_digital == "Yes" and has_home == "Yes":
                services = "Auto+Dig+Home"
            elif has_auto == "Yes" and has_digital == "Yes" and has_credit_card == "Yes":
                services = "Auto+Dig+Card"
            elif has_auto == "Yes" and has_home == "Yes" and has_credit_card == "Yes":
                services = "Auto+Home+Card"
            elif has_digital == "Yes" and has_home == "Yes" and has_credit_card == "Yes":
                services = "Dig+Home+Credit"
        elif service_count == 2:
            if has_auto == "Yes" and has_digital == "Yes":
                services = "Auto+Digital"
            elif has_auto == "Yes" and has_home == "Yes":
                services = "Auto+Home"
            elif has_auto == "Yes" and has_credit_card == "Yes":
                services = "Auto+Credit"
            elif has_digital == "Yes" and has_home == "Yes":
                services = "Digital+Home"
            elif has_digital == "Yes" and has_credit_card == "Yes":
                services = "Digital+Credit"
            elif has_home == "Yes" and has_credit_card == "Yes":
                services = "Home+Credit"
        elif service_count == 1:
            if has_auto == "Yes":
                services = "Auto Only"
            elif has_digital == "Yes":
                services = "Digital Only"
            elif has_home == "Yes":
                services = "Home Only"
            elif has_credit_card == "Yes":
                services = "Credit Only"
        else:
            services = "None"
        
        # Credit score range classification
        if actual >= 750:
            score_range = "Excellent"
        elif actual >= 700:
            score_range = "Good"
        elif actual >= 650:
            score_range = "Fair"
        elif actual >= 600:
            score_range = "Poor"
        else:
            score_range = "Very Poor"
            
        logger.info(f"{sample_num+1:<3} {i:<7} {tax_id:<15} {actual:<8.0f} {predicted:<10.1f} {error:<8.1f} {error_pct:<8.1f} {has_auto:<6} {has_digital:<8} {has_home:<6} {has_credit_card:<8} {services:<20} {score_range:<12}")
    
    # Summary statistics for the random sample
    avg_error = total_error / n_samples
    sample_actual_scores = y_test[random_indices]
    sample_predicted_scores = y_pred.flatten()[random_indices]
    
    logger.info("-" * 140)
    logger.info(f"📈 Random Sample Statistics ({n_samples} customers - {phase}):")
    logger.info(f"   Average Error: {avg_error:.1f} points")
    logger.info(f"   Min Actual Score: {sample_actual_scores.min():.0f}")
    logger.info(f"   Max Actual Score: {sample_actual_scores.max():.0f}")
    logger.info(f"   Actual Score Range: {sample_actual_scores.max() - sample_actual_scores.min():.0f} points")
    logger.info(f"   Prediction Range: {sample_predicted_scores.min():.1f} - {sample_predicted_scores.max():.1f}")
    logger.info(f"   Sample Standard Deviation (Actual): {sample_actual_scores.std():.1f}")
    logger.info(f"   Sample Standard Deviation (Predicted): {sample_predicted_scores.std():.1f}")
    
    # Service distribution in random sample for four banks
    auto_mask_pos = auto_repr_size
    digital_mask_pos = auto_repr_size + 1 + digital_repr_size
    home_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size
    credit_card_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size + 1 + credit_card_repr_size
    
    sample_auto = sum(1 for i in random_indices if X_test[i, auto_mask_pos] == 1)
    sample_digital = sum(1 for i in random_indices if X_test[i, digital_mask_pos] == 1)
    sample_home = sum(1 for i in random_indices if X_test[i, home_mask_pos] == 1)
    sample_credit_card = sum(1 for i in random_indices if X_test[i, credit_card_mask_pos] == 1)
    
    sample_all_four = sum(1 for i in random_indices if 
                         X_test[i, auto_mask_pos] == 1 and 
                         X_test[i, digital_mask_pos] == 1 and 
                         X_test[i, home_mask_pos] == 1 and
                         X_test[i, credit_card_mask_pos] == 1)
    
    sample_none = sum(1 for i in random_indices if 
                     X_test[i, auto_mask_pos] == 0 and 
                     X_test[i, digital_mask_pos] == 0 and 
                     X_test[i, home_mask_pos] == 0 and
                     X_test[i, credit_card_mask_pos] == 0)
    
    logger.info(f"   Service Distribution in Sample:")
    logger.info(f"     - Auto: {sample_auto}/{n_samples} ({sample_auto/n_samples*100:.0f}%)")
    logger.info(f"     - Digital: {sample_digital}/{n_samples} ({sample_digital/n_samples*100:.0f}%)")
    logger.info(f"     - Home: {sample_home}/{n_samples} ({sample_home/n_samples*100:.0f}%)")
    logger.info(f"     - Credit: {sample_credit_card}/{n_samples} ({sample_credit_card/n_samples*100:.0f}%)")
    logger.info(f"   Service Combinations in Sample:")
    logger.info(f"     - All Four Services: {sample_all_four}/{n_samples} ({sample_all_four/n_samples*100:.0f}%)")
    logger.info(f"     - No Services: {sample_none}/{n_samples} ({sample_none/n_samples*100:.0f}%)")
    
    if phase == "Phase 1":
        logger.info(f"   📝 Note: Random sample from AutoML search with {AUTOML_SAMPLE_SIZE:,} customers")
    else:
        logger.info(f"   📝 Note: Random sample from final training with {FINAL_SAMPLE_SIZE:,} customers")
    logger.info(f"   🎲 Random seed used: {RANDOM_SEED} (for reproducibility)")
    logger.info("=" * 140)

def add_predictions_to_representations_csv(model, X_combined, y_combined, ids_combined):
    """Add predictions and confidence scores to the latest intermediate representations CSV file"""
    
    try:
        # Find the most recent intermediate representations CSV file
        csv_pattern = 'data/intermediate_representations/vfl_representations_sample_*.csv'
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            logger.warning("⚠️  No intermediate representations CSV files found to update")
            return
        
        # Get the most recent file
        latest_csv = max(csv_files, key=os.path.getmtime)
        logger.info(f"📝 Updating CSV file: {latest_csv}")
        
        # Read the existing CSV
        df = pd.read_csv(latest_csv)
        
        # Generate predictions and confidence scores for all customers in the combined dataset
        logger.debug("Generating predictions and confidence scores for all customers...")
        
        if ENABLE_CONFIDENCE_SCORES:
            conf_results = calculate_confidence_scores(model, X_combined, enable_mc_dropout=True, mc_samples=MC_DROPOUT_SAMPLES)
            all_predictions = conf_results['predictions']
            all_confidence_scores = conf_results['confidence_scores']
            all_prediction_std = conf_results['prediction_std']
            all_confidence_categories = conf_results['confidence_categories']
            all_confidence_intervals = conf_results['confidence_intervals']
        else:
            all_predictions = model.predict(X_combined, verbose=0).flatten()
            all_confidence_scores = np.ones_like(all_predictions) * 0.5
            all_prediction_std = np.zeros_like(all_predictions)
            all_confidence_categories = np.array(['N/A'] * len(all_predictions))
            all_confidence_intervals = {}
        
        # Create mappings from tax_id to predictions and confidence metrics
        prediction_map = dict(zip(ids_combined, all_predictions))
        confidence_map = dict(zip(ids_combined, all_confidence_scores))
        uncertainty_map = dict(zip(ids_combined, all_prediction_std))
        confidence_category_map = dict(zip(ids_combined, all_confidence_categories))
        
        # Add predictions and confidence scores to the dataframe
        df['predicted_credit_score'] = df['tax_id'].map(prediction_map)
        
        if ENABLE_CONFIDENCE_SCORES:
            df['confidence_score'] = df['tax_id'].map(confidence_map)
            df['prediction_uncertainty'] = df['tax_id'].map(uncertainty_map)
            df['confidence_category'] = df['tax_id'].map(confidence_category_map)
            
            # Add confidence intervals if available
            if '68%' in all_confidence_intervals:
                ci_68_lower_map = dict(zip(ids_combined, all_confidence_intervals['68%']['lower']))
                ci_68_upper_map = dict(zip(ids_combined, all_confidence_intervals['68%']['upper']))
                df['ci_68_lower'] = df['tax_id'].map(ci_68_lower_map)
                df['ci_68_upper'] = df['tax_id'].map(ci_68_upper_map)
                df['ci_68_width'] = df['ci_68_upper'] - df['ci_68_lower']
            
            if '95%' in all_confidence_intervals:
                ci_95_lower_map = dict(zip(ids_combined, all_confidence_intervals['95%']['lower']))
                ci_95_upper_map = dict(zip(ids_combined, all_confidence_intervals['95%']['upper']))
                df['ci_95_lower'] = df['tax_id'].map(ci_95_lower_map)
                df['ci_95_upper'] = df['tax_id'].map(ci_95_upper_map)
                df['ci_95_width'] = df['ci_95_upper'] - df['ci_95_lower']
        
        # Calculate prediction error and accuracy metrics for the sample
        df['prediction_error'] = abs(df['credit_score'] - df['predicted_credit_score'])
        df['prediction_error_pct'] = (df['prediction_error'] / df['credit_score']) * 100
        
        # Add prediction quality categories
        def categorize_prediction_quality(error_pct):
            if error_pct <= 5:
                return "Excellent"
            elif error_pct <= 10:
                return "Good"
            elif error_pct <= 15:
                return "Fair"
            elif error_pct <= 25:
                return "Poor"
            else:
                return "Very Poor"
        
        df['prediction_quality'] = df['prediction_error_pct'].apply(categorize_prediction_quality)
        
        # Add credit score range categories for both actual and predicted
        def categorize_credit_score(score):
            if score >= 750:
                return "Excellent"
            elif score >= 700:
                return "Good"
            elif score >= 650:
                return "Fair"
            elif score >= 600:
                return "Poor"
            else:
                return "Very Poor"
        
        df['actual_score_category'] = df['credit_score'].apply(categorize_credit_score)
        df['predicted_score_category'] = df['predicted_credit_score'].apply(categorize_credit_score)
        df['category_match'] = df['actual_score_category'] == df['predicted_score_category']
        
        # Add high confidence flag
        if ENABLE_CONFIDENCE_SCORES:
            df['high_confidence'] = df['confidence_score'] >= MIN_CONFIDENCE_THRESHOLD
        
        # Reorder columns for better readability
        key_columns = [
            'tax_id', 'credit_score', 'predicted_credit_score', 'prediction_error', 'prediction_error_pct', 
            'prediction_quality', 'actual_score_category', 'predicted_score_category', 'category_match'
        ]
        
        # Add confidence columns if enabled
        if ENABLE_CONFIDENCE_SCORES:
            confidence_columns = ['confidence_score', 'prediction_uncertainty', 'confidence_category', 'high_confidence']
            key_columns.extend(confidence_columns)
            
            # Add confidence interval columns if available
            if 'ci_68_lower' in df.columns:
                key_columns.extend(['ci_68_lower', 'ci_68_upper', 'ci_68_width'])
            if 'ci_95_lower' in df.columns:
                key_columns.extend(['ci_95_lower', 'ci_95_upper', 'ci_95_width'])
        
        # Add service information
        service_columns = ['service_label', 'service_combination', 'service_count', 'has_auto', 'has_digital', 'has_home', 'has_credit_card']
        key_columns.extend(service_columns)
        
        # Get representation columns (all columns starting with bank names)
        repr_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in 
                       ['auto_repr_', 'digital_repr_', 'home_repr_', 'credit_card_repr_', 'auto_mask', 'digital_mask', 'home_mask', 'credit_card_mask'])]
        
        # Combine and reorder columns
        final_columns = key_columns + repr_columns
        missing_columns = [col for col in final_columns if col in df.columns]
        df = df[missing_columns]
        
        # Save the updated CSV
        df.to_csv(latest_csv, index=False)
        
        # Log summary statistics
        logger.info("✅ CSV updated with predictions and confidence scores!")
        logger.info(f"📊 Prediction Summary for Sample ({len(df)} customers):")
        logger.info(f"   - Average Error: {df['prediction_error'].mean():.1f} points")
        logger.info(f"   - Average Error %: {df['prediction_error_pct'].mean():.1f}%")
        logger.info(f"   - Best Prediction Error: {df['prediction_error'].min():.1f} points")
        logger.info(f"   - Worst Prediction Error: {df['prediction_error'].max():.1f} points")
        
        if ENABLE_CONFIDENCE_SCORES:
            logger.info(f"🔮 Confidence Score Summary:")
            logger.info(f"   - Mean Confidence: {df['confidence_score'].mean():.3f}")
            logger.info(f"   - High Confidence Predictions (≥{MIN_CONFIDENCE_THRESHOLD}): {df['high_confidence'].sum()}/{len(df)} ({df['high_confidence'].mean()*100:.1f}%)")
            logger.info(f"   - Mean Prediction Uncertainty: {df['prediction_uncertainty'].mean():.1f} points")
            
            # Confidence category distribution
            conf_cat_dist = df['confidence_category'].value_counts()
            logger.info(f"📈 Confidence Category Distribution:")
            for category, count in conf_cat_dist.items():
                percentage = (count / len(df)) * 100
                logger.info(f"   - {category}: {count} customers ({percentage:.1f}%)")
        
        # Quality distribution
        quality_dist = df['prediction_quality'].value_counts()
        logger.info(f"📈 Prediction Quality Distribution:")
        for quality, count in quality_dist.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   - {quality}: {count} customers ({percentage:.1f}%)")
        
        # Category matching
        category_matches = df['category_match'].sum()
        category_match_pct = (category_matches / len(df)) * 100
        logger.info(f"🎯 Credit Score Category Accuracy: {category_matches}/{len(df)} ({category_match_pct:.1f}%)")
        
        # Service-specific performance
        logger.info(f"📊 Performance by Service Combination:")
        performance_cols = ['prediction_error', 'prediction_error_pct', 'category_match']
        if ENABLE_CONFIDENCE_SCORES:
            performance_cols.extend(['confidence_score', 'high_confidence'])
        
        service_performance = df.groupby('service_label')[performance_cols].agg({
            'prediction_error': 'mean',
            'prediction_error_pct': 'mean',
            'category_match': 'mean',
            'confidence_score': 'mean' if ENABLE_CONFIDENCE_SCORES else lambda x: None,
            'high_confidence': 'mean' if ENABLE_CONFIDENCE_SCORES else lambda x: None
        }).round(3)
        
        for service_type in service_performance.index:
            error = service_performance.loc[service_type, 'prediction_error']
            error_pct = service_performance.loc[service_type, 'prediction_error_pct']
            accuracy = service_performance.loc[service_type, 'category_match'] * 100
            
            if ENABLE_CONFIDENCE_SCORES:
                conf = service_performance.loc[service_type, 'confidence_score']
                high_conf = service_performance.loc[service_type, 'high_confidence'] * 100
                logger.info(f"   - {service_type}: {error:.1f} pts error ({error_pct:.1f}%), {accuracy:.0f}% category accuracy, {conf:.3f} avg confidence, {high_conf:.0f}% high confidence")
            else:
                logger.info(f"   - {service_type}: {error:.1f} pts error ({error_pct:.1f}%), {accuracy:.0f}% category accuracy")
        
        logger.info(f"💾 Updated file saved: {latest_csv}")
        
    except Exception as e:
        logger.error(f"❌ Error updating intermediate representations CSV: {str(e)}")
        logger.debug(f"Error details: {e}", exc_info=True)

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
        
        # Calculate confidence intervals
        for confidence_level in CONFIDENCE_INTERVALS:
            alpha = (100 - confidence_level) / 100
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(mc_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(mc_predictions, upper_percentile, axis=0)
            
            results['confidence_intervals'][f'{confidence_level}%'] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        # Calculate confidence scores (inverse of normalized uncertainty)
        max_std = np.max(prediction_std) if np.max(prediction_std) > 0 else 1.0
        normalized_uncertainty = prediction_std / max_std
        confidence_scores = 1.0 - normalized_uncertainty
        
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
        predictions = model.predict(X_data, verbose=0).flatten()
        
        # For standard inference, use a simple heuristic for confidence
        # Based on distance from mean prediction
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Confidence based on how close prediction is to the mean
        if std_pred > 0:
            normalized_distance = np.abs(predictions - mean_pred) / std_pred
            confidence_scores = np.exp(-normalized_distance / 2)  # Gaussian-like confidence
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
        predictions = model.predict(X_test, verbose=0).flatten()
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

def print_detailed_sample_predictions_with_confidence(X_test, y_test, ids_test, model, auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size, phase="Final"):
    """Log detailed sample predictions with confidence scores for the model - NOW WITH FOUR BANKS AND CONFIDENCE"""
    
    logger.info("\n" + "=" * 180)
    if phase == "Phase 1":
        logger.info(f"📊 DETAILED SAMPLE PREDICTIONS WITH CONFIDENCE - {phase} MODEL (AutoML Search Results)")
        logger.info(f"📋 Note: These predictions are from the best architecture found during AutoML search")
        logger.info(f"📊 Dataset: {len(y_test):,} test samples from AutoML search dataset")
    else:
        logger.info(f"📊 DETAILED SAMPLE PREDICTIONS WITH CONFIDENCE - FINAL VFL MODEL (Four Banks)")
        logger.info(f"📋 Note: These predictions are from the final model trained on larger dataset")
        logger.info(f"📊 Dataset: {len(y_test):,} test samples from final training dataset")
    logger.info("=" * 180)
    logger.info(f"Showing 10 randomly selected customers from test set with actual vs predicted credit scores")
    logger.info(f"Credit score range: 300-850 points")
    logger.info("")
    
    # Calculate confidence scores for the sample
    if ENABLE_CONFIDENCE_SCORES:
        logger.info("🔮 Calculating confidence scores for sample predictions...")
        conf_results = calculate_confidence_scores(model, X_test, enable_mc_dropout=True, mc_samples=MC_DROPOUT_SAMPLES)
        predictions = conf_results['predictions']
        confidence_scores = conf_results['confidence_scores']
        prediction_std = conf_results['prediction_std']
        confidence_categories = conf_results['confidence_categories']
        confidence_intervals = conf_results['confidence_intervals']
    else:
        predictions = model.predict(X_test, verbose=0).flatten()
        confidence_scores = np.ones_like(predictions) * 0.5
        prediction_std = np.zeros_like(predictions)
        confidence_categories = np.array(['N/A'] * len(predictions))
        confidence_intervals = {}
    
    # Randomly sample 10 customers for detailed analysis
    n_samples = min(10, len(ids_test))
    np.random.seed(RANDOM_SEED)  # For reproducible results
    random_indices = np.random.choice(len(ids_test), size=n_samples, replace=False)
    random_indices = sorted(random_indices)  # Sort for consistent ordering
    
    logger.info(f"🎯 Random sample indices (seed={RANDOM_SEED}): {random_indices}")
    logger.info("")
    
    # Enhanced table header for four banks with confidence
    if ENABLE_CONFIDENCE_SCORES:
        logger.info(f"{'#':<3} {'Index':<7} {'Tax ID':<15} {'Actual':<8} {'Predicted':<10} {'Confidence':<10} {'Uncertainty':<11} {'CI (68%)':<15} {'CI (95%)':<15} {'Error':<8} {'Error%':<8} {'Services':<20} {'Conf. Level':<12}")
        logger.info("-" * 180)
    else:
        logger.info(f"{'#':<3} {'Index':<7} {'Tax ID':<15} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Error%':<8} {'Auto':<6} {'Digital':<8} {'Home':<6} {'Credit':<8} {'Services':<20} {'Score Range':<12}")
        logger.info("-" * 140)
    
    total_error = 0
    for sample_num, i in enumerate(random_indices):
        tax_id = ids_test[i]
        actual = y_test[i]
        predicted = predictions[i]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100 if actual > 0 else 0
        total_error += error
        
        # Service information for four banks - Calculate positions correctly
        auto_mask_pos = auto_repr_size
        digital_mask_pos = auto_repr_size + 1 + digital_repr_size
        home_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size
        credit_card_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size + 1 + credit_card_repr_size
        
        has_auto = "Yes" if X_test[i, auto_mask_pos] == 1 else "No"
        has_digital = "Yes" if X_test[i, digital_mask_pos] == 1 else "No"
        has_home = "Yes" if X_test[i, home_mask_pos] == 1 else "No"
        has_credit_card = "Yes" if X_test[i, credit_card_mask_pos] == 1 else "No"
        
        # Determine service combination
        service_count = sum([has_auto == "Yes", has_digital == "Yes", has_home == "Yes", has_credit_card == "Yes"])
        if service_count == 4:
            services = "All Four"
        elif service_count == 3:
            if has_auto == "Yes" and has_digital == "Yes" and has_home == "Yes":
                services = "Auto+Dig+Home"
            elif has_auto == "Yes" and has_digital == "Yes" and has_credit_card == "Yes":
                services = "Auto+Dig+Card"
            elif has_auto == "Yes" and has_home == "Yes" and has_credit_card == "Yes":
                services = "Auto+Home+Card"
            elif has_digital == "Yes" and has_home == "Yes" and has_credit_card == "Yes":
                services = "Dig+Home+Credit"
        elif service_count == 2:
            if has_auto == "Yes" and has_digital == "Yes":
                services = "Auto+Digital"
            elif has_auto == "Yes" and has_home == "Yes":
                services = "Auto+Home"
            elif has_auto == "Yes" and has_credit_card == "Yes":
                services = "Auto+Credit"
            elif has_digital == "Yes" and has_home == "Yes":
                services = "Digital+Home"
            elif has_digital == "Yes" and has_credit_card == "Yes":
                services = "Digital+Credit"
            elif has_home == "Yes" and has_credit_card == "Yes":
                services = "Home+Credit"
        elif service_count == 1:
            if has_auto == "Yes":
                services = "Auto Only"
            elif has_digital == "Yes":
                services = "Digital Only"
            elif has_home == "Yes":
                services = "Home Only"
            elif has_credit_card == "Yes":
                services = "Credit Only"
        else:
            services = "None"
        
        if ENABLE_CONFIDENCE_SCORES:
            # Enhanced output with confidence information including both intervals
            conf = confidence_scores[i]
            uncertainty = prediction_std[i]
            conf_category = confidence_categories[i]
            
            # 68% confidence interval
            if '68%' in confidence_intervals:
                ci_68_lower = confidence_intervals['68%']['lower'][i]
                ci_68_upper = confidence_intervals['68%']['upper'][i]
                ci_68_range = f"{ci_68_lower:.0f}-{ci_68_upper:.0f}"
            else:
                ci_68_range = "N/A"
            
            # 95% confidence interval
            if '95%' in confidence_intervals:
                ci_95_lower = confidence_intervals['95%']['lower'][i]
                ci_95_upper = confidence_intervals['95%']['upper'][i]
                ci_95_range = f"{ci_95_lower:.0f}-{ci_95_upper:.0f}"
            else:
                ci_95_range = "N/A"
            
            logger.info(f"{sample_num+1:<3} {i:<7} {tax_id:<15} {actual:<8.0f} {predicted:<10.1f} {conf:<10.3f} {uncertainty:<11.1f} {ci_68_range:<15} {ci_95_range:<15} {error:<8.1f} {error_pct:<8.1f} {services:<20} {conf_category:<12}")
        else:
            # Credit score range classification
            if actual >= 750:
                score_range = "Excellent"
            elif actual >= 700:
                score_range = "Good"
            elif actual >= 650:
                score_range = "Fair"
            elif actual >= 600:
                score_range = "Poor"
            else:
                score_range = "Very Poor"
                
            logger.info(f"{sample_num+1:<3} {i:<7} {tax_id:<15} {actual:<8.0f} {predicted:<10.1f} {error:<8.1f} {error_pct:<8.1f} {has_auto:<6} {has_digital:<8} {has_home:<6} {has_credit_card:<8} {services:<20} {score_range:<12}")
    
    # Summary statistics for the random sample
    avg_error = total_error / n_samples
    sample_actual_scores = y_test[random_indices]
    sample_predicted_scores = predictions[random_indices]
    
    logger.info("-" * (180 if ENABLE_CONFIDENCE_SCORES else 140))
    logger.info(f"📈 Random Sample Statistics ({n_samples} customers - {phase}):")
    logger.info(f"   Average Error: {avg_error:.1f} points")
    logger.info(f"   Min Actual Score: {sample_actual_scores.min():.0f}")
    logger.info(f"   Max Actual Score: {sample_actual_scores.max():.0f}")
    logger.info(f"   Actual Score Range: {sample_actual_scores.max() - sample_actual_scores.min():.0f} points")
    logger.info(f"   Prediction Range: {sample_predicted_scores.min():.1f} - {sample_predicted_scores.max():.1f}")
    logger.info(f"   Sample Standard Deviation (Actual): {sample_actual_scores.std():.1f}")
    logger.info(f"   Sample Standard Deviation (Predicted): {sample_predicted_scores.std():.1f}")
    
    if ENABLE_CONFIDENCE_SCORES:
        sample_confidence = confidence_scores[random_indices]
        sample_uncertainty = prediction_std[random_indices]
        logger.info(f"🔮 Confidence Statistics for Sample:")
        logger.info(f"   Mean Confidence: {sample_confidence.mean():.3f}")
        logger.info(f"   Min Confidence: {sample_confidence.min():.3f}")
        logger.info(f"   Max Confidence: {sample_confidence.max():.3f}")
        logger.info(f"   Mean Uncertainty: {sample_uncertainty.mean():.1f} points")
        logger.info(f"   High Confidence Predictions (≥{MIN_CONFIDENCE_THRESHOLD}): {np.sum(sample_confidence >= MIN_CONFIDENCE_THRESHOLD)}/{n_samples}")
    
    # Service distribution in random sample for four banks
    auto_mask_pos = auto_repr_size
    digital_mask_pos = auto_repr_size + 1 + digital_repr_size
    home_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size
    credit_card_mask_pos = auto_repr_size + 1 + digital_repr_size + 1 + home_repr_size + 1 + credit_card_repr_size
    
    sample_auto = sum(1 for i in random_indices if X_test[i, auto_mask_pos] == 1)
    sample_digital = sum(1 for i in random_indices if X_test[i, digital_mask_pos] == 1)
    sample_home = sum(1 for i in random_indices if X_test[i, home_mask_pos] == 1)
    sample_credit_card = sum(1 for i in random_indices if X_test[i, credit_card_mask_pos] == 1)
    
    sample_all_four = sum(1 for i in random_indices if 
                         X_test[i, auto_mask_pos] == 1 and 
                         X_test[i, digital_mask_pos] == 1 and 
                         X_test[i, home_mask_pos] == 1 and
                         X_test[i, credit_card_mask_pos] == 1)
    
    sample_none = sum(1 for i in random_indices if 
                     X_test[i, auto_mask_pos] == 0 and 
                     X_test[i, digital_mask_pos] == 0 and 
                     X_test[i, home_mask_pos] == 0 and
                     X_test[i, credit_card_mask_pos] == 0)
    
    logger.info(f"   Service Distribution in Sample:")
    logger.info(f"     - Auto: {sample_auto}/{n_samples} ({sample_auto/n_samples*100:.0f}%)")
    logger.info(f"     - Digital: {sample_digital}/{n_samples} ({sample_digital/n_samples*100:.0f}%)")
    logger.info(f"     - Home: {sample_home}/{n_samples} ({sample_home/n_samples*100:.0f}%)")
    logger.info(f"     - Credit: {sample_credit_card}/{n_samples} ({sample_credit_card/n_samples*100:.0f}%)")
    logger.info(f"   Service Combinations in Sample:")
    logger.info(f"     - All Four Services: {sample_all_four}/{n_samples} ({sample_all_four/n_samples*100:.0f}%)")
    logger.info(f"     - No Services: {sample_none}/{n_samples} ({sample_none/n_samples*100:.0f}%)")
    
    if phase == "Phase 1":
        logger.info(f"   📝 Note: Random sample from AutoML search with {AUTOML_SAMPLE_SIZE:,} customers")
    else:
        logger.info(f"   📝 Note: Random sample from final training with {FINAL_SAMPLE_SIZE:,} customers")
    logger.info(f"   🎲 Random seed used: {RANDOM_SEED} (for reproducibility)")
    if ENABLE_CONFIDENCE_SCORES:
        logger.info(f"   🔮 Confidence scores calculated using Monte Carlo Dropout ({MC_DROPOUT_SAMPLES} samples)")
    logger.info("=" * (180 if ENABLE_CONFIDENCE_SCORES else 140))

if __name__ == "__main__":
    run_automl_search()