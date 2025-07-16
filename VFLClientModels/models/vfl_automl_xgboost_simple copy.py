import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import logging
import sys
from logging.handlers import RotatingFileHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# ===================== Logging Setup =====================
def setup_logging():
    os.makedirs('logs', exist_ok=True)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('VFL_AutoML_XGBoost_Simple')
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = RotatingFileHandler(f'logs/vfl_automl_xgboost_simple_{timestamp}.log', maxBytes=10*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    tf.get_logger().setLevel('ERROR')
    logger.info("=" * 80)
    logger.info("VFL AutoML XGBoost Simple Logging Initialized")
    logger.info(f"Log file: logs/vfl_automl_xgboost_simple_{timestamp}.log")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    return logger
logger = setup_logging()

# ===================== Config =====================
RANDOM_SEED = 42
XGBOOST_OUTPUT_DIM = 12
ENABLE_CONFIDENCE_SCORES = True
MC_DROPOUT_SAMPLES = 30
CONFIDENCE_INTERVALS = [68, 95]
MIN_CONFIDENCE_THRESHOLD = 0.7

# ===================== XGBoost Credit Card Feature Extraction =====================
def load_xgboost_credit_card_model():
    model_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_independent.pkl'
    scaler_path = 'VFLClientModels/models/saved_models/credit_card_scaler.pkl'
    feature_names_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_feature_names.npy'
    pca_path = 'VFLClientModels/models/saved_models/credit_card_xgboost_pca.pkl'
    model_data = joblib.load(model_path)
    if isinstance(model_data, dict):
        classifier = model_data['classifier']
        scaler = model_data['scaler']
        feature_names = model_data.get('feature_names', None)
        # Always load from .npy if not present
        if feature_names is None:
            feature_names = np.load(feature_names_path, allow_pickle=True)
    else:
        classifier = model_data
        scaler = joblib.load(scaler_path)
        feature_names = np.load(feature_names_path, allow_pickle=True)
    pca = joblib.load(pca_path)
    class Wrapper:
        def __init__(self, classifier, scaler, feature_names, pca):
            self.classifier = classifier
            self.scaler = scaler
            self.feature_names = feature_names
            self.pca = pca
    return Wrapper(classifier, scaler, feature_names, pca)

def extract_xgboost_representations(model, customer_data, target_dim=XGBOOST_OUTPUT_DIM):
    features_to_use = list(model.feature_names)
    # Derived features (must match training)
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
    feature_data = customer_data[features_to_use].copy()
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
    feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
    X_scaled = model.scaler.transform(feature_data)
    leaf_indices = model.classifier.apply(X_scaled)
    representations = model.pca.transform(leaf_indices.astype(np.float32))
    if representations.max() > representations.min():
        representations = (representations - representations.min()) / (representations.max() - representations.min())
    return representations

# ===================== Neural Network Feature Extraction =====================
def get_penultimate_layer_model(model):
    for i, layer in enumerate(reversed(model.layers)):
        if hasattr(layer, 'activation') and layer.activation is not None:
            if i > 0:
                penultimate_layer = model.layers[-(i+1)]
                break
    else:
        penultimate_layer = model.layers[-2]
    feature_extractor = models.Model(inputs=model.inputs, outputs=penultimate_layer.output)
    return feature_extractor

# ===================== Data Loading and Preprocessing =====================
def load_client_models():
    from tensorflow.keras.models import load_model
    auto_loans_model = load_model('VFLClientModels/models/saved_models/auto_loans_model.keras', compile=False)
    digital_bank_model = load_model('VFLClientModels/models/saved_models/digital_bank_model.keras', compile=False)
    home_loans_model = load_model('VFLClientModels/models/saved_models/home_loans_model.keras', compile=False)
    credit_card_model = load_xgboost_credit_card_model()
    return auto_loans_model, digital_bank_model, home_loans_model, credit_card_model

def load_and_preprocess_data(sample_size=None):
    # Load datasets
    auto_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/auto_loans_bank.csv')
    digital_bank_df = pd.read_csv('VFLClientModels/dataset/data/banks/digital_savings_bank.csv')
    home_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/home_loans_bank.csv')
    credit_card_df = pd.read_csv('VFLClientModels/dataset/data/banks/credit_card_bank.csv')
    master_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    # Optionally sample
    if sample_size is not None and sample_size < len(master_df):
        master_df = master_df.sample(n=sample_size, random_state=RANDOM_SEED)
        auto_loans_df = auto_loans_df[auto_loans_df['tax_id'].isin(master_df['tax_id'])]
        digital_bank_df = digital_bank_df[digital_bank_df['tax_id'].isin(master_df['tax_id'])]
        home_loans_df = home_loans_df[home_loans_df['tax_id'].isin(master_df['tax_id'])]
        credit_card_df = credit_card_df[credit_card_df['tax_id'].isin(master_df['tax_id'])]
    # Alignment
    all_customers = set(master_df['tax_id'])
    customer_df = pd.DataFrame({'tax_id': sorted(list(all_customers))})
    customer_df['has_auto'] = customer_df['tax_id'].isin(auto_loans_df['tax_id'])
    customer_df['has_digital'] = customer_df['tax_id'].isin(digital_bank_df['tax_id'])
    customer_df['has_home'] = customer_df['tax_id'].isin(home_loans_df['tax_id'])
    customer_df['has_credit_card'] = customer_df['tax_id'].isin(credit_card_df['tax_id'])
    # Sort
    auto_loans_df = auto_loans_df.sort_values('tax_id').reset_index(drop=True)
    digital_bank_df = digital_bank_df.sort_values('tax_id').reset_index(drop=True)
    home_loans_df = home_loans_df.sort_values('tax_id').reset_index(drop=True)
    credit_card_df = credit_card_df.sort_values('tax_id').reset_index(drop=True)
    master_df = master_df.sort_values('tax_id').reset_index(drop=True)
    # Load models
    auto_loans_model, digital_bank_model, home_loans_model, credit_card_model = load_client_models()
    auto_loans_extractor = get_penultimate_layer_model(auto_loans_model)
    digital_bank_extractor = get_penultimate_layer_model(digital_bank_model)
    home_loans_extractor = get_penultimate_layer_model(home_loans_model)
    # Feature sets
    auto_features = [
        'annual_income', 'credit_score', 'payment_history', 'employment_length', 'debt_to_income_ratio', 'age',
        'credit_history_length', 'num_credit_cards', 'num_loan_accounts', 'total_credit_limit', 'credit_utilization_ratio',
        'late_payments', 'credit_inquiries', 'last_late_payment_days', 'current_debt', 'monthly_expenses',
        'savings_balance', 'checking_balance', 'investment_balance', 'auto_loan_balance', 'mortgage_balance'
    ]
    digital_features = [
        'annual_income', 'savings_balance', 'checking_balance', 'investment_balance', 'payment_history', 'credit_score', 'age', 'employment_length',
        'avg_monthly_transactions', 'avg_transaction_value', 'digital_banking_score', 'mobile_banking_usage', 'online_transactions_ratio',
        'international_transactions_ratio', 'e_statement_enrolled', 'monthly_expenses', 'total_credit_limit', 'credit_utilization_ratio',
        'num_credit_cards', 'credit_history_length', 'current_debt', 'mortgage_balance', 'total_wealth', 'net_worth', 'credit_efficiency',
        'financial_stability_score'
    ]
    home_features = [
        'annual_income', 'credit_score', 'payment_history', 'employment_length', 'debt_to_income_ratio', 'age',
        'credit_history_length', 'num_credit_cards', 'num_loan_accounts', 'total_credit_limit', 'credit_utilization_ratio', 'late_payments',
        'credit_inquiries', 'last_late_payment_days', 'current_debt', 'monthly_expenses', 'savings_balance', 'checking_balance',
        'investment_balance', 'mortgage_balance', 'auto_loan_balance', 'estimated_property_value', 'required_down_payment',
        'available_down_payment_funds', 'mortgage_risk_score', 'loan_to_value_ratio', 'min_down_payment_pct', 'interest_rate', 'dti_after_mortgage'
    ]
    # Scalers
    auto_scaler = StandardScaler()
    digital_scaler = StandardScaler()
    home_scaler = StandardScaler()
    # Representation extraction
    def extract_bank_representations(bank_df, features, scaler, extractor, customers_with_service):
        output_size = extractor.output_shape[-1]
        all_representations = np.zeros((len(customer_df), output_size))
        if len(bank_df) > 0:
            feature_data = bank_df[features].copy()
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
            feature_data[numeric_cols] = feature_data[numeric_cols].fillna(feature_data[numeric_cols].median())
            X_scaled = scaler.fit_transform(feature_data)
            representations = extractor.predict(X_scaled, verbose=0)
            bank_customer_ids = bank_df['tax_id'].values
            for i, customer_id in enumerate(bank_customer_ids):
                customer_idx = customer_df[customer_df['tax_id'] == customer_id].index[0]
                all_representations[customer_idx] = representations[i]
        service_mask = customers_with_service.values.astype(np.float32).reshape(-1, 1)
        return all_representations, service_mask, scaler
    auto_repr, auto_mask, fitted_auto_scaler = extract_bank_representations(auto_loans_df, auto_features, auto_scaler, auto_loans_extractor, customer_df['has_auto'])
    digital_repr, digital_mask, fitted_digital_scaler = extract_bank_representations(digital_bank_df, digital_features, digital_scaler, digital_bank_extractor, customer_df['has_digital'])
    home_repr, home_mask, fitted_home_scaler = extract_bank_representations(home_loans_df, home_features, home_scaler, home_loans_extractor, customer_df['has_home'])
    # XGBoost credit card representations
    credit_card_repr = np.zeros((len(customer_df), XGBOOST_OUTPUT_DIM))
    if len(credit_card_df) > 0:
        xgb_representations = extract_xgboost_representations(credit_card_model, credit_card_df, XGBOOST_OUTPUT_DIM)
        customer_id_to_index = {tax_id: idx for idx, tax_id in enumerate(customer_df['tax_id'])}
        credit_card_customer_ids = credit_card_df['tax_id'].values
        for i, customer_id in enumerate(credit_card_customer_ids):
            customer_idx = customer_id_to_index[customer_id]
            credit_card_repr[customer_idx] = xgb_representations[i]
    credit_card_mask = customer_df['has_credit_card'].values.astype(np.float32).reshape(-1, 1)
    # Combine features
    X_combined = np.concatenate([
        auto_repr, auto_mask, digital_repr, digital_mask, home_repr, home_mask, credit_card_repr, credit_card_mask
    ], axis=1)
    y_combined = master_df['credit_score'].values
    ids_combined = master_df['tax_id'].values
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_combined, y_combined, ids_combined, test_size=0.2, random_state=RANDOM_SEED, stratify=pd.cut(y_combined, bins=5, labels=False)
    )
    return (X_train, X_test, y_train, y_test, ids_train, ids_test, fitted_auto_scaler, fitted_digital_scaler, fitted_home_scaler, None, auto_repr.shape[1], digital_repr.shape[1], home_repr.shape[1], credit_card_repr.shape[1], X_combined, y_combined, ids_combined)

# ===================== Model Training =====================
def train_model(X_train, y_train, X_test, y_test, model_name, model_type, model_config, scaler, output_dim,
                confidence_intervals=CONFIDENCE_INTERVALS, mc_dropout_samples=MC_DROPOUT_SAMPLES,
                enable_confidence_scores=ENABLE_CONFIDENCE_SCORES, min_confidence_threshold=MIN_CONFIDENCE_THRESHOLD):
    """
    Trains a neural network model for a given bank.
    """
    logger.info(f"Training {model_name} ({model_type})...")
    input_dim = X_train.shape[1]
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_dim, activation='linear')) # Linear activation for regression

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    if model_type == 'neural_network':
        # Neural network training logic (e.g., using Keras)
        # This part would involve loading pre-trained models, freezing layers, etc.
        # For simplicity, we'll just compile and return the model.
        logger.info(f"Compiled {model_name} ({model_type}) model.")
        return model
    elif model_type == 'xgboost':
        # XGBoost training logic (e.g., using XGBoost)
        # This part would involve loading pre-trained models, freezing layers, etc.
        # For simplicity, we'll just compile and return the model.
        logger.info(f"Compiled {model_name} ({model_type}) model.")
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# ===================== Confidence Scoring =====================
def calculate_confidence_intervals(model, X, y, n_samples=MC_DROPOUT_SAMPLES):
    """
    Calculates confidence intervals for predictions using Monte Carlo dropout.
    """
    predictions = []
    for _ in range(n_samples):
        predictions.append(model.predict(X, verbose=0))
    predictions = np.array(predictions)
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    lower_bounds = []
    upper_bounds = []
    for ci in CONFIDENCE_INTERVALS:
        lower_bounds.append(mean_predictions - (std_predictions * (100 - ci) / 200))
        upper_bounds.append(mean_predictions + (std_predictions * (100 - ci) / 200))

    return mean_predictions, lower_bounds, upper_bounds

# ===================== Single Tax ID Inference =====================
def predict_credit_score_by_tax_id(tax_id, customer_data, models, scalers, feature_sets, output_dim,
                                  confidence_intervals=CONFIDENCE_INTERVALS, mc_dropout_samples=MC_DROPOUT_SAMPLES,
                                  enable_confidence_scores=ENABLE_CONFIDENCE_SCORES, min_confidence_threshold=MIN_CONFIDENCE_THRESHOLD):
    """
    Predicts the credit score for a single tax ID using the combined VFL pipeline.
    """
    logger.info(f"Predicting credit score for tax ID: {tax_id}")
    # Ensure customer_data is a DataFrame
    if isinstance(customer_data, pd.Series):
        customer_data = pd.DataFrame([customer_data])
    elif not isinstance(customer_data, pd.DataFrame):
        raise ValueError("customer_data must be a pandas DataFrame or Series.")

    # Add derived features if not present
    if 'credit_capacity_ratio' not in customer_data.columns:
        customer_data['credit_capacity_ratio'] = customer_data['credit_card_limit'] / customer_data['total_credit_limit'].replace(0, 1)
    if 'income_to_limit_ratio' not in customer_data.columns:
        customer_data['income_to_limit_ratio'] = customer_data['annual_income'] / customer_data['credit_card_limit'].replace(0, 1)
    if 'debt_service_ratio' not in customer_data.columns:
        customer_data['debt_service_ratio'] = (customer_data['current_debt'] * 0.03) / (customer_data['annual_income'] / 12)
    if 'risk_adjusted_income' not in customer_data.columns:
        customer_data['risk_adjusted_income'] = customer_data['annual_income'] * (customer_data['risk_score'] / 100)
    if 'credit_to_income_ratio' not in customer_data.columns:
        customer_data['credit_to_income_ratio'] = customer_data['credit_card_limit'] / customer_data['annual_income'].replace(0, 1)

    # Extract representations for each bank
    representations = {}
    for bank_name, model_data in models.items():
        if bank_name == 'credit_card':
            # For XGBoost, we need to pass the specific model and scaler
            xgb_model = model_data
            xgb_scaler = scalers[bank_name]
            xgb_feature_names = feature_sets[bank_name]
            xgb_output_dim = output_dim[bank_name]

            # Ensure derived features are present for XGBoost
            if 'credit_capacity_ratio' not in customer_data.columns:
                customer_data['credit_capacity_ratio'] = customer_data['credit_card_limit'] / customer_data['total_credit_limit'].replace(0, 1)
            if 'income_to_limit_ratio' not in customer_data.columns:
                customer_data['income_to_limit_ratio'] = customer_data['annual_income'] / customer_data['credit_card_limit'].replace(0, 1)
            if 'debt_service_ratio' not in customer_data.columns:
                customer_data['debt_service_ratio'] = (customer_data['current_debt'] * 0.03) / (customer_data['annual_income'] / 12)
            if 'risk_adjusted_income' not in customer_data.columns:
                customer_data['risk_adjusted_income'] = customer_data['annual_income'] * (customer_data['risk_score'] / 100)
            if 'credit_to_income_ratio' not in customer_data.columns:
                customer_data['credit_to_income_ratio'] = customer_data['credit_card_limit'] / customer_data['annual_income'].replace(0, 1)

            # Extract XGBoost representation
            xgb_representations = extract_xgboost_representations(xgb_model, customer_data, xgb_output_dim)
            representations['credit_card'] = xgb_representations
        else:
            # For neural networks, we need to pass the specific model and scaler
            model = model_data
            scaler = scalers[bank_name]
            feature_names = feature_sets[bank_name]
            output_dim = output_dim[bank_name]

            # Ensure derived features are present for neural networks
            if 'credit_capacity_ratio' not in customer_data.columns:
                customer_data['credit_capacity_ratio'] = customer_data['credit_card_limit'] / customer_data['total_credit_limit'].replace(0, 1)
            if 'income_to_limit_ratio' not in customer_data.columns:
                customer_data['income_to_limit_ratio'] = customer_data['annual_income'] / customer_data['credit_card_limit'].replace(0, 1)
            if 'debt_service_ratio' not in customer_data.columns:
                customer_data['debt_service_ratio'] = (customer_data['current_debt'] * 0.03) / (customer_data['annual_income'] / 12)
            if 'risk_adjusted_income' not in customer_data.columns:
                customer_data['risk_adjusted_income'] = customer_data['annual_income'] * (customer_data['risk_score'] / 100)
            if 'credit_to_income_ratio' not in customer_data.columns:
                customer_data['credit_to_income_ratio'] = customer_data['credit_card_limit'] / customer_data['annual_income'].replace(0, 1)

            # Extract neural network representation
            # For neural networks, we need to get the penultimate layer output
            feature_extractor = get_penultimate_layer_model(model)
            representations[bank_name] = feature_extractor.predict(scaler.transform(customer_data[feature_names]), verbose=0)

    # Combine representations and predict
    X_combined = np.concatenate([representations[bank] for bank in sorted(representations.keys())], axis=1)

    # Ensure X_combined has the correct number of features for the combined model
    # This part might need adjustment based on how the combined model expects input
    # For now, we'll assume the combined model expects all representations concatenated
    # and that the output_dim for the combined model is the sum of all output_dims.
    # This is a simplification; a more robust approach would involve a single model.
    # For this simple example, we'll just concatenate and pass to a dummy model.
    # In a real scenario, you'd have a single model that takes all representations.

    # Placeholder for combined model prediction
    # This part would involve loading the final combined model and predicting
    # For now, we'll just return a placeholder
    logger.info(f"Placeholder prediction for tax ID {tax_id}: 700 (assuming a default score)")
    return 700, 0.95 # Placeholder confidence

def build_vfl_model(input_dim):
    """Builds a simple VFL model similar to vfl_automl_model.py"""
    model = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def mc_dropout_intervals(model, X, n_samples=30, ci_levels=[68, 95]):
    """Calculate MC Dropout mean and confidence intervals for each sample in X."""
    preds = []
    for _ in range(n_samples):
        preds.append(model(X, training=True).numpy().flatten())
    preds = np.array(preds)  # shape: (n_samples, batch_size)
    mean = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    intervals = {}
    for ci in ci_levels:
        z = 1.0 if ci == 68 else 2.0  # 68% ~ 1 std, 95% ~ 2 std
        lower = mean - z * std
        upper = mean + z * std
        intervals[ci] = (lower, upper)
    return mean, intervals

def run_training():
    logger.info("Starting VFL AutoML XGBoost Simple Training...")
    (X_train, X_test, y_train, y_test, ids_train, ids_test, 
     auto_scaler, digital_scaler, home_scaler, _, 
     auto_repr_size, digital_repr_size, home_repr_size, credit_card_repr_size,
     X_combined, y_combined, ids_combined) = load_and_preprocess_data(sample_size=10000)

    # Save fitted scalers for inference
    import joblib
    joblib.dump(auto_scaler, 'VFLClientModels/models/saved_models/auto_loans_scaler.pkl')
    logger.info('Saved auto_loans_scaler.pkl')
    joblib.dump(digital_scaler, 'VFLClientModels/models/saved_models/digital_bank_scaler.pkl')
    logger.info('Saved digital_bank_scaler.pkl')
    joblib.dump(home_scaler, 'VFLClientModels/models/saved_models/home_loans_scaler.pkl')
    logger.info('Saved home_loans_scaler.pkl')

    input_dim = X_train.shape[1]
    model = build_vfl_model(input_dim)
    logger.info(model.summary())

    # Train the model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=2)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
    model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=256,
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping, reduce_lr]
    )
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    logger.info(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

    # Predict and cache results for test set
    y_pred = model.predict(X_test).flatten()
    mean_pred, intervals = mc_dropout_intervals(model, X_test, n_samples=MC_DROPOUT_SAMPLES, ci_levels=[68, 95])
    global _prediction_cache
    _prediction_cache = {}
    for i, idx in enumerate(range(len(X_test))):
        tax_id = ids_test[idx]
        result = {
            'tax_id': tax_id,
            'predicted': float(mean_pred[i]),
            'actual': float(y_test[idx]),
            '68_CI': (float(intervals[68][0][i]), float(intervals[68][1][i])),
            '95_CI': (float(intervals[95][0][i]), float(intervals[95][1][i])),
            'deterministic': float(y_pred[i])
        }
        logger.info(f"Prediction for {tax_id}: Predicted={result['predicted']:.1f}, Deterministic={result['deterministic']:.1f}, Actual={result['actual']:.1f}, 68% CI=({result['68_CI'][0]:.1f}, {result['68_CI'][1]:.1f}), 95% CI=({result['95_CI'][0]:.1f}, {result['95_CI'][1]:.1f})")
        _prediction_cache[tax_id] = result
    _save_prediction_cache()
    logger.info(f"Saved predictions for {len(_prediction_cache)} test customers to cache.")

    # Save model
    model.save('VFLClientModels/models/saved_models/vfl_automl_xgboost_simple_model.keras')
    logger.info('Model saved to VFLClientModels/models/saved_models/vfl_automl_xgboost_simple_model.keras')


# Persistent cache file path
_CACHE_PATH = 'VFLClientModels/models/saved_models/prediction_cache.pkl'

# In-memory cache for predictions
if os.path.exists(_CACHE_PATH):
    _prediction_cache = joblib.load(_CACHE_PATH)
    logger.info(f"Loaded prediction cache from {_CACHE_PATH} ({len(_prediction_cache)} entries)")
else:
    _prediction_cache = {}
    logger.info("Initialized empty prediction cache")

def _save_prediction_cache():
    joblib.dump(_prediction_cache, _CACHE_PATH)
    logger.info(f"Saved prediction cache to {_CACHE_PATH} ({len(_prediction_cache)} entries)")

def predict_with_confidence_by_tax_id(tax_id):
    """
    Return cached credit score prediction for a single tax_id if available. Do not perform any computation or prediction if not in the cache.
    """
    if tax_id in _prediction_cache:
        logger.info(f"[predict_with_confidence_by_tax_id] Returning cached result for tax_id: {tax_id}")
        return _prediction_cache[tax_id]
    logger.warning(f"[predict_with_confidence_by_tax_id] No cached result for tax_id: {tax_id}. Returning None.")
    return None

# Example usage:
# if __name__ == "__main__":
#     res = predict_with_confidence_by_tax_id('TAX001')
#     print(res)

if __name__ == "__main__":
    run_training() 