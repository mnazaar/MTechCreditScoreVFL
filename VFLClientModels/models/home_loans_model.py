import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Setup comprehensive logging for Home Loans model training"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger
    logger = logging.getLogger('Home_Loans_Model')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Single comprehensive log file for this run
    file_handler = RotatingFileHandler(
        f'VFLClientModels/logs/home_loans_model_{run_timestamp}.log',
        maxBytes=15*1024*1024,  # 15MB per file (increased since it's the only log)
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # TensorFlow logging configuration
    tf.get_logger().setLevel('ERROR')  # Reduce TF verbosity
    
    logger.info("=" * 80)
    logger.info("Home Loans Model Logging System Initialized")
    logger.info(f"Log file: VFLClientModels/logs/home_loans_model_{run_timestamp}.log")
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return logger

# Initialize logging
logger = setup_logging()

class TrainingLogger(tf.keras.callbacks.Callback):
    """Custom callback to log training progress to our logging system"""
    
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        
    def on_epoch_begin(self, epoch, logs=None):
        self.logger.info(f"Epoch {epoch + 1} started...")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Log training metrics
        loss = logs.get('loss', 0)
        mae = logs.get('mae', 0)
        mape = logs.get('mape', 0)
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        val_mape = logs.get('val_mape', 0)
        
        self.logger.info(f"Epoch {epoch + 1:3d} - "
                        f"loss: {loss:.6f} - "
                        f"mae: {mae:.4f} - "
                        f"mape: {mape:.4f} - "
                        f"val_loss: {val_loss:.6f} - "
                        f"val_mae: {val_mae:.4f} - "
                        f"val_mape: {val_mape:.4f}")
        
        # Log to training-specific handler with more detail
        training_logger = logging.getLogger('Home_Loans_Model')
        training_logger.debug(f"Epoch {epoch + 1} detailed metrics:")
        for metric, value in logs.items():
            training_logger.debug(f"  {metric}: {value}")
    
    def on_train_begin(self, logs=None):
        self.logger.info("🚀 Training started - logging epoch progress...")
    
    def on_train_end(self, logs=None):
        self.logger.info("✅ Training completed - final metrics logged")

def load_and_preprocess_data():
    """Load and preprocess data for home loans model with robust scaling"""
    logger.info("🔄 Loading and preprocessing home loans data...")
    
    # Load data
    logger.debug("Loading home loans dataset from CSV...")
    df = pd.read_csv('VFLClientModels/dataset/data/banks/home_loans_bank.csv')
    logger.info(f"📊 Dataset loaded: {len(df):,} total records")
    
    # Select features - comprehensive feature set for home loan prediction
    features = [
        # Core financial features
        'annual_income',              # Primary factor for loan calculation
        'credit_score',              # Credit worthiness (critical for mortgages)
        'payment_history',           # Payment reliability
        'employment_length',         # Job stability (important for long-term loans)
        'debt_to_income_ratio',      # Existing debt burden
        'age',                       # Age considerations
        
        # Credit history and behavior
        'credit_history_length',     # Credit maturity
        'num_credit_cards',          # Credit relationships
        'num_loan_accounts',         # Existing loan burden
        'total_credit_limit',        # Credit capacity
        'credit_utilization_ratio',  # Credit usage
        'late_payments',             # Payment behavior
        'credit_inquiries',          # Recent credit activity
        'last_late_payment_days',    # Recent payment behavior
        
        # Financial position and assets
        'current_debt',              # Current debt amount
        'monthly_expenses',          # Monthly obligations
        'savings_balance',           # Down payment source
        'checking_balance',          # Liquid assets
        'investment_balance',        # Additional assets
        'mortgage_balance',          # Existing mortgage
        'auto_loan_balance',         # Other secured debt
        
        # Home loan specific calculated features
        'estimated_property_value',   # Property value estimate
        'required_down_payment',      # Down payment needed
        'available_down_payment_funds', # Available funds
        'mortgage_risk_score',        # Comprehensive mortgage risk
        'loan_to_value_ratio',       # LTV ratio
        'min_down_payment_pct',      # Down payment percentage
        'interest_rate',             # Risk-based interest rate
        'dti_after_mortgage'         # DTI including new mortgage
    ]
    
    logger.info(f"📋 Selected {len(features)} features for home loans prediction")
    logger.debug(f"Features: {features}")
    
    # Prepare features and target
    X = df[features].copy()
    y = df['max_loan_amount']
    eligible = df['loan_eligible']
    
    logger.info(f"📊 Dataset composition:")
    logger.info(f"   - Total customers: {len(df):,}")
    logger.info(f"   - Eligible customers: {eligible.sum():,} ({eligible.sum()/len(df)*100:.1f}%)")
    logger.info(f"   - Ineligible customers: {(~eligible).sum():,} ({(~eligible).sum()/len(df)*100:.1f}%)")
    logger.info(f"   - Training on ALL customers (eligible + ineligible)")
    
    # Train on ALL customers - don't filter by eligibility
    logger.info(f"📊 Target variable statistics (all customers):")
    logger.info(f"   - Loan amount range: ${y.min():,.0f} - ${y.max():,.0f}")
    logger.info(f"   - Mean loan amount: ${y.mean():,.0f}")
    logger.info(f"   - Median loan amount: ${y.median():,.0f}")
    logger.info(f"   - Zero loan amounts: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    # Handle any infinite or missing values
    logger.debug("Handling infinite and missing values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()
    logger.debug(f"Missing/infinite values handled: {missing_before} → {missing_after}")
    
    # Split data
    logger.debug("Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"📊 Data split results:")
    logger.info(f"   - Training samples: {len(X_train):,}")
    logger.info(f"   - Test samples: {len(X_test):,}")
    logger.info(f"   - Training target range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    logger.info(f"   - Test target range: ${y_test.min():,.0f} - ${y_test.max():,.0f}")
    
    # Scale features
    logger.debug("Applying StandardScaler to features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.debug("✅ Feature scaling completed")
    
    # Log transform for target (loan amounts are highly skewed)
    logger.debug("Applying log transformation to target variable...")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    logger.debug(f"Target transformation results:")
    logger.debug(f"   - Original target range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    logger.debug(f"   - Log-transformed range: {y_train_log.min():.3f} - {y_train_log.max():.3f}")
    
    # Return feature names for later use
    feature_names = list(X.columns)
    logger.info("✅ Data preprocessing completed successfully")
    
    return X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train, y_test, scaler, feature_names

def build_model(input_shape):
    """Build neural network model with 16 units in penultimate layer for home loans"""
    # L1L2 regularization
    regularizer = tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-5)
    
    model = models.Sequential([
        # Input layer - sized for comprehensive home loan features
        layers.Dense(512, activation='relu', input_shape=input_shape,
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layer 1
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Hidden layer 2
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Hidden layer 3
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        # Hidden layer 4
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Penultimate layer with 16 units (for VFL integration)
        layers.Dense(16, activation='relu', name='penultimate',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        
        # Output layer (no activation for regression)
        layers.Dense(1)
    ])
    
    return model

def print_test_predictions(X_test, y_test, model, scaler, feature_names, n_samples=10):
    """Print detailed test predictions for home loan samples"""
    # Get random indices
    test_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions
    X_test_samples = X_test[test_indices]
    y_test_samples = y_test.iloc[test_indices]
    y_pred_log = model.predict(X_test_samples)
    y_pred = np.expm1(y_pred_log)
    
    logger.info("\nDetailed Home Loan Test Predictions:")
    logger.info("=" * 140)
    logger.info(f"{'Actual Amount':>15} {'Predicted Amount':>15} {'Difference':>15} {'% Error':>10} {'Key Features':<65}")
    logger.info("-" * 140)
    
    for idx, (actual, pred) in enumerate(zip(y_test_samples, y_pred)):
        diff = actual - pred[0]
        pct_error = (abs(diff) / actual * 100) if actual != 0 else float('inf')
        
        # Get key features for this sample
        sample_features = X_test_samples[idx]
        # Inverse transform to get original scale
        sample_features_orig = scaler.inverse_transform(sample_features.reshape(1, -1))[0]
        
        # Extract key information for home loans
        try:
            income_idx = feature_names.index('annual_income')
            score_idx = feature_names.index('credit_score')
            debt_idx = feature_names.index('debt_to_income_ratio')
            property_idx = feature_names.index('estimated_property_value')
            down_idx = feature_names.index('available_down_payment_funds')
            ltv_idx = feature_names.index('loan_to_value_ratio')
            
            key_info = (f"Income: ${sample_features_orig[income_idx]:,.0f}, Score: {sample_features_orig[score_idx]:.0f}, "
                       f"Property: ${sample_features_orig[property_idx]:,.0f}, LTV: {sample_features_orig[ltv_idx]:.1%}, "
                       f"Down Payment: ${sample_features_orig[down_idx]:,.0f}")
        except ValueError:
            # Fallback to first few features if specific ones not found
            key_info = f"Features: {sample_features_orig[:5]}"
        
        logger.info(f"${actual:>14,.0f} ${pred[0]:>14,.0f} ${diff:>14,.0f} {pct_error:>9.1f}% {key_info:<65}")
    logger.info("-" * 140)

def analyze_feature_importance(model, X_train_scaled, scaler, feature_names):
    """Analyze feature importance using permutation importance"""
    
    # Get baseline predictions
    baseline_pred = model.predict(X_train_scaled)
    baseline_mse = np.mean((baseline_pred.flatten()) ** 2)
    
    # Calculate importance for each feature
    importance_scores = []
    for i in range(X_train_scaled.shape[1]):
        # Create a copy and permute one feature
        X_permuted = X_train_scaled.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Get predictions with permuted feature
        permuted_pred = model.predict(X_permuted)
        permuted_mse = np.mean((permuted_pred.flatten()) ** 2)
        
        # Calculate importance (increase in MSE)
        importance = (permuted_mse - baseline_mse) / baseline_mse
        importance_scores.append((feature_names[i], importance))
    
    # Sort by importance
    return sorted(importance_scores, key=lambda x: x[1], reverse=True)

def plot_feature_importance(importance_scores, top_n=15):
    """Plot feature importance"""
    top_features = importance_scores[:top_n]
    features, scores = zip(*top_features)
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))
    
    bars = plt.barh(y_pos, scores)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score (Increase in MSE)')
    plt.title(f'Top {top_n} Feature Importance for Home Loans Model')
    plt.gca().invert_yaxis()
    
    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig('VFLClientModels/plots/home_loans_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_plots(y_test, y_pred, history):
    """Create comprehensive visualization plots"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training History - Loss
    plt.subplot(3, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Training History - MAE
    plt.subplot(3, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model Training MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Predictions vs Actual
    plt.subplot(3, 3, 3)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Loan Amount ($)')
    plt.ylabel('Predicted Loan Amount ($)')
    plt.title('Predicted vs Actual Loan Amounts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Residuals Plot
    plt.subplot(3, 3, 4)
    residuals = y_test - y_pred.flatten()
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Loan Amount ($)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # 5. Distribution of Predictions
    plt.subplot(3, 3, 5)
    plt.hist(y_pred, bins=50, alpha=0.7, label='Predicted', color='orange', density=True)
    plt.hist(y_test, bins=50, alpha=0.7, label='Actual', color='blue', density=True)
    plt.xlabel('Loan Amount ($)')
    plt.ylabel('Density')
    plt.title('Distribution of Loan Amounts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Error Distribution
    plt.subplot(3, 3, 6)
    errors = np.abs(y_test - y_pred.flatten())
    plt.hist(errors, bins=50, alpha=0.7, color='red', density=True)
    plt.xlabel('Absolute Error ($)')
    plt.ylabel('Density')
    plt.title('Distribution of Absolute Errors')
    plt.grid(True, alpha=0.3)
    
    # 7. Percentage Error Distribution
    plt.subplot(3, 3, 7)
    pct_errors = np.abs((y_test - y_pred.flatten()) / y_test) * 100
    pct_errors = pct_errors[np.isfinite(pct_errors)]  # Remove inf values
    plt.hist(pct_errors, bins=50, alpha=0.7, color='green', density=True)
    plt.xlabel('Absolute Percentage Error (%)')
    plt.ylabel('Density')
    plt.title('Distribution of Percentage Errors')
    plt.grid(True, alpha=0.3)
    
    # 8. Loan Amount Ranges
    plt.subplot(3, 3, 8)
    loan_ranges = ['<$100K', '$100K-$200K', '$200K-$300K', '$300K-$500K', '$500K+']
    actual_counts = [
        (y_test < 100000).sum(),
        ((y_test >= 100000) & (y_test < 200000)).sum(),
        ((y_test >= 200000) & (y_test < 300000)).sum(),
        ((y_test >= 300000) & (y_test < 500000)).sum(),
        (y_test >= 500000).sum()
    ]
    pred_counts = [
        (y_pred < 100000).sum(),
        ((y_pred >= 100000) & (y_pred < 200000)).sum(),
        ((y_pred >= 200000) & (y_pred < 300000)).sum(),
        ((y_pred >= 300000) & (y_pred < 500000)).sum(),
        (y_pred >= 500000).sum()
    ]
    
    x = np.arange(len(loan_ranges))
    width = 0.35
    plt.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.7)
    plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    plt.xlabel('Loan Amount Range')
    plt.ylabel('Count')
    plt.title('Loan Amount Distribution by Range')
    plt.xticks(x, loan_ranges, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Model Performance Summary
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
    r2 = 1 - np.sum((y_test - y_pred.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    metrics_text = f"""
    Model Performance Metrics:
    
    RMSE: ${rmse:,.0f}
    MAE: ${mae:,.0f}
    MAPE: {mape:.2f}%
    R²: {r2:.4f}
    
    Data Summary:
    Training Samples: {len(y_test) * 4}  # Approx
    Test Samples: {len(y_test)}
    Mean Loan Amount: ${np.mean(y_test):,.0f}
    Std Loan Amount: ${np.std(y_test):,.0f}
    """
    
    plt.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Home Loans Model - Comprehensive Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('VFLClientModels/plots/home_loans_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_model():
    """Train the home loans prediction model with comprehensive analysis"""
    # Create necessary directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load and preprocess data
    logger.info("🏠 Home Loans Model Training")
    logger.info("=" * 60)
    X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
    
    # Print feature information
    logger.info(f"\nUsing {len(feature_names)} features for home loans prediction:")
    logger.info("=" * 80)
    for i, feature in enumerate(feature_names, 1):
        logger.info(f"{i:2d}. {feature:<30}")
        if i % 3 == 0:  # New line every 3 features for better readability
            logger.info("")
    logger.info("=" * 80)
    
    # Build model
    model = build_model(input_shape=(X_train_scaled.shape[1],))
    
    # Print model summary
    logger.info("\nModel Architecture:")
    model.summary()
    
    # Compile model with advanced optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'VFLClientModels/saved_models/home_loans_model.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    training_logger = TrainingLogger(logger)
    
    logger.info("")
    logger.info("🎯 Training Configuration:")
    logger.info(f"   - Early stopping: patience=5 (monitor val_loss)")
    logger.info(f"   - Learning rate reduction: factor=0.2, patience=3")
    logger.info(f"   - Model checkpoint: save best model by val_loss")
    logger.info(f"   - Training progress logger: epoch-by-epoch metrics")
    logger.info(f"   - Max epochs: 100")
    logger.info(f"   - Batch size: 32")
    logger.info(f"   - Validation split: 20%")
    
    logger.info("")
    logger.info("📊 Training Metrics to be Logged:")
    logger.info("   - Training Loss (MSE)")
    logger.info("   - Training MAE")
    logger.info("   - Training MAPE") 
    logger.info("   - Validation Loss")
    logger.info("   - Validation MAE")
    logger.info("   - Validation MAPE")
    logger.info("   - Epoch timing and progress")
    
    # Train model
    logger.info("\nTraining model...")
    history = model.fit(
        X_train_scaled,
        y_train_log,  # Using log-transformed target
        epochs=100,
        batch_size=32,  # Smaller batch size for more stable training
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, training_logger],
        verbose=1
    )
    
    # Evaluate model
    logger.info("\nEvaluating model...")
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)  # Convert back from log scale
    
    # DEBUG: Add detailed debugging for negative R²
    logger.info("\n🔍 DEBUGGING NEGATIVE R² ISSUE:")
    logger.info("=" * 80)
    
    # Check prediction statistics
    logger.info(f"📊 Prediction Statistics:")
    logger.info(f"   - y_pred_log range: [{y_pred_log.min():.3f}, {y_pred_log.max():.3f}]")
    logger.info(f"   - y_pred range: [${y_pred.min():,.0f}, ${y_pred.max():,.0f}]")
    logger.info(f"   - y_test range: [${y_test.min():,.0f}, ${y_test.max():,.0f}]")
    logger.info(f"   - y_pred mean: ${y_pred.mean():,.0f}")
    logger.info(f"   - y_test mean: ${y_test.mean():,.0f}")
    logger.info(f"   - y_pred std: ${y_pred.std():,.0f}")
    logger.info(f"   - y_test std: ${y_test.std():,.0f}")
    
    # Check for any invalid predictions
    invalid_preds = np.isnan(y_pred.flatten()) | np.isinf(y_pred.flatten())
    if invalid_preds.any():
        logger.warning(f"⚠️  Found {invalid_preds.sum()} invalid predictions (NaN/Inf)")
    
    # Calculate R² on log scale vs original scale
    r2_log_scale = r2_score(y_test_log, y_pred_log)
    r2_original_scale = r2_score(y_test, y_pred.flatten())
    
    logger.info(f"📊 R² Comparison:")
    logger.info(f"   - R² on log scale: {r2_log_scale:.4f}")
    logger.info(f"   - R² on original scale: {r2_original_scale:.4f}")
    
    # Manual R² calculation breakdown
    ss_res = np.sum((y_test - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot)
    
    logger.info(f"📊 Manual R² Breakdown:")
    logger.info(f"   - SS_res (residual sum of squares): {ss_res:,.0f}")
    logger.info(f"   - SS_tot (total sum of squares): {ss_tot:,.0f}")
    logger.info(f"   - SS_res/SS_tot ratio: {ss_res/ss_tot:.4f}")
    logger.info(f"   - Manual R²: {r2_manual:.4f}")
    
    # Check baseline performance (predicting mean)
    baseline_pred = np.full_like(y_test, np.mean(y_test))
    baseline_mse = np.mean((y_test - baseline_pred) ** 2)
    model_mse = np.mean((y_test - y_pred.flatten()) ** 2)
    
    logger.info(f"📊 Baseline Comparison:")
    logger.info(f"   - Baseline MSE (predicting mean): {baseline_mse:,.0f}")
    logger.info(f"   - Model MSE: {model_mse:,.0f}")
    logger.info(f"   - Model is {'WORSE' if model_mse > baseline_mse else 'BETTER'} than baseline")
    
    logger.info("=" * 80)
    
    # Calculate comprehensive metrics
    mse = np.mean((y_test - y_pred.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
    r2 = r2_original_scale  # Use sklearn's implementation for consistency
    
    # POTENTIAL FIX: If R² is still very negative, suggest alternative approaches
    if r2 < -0.5:
        logger.warning("\n⚠️  NEGATIVE R² DETECTED - POTENTIAL FIXES:")
        logger.warning("=" * 80)
        logger.warning("1. Consider training without log transformation")
        logger.warning("2. Check for data leakage or corruption")
        logger.warning("3. Try simpler model architecture")
        logger.warning("4. Increase training epochs or adjust learning rate")
        logger.warning("5. Check feature engineering - some features might be harmful")
        logger.warning("6. Consider using robust scaling instead of standard scaling")
        
        # Try a simple linear baseline for comparison
        from sklearn.linear_model import LinearRegression
        baseline_model = LinearRegression()
        baseline_model.fit(X_train_scaled, y_train)  # Train on original scale
        baseline_pred = baseline_model.predict(X_test_scaled)
        baseline_r2 = r2_score(y_test, baseline_pred)
        
        logger.warning(f"📊 Linear Regression Baseline R²: {baseline_r2:.4f}")
        logger.warning("If baseline R² is much better, the neural network isn't learning properly")
        logger.warning("=" * 80)
    
    # Additional metrics
    median_error = np.median(np.abs(y_test - y_pred.flatten()))
    max_error = np.max(np.abs(y_test - y_pred.flatten()))
    q75_error = np.percentile(np.abs(y_test - y_pred.flatten()), 75)
    q25_error = np.percentile(np.abs(y_test - y_pred.flatten()), 25)
    
    logger.info("\n" + "="*80)
    logger.info("HOME LOANS MODEL PERFORMANCE SUMMARY")
    logger.info("="*80)
    logger.info(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
    logger.info(f"Mean Absolute Error (MAE):     ${mae:,.2f}")
    logger.info(f"Median Absolute Error:         ${median_error:,.2f}")
    logger.info(f"Mean Absolute Percentage Error: {mape:.2f}%")
    logger.info(f"R-squared Score:               {r2:.4f}")
    logger.info(f"Maximum Error:                 ${max_error:,.2f}")
    logger.info(f"75th Percentile Error:         ${q75_error:,.2f}")
    logger.info(f"25th Percentile Error:         ${q25_error:,.2f}")
    logger.info("="*80)
    
    # Print detailed test predictions
    print_test_predictions(X_test_scaled, y_test, model, scaler, feature_names, n_samples=10)
    
    # Analyze feature importance
    logger.info("\nAnalyzing feature importance...")
    importance_scores = analyze_feature_importance(model, X_train_scaled, scaler, feature_names)
    
    logger.info("\nTop 15 Most Important Features:")
    logger.info("=" * 70)
    logger.info(f"{'Rank':<4} {'Feature':<35} {'Importance':<15} {'Category'}")
    logger.info("-" * 70)
    for i, (feature, importance) in enumerate(importance_scores[:15], 1):
        # Categorize features
        if feature in ['annual_income', 'credit_score', 'payment_history']:
            category = 'Core Financial'
        elif feature in ['estimated_property_value', 'loan_to_value_ratio', 'required_down_payment']:
            category = 'Property/Loan'
        elif feature in ['mortgage_risk_score', 'interest_rate', 'dti_after_mortgage']:
            category = 'Risk Assessment'
        elif 'balance' in feature:
            category = 'Assets/Debt'
        else:
            category = 'Other'
        
        logger.info(f"{i:<4} {feature:<35} {importance:<15.6f} {category}")
    logger.info("-" * 70)
    
    # Create comprehensive plots
    logger.info("\nGenerating comprehensive visualizations...")
    create_comprehensive_plots(y_test, y_pred, history)
    plot_feature_importance(importance_scores)
    
    # Save additional artifacts
    np.save('VFLClientModels/saved_models/home_loans_feature_names.npy', feature_names)
    
    # Save scaler for future use
    import joblib
    joblib.dump(scaler, 'VFLClientModels/saved_models/home_loans_scaler.pkl')
    
    logger.info("\n" + "="*80)
    logger.info("MODEL ARTIFACTS SAVED")
    logger.info("="*80)
    logger.info("✅ Model saved as: 'VFLClientModels/saved_models/home_loans_model.keras'")
    logger.info("✅ Feature names saved as: 'VFLClientModels/saved_models/home_loans_feature_names.npy'")
    logger.info("✅ Scaler saved as: 'VFLClientModels/saved_models/home_loans_scaler.pkl'")
    logger.info("✅ Comprehensive analysis plot: 'VFLClientModels/plots/home_loans_comprehensive_analysis.png'")
    logger.info("✅ Feature importance plot: 'VFLClientModels/plots/home_loans_feature_importance.png'")
    logger.info("="*80)
    
    return model, history, importance_scores

def analyze_loan_segments(y_test, y_pred):
    """Analyze model performance across different loan segments"""
    logger.info("\nLoan Segment Analysis:")
    logger.info("=" * 80)
    
    segments = [
        ("Small Loans (<$150K)", y_test < 150000),
        ("Medium Loans ($150K-$300K)", (y_test >= 150000) & (y_test < 300000)),
        ("Large Loans ($300K-$500K)", (y_test >= 300000) & (y_test < 500000)),
        ("Jumbo Loans (≥$500K)", y_test >= 500000)
    ]
    
    for segment_name, mask in segments:
        if mask.sum() > 0:
            segment_actual = y_test[mask]
            segment_pred = y_pred.flatten()[mask]
            
            segment_mae = np.mean(np.abs(segment_actual - segment_pred))
            segment_mape = np.mean(np.abs((segment_actual - segment_pred) / segment_actual)) * 100
            segment_r2 = 1 - np.sum((segment_actual - segment_pred) ** 2) / np.sum((segment_actual - np.mean(segment_actual)) ** 2)
            
            logger.info(f"{segment_name:<25} Count: {mask.sum():>5} | MAE: ${segment_mae:>8,.0f} | MAPE: {segment_mape:>6.2f}% | R²: {segment_r2:>7.4f}")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    logger.info("🏠 Starting Home Loans Model Training...")
    logger.info("This may take several minutes...")
    
    # Train the model
    model, history, importance_scores = train_model()
    
    logger.info("\n🎉 Home Loans Model Training Completed Successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated plots in the 'plots/' directory")
    logger.info("2. Check feature importance rankings")
    logger.info("3. Analyze model performance across different loan segments")
    logger.info("4. Consider the model for integration into the VFL system") 