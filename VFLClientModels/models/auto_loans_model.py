import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import warnings
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Setup comprehensive logging for Auto Loans model training"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger
    logger = logging.getLogger('Auto_Loans_Model')
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
        f'VFLClientModels/logs/auto_loans_model_{run_timestamp}.log',
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
    logger.info("Auto Loans Model Logging System Initialized")
    logger.info(f"Log file: VFLClientModels/logs/auto_loans_model_{run_timestamp}.log")
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
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        
        self.logger.info(f"Epoch {epoch + 1:3d} - "
                        f"loss: {loss:.6f} - "
                        f"mae: {mae:.4f} - "
                        f"val_loss: {val_loss:.6f} - "
                        f"val_mae: {val_mae:.4f}")
        
        # Log to training-specific handler with more detail
        training_logger = logging.getLogger('Auto_Loans_Model')
        training_logger.debug(f"Epoch {epoch + 1} detailed metrics:")
        for metric, value in logs.items():
            training_logger.debug(f"  {metric}: {value}")
    
    def on_train_begin(self, logs=None):
        self.logger.info("🚀 Training started - logging epoch progress...")
    
    def on_train_end(self, logs=None):
        self.logger.info("✅ Training completed - final metrics logged")

def load_and_preprocess_data():
    """Load and preprocess data for auto loans model with robust scaling"""
    logger.info("🔄 Loading and preprocessing auto loans data...")
    
    # Load data
    logger.debug("Loading auto loans dataset from CSV...")
    df = pd.read_csv('VFLClientModels/dataset/data/banks/auto_loans_bank.csv')
    logger.info(f"📊 Dataset loaded: {len(df):,} total records")
    
    # Select features - expanded to use new rich feature set (exclude calculated/target features)
    features = [
        # Core financial features
        'annual_income',              # Strong predictor of loan amount
        'credit_score',              # Credit worthiness
        'payment_history',           # Payment reliability  
        'employment_length',         # Job stability
        'debt_to_income_ratio',      # Debt burden
        'age',                       # Age factor
        
        # Credit history and behavior
        'credit_history_length',     # Length of credit history
        'num_credit_cards',          # Credit relationships
        'num_loan_accounts',         # Existing loan burden
        'total_credit_limit',        # Overall borrowing capacity
        'credit_utilization_ratio',  # Credit usage pattern
        'late_payments',             # Payment behavior indicator
        'credit_inquiries',          # Recent credit activity
        'last_late_payment_days',    # Recent payment behavior
        
        # Financial position
        'current_debt',              # Current debt amount
        'monthly_expenses',          # Monthly spending pattern
        'savings_balance',           # Available savings/down payment
        'checking_balance',          # Liquid assets
        'investment_balance',        # Investment portfolio
        
        # Existing loans (important for auto loan decisions)
        'auto_loan_balance',         # Existing auto loan
        'mortgage_balance'           # Existing mortgage
    ]
    
    logger.info(f"📋 Selected {len(features)} features for auto loans prediction")
    logger.debug(f"Features: {features}")
    
    # Prepare features and target
    X = df[features].copy()
    y = df['auto_loan_limit']
    
    logger.debug("Analyzing target variable distribution...")
    logger.info(f"📊 Target variable statistics (all customers):")
    logger.info(f"   - Loan amount range: ${y.min():,.0f} - ${y.max():,.0f}")
    logger.info(f"   - Mean loan amount: ${y.mean():,.0f}")
    logger.info(f"   - Median loan amount: ${y.median():,.0f}")
    logger.info(f"   - Zero loan amounts: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    logger.info(f"   - Non-zero loan amounts: {(y > 0).sum():,} ({(y > 0).sum()/len(y)*100:.1f}%)")
    logger.info(f"   - Training on ALL customers (including zero amounts)")
    
    # Train on ALL customers - don't filter out zero amounts
    logger.info(f"📊 Training dataset composition:")
    logger.info(f"   - Total customers: {len(X):,}")
    logger.info(f"   - Features: {len(features)}")
    
    # Handle any missing values
    logger.debug("Handling missing values...")
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()
    logger.debug(f"Missing values handled: {missing_before} → {missing_after}")
    
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
    logger.info(f"   - Training zero amounts: {(y_train == 0).sum():,}")
    logger.info(f"   - Test zero amounts: {(y_test == 0).sum():,}")
    
    # Scale features
    logger.debug("Applying StandardScaler to features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.debug("✅ Feature scaling completed")
    
    # Simple log transform for target
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
    """Build neural network model with 16 units in penultimate layer - updated for more features"""
    logger.info("🏗️ Building auto loans neural network model...")
    logger.debug(f"Input shape: {input_shape}")
    
    # L1L2 regularization
    regularizer = tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-5)
    logger.debug(f"Regularization: L1=1e-6, L2=1e-5")
    
    model = models.Sequential([
        # Input layer - increased size to handle more features
        layers.Dense(256, activation='relu', input_shape=input_shape,
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layer 1
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Hidden layer 2
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Hidden layer 3 - additional layer for complexity
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Penultimate layer with 16 units
        layers.Dense(16, activation='relu', name='penultimate',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        
        # Output layer (no activation for regression)
        layers.Dense(1)
    ])
    
    total_params = model.count_params()
    logger.info(f"📊 Model architecture created:")
    logger.info(f"   - Architecture: 256→128→64→32→16→1")
    logger.info(f"   - Penultimate layer: 16 units (for VFL compatibility)")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - Regularization: L1L2 with dropout")
    logger.debug("Model layers:")
    for i, layer in enumerate(model.layers):
        layer_info = f"   Layer {i+1}: {layer.__class__.__name__}"
        if hasattr(layer, 'units'):
            layer_info += f" ({layer.units} units)"
        if hasattr(layer, 'activation') and layer.activation:
            layer_info += f" - {layer.activation.__name__}"
        logger.debug(layer_info)
    
    return model

def print_test_predictions(X_test, y_test, model, scaler, feature_names, n_samples=5):
    """Print detailed test predictions for random samples"""
    # Get random indices
    test_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions
    X_test_samples = X_test[test_indices]
    y_test_samples = y_test.iloc[test_indices]
    y_pred_log = model.predict(X_test_samples)
    y_pred = np.expm1(y_pred_log)
    
    logger.info("\nDetailed Test Predictions:")
    logger.info("=" * 120)
    logger.info(f"{'Actual Amount':>15} {'Predicted Amount':>15} {'Difference':>15} {'% Error':>10} {'Key Features':<50}")
    logger.info("-" * 120)
    
    for idx, (actual, pred) in enumerate(zip(y_test_samples, y_pred)):
        diff = actual - pred[0]
        pct_error = (abs(diff) / actual * 100) if actual != 0 else float('inf')
        
        # Get key features for this sample
        sample_features = X_test_samples[idx]
        # Inverse transform to get original scale (only for numerical features)
        sample_features_orig = scaler.inverse_transform(sample_features.reshape(1, -1))[0]
        
        # Extract key information - find indices of important features
        try:
            income_idx = feature_names.index('annual_income')
            score_idx = feature_names.index('credit_score')
            debt_idx = feature_names.index('debt_to_income_ratio')
            savings_idx = feature_names.index('savings_balance')
            
            key_info = f"Income: ${sample_features_orig[income_idx]:,.0f}, Score: {sample_features_orig[score_idx]:.0f}, DTI: {sample_features_orig[debt_idx]:.2f}, Savings: ${sample_features_orig[savings_idx]:,.0f}"
        except ValueError:
            # Fallback to first few features if specific ones not found
            key_info = f"Features: {sample_features_orig[:4]}"
        
        logger.info(f"${actual:>14,.0f} ${pred[0]:>14,.0f} ${diff:>14,.0f} {pct_error:>9.1f}% {key_info:<50}")
    logger.info("-" * 120)

def train_model():
    """Train the auto loans prediction model with enhanced training process"""
    training_start = datetime.now()
    logger.info("🚀 Starting Auto Loans Model Training")
    logger.info("=" * 80)
    logger.info(f"Training session started: {training_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    logger.debug("Creating output directories...")
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load and preprocess data
        logger.info("📊 Data Loading and Preprocessing Phase")
        logger.info("-" * 60)
        data_start = datetime.now()
        X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train, y_test, scaler, feature_names = load_and_preprocess_data()
        data_duration = datetime.now() - data_start
        logger.info(f"⏱️ Data preprocessing completed in {data_duration}")
        
        # Print feature information
        logger.info("")
        logger.info("📋 Feature Information:")
        logger.info(f"Using {len(feature_names)} features for auto loans prediction:")
        for i, feature in enumerate(feature_names, 1):
            logger.debug(f"{i:2d}. {feature}")
        
        # Build model
        logger.info("")
        logger.info("🏗️ Model Building Phase")
        logger.info("-" * 60)
        model_start = datetime.now()
        model = build_model(input_shape=(X_train_scaled.shape[1],))
        model_duration = datetime.now() - model_start
        logger.info(f"⏱️ Model building completed in {model_duration}")
        
        # Compile model with gradient clipping
        logger.info("")
        logger.info("⚙️ Model Compilation Phase")
        logger.info("-" * 60)
        logger.debug("Configuring optimizer with gradient clipping...")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Back to original learning rate
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("📊 Model compilation settings:")
        logger.info(f"   - Optimizer: Adam (lr=0.001, clipnorm=1.0)")
        logger.info(f"   - Loss function: MSE")
        logger.info(f"   - Metrics: MAE")
        
        # Callbacks
        logger.debug("Setting up training callbacks...")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            mode='min'
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
        
        training_logger = TrainingLogger(logger)
        
        logger.info("🎯 Training callbacks configured:")
        logger.info(f"   - Early stopping: patience=3 (monitor val_loss)")
        logger.info(f"   - Learning rate reduction: factor=0.2, patience=2")
        logger.info(f"   - Training progress logger: epoch-by-epoch metrics")
        
        logger.info("")
        logger.info("📊 Training Metrics to be Logged:")
        logger.info("   - Training Loss (MSE)")
        logger.info("   - Training MAE") 
        logger.info("   - Validation Loss")
        logger.info("   - Validation MAE")
        logger.info("   - Epoch timing and progress")
        
        # Train model
        logger.info("")
        logger.info("🚀 Model Training Phase")
        logger.info("-" * 60)
        training_actual_start = datetime.now()
        logger.info(f"Starting training with {len(X_train_scaled):,} samples...")
        logger.info(f"Training configuration:")
        logger.info(f"   - Max epochs: 100")
        logger.info(f"   - Batch size: 64")
        logger.info(f"   - Validation split: 20%")
        
        history = model.fit(
            X_train_scaled,
            y_train_log,  # Using log-transformed target
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr, training_logger],
            verbose=1
        )
        
        training_actual_duration = datetime.now() - training_actual_start
        logger.info(f"⏱️ Model training completed in {training_actual_duration}")
        logger.info(f"📊 Training summary:")
        logger.info(f"   - Epochs completed: {len(history.history['loss'])}")
        logger.info(f"   - Final training loss: {history.history['loss'][-1]:.6f}")
        logger.info(f"   - Final validation loss: {history.history['val_loss'][-1]:.6f}")
        logger.info(f"   - Best validation loss: {min(history.history['val_loss']):.6f}")
        
        # Evaluate model
        logger.info("")
        logger.info("📊 Model Evaluation Phase")
        logger.info("-" * 60)
        eval_start = datetime.now()
        logger.debug("Generating predictions on test set...")
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)  # Convert back from log scale
        
        # Calculate metrics
        logger.debug("Calculating performance metrics...")
        mse = np.mean((y_test - y_pred.flatten()) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred.flatten()))
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
        r2 = 1 - np.sum((y_test - y_pred.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        eval_duration = datetime.now() - eval_start
        logger.info(f"⏱️ Model evaluation completed in {eval_duration}")
        
        logger.info("")
        logger.info("🎯 MODEL PERFORMANCE RESULTS")
        logger.info("=" * 80)
        logger.info(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
        logger.info(f"Mean Absolute Error (MAE):     ${mae:,.2f}")
        logger.info(f"Mean Absolute Percentage Error: {mape:.2f}%")
        logger.info(f"R-squared Score:               {r2:.4f}")
        logger.info("=" * 80)
        
        # Log additional statistics
        logger.debug("Additional performance statistics:")
        logger.debug(f"   - Mean prediction: ${np.mean(y_pred):,.2f}")
        logger.debug(f"   - Prediction std: ${np.std(y_pred):,.2f}")
        logger.debug(f"   - Prediction range: ${np.min(y_pred):,.2f} - ${np.max(y_pred):,.2f}")
        logger.debug(f"   - Actual mean: ${np.mean(y_test):,.2f}")
        logger.debug(f"   - Actual std: ${np.std(y_test):,.2f}")
        
        # Print detailed test predictions
        logger.info("")
        logger.info("📋 Sample Predictions Analysis")
        logger.info("-" * 60)
        print_test_predictions(X_test_scaled, y_test, model, scaler, feature_names)
        
        # Analyze feature importance
        logger.info("")
        logger.info("🔍 Feature Importance Analysis")
        logger.info("-" * 60)
        importance_start = datetime.now()
        logger.debug("Calculating feature importance using permutation method...")
        importance_scores = analyze_feature_importance(model, X_train_scaled, scaler, feature_names)
        importance_duration = datetime.now() - importance_start
        logger.info(f"⏱️ Feature importance analysis completed in {importance_duration}")
        
        logger.info("📊 Top 10 Most Important Features:")
        for i, (feature, importance) in enumerate(importance_scores[:10]):
            logger.info(f"{i+1:2d}. {feature:<25} {importance:8.4f}")
        
        # Log all feature importance for debugging
        logger.debug("Complete feature importance ranking:")
        for i, (feature, importance) in enumerate(importance_scores):
            logger.debug(f"{i+1:2d}. {feature:<25} {importance:8.4f}")
        
        # Plot training history and save plots
        logger.info("")
        logger.info("📈 Generating Visualizations")
        logger.info("-" * 60)
        plot_start = datetime.now()
        logger.debug("Creating training history plot...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot loss and validation metrics
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.plot(history.history['mae'], label='Training MAE', linestyle='--')
        plt.plot(history.history['val_mae'], label='Validation MAE', linestyle='--')
        plt.title('Auto Loans Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('VFLClientModels/plots/auto_loans_training_history.png')
        plt.close()
        logger.debug("✅ Training history plot saved")
        
        # Plot predictions vs actual
        logger.debug("Creating predictions vs actual plot...")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label='Perfect Prediction')
        plt.xlabel('Actual Auto Loan Limit')
        plt.ylabel('Predicted Auto Loan Limit')
        plt.title('Predicted vs Actual Auto Loan Limits')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('VFLClientModels/plots/auto_loans_predictions.png')
        plt.close()
        logger.debug("✅ Predictions plot saved")
        
        plot_duration = datetime.now() - plot_start
        logger.info(f"⏱️ Visualization generation completed in {plot_duration}")
        
        # Save model and features
        logger.info("")
        logger.info("💾 Saving Model and Artifacts")
        logger.info("-" * 60)
        save_start = datetime.now()
        
        try:
            logger.debug("Saving Keras model...")
            model.save('VFLClientModels/saved_models/auto_loans_model.keras')
            logger.info("✅ Model saved as: saved_models/auto_loans_model.keras")
            
            logger.debug("Saving feature names...")
            np.save('VFLClientModels/saved_models/auto_loans_feature_names.npy', feature_names)
            logger.info("✅ Feature names saved as: VFLClientModels/saved_models/auto_loans_feature_names.npy")
            
            logger.debug("Saving plots...")
            logger.info("✅ Plots saved in plots/ directory:")
            logger.info("   - auto_loans_training_history.png")
            logger.info("   - auto_loans_predictions.png")
            
            save_duration = datetime.now() - save_start
            logger.info(f"⏱️ Model saving completed in {save_duration}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {str(e)}")
            raise
        
        # Final summary
        total_duration = datetime.now() - training_start
        logger.info("")
        logger.info("🎉 AUTO LOANS MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total training time: {total_duration}")
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"   - Test MAE: ${mae:,.2f}")
        logger.info(f"   - Test RMSE: ${rmse:,.2f}")
        logger.info(f"   - Test R²: {r2:.4f}")
        logger.info(f"   - Model Parameters: {model.count_params():,}")
        logger.info(f"   - Training Samples: {len(X_train_scaled):,}")
        logger.info(f"   - Test Samples: {len(X_test_scaled):,}")
        logger.info(f"📁 Artifacts saved:")
        logger.info(f"   - Model: saved_models/auto_loans_model.keras")
        logger.info(f"   - Features: saved_models/auto_loans_feature_names.npy")
        logger.info(f"   - Plots: VFLClientModels/plots/auto_loans_*.png")
        logger.info("=" * 80)
        
        return model, history, importance_scores
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Stack trace:", exc_info=True)
        raise

def analyze_feature_importance(model, X_train_scaled, scaler, feature_names):
    """Analyze feature importance using permutation importance"""
    
    # Get baseline predictions
    baseline_pred = model.predict(X_train_scaled)
    baseline_mse = np.mean(baseline_pred ** 2)
    
    # Calculate importance for each feature
    importance_scores = []
    for i in range(X_train_scaled.shape[1]):
        # Create a copy and permute one feature
        X_permuted = X_train_scaled.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Get predictions with permuted feature
        permuted_pred = model.predict(X_permuted)
        permuted_mse = np.mean(permuted_pred ** 2)
        
        # Calculate importance
        importance = (permuted_mse - baseline_mse) / baseline_mse
        importance_scores.append((feature_names[i], importance))
    
    # Sort by importance
    return sorted(importance_scores, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    train_model() 