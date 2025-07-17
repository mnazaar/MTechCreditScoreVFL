import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import os
import warnings
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import tensorflow as tf

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Setup comprehensive logging for Digital Savings model training"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger
    logger = logging.getLogger('Digital_Savings_Model')
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
        f'VFLClientModels/logs/digital_savings_model_{run_timestamp}.log',
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
    logger.info("Digital Savings Model Logging System Initialized")
    logger.info(f"Log file: VFLClientModels/logs/digital_savings_model_{run_timestamp}.log")
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
        accuracy = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_accuracy = logs.get('val_accuracy', 0)
        
        self.logger.info(f"Epoch {epoch + 1:3d} - "
                        f"loss: {loss:.6f} - "
                        f"accuracy: {accuracy:.4f} - "
                        f"val_loss: {val_loss:.6f} - "
                        f"val_accuracy: {val_accuracy:.4f}")
        
        # Log to training-specific handler with more detail
        training_logger = logging.getLogger('Digital_Savings_Model')
        training_logger.debug(f"Epoch {epoch + 1} detailed metrics:")
        for metric, value in logs.items():
            training_logger.debug(f"  {metric}: {value}")
    
    def on_train_begin(self, logs=None):
        self.logger.info("🚀 Training started - logging epoch progress...")
    
    def on_train_end(self, logs=None):
        self.logger.info("✅ Training completed - final metrics logged")

def create_model(input_dim, num_classes):
    """Create a neural network model with 8-unit penultimate layer - enhanced for more features"""
    logger.info("🏗️ Building digital banking classification model...")
    logger.debug(f"Input dimension: {input_dim}")
    logger.debug(f"Number of classes: {num_classes}")
    
    model = Sequential([
        # Input layer - increased capacity for more features
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 1
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 3 - additional layer for complexity
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Penultimate layer with 8 units
        Dense(8, activation='relu', name='penultimate_layer'),
        BatchNormalization(),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_params = model.count_params()
    logger.info(f"📊 Model architecture created:")
    logger.info(f"   - Architecture: 256→128→64→32→8→{num_classes}")
    logger.info(f"   - Penultimate layer: 8 units (for VFL compatibility)")
    logger.info(f"   - Total parameters: {total_params:,}")
    logger.info(f"   - Optimizer: Adam")
    logger.info(f"   - Loss: Categorical Crossentropy")
    logger.debug("Model layers:")
    for i, layer in enumerate(model.layers):
        layer_info = f"   Layer {i+1}: {layer.__class__.__name__}"
        if hasattr(layer, 'units'):
            layer_info += f" ({layer.units} units)"
        if hasattr(layer, 'activation') and layer.activation:
            layer_info += f" - {layer.activation.__name__}"
        logger.debug(layer_info)
    
    return model

def load_and_preprocess_data():
    """Load and preprocess the digital savings bank data"""
    logger.info("🔄 Loading and preprocessing digital banking data...")
    
    # Load the data
    logger.debug("Loading digital savings dataset from CSV...")
    # Try multiple possible paths for the dataset
    possible_paths = [
        'VFLClientModels/dataset/data/banks/digital_savings_bank.csv'
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            logger.debug(f"Successfully loaded data from: {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError(f"Could not find digital_savings_bank.csv in any of the expected locations: {possible_paths}")
    
    logger.info(f"📊 Dataset loaded: {len(df):,} total records")
    
    # Enhanced feature engineering - using new rich dataset features
    logger.debug("Creating derived features...")
    df['transaction_volume'] = df['avg_monthly_transactions'] * df['avg_transaction_value']
    df['digital_engagement_score'] = (df['digital_banking_score'] + df['mobile_banking_usage']) / 2
    logger.debug("✅ Derived features created: transaction_volume, digital_engagement_score")
    
    # Expanded feature set - leveraging the richer dataset
    feature_columns = [
        # Core financial features
        'annual_income',
        'savings_balance',
        'checking_balance',
        'investment_balance',              # New: Investment portfolio
        'payment_history',
        'credit_score',                    # New: Credit worthiness
        'age',
        'employment_length',               # New: Job stability
        
        # Transaction and banking behavior
        'avg_monthly_transactions',
        'avg_transaction_value',
        'transaction_volume',              # Calculated
        'digital_banking_score',
        'mobile_banking_usage',
        'online_transactions_ratio',
        'international_transactions_ratio', # New: International activity
        'e_statement_enrolled',
        'digital_engagement_score',        # Calculated
        
        # Financial behavior and credit
        'monthly_expenses',                # New: Spending pattern
        'total_credit_limit',              # New: Credit capacity
        'credit_utilization_ratio',        # New: Credit usage
        'num_credit_cards',                # New: Credit relationships
        'credit_history_length',           # New: Credit maturity
        
        # Additional calculated metrics from dataset
        'total_wealth',                    # New: Combined assets
        'net_worth',                       # New: Assets minus debts
        'credit_efficiency',               # New: Credit usage efficiency
        'financial_stability_score'       # New: Overall financial health
    ]
    
    logger.info(f"📋 Selected {len(feature_columns)} features for digital banking classification")
    logger.debug(f"Features: {feature_columns}")
    
    X = df[feature_columns]
    y = df['customer_category']
    customer_ids = df['tax_id']
    
    logger.info(f"📊 Dataset composition:")
    logger.info(f"   - Total customers: {len(df):,}")
    logger.info(f"   - Features: {len(feature_columns)}")
    logger.info(f"   - Target classes: {len(y.unique())}")
    
    logger.info(f"📊 Target variable distribution:")
    class_counts = y.value_counts()
    for category, count in class_counts.items():
        percentage = (count / len(y)) * 100
        logger.info(f"   - {category}: {count:,} samples ({percentage:.1f}%)")
    
    # Handle any infinite values
    logger.debug("Handling infinite and missing values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.mean())
    missing_after = X.isnull().sum().sum()
    logger.debug(f"Missing/infinite values: {missing_before} → {missing_after}")
    
    # Log transform monetary values - expanded for new features
    logger.debug("Applying log transformations to monetary features...")
    monetary_columns = [
        'annual_income', 'savings_balance', 'checking_balance', 'investment_balance',
        'avg_transaction_value', 'transaction_volume', 'monthly_expenses', 
        'total_credit_limit', 'total_wealth'
    ]
    transformed_count = 0
    for col in monetary_columns:
        # Handle negative values in net_worth separately
        if col in X.columns:
            X[col] = np.log1p(np.maximum(X[col], 0))  # Ensure non-negative before log transform
            transformed_count += 1
    logger.debug(f"Log transformed {transformed_count} monetary columns")
    
    # Handle net_worth separately (can be negative, so no log transform)
    # Just ensure it's properly scaled with the rest
    logger.debug("Handling net_worth (can be negative - no log transform)")
    
    # Scale features
    logger.debug("Applying StandardScaler to features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.debug("✅ Feature scaling completed")
    
    # Encode target labels
    logger.debug("Encoding target labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"📊 Label encoding results:")
    logger.info(f"   - Classes: {list(label_encoder.classes_)}")
    logger.info(f"   - Encoded range: {y_encoded.min()} to {y_encoded.max()}")
    
    # Split the data
    logger.debug(f"Splitting data (80/20 split, stratified by class)...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_scaled, y_encoded, customer_ids,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    logger.info(f"📊 Data split results:")
    logger.info(f"   - Training samples: {len(X_train):,}")
    logger.info(f"   - Test samples: {len(X_test):,}")
    
    # Log class distribution in splits
    logger.debug("Class distribution in training set:")
    train_class_counts = np.bincount(y_train)
    for i, count in enumerate(train_class_counts):
        class_name = label_encoder.classes_[i]
        logger.debug(f"   - {class_name}: {count:,} samples")
    
    logger.debug("Class distribution in test set:")
    test_class_counts = np.bincount(y_test)
    for i, count in enumerate(test_class_counts):
        class_name = label_encoder.classes_[i]
        logger.debug(f"   - {class_name}: {count:,} samples")
    
    # Apply SMOTE
    logger.info("🔄 Applying SMOTE for class balancing...")
    logger.debug("Class distribution before SMOTE:")
    for i, count in enumerate(train_class_counts):
        logger.debug(f"   - Class {i}: {count:,} samples")
    
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info("📊 SMOTE balancing results:")
    logger.info(f"   - Original training samples: {len(X_train):,}")
    logger.info(f"   - Balanced training samples: {len(X_train_balanced):,}")
    balanced_class_counts = np.bincount(y_train_balanced)
    for i, count in enumerate(balanced_class_counts):
        class_name = label_encoder.classes_[i]
        logger.info(f"   - {class_name}: {count:,} samples")
    
    # Convert to categorical
    logger.debug("Converting labels to categorical format...")
    y_train_cat = to_categorical(y_train_balanced)
    y_test_cat = to_categorical(y_test)
    logger.debug(f"Categorical shape: train {y_train_cat.shape}, test {y_test_cat.shape}")
    
    logger.info("✅ Data preprocessing completed successfully")
    
    return (X_train_balanced, X_test, y_train_cat, y_test_cat,
            label_encoder.classes_, feature_columns, ids_test, scaler, label_encoder)

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Digital Bank Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Digital Bank Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('VFLClientModels/plots/digital_bank_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Digital Bank Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('VFLClientModels/plots/digital_bank_confusion_matrix.png')
    plt.close()

def test_model_samples(X_test, y_test, customer_ids, classes, model, n_samples=5):
    """Test the model with sample records"""
    logger.info("\nTesting model with random sample records:")
    logger.info("-" * 100)
    logger.info(f"{'Customer ID':<15} {'Original Category':<20} {'Predicted Category':<20} {'Confidence':<10} {'Correct':<10}")
    logger.info("-" * 100)
    
    # Get random indices for sampling
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions
    y_pred_prob = model.predict(X_test[indices])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test[indices], axis=1)
    
    # Print results
    for idx, (true, pred, prob) in enumerate(zip(y_true, y_pred, y_pred_prob)):
        customer_id = customer_ids.iloc[indices[idx]]
        confidence = prob[pred] * 100
        is_correct = "✓" if true == pred else "✗"
        logger.info(f"{customer_id:<15} {classes[true]:<20} {classes[pred]:<20} {confidence:>6.2f}% {is_correct:^10}")
    
    logger.info("-" * 100)

def main():
    training_start = datetime.now()
    logger.info("🚀 Starting Digital Banking Classification Model Training")
    logger.info("=" * 80)
    logger.info(f"Training session started: {training_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    
    # Create necessary directories
    logger.debug("Creating output directories...")
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    try:
        # Load and preprocess data
        logger.info("")
        logger.info("📊 Data Loading and Preprocessing Phase")
        logger.info("-" * 60)
        data_start = datetime.now()
        (X_train, X_test, y_train, y_test, classes, feature_names,
         customer_ids, scaler, label_encoder) = load_and_preprocess_data()
        data_duration = datetime.now() - data_start
        logger.info(f"⏱️ Data preprocessing completed in {data_duration}")
        
        # Create and train model
        logger.info("")
        logger.info("🏗️ Model Building Phase")
        logger.info("-" * 60)
        model_start = datetime.now()
        model = create_model(input_dim=len(feature_names), num_classes=len(classes))
        model_duration = datetime.now() - model_start
        logger.info(f"⏱️ Model building completed in {model_duration}")
        
        # Print model summary
        logger.info("")
        logger.info("📊 Model Architecture Summary:")
        logger.debug("Generating model summary...")
        model.summary()
        
        # Define callbacks
        logger.info("")
        logger.info("⚙️ Training Configuration Phase")
        logger.info("-" * 60)
        logger.debug("Setting up training callbacks...")
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'VFLClientModels/saved_models/digital_bank_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TrainingLogger(logger)
        ]
        
        logger.info("🎯 Training callbacks configured:")
        logger.info(f"   - Early stopping: patience=5 (monitor val_loss)")
        logger.info(f"   - Model checkpoint: save best model by val_loss")
        logger.info(f"   - Training progress logger: epoch-by-epoch metrics")
        logger.info(f"   - Max epochs: 50")
        logger.info(f"   - Batch size: 128")
        logger.info(f"   - Validation split: 20%")
        
        logger.info("")
        logger.info("📊 Training Metrics to be Logged:")
        logger.info("   - Training Loss (categorical crossentropy)")
        logger.info("   - Training Accuracy") 
        logger.info("   - Validation Loss")
        logger.info("   - Validation Accuracy")
        logger.info("   - Epoch timing and progress")
        
        # Train model
        logger.info("")
        logger.info("🚀 Model Training Phase")
        logger.info("-" * 60)
        training_actual_start = datetime.now()
        logger.info(f"Starting training with {len(X_train):,} balanced samples...")
        logger.info(f"Training on {len(feature_names)} features, {len(classes)} classes")
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=128,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        training_actual_duration = datetime.now() - training_actual_start
        logger.info(f"⏱️ Model training completed in {training_actual_duration}")
        
        # Plot training history
        logger.info("")
        logger.info("📈 Generating Training Visualizations")
        logger.info("-" * 60)
        plot_start = datetime.now()
        logger.debug("Creating training history plots...")
        plot_training_history(history)
        logger.debug("✅ Training history plots saved")
        
        # Evaluate model
        logger.info("")
        logger.info("📊 Model Evaluation Phase")
        logger.info("-" * 60)
        eval_start = datetime.now()
        logger.debug("Generating predictions on test set...")
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        eval_duration = datetime.now() - eval_start
        logger.info(f"⏱️ Model evaluation completed in {eval_duration}")
        
        # Calculate and log performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test_labels, y_pred)
        precision = precision_score(y_test_labels, y_pred, average='weighted')
        recall = recall_score(y_test_labels, y_pred, average='weighted')
        f1 = f1_score(y_test_labels, y_pred, average='weighted')
        
        logger.info("")
        logger.info("🎯 MODEL PERFORMANCE RESULTS")
        logger.info("=" * 80)
        logger.info(f"Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Weighted Precision:   {precision:.4f}")
        logger.info(f"Weighted Recall:      {recall:.4f}")
        logger.info(f"Weighted F1-Score:    {f1:.4f}")
        logger.info("=" * 80)
        
        # Detailed classification report
        logger.info("")
        logger.info("📋 Detailed Classification Report:")
        logger.info(classification_report(y_test_labels, y_pred, target_names=classes))
        
        # Log per-class performance
        logger.debug("Per-class performance details:")
        report = classification_report(y_test_labels, y_pred, target_names=classes, output_dict=True)
        for class_name in classes:
            if class_name in report:
                class_metrics = report[class_name]
                logger.debug(f"   {class_name}: precision={class_metrics['precision']:.3f}, "
                           f"recall={class_metrics['recall']:.3f}, f1={class_metrics['f1-score']:.3f}")
        
        # Create confusion matrix plot
        logger.debug("Creating confusion matrix plot...")
        plot_confusion_matrix(y_test_labels, y_pred, classes)
        logger.debug("✅ Confusion matrix plot saved")
        
        plot_duration = datetime.now() - plot_start
        logger.info(f"⏱️ Visualization generation completed in {plot_duration}")
        
        # Save additional artifacts
        logger.info("")
        logger.info("💾 Saving Model and Artifacts")
        logger.info("-" * 60)
        save_start = datetime.now()
        
        try:
            logger.debug("Saving feature names...")
            np.save('VFLClientModels/saved_models/digital_bank_feature_names.npy', feature_names)
            logger.info("✅ Feature names saved as: VFLClientModels/saved_models/digital_bank_feature_names.npy")
            
            logger.debug("Saving additional artifacts...")
            # The model was already saved by ModelCheckpoint callback
            logger.info("✅ Model saved as: VFLClientModels/saved_models/digital_bank_model.keras")
            logger.info("✅ Plots saved in VFLClientModels/plots/ directory:")
            logger.info("   - digital_bank_training_history.png")
            logger.info("   - digital_bank_confusion_matrix.png")
            
            save_duration = datetime.now() - save_start
            logger.info(f"⏱️ Model saving completed in {save_duration}")
            
        except Exception as e:
            logger.error(f"❌ Error saving artifacts: {str(e)}")
            raise
        
        # Test model with samples
        logger.info("")
        logger.info("🔍 Sample Predictions Analysis")
        logger.info("-" * 60)
        test_model_samples(X_test, y_test, customer_ids, classes, model)
        
        # Final summary
        total_duration = datetime.now() - training_start
        logger.info("")
        logger.info("🎉 DIGITAL BANKING MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total training time: {total_duration}")
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"   - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   - Weighted F1-Score: {f1:.4f}")
        logger.info(f"   - Model Parameters: {model.count_params():,}")
        logger.info(f"   - Training Samples: {len(X_train):,} (after SMOTE)")
        logger.info(f"   - Test Samples: {len(X_test):,}")
        logger.info(f"   - Features: {len(feature_names)}")
        logger.info(f"   - Classes: {len(classes)} ({', '.join(classes)})")
        logger.info(f"📁 Artifacts saved:")
        logger.info(f"   - Model: VFLClientModels/saved_models/digital_bank_model.keras")
        logger.info(f"   - Features: VFLClientModels/saved_models/digital_bank_feature_names.npy")
        logger.info(f"   - Plots: VFLClientModels/plots/digital_bank_*.png")
        logger.info("=" * 80)
        
        return model, history, classes, feature_names
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    main() 