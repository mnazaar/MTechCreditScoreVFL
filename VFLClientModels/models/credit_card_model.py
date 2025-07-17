import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    """Setup comprehensive logging for Credit Card model training"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger
    logger = logging.getLogger('Credit_Card_Model')
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
        f'logs/credit_card_model_{run_timestamp}.log',
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
    logger.info("Credit Card Model Logging System Initialized")
    logger.info(f"Log file: logs/credit_card_model_{run_timestamp}.log")
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
        training_logger = logging.getLogger('Credit_Card_Model')
        training_logger.debug(f"Epoch {epoch + 1} detailed metrics:")
        for metric, value in logs.items():
            training_logger.debug(f"  {metric}: {value}")
    
    def on_train_begin(self, logs=None):
        self.logger.info("🚀 Training started - logging epoch progress...")
    
    def on_train_end(self, logs=None):
        self.logger.info("✅ Training completed - final metrics logged")

def create_model(input_dim, num_classes):
    """Create a neural network model with 8-unit penultimate layer for credit card tier classification"""
    logger.info("🏗️ Building credit card tier classification model...")
    logger.debug(f"Input dimension: {input_dim}")
    logger.debug(f"Number of card tiers: {num_classes}")
    
    model = Sequential([
        # Input layer - robust capacity for credit card features
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 1
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),
        
        # Hidden layer 2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 3 - additional complexity for credit decisions
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        
        # Penultimate layer with 8 units (for VFL compatibility)
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
    """Load and preprocess the credit card data"""
    logger.info("🔄 Loading and preprocessing credit card data...")
    
    # Load the data
    logger.debug("Loading credit card dataset from CSV...")
    df = pd.read_csv('../dataset/data/banks/credit_card_bank.csv')
    logger.info(f"📊 Dataset loaded: {len(df):,} total records")
    
    # Enhanced feature engineering for credit card classification
    logger.debug("Creating derived features for credit card prediction...")
    
    # Credit capacity utilization
    df['credit_capacity_ratio'] = df['credit_card_limit'] / df['total_credit_limit'].replace(0, 1)
    
    # Income to limit ratio
    df['income_to_limit_ratio'] = df['annual_income'] / df['credit_card_limit'].replace(0, 1)
    
    # Debt service ratio
    df['debt_service_ratio'] = (df['current_debt'] * 0.03) / (df['annual_income'] / 12)  # Assuming 3% monthly payment
    
    # Risk-adjusted income
    df['risk_adjusted_income'] = df['annual_income'] * (df['risk_score'] / 100)
    
    logger.debug("✅ Derived features created")
    
    # Select comprehensive features for credit card tier prediction
    feature_columns = [
        # Core financial features
        'annual_income',                # Primary income factor
        'credit_score',                 # Most important for credit cards
        'payment_history',              # Payment reliability
        'employment_length',            # Income stability
        'debt_to_income_ratio',         # Debt burden
        'age',                          # Customer demographics
        
        # Credit behavior and history
        'credit_history_length',        # Credit experience
        'num_credit_cards',             # Existing credit relationships
        'num_loan_accounts',            # Other credit relationships
        'total_credit_limit',           # Current credit exposure
        'credit_utilization_ratio',     # Credit usage behavior
        'late_payments',                # Payment behavior
        'credit_inquiries',             # Recent credit activity
        'last_late_payment_days',       # Recent payment behavior
        
        # Financial position
        'current_debt',                 # Current debt level
        'monthly_expenses',             # Monthly obligations
        'savings_balance',              # Financial cushion
        'checking_balance',             # Liquid assets
        'investment_balance',           # Investment portfolio
        'auto_loan_balance',            # Auto debt
        'mortgage_balance',             # Mortgage debt
        
        # Credit card specific metrics
        'apr',                          # Interest rate offered
        'risk_score',                   # Risk assessment
        'total_available_credit',       # Total available credit
        'credit_to_income_ratio',       # Credit to income ratio
        'cash_advance_limit',           # Cash advance capacity
        
        # Derived features
        'credit_capacity_ratio',        # Credit capacity utilization
        'income_to_limit_ratio',        # Income to limit ratio
        'debt_service_ratio',           # Debt service capacity
        'risk_adjusted_income'          # Risk-adjusted income
    ]
    
    logger.info(f"📋 Selected {len(feature_columns)} features for credit card tier classification")
    logger.debug(f"Features: {feature_columns}")
    
    X = df[feature_columns]
    y = df['card_tier']
    customer_ids = df['tax_id']
    
    logger.info(f"📊 Dataset composition:")
    logger.info(f"   - Total customers: {len(df):,}")
    logger.info(f"   - Features: {len(feature_columns)}")
    logger.info(f"   - Target classes: {len(y.unique())}")
    
    logger.info(f"📊 Target variable distribution:")
    class_counts = y.value_counts()
    for tier, count in class_counts.items():
        percentage = (count / len(y)) * 100
        logger.info(f"   - {tier}: {count:,} samples ({percentage:.1f}%)")
    
    # Handle any infinite or missing values
    logger.debug("Handling infinite and missing values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()
    logger.debug(f"Missing/infinite values: {missing_before} → {missing_after}")
    
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
    logger.info(f"   - Card tiers: {list(label_encoder.classes_)}")
    logger.info(f"   - Encoded range: {y_encoded.min()} to {y_encoded.max()}")
    
    # Split the data
    logger.debug(f"Splitting data (80/20 split, stratified by card tier)...")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_scaled, y_encoded, customer_ids,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    logger.info(f"📊 Data split results:")
    logger.info(f"   - Training samples: {len(X_train):,}")
    logger.info(f"   - Test samples: {len(X_test):,}")
    
    # Log tier distribution in splits
    logger.debug("Tier distribution in training set:")
    train_class_counts = np.bincount(y_train)
    for i, count in enumerate(train_class_counts):
        tier_name = label_encoder.classes_[i]
        logger.debug(f"   - {tier_name}: {count:,} samples")
    
    logger.debug("Tier distribution in test set:")
    test_class_counts = np.bincount(y_test)
    for i, count in enumerate(test_class_counts):
        tier_name = label_encoder.classes_[i]
        logger.debug(f"   - {tier_name}: {count:,} samples")
    
    # Apply SMOTE for class balancing
    logger.info("🔄 Applying SMOTE for class balancing...")
    logger.debug("Class distribution before SMOTE:")
    for i, count in enumerate(train_class_counts):
        tier_name = label_encoder.classes_[i]
        logger.debug(f"   - {tier_name}: {count:,} samples")
    
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info("📊 SMOTE balancing results:")
    logger.info(f"   - Original training samples: {len(X_train):,}")
    logger.info(f"   - Balanced training samples: {len(X_train_balanced):,}")
    balanced_class_counts = np.bincount(y_train_balanced)
    for i, count in enumerate(balanced_class_counts):
        tier_name = label_encoder.classes_[i]
        logger.info(f"   - {tier_name}: {count:,} samples")
    
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
    logger.debug("Creating training history plots...")
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Credit Card Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Credit Card Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot learning rate (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate', linewidth=2)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Tracked', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig('plots/credit_card_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    logger.debug("Creating confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Credit Card Tier Model Confusion Matrix')
    plt.ylabel('True Card Tier')
    plt.xlabel('Predicted Card Tier')
    plt.tight_layout()
    plt.savefig('plots/credit_card_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tier_analysis(df, classes):
    """Plot comprehensive tier analysis"""
    logger.debug("Creating tier analysis plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Credit Card Tier Analysis', fontsize=16, fontweight='bold')
    
    # 1. Tier distribution
    axes[0, 0].pie(df['card_tier'].value_counts().values, 
                   labels=df['card_tier'].value_counts().index,
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Card Tier Distribution')
    
    # 2. Credit score by tier
    tier_order = ['Secured', 'Basic', 'Silver', 'Gold', 'Platinum']
    df_plot = df[df['card_tier'].isin(tier_order)]
    credit_scores = [df_plot[df_plot['card_tier'] == tier]['credit_score'].values 
                    for tier in tier_order if tier in df_plot['card_tier'].values]
    axes[0, 1].boxplot(credit_scores, labels=[t for t in tier_order if t in df_plot['card_tier'].values])
    axes[0, 1].set_title('Credit Score Distribution by Tier')
    axes[0, 1].set_ylabel('Credit Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Income by tier
    incomes = [df_plot[df_plot['card_tier'] == tier]['annual_income'].values 
              for tier in tier_order if tier in df_plot['card_tier'].values]
    axes[0, 2].boxplot(incomes, labels=[t for t in tier_order if t in df_plot['card_tier'].values])
    axes[0, 2].set_title('Annual Income Distribution by Tier')
    axes[0, 2].set_ylabel('Annual Income ($)')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Credit limit by tier
    limits = [df_plot[df_plot['card_tier'] == tier]['credit_card_limit'].values 
             for tier in tier_order if tier in df_plot['card_tier'].values]
    axes[1, 0].boxplot(limits, labels=[t for t in tier_order if t in df_plot['card_tier'].values])
    axes[1, 0].set_title('Credit Limit Distribution by Tier')
    axes[1, 0].set_ylabel('Credit Limit ($)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. APR by tier
    aprs = [df_plot[df_plot['card_tier'] == tier]['apr'].values 
           for tier in tier_order if tier in df_plot['card_tier'].values]
    axes[1, 1].boxplot(aprs, labels=[t for t in tier_order if t in df_plot['card_tier'].values])
    axes[1, 1].set_title('APR Distribution by Tier')
    axes[1, 1].set_ylabel('APR (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Risk score by tier
    risks = [df_plot[df_plot['card_tier'] == tier]['risk_score'].values 
            for tier in tier_order if tier in df_plot['card_tier'].values]
    axes[1, 2].boxplot(risks, labels=[t for t in tier_order if t in df_plot['card_tier'].values])
    axes[1, 2].set_title('Risk Score Distribution by Tier')
    axes[1, 2].set_ylabel('Risk Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/credit_card_tier_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def detailed_prediction_analysis(X_test, y_test, customer_ids, classes, model, scaler, feature_names, original_df, n_samples=15):
    """Detailed prediction analysis with confidence scores, actual vs predicted, and customer profiles"""
    logger.info("🔍 Performing detailed prediction analysis...")
    
    # Get random indices for sampling
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions and probabilities
    y_pred_prob = model.predict(X_test[indices], verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test[indices], axis=1)
    
    logger.info("\n" + "="*120)
    logger.info("DETAILED CREDIT CARD TIER PREDICTION ANALYSIS")
    logger.info("="*120)
    logger.info(f"{'Tax ID':<12} {'Actual':<9} {'Predicted':<9} {'Confidence':<11} {'Status':<8} {'Credit Score':<12} {'Income':<12} {'Key Profile'}")
    logger.info("-"*120)
    
    # Track statistics
    correct_predictions = 0
    high_confidence_correct = 0
    low_confidence_incorrect = 0
    
    # Analyze each sample
    for idx, (true_idx, pred_idx, prob_dist) in enumerate(zip(y_true, y_pred, y_pred_prob)):
        customer_id = customer_ids.iloc[indices[idx]]
        true_tier = classes[true_idx]
        pred_tier = classes[pred_idx]
        confidence = prob_dist[pred_idx] * 100
        
        # Status indicators
        is_correct = true_idx == pred_idx
        if is_correct:
            correct_predictions += 1
            status = "✓"
            if confidence >= 80:
                high_confidence_correct += 1
        else:
            status = "✗"
            if confidence < 60:
                low_confidence_incorrect += 1
        
        # Get customer financial profile from original data
        customer_data = original_df[original_df['tax_id'] == customer_id].iloc[0]
        credit_score = int(customer_data['credit_score'])
        annual_income = customer_data['annual_income']
        debt_ratio = customer_data['debt_to_income_ratio']
        risk_score = customer_data['risk_score']
        
        # Create key profile summary
        if annual_income >= 100000:
            income_level = "High"
        elif annual_income >= 50000:
            income_level = "Med"
        else:
            income_level = "Low"
            
        if credit_score >= 750:
            credit_level = "Exc"
        elif credit_score >= 700:
            credit_level = "VG"
        elif credit_score >= 650:
            credit_level = "Good"
        else:
            credit_level = "Fair"
            
        profile = f"{income_level}-{credit_level}, DTI:{debt_ratio:.2f}, Risk:{risk_score:.0f}"
        
        # Format output
        logger.info(f"{customer_id:<12} {true_tier:<9} {pred_tier:<9} {confidence:>8.1f}% {status:^8} "
              f"{credit_score:<12} ${annual_income/1000:>8.0f}K {profile}")
        
        # Show probability distribution for misclassifications
        if not is_correct and confidence < 70:
            prob_info = f"{'':>12} Probability Distribution: "
            sorted_indices = np.argsort(prob_dist)[::-1]
            for i, class_idx in enumerate(sorted_indices[:3]):
                if i > 0:
                    prob_info += " | "
                prob_info += f"{classes[class_idx]}: {prob_dist[class_idx]*100:.1f}%"
            logger.info(prob_info)
    
    logger.info("-"*120)
    
    # Summary statistics
    total_samples = len(indices)
    accuracy = (correct_predictions / total_samples) * 100
    
    logger.info(f"\nPREDICTION SUMMARY:")
    logger.info(f"Total Samples Analyzed: {total_samples}")
    logger.info(f"Correct Predictions: {correct_predictions} ({accuracy:.1f}%)")
    logger.info(f"Incorrect Predictions: {total_samples - correct_predictions} ({100-accuracy:.1f}%)")
    logger.info(f"High Confidence Correct (≥80%): {high_confidence_correct}")
    logger.info(f"Low Confidence Incorrect (<60%): {low_confidence_incorrect}")
    
    # Tier-specific analysis
    logger.info(f"\nTIER-SPECIFIC PERFORMANCE:")
    tier_correct = {}
    tier_total = {}
    
    for true_idx, pred_idx in zip(y_true, y_pred):
        tier = classes[true_idx]
        if tier not in tier_total:
            tier_total[tier] = 0
            tier_correct[tier] = 0
        tier_total[tier] += 1
        if true_idx == pred_idx:
            tier_correct[tier] += 1
    
    for tier in sorted(tier_total.keys()):
        if tier_total[tier] > 0:
            tier_accuracy = (tier_correct[tier] / tier_total[tier]) * 100
            logger.info(f"  {tier:<10}: {tier_correct[tier]}/{tier_total[tier]} ({tier_accuracy:.1f}%)")
    
    # Confidence distribution analysis
    logger.info(f"\nCONFIDENCE DISTRIBUTION:")
    confidences = [prob_dist[pred_idx] * 100 for pred_idx, prob_dist in zip(y_pred, y_pred_prob)]
    high_conf = sum(1 for c in confidences if c >= 80)
    med_conf = sum(1 for c in confidences if 60 <= c < 80)
    low_conf = sum(1 for c in confidences if c < 60)
    
    logger.info(f"  High Confidence (≥80%): {high_conf} ({high_conf/total_samples*100:.1f}%)")
    logger.info(f"  Medium Confidence (60-80%): {med_conf} ({med_conf/total_samples*100:.1f}%)")
    logger.info(f"  Low Confidence (<60%): {low_conf} ({low_conf/total_samples*100:.1f}%)")
    
    logger.info("="*120)
    
    # Log summary to logger
    logger.info(f"Sample prediction analysis completed:")
    logger.info(f"   - Samples analyzed: {total_samples}")
    logger.info(f"   - Overall accuracy: {accuracy:.1f}%")
    logger.info(f"   - High confidence predictions: {high_conf} ({high_conf/total_samples*100:.1f}%)")
    logger.info(f"   - Average confidence: {np.mean(confidences):.1f}%")
    
    return {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'high_confidence_correct': high_confidence_correct,
        'confidence_distribution': confidences
    }

def test_model_samples(X_test, y_test, customer_ids, classes, model, n_samples=10):
    """Test the model with sample records"""
    logger.info("🔍 Testing model with sample records...")
    logger.info("\nTesting Credit Card Tier Model with sample records:")
    logger.info("-" * 100)
    logger.info(f"{'Customer ID':<15} {'True Tier':<12} {'Predicted Tier':<15} {'Confidence':<12} {'Correct':<8} {'Details'}")
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
        
        # Get second highest probability for comparison
        sorted_probs = np.sort(prob)[::-1]
        margin = (sorted_probs[0] - sorted_probs[1]) * 100
        
        details = f"Margin: {margin:.1f}%"
        
        logger.info(f"{customer_id:<15} {classes[true]:<12} {classes[pred]:<15} "
              f"{confidence:>8.1f}% {is_correct:^8} {details}")
    
    logger.info("-" * 100)
    
    # Calculate sample accuracy
    sample_accuracy = np.mean(y_true == y_pred) * 100
    logger.info(f"Sample accuracy: {sample_accuracy:.1f}%")

def main():
    training_start = datetime.now()
    logger.info("🚀 Starting Credit Card Tier Classification Model Training")
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
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'saved_models/credit_card_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TrainingLogger(logger)
        ]
        
        logger.info("🎯 Training callbacks configured:")
        logger.info(f"   - Early stopping: patience=10 (monitor val_accuracy)")
        logger.info(f"   - Model checkpoint: save best model by val_accuracy")
        logger.info(f"   - Training progress logger: epoch-by-epoch metrics")
        logger.info(f"   - Max epochs: 100")
        logger.info(f"   - Batch size: 32")
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
        logger.info(f"Training on {len(feature_names)} features, {len(classes)} card tiers")
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
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
        plot_training_history(history)
        logger.debug("✅ Training history plots saved")
        
        # Load original data for tier analysis
        original_df = pd.read_csv('../dataset/data/banks/credit_card_bank.csv')
        plot_tier_analysis(original_df, classes)
        logger.debug("✅ Tier analysis plots saved")
        
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
            np.save('saved_models/credit_card_feature_names.npy', feature_names)
            logger.info("✅ Feature names saved as: saved_models/credit_card_feature_names.npy")
            
            logger.debug("Saving label encoder...")
            import joblib
            joblib.dump(label_encoder, 'saved_models/credit_card_label_encoder.pkl')
            logger.info("✅ Label encoder saved as: saved_models/credit_card_label_encoder.pkl")
            
            logger.debug("Saving scaler...")
            joblib.dump(scaler, 'saved_models/credit_card_scaler.pkl')
            logger.info("✅ Scaler saved as: saved_models/credit_card_scaler.pkl")
            
            # The model was already saved by ModelCheckpoint callback
            logger.info("✅ Model saved as: saved_models/credit_card_model.keras")
            logger.info("✅ Plots saved in plots/ directory:")
            logger.info("   - credit_card_training_history.png")
            logger.info("   - credit_card_confusion_matrix.png")
            logger.info("   - credit_card_tier_analysis.png")
            
            save_duration = datetime.now() - save_start
            logger.info(f"⏱️ Model saving completed in {save_duration}")
            
        except Exception as e:
            logger.error(f"❌ Error saving artifacts: {str(e)}")
            raise
        
        # Test model with samples
        logger.info("")
        logger.info("🔍 Sample Predictions Analysis")
        logger.info("-" * 60)
        
        # Load original data for detailed analysis
        original_df = pd.read_csv('../dataset/data/banks/credit_card_bank.csv')
        
        # Detailed prediction analysis
        prediction_results = detailed_prediction_analysis(
            X_test, y_test, customer_ids, classes, model, 
            scaler, feature_names, original_df, n_samples=15
        )
        
        # Quick test samples (simpler format)
        test_model_samples(X_test, y_test, customer_ids, classes, model)
        
        # Final summary
        total_duration = datetime.now() - training_start
        logger.info("")
        logger.info("🎉 CREDIT CARD TIER MODEL TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total training time: {total_duration}")
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"   - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   - Weighted F1-Score: {f1:.4f}")
        logger.info(f"   - Model Parameters: {model.count_params():,}")
        logger.info(f"   - Training Samples: {len(X_train):,} (after SMOTE)")
        logger.info(f"   - Test Samples: {len(X_test):,}")
        logger.info(f"   - Features: {len(feature_names)}")
        logger.info(f"   - Card Tiers: {len(classes)} ({', '.join(classes)})")
        logger.info(f"📁 Artifacts saved:")
        logger.info(f"   - Model: saved_models/credit_card_model.keras")
        logger.info(f"   - Features: saved_models/credit_card_feature_names.npy")
        logger.info(f"   - Label Encoder: saved_models/credit_card_label_encoder.pkl")
        logger.info(f"   - Scaler: saved_models/credit_card_scaler.pkl")
        logger.info(f"   - Plots: plots/credit_card_*.png")
        logger.info("=" * 80)
        
        return model, history, classes, feature_names
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("💳 Starting Credit Card Tier Model Training...")
    logger.info("This may take several minutes...")
    
    # Train the model
    model, history, classes, feature_names = main()
    
    logger.info("\n🎉 Credit Card Tier Model Training Completed Successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated plots in the 'plots/' directory")
    logger.info("2. Check tier classification performance across different customer segments")
    logger.info("3. Analyze feature importance for tier decisions")
    logger.info("4. Consider the model for integration into the VFL system as the 4th bank") 