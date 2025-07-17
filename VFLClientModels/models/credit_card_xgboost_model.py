import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
import joblib
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Setup comprehensive logging for Credit Card XGBoost model training"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create unique timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main logger
    logger = logging.getLogger('Credit_Card_XGBoost_Model')
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
        f'VFLClientModels/logs/credit_card_xgboost_{run_timestamp}.log',
        maxBytes=15*1024*1024,  # 15MB per file
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("Credit Card XGBoost Model Logging System Initialized")
    logger.info(f"Log file: VFLClientModels/logs/credit_card_xgboost_{run_timestamp}.log")
    logger.info(f"Run timestamp: {run_timestamp}")
    logger.info(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    return logger

# Initialize logging
logger = setup_logging()

class IndependentXGBoostModel:
    """
    Independent XGBoost model for VFL framework
    Works with natural feature dimensions - no forced dimensionality reduction
    Central VFL model will handle dimension alignment
    """
    
    def __init__(self, random_state=RANDOM_SEED):
        self.random_state = random_state
        
        # Main XGBoost classifier - optimized for credit card classification
        self.classifier = xgb.XGBClassifier(
            random_state=random_state,
            n_estimators=150,  # More trees for better performance
            max_depth=8,       # Deeper trees for complex patterns
            learning_rate=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,     # L1 regularization
            reg_lambda=0.1,    # L2 regularization
            objective='multi:softprob',
            eval_metric='mlogloss',
            verbosity=0
        )
        
        # Feature scaler for preprocessing
        self.scaler = StandardScaler()
        
        # Pipeline for easy prediction
        self.pipeline = None
        self.is_fitted = False
        self.feature_dim = None  # Will be set during training
        
        logger.info(f"🏗️ Independent XGBoost Model initialized:")
        logger.info(f"   - Classifier: XGBoost (multi:softprob)")
        logger.info(f"   - No forced dimensionality reduction")
        logger.info(f"   - Works with natural feature space")
        logger.info(f"   - Central VFL model handles dimension alignment")
        logger.info(f"   - Random state: {random_state}")
        logger.info(f"   - XGBoost parameters: n_estimators=150, max_depth=8")
    
    def fit(self, X, y):
        """Fit the model with natural feature dimensions"""
        logger.info("🚀 Training Independent XGBoost Model...")
        
        # Store feature dimensions
        self.feature_dim = X.shape[1]
        logger.info(f"📊 Working with {self.feature_dim} natural features")
        
        # Scale features
        logger.debug("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train XGBoost classifier directly on scaled features
        logger.debug("Training XGBoost classifier on natural feature space...")
        self.classifier.fit(X_scaled, y)
        
        # Create complete pipeline
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('classifier', self.classifier)
        ])
        
        self.is_fitted = True
        
        # Log model characteristics
        n_classes = len(np.unique(y))
        logger.info(f"✅ Model training completed:")
        logger.info(f"   - Input features: {self.feature_dim}")
        logger.info(f"   - Output classes: {n_classes}")
        logger.info(f"   - Samples: {len(X):,}")
        logger.info(f"   - XGBoost trees: {self.classifier.n_estimators}")
        logger.info(f"   - Natural feature space preserved")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.pipeline.predict_proba(X)
    
    def get_feature_representation(self, X):
        """Get scaled feature representation for VFL central model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before feature extraction")
        
        # Return scaled features in natural dimensions
        # Central VFL model will handle dimension alignment
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def get_local_prediction(self, X):
        """Get local prediction and confidence for VFL aggregation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = self.predict_proba(X)
        predictions = self.predict(X)
        confidence_scores = np.max(probabilities, axis=1)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence_scores,
            'features': self.get_feature_representation(X),
            'feature_dim': self.feature_dim
        }
    
    def get_feature_importance(self):
        """Get XGBoost feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.classifier.feature_importances_
    
    def get_model_info(self):
        """Get model information for VFL central coordinator"""
        return {
            'model_type': 'XGBoost',
            'feature_dim': self.feature_dim,
            'n_classes': self.classifier.n_classes_,
            'n_estimators': self.classifier.n_estimators,
            'max_depth': self.classifier.max_depth,
            'is_fitted': self.is_fitted
        }
    
    def save_model(self, filepath):
        """Save the complete model"""
        logger.info(f"💾 Saving Independent XGBoost Model to {filepath}")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'pipeline': self.pipeline,
            'feature_dim': self.feature_dim,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved successfully")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model"""
        logger.info(f"📂 Loading Independent XGBoost Model from {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create new instance
        model = cls(random_state=model_data['random_state'])
        
        # Restore components
        model.classifier = model_data['classifier']
        model.scaler = model_data['scaler']
        model.pipeline = model_data['pipeline']
        model.feature_dim = model_data['feature_dim']
        model.is_fitted = model_data['is_fitted']
        
        logger.info(f"✅ Model loaded successfully")
        return model

def load_and_preprocess_data():
    """Load and preprocess the credit card data - identical to neural network version"""
    logger.info("🔄 Loading and preprocessing credit card data...")
    
    # Load the data
    logger.debug("Loading credit card dataset from CSV...")
    # Try multiple possible paths for the dataset
    possible_paths = [
        'VFLClientModels/dataset/data/banks/credit_card_bank.csv'
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
        raise FileNotFoundError(f"Could not find credit_card_bank.csv in any of the expected locations: {possible_paths}")
    
    logger.info(f"📊 Dataset loaded: {len(df):,} total records")
    
    # Enhanced feature engineering for credit card classification (same as neural network)
    logger.debug("Creating derived features for credit card prediction...")
    
    # Credit capacity utilization (using total_credit_limit instead of credit_card_limit)
    df['credit_capacity_ratio'] = df['total_credit_limit'] / df['total_credit_limit'].replace(0, 1)
    
    # Income to limit ratio (using total_credit_limit instead of credit_card_limit)
    df['income_to_limit_ratio'] = df['annual_income'] / df['total_credit_limit'].replace(0, 1)
    
    # Debt service ratio
    df['debt_service_ratio'] = (df['current_debt'] * 0.03) / (df['annual_income'] / 12)  # Assuming 3% monthly payment
    
    # Risk-adjusted income (using a simplified risk calculation instead of risk_score)
    df['risk_adjusted_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    
    logger.debug("✅ Derived features created")
    
    # Select comprehensive features for credit card tier prediction (same as neural network)
    feature_columns = [
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
        # Derived features (removed correlated features)
        'credit_capacity_ratio', 'income_to_limit_ratio', 'debt_service_ratio', 'risk_adjusted_income'
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
        X, y_encoded, customer_ids,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    logger.info(f"📊 Data split results:")
    logger.info(f"   - Training samples: {len(X_train):,}")
    logger.info(f"   - Test samples: {len(X_test):,}")
    
    # Apply SMOTE for class balancing
    logger.info("🔄 Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info("📊 SMOTE balancing results:")
    logger.info(f"   - Original training samples: {len(X_train):,}")
    logger.info(f"   - Balanced training samples: {len(X_train_balanced):,}")
    balanced_class_counts = np.bincount(y_train_balanced)
    for i, count in enumerate(balanced_class_counts):
        tier_name = label_encoder.classes_[i]
        logger.info(f"   - {tier_name}: {count:,} samples")
    
    logger.info("✅ Data preprocessing completed successfully")
    
    return (X_train_balanced, X_test, y_train_balanced, y_test,
            label_encoder.classes_, feature_columns, ids_test, label_encoder)

def plot_training_analysis(y_test, y_pred, y_pred_proba, classes, model):
    """Plot comprehensive analysis for XGBoost"""
    logger.debug("Creating XGBoost analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Credit Card XGBoost Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=classes, yticklabels=classes)
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Card Tier')
    axes[0, 0].set_xlabel('Predicted Card Tier')
    
    # 2. Class Prediction Confidence
    avg_confidence = []
    for i, class_name in enumerate(classes):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_confidence = np.mean(np.max(y_pred_proba[class_mask], axis=1))
            avg_confidence.append(class_confidence)
        else:
            avg_confidence.append(0)
    
    axes[0, 1].bar(classes, avg_confidence, color='skyblue', alpha=0.7)
    axes[0, 1].set_title('Average Prediction Confidence by Class')
    axes[0, 1].set_ylabel('Average Confidence')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Prediction Confidence Distribution
    max_probabilities = np.max(y_pred_proba, axis=1)
    axes[0, 2].hist(max_probabilities, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_title('Prediction Confidence Distribution')
    axes[0, 2].set_xlabel('Max Probability')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(max_probabilities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(max_probabilities):.3f}')
    axes[0, 2].legend()
    
    # 4. Per-Class Accuracy
    class_accuracies = []
    for i, class_name in enumerate(classes):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_test[class_mask], y_pred[class_mask])
            class_accuracies.append(class_accuracy)
        else:
            class_accuracies.append(0)
    
    axes[1, 0].bar(classes, class_accuracies, color='orange', alpha=0.7)
    axes[1, 0].set_title('Per-Class Accuracy')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. XGBoost Feature Importance (from PCA components)
    if model.is_fitted:
        feature_importance = model.get_feature_importance()
        feature_names = [f'PC{i+1}' for i in range(len(feature_importance))]
        
        axes[1, 1].barh(feature_names, feature_importance, color='purple', alpha=0.7)
        axes[1, 1].set_title('XGBoost Feature Importance (PCA Components)')
        axes[1, 1].set_xlabel('Importance Score')
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # 6. Classification Report Heatmap
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    metrics_df = pd.DataFrame(report).iloc[:-1, :-3].T  # Remove support and averages
    sns.heatmap(metrics_df.astype(float), annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[1, 2])
    axes[1, 2].set_title('Classification Metrics Heatmap')
    
    plt.tight_layout()
    plt.savefig('VFLClientModels/plots/credit_card_xgboost_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def detailed_prediction_analysis(X_test, y_test, customer_ids, classes, model, feature_names, original_df, n_samples=15):
    """Detailed prediction analysis for independent XGBoost model"""
    logger.info("🔍 Performing detailed XGBoost prediction analysis...")
    
    # Get random indices for sampling
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions and probabilities
    y_pred_prob = model.predict_proba(X_test.iloc[indices])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test[indices]
    
    # Get feature representations for VFL (natural dimensions)
    feature_representations = model.get_feature_representation(X_test.iloc[indices])
    
    logger.info("\n" + "="*140)
    logger.info("DETAILED CREDIT CARD XGBOOST PREDICTION ANALYSIS (INDEPENDENT CLIENT)")
    logger.info("="*140)
    logger.info(f"{'Tax ID':<12} {'Actual':<9} {'Predicted':<9} {'Confidence':<11} {'Status':<8} {'Credit Score':<12} {'Income':<12} {'Features (sample)':<30} {'Key Profile'}")
    logger.info("-"*140)
    
    # Track statistics
    correct_predictions = 0
    high_confidence_correct = 0
    low_confidence_incorrect = 0
    
    # Analyze each sample
    for idx, (true_idx, pred_idx, prob_dist, features_natural) in enumerate(zip(y_true, y_pred, y_pred_prob, feature_representations)):
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
        
        # Show first 3 components of natural feature representation
        features_sample = f"[{features_natural[0]:.2f}, {features_natural[1]:.2f}, {features_natural[2]:.2f}, ...]"
        
        # Format output
        logger.info(f"{customer_id:<12} {true_tier:<9} {pred_tier:<9} {confidence:>8.1f}% {status:^8} "
                   f"{credit_score:<12} ${annual_income/1000:>8.0f}K {features_sample:<30} {profile}")
    
    logger.info("-"*140)
    
    # Summary statistics
    total_samples = len(indices)
    accuracy = (correct_predictions / total_samples) * 100
    
    logger.info(f"\nINDEPENDENT XGBOOST PREDICTION SUMMARY:")
    logger.info(f"Total Samples Analyzed: {total_samples}")
    logger.info(f"Correct Predictions: {correct_predictions} ({accuracy:.1f}%)")
    logger.info(f"High Confidence Correct (≥80%): {high_confidence_correct}")
    logger.info(f"Low Confidence Incorrect (<60%): {low_confidence_incorrect}")
    
    # Natural Feature Analysis
    logger.info(f"\nNATURAL FEATURE REPRESENTATION ANALYSIS:")
    logger.info(f"Feature representation shape: {feature_representations.shape}")
    logger.info(f"Natural dimensions: {model.feature_dim} (no forced reduction)")
    logger.info(f"Feature range: [{feature_representations.min():.3f}, {feature_representations.max():.3f}]")
    logger.info(f"Feature means: {np.mean(feature_representations, axis=0)[:5]}... (first 5)")
    logger.info(f"Feature stds: {np.std(feature_representations, axis=0)[:5]}... (first 5)")
    
    # XGBoost specific analysis
    logger.info(f"\nINDEPENDENT XGBOOST MODEL ANALYSIS:")
    logger.info(f"Number of trees: {model.classifier.n_estimators}")
    logger.info(f"Max depth: {model.classifier.max_depth}")
    logger.info(f"Learning rate: {model.classifier.learning_rate}")
    feature_importance = model.get_feature_importance()
    top_feature_idx = np.argmax(feature_importance)
    logger.info(f"Most important feature: {feature_names[top_feature_idx]} ({feature_importance[top_feature_idx]:.3f})")
    
    # VFL Integration Readiness
    logger.info(f"\nVFL INTEGRATION READINESS:")
    logger.info(f"✅ Independent client operation")
    logger.info(f"✅ Natural feature space: {model.feature_dim}D")
    logger.info(f"✅ Local predictions with confidence")
    logger.info(f"✅ Standardized feature representations")
    logger.info(f"🔄 Central VFL model will handle dimension alignment")
    
    logger.info("="*140)
    
    return {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'high_confidence_correct': high_confidence_correct,
        'feature_representations': feature_representations,
        'natural_dimensions': model.feature_dim
    }

def main():
    training_start = datetime.now()
    logger.info("🚀 Starting Credit Card XGBoost Model Training (Independent Client)")
    logger.info("=" * 80)
    logger.info(f"Training session started: {training_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Model type: Independent XGBoost (Natural Feature Space)")
    logger.info(f"Purpose: Client-side model for heterogeneous VFL")
    logger.info(f"Central VFL model will handle dimension alignment")
    
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
         customer_ids, label_encoder) = load_and_preprocess_data()
        data_duration = datetime.now() - data_start
        logger.info(f"⏱️ Data preprocessing completed in {data_duration}")
        
        # Create and train independent XGBoost model
        logger.info("")
        logger.info("🏗️ Independent XGBoost Model Building Phase")
        logger.info("-" * 60)
        model_start = datetime.now()
        
        # Initialize the independent model
        global model
        model = IndependentXGBoostModel(random_state=RANDOM_SEED)
        
        # Train the model
        logger.info("🚀 Training XGBoost on natural feature space...")
        model.fit(X_train, y_train)
        
        model_duration = datetime.now() - model_start
        logger.info(f"⏱️ Model building and training completed in {model_duration}")
        
        # Evaluate model
        logger.info("")
        logger.info("📊 Model Evaluation Phase")
        logger.info("-" * 60)
        eval_start = datetime.now()
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        
        # Feature representations (in natural dimensions)
        test_representations = model.get_feature_representation(X_test)
        
        # Get local predictions with metadata
        local_predictions = model.get_local_prediction(X_test)
        
        eval_duration = datetime.now() - eval_start
        logger.info(f"⏱️ Model evaluation completed in {eval_duration}")
        
        # Calculate and log performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        logger.info("")
        logger.info("🎯 INDEPENDENT XGBOOST MODEL PERFORMANCE RESULTS")
        logger.info("=" * 80)
        logger.info(f"Overall Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"Weighted Precision:   {precision:.4f}")
        logger.info(f"Weighted Recall:      {recall:.4f}")
        logger.info(f"Weighted F1-Score:    {f1:.4f}")
        logger.info("=" * 80)
        
        # --- PCA for VFL: Fit and Save ---
        logger.info("")
        logger.info("🔄 Fitting PCA on XGBoost leaf indices for VFL integration...")
        # Get XGBoost leaf indices for the test set
        X_test_scaled = model.scaler.transform(X_test)
        leaf_indices = model.classifier.apply(X_test_scaled)
        pca = PCA(n_components=12, random_state=RANDOM_SEED)
        pca.fit(leaf_indices.astype(np.float32))
        pca_path = 'VFLClientModels/saved_models/credit_card_xgboost_pca.pkl'
        joblib.dump(pca, pca_path)
        logger.info(f"✅ PCA fitted and saved to {pca_path}")
        
        # Detailed classification report
        logger.info("")
        logger.info("📋 Detailed Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=classes))
        
        # Model Information for VFL Central Coordinator
        model_info = model.get_model_info()
        logger.info("")
        logger.info("🔗 VFL CLIENT MODEL INFORMATION")
        logger.info("-" * 60)
        logger.info(f"Model Type: {model_info['model_type']}")
        logger.info(f"Feature Dimensions: {model_info['feature_dim']} (natural space)")
        logger.info(f"Number of Classes: {model_info['n_classes']}")
        logger.info(f"XGBoost Trees: {model_info['n_estimators']}")
        logger.info(f"Max Depth: {model_info['max_depth']}")
        logger.info(f"Model Status: {'✅ Fitted' if model_info['is_fitted'] else '❌ Not Fitted'}")
        
        # Feature Analysis
        logger.info("")
        logger.info("📊 FEATURE REPRESENTATION ANALYSIS")
        logger.info("-" * 60)
        logger.info(f"Feature representation shape: {test_representations.shape}")
        logger.info(f"Natural dimensions: {model.feature_dim}")
        logger.info(f"Feature range: [{test_representations.min():.3f}, {test_representations.max():.3f}]")
        logger.info(f"Feature mean: {np.mean(test_representations):.3f}")
        logger.info(f"Feature std: {np.std(test_representations):.3f}")
        
        # Client Model Capabilities
        logger.info("")
        logger.info("🎯 CLIENT MODEL CAPABILITIES FOR VFL")
        logger.info("-" * 60)  
        logger.info("✅ Independent operation - no dimensional constraints")
        logger.info("✅ Natural feature space optimization")
        logger.info("✅ Local predictions with confidence scores")
        logger.info("✅ Scaled feature representations for central model")
        logger.info("✅ Model metadata for VFL coordination")
        logger.info("🔄 Central VFL model handles dimension alignment")
        logger.info("🔄 Heterogeneous federated learning ready")
        
        # Create analysis plots
        logger.info("")
        logger.info("📈 Generating Analysis Visualizations")
        logger.info("-" * 60)
        plot_start = datetime.now()
        plot_training_analysis(y_test, y_pred, y_pred_prob, classes, model)
        logger.debug("✅ XGBoost analysis plots saved")
        
        # Load original data for detailed analysis
        # Try multiple possible paths for the dataset
        possible_paths = [
            'VFLClientModels/dataset/data/banks/credit_card_bank.csv'
        ]
        
        original_df = None
        for path in possible_paths:
            try:
                original_df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if original_df is None:
            raise FileNotFoundError(f"Could not find credit_card_bank.csv in any of the expected locations: {possible_paths}")
        
        # Detailed prediction analysis
        prediction_results = detailed_prediction_analysis(
            X_test, y_test, customer_ids, classes, model, 
            feature_names, original_df, n_samples=15
        )
        
        plot_duration = datetime.now() - plot_start
        logger.info(f"⏱️ Visualization generation completed in {plot_duration}")
        
        # Save model and artifacts
        logger.info("")
        logger.info("💾 Saving Independent Client Model and Artifacts")
        logger.info("-" * 60)
        save_start = datetime.now()
        
        try:
            # Save the independent model
            model.save_model('VFLClientModels/saved_models/credit_card_xgboost_independent.pkl')
            
            # Save additional artifacts
            logger.debug("Saving feature names...")
            np.save('VFLClientModels/saved_models/credit_card_xgboost_feature_names.npy', feature_names)
            logger.info("✅ Feature names saved as: saved_models/credit_card_xgboost_feature_names.npy")
            
            logger.debug("Saving label encoder...")
            joblib.dump(label_encoder, 'VFLClientModels/saved_models/credit_card_xgboost_label_encoder.pkl')
            logger.info("✅ Label encoder saved as: VFLClientModels/saved_models/credit_card_xgboost_label_encoder.pkl")
            
            # Save sample representations for VFL testing (natural dimensions)
            np.save('VFLClientModels/saved_models/credit_card_xgboost_sample_representations.npy', test_representations[:100])
            logger.info("✅ Sample feature representations saved for VFL testing")
            
            # Save model information for VFL central coordinator
            joblib.dump(model_info, 'VFLClientModels/saved_models/credit_card_xgboost_model_info.pkl')
            logger.info("✅ Model info saved for VFL central coordinator")
            
            logger.info("✅ Independent XGBoost Model saved as: VFLClientModels/saved_models/credit_card_xgboost_independent.pkl")
            logger.info("✅ Plots saved: VFLClientModels/plots/credit_card_xgboost_analysis.png")
            
            save_duration = datetime.now() - save_start
            logger.info(f"⏱️ Model saving completed in {save_duration}")
            
        except Exception as e:
            logger.error(f"❌ Error saving artifacts: {str(e)}")
            raise
        
        # Final summary
        total_duration = datetime.now() - training_start
        logger.info("")
        logger.info("🎉 CREDIT CARD XGBOOST CLIENT MODEL TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"⏱️ Total training time: {total_duration}")
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"   - Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"   - Weighted F1-Score: {f1:.4f}")
        logger.info(f"   - Training Samples: {len(X_train):,} (after SMOTE)")
        logger.info(f"   - Test Samples: {len(X_test):,}")
        logger.info(f"   - Natural Features: {len(feature_names)}")
        logger.info(f"   - Card Tiers: {len(classes)} ({', '.join(classes)})")
        logger.info(f"🔗 VFL Integration Ready:")
        logger.info(f"   - Independent Client Model: ✅")
        logger.info(f"   - Natural Feature Space: ✅ ({model.feature_dim}D)")
        logger.info(f"   - Gradient Boosting Model: ✅ (XGBoost)")
        logger.info(f"   - Heterogeneous VFL Compatible: ✅")
        logger.info(f"   - Central Model Handles Dimensions: ✅")
        logger.info(f"📁 Artifacts saved:")
        logger.info(f"   - Model: VFLClientModels/saved_models/credit_card_xgboost_independent.pkl")
        logger.info(f"   - Model Info: VFLClientModels/saved_models/credit_card_xgboost_model_info.pkl")
        logger.info(f"   - Features: VFLClientModels/saved_models/credit_card_xgboost_feature_names.npy")
        logger.info(f"   - Label Encoder: VFLClientModels/saved_models/credit_card_xgboost_label_encoder.pkl")
        logger.info(f"   - Plots: VFLClientModels/plots/credit_card_xgboost_analysis.png")
        logger.info("=" * 80)
        
        return model, classes, feature_names
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("💳 Starting Credit Card XGBoost Model Training (Independent Client)...")
    logger.info("This creates an independent client model for heterogeneous VFL...")
    logger.info("Central VFL model will handle dimension alignment across all clients...")
    
    # Train the model
    model, classes, feature_names = main()
    
    logger.info("\n🎉 Credit Card XGBoost Client Model Training Completed Successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated analysis plots")
    logger.info("2. Each client model works independently with optimal architecture")
    logger.info("3. Create central VFL model to handle heterogeneous dimensions")
    logger.info("4. Central model aligns: Auto(NN), Digital(NN), Home(NN), Credit(XGB)")

def load_credit_card_xgboost_pipeline(dataframe, saved_models_dir="VFLClientModels/models/saved_models"):
    # Load feature names
    feature_names_path = os.path.join(saved_models_dir, "credit_card_xgboost_feature_names.npy")
    feature_names = list(np.load(feature_names_path, allow_pickle=True))
    
    # Check DataFrame columns
    missing = [f for f in feature_names if f not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required features in DataFrame: {missing}")
    extra = [f for f in dataframe.columns if f not in feature_names]
    if extra:
        print(f"Warning: Extra features in DataFrame will be ignored: {extra}")
    
    # Select only required features, in correct order
    X = dataframe.loc[:, feature_names].copy()
    
    # Load scaler
    scaler_path = os.path.join(saved_models_dir, "credit_card_scaler.pkl")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    
    # Load XGBoost model
    model_path = os.path.join(saved_models_dir, "credit_card_xgboost_independent.pkl")
    xgb_model = joblib.load(model_path)
    
    # Predict (or get intermediate representation)
    preds = xgb_model.predict(X_scaled)
    # If you want leaf indices or SHAP values, use:
    # leaf_indices = xgb_model.apply(X_scaled)
    # shap_values = xgb_model.predict(X_scaled, pred_contribs=True)
    
    return preds  # or return leaf_indices, shap_values, etc.

# Example usage:
# df = pd.read_csv("path/to/your/customer_data.csv")
# predictions = load_credit_card_xgboost_pipeline(df) 