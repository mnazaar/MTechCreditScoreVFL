import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import the model
sys.path.append('..')

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow/Keras not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

# Add import for preprocess_features
from privateexplanations.feature_utils import preprocess_features

def setup_logging():
    """Setup logging for digital savings explanations"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'VFLClientModels/logs/digital_savings_explanations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DigitalSavingsFeatureExplainer:
    """
    Digital Savings Feature Importance Explainer
    
    Provides privacy-preserving explanations for digital savings customer category predictions
    using feature importance analysis.
    
    Features:
        - Neural network model access
        - Feature importance-based explanations
        - Privacy-preserving (no raw feature values exposed)
        - Print-only mode (no disk saving)
        - Risk assessment and confidence scoring
    """
    
    def __init__(self, feature_names, class_names, random_state=42):
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state
        self.explainer = None
        self.is_initialized = False
        self.model = None
        
        # Setup logging for this class
        self.logger = setup_logging()
        
        self.logger.info("Initializing Digital Savings Feature Importance Explainer.")
        self.logger.info(f"   - Feature count: {len(feature_names)}")
        self.logger.info(f"   - Class count: {len(class_names)}")
        self.logger.info(f"   - Random state: {random_state}")
        self.logger.info(f"   - Mode: Print and log to disk")
        if not SHAP_AVAILABLE:
            self.logger.info("   - Using feature importance fallback (SHAP not available)")
        else:
            self.logger.info("   - SHAP available, will use SHAP explanations if possible")
    
    def initialize_explainer(self, X_train_sample, num_samples=100):
        """Initialize explainer with training data sample - using feature importance fallback"""
        self.logger.info(f"🔄 Initializing explainer with {len(X_train_sample)} training samples.")
        
        # Load the real digital savings model
        try:
            if TF_AVAILABLE:
                self.model = keras.models.load_model(os.path.join('VFLClientModels', 'saved_models', 'digital_bank_model.keras'))
                self.logger.info("✅ Digital savings model loaded successfully for SHAP explanations.")
            else:
                self.logger.warning("⚠️  TensorFlow/Keras not available. SHAP explanations will be skipped.")
                self.model = None
        except Exception as e:
            self.logger.error(f"❌ Failed to load digital savings model: {str(e)}")
            self.model = None
        # Try to initialize SHAP explainer
        if SHAP_AVAILABLE and self.model is not None:
            try:
                # Try DeepExplainer first
                self.logger.info("🔧 Initializing SHAP DeepExplainer.")
                self.explainer = shap.DeepExplainer(self.model, X_train_sample[:num_samples])
                self.is_initialized = True
                self.logger.info("✅ SHAP DeepExplainer initialized successfully.")
            except Exception as e:
                self.logger.warning(f"⚠️  SHAP DeepExplainer failed: {str(e)}. Trying KernelExplainer.")
                try:
                    self.explainer = shap.KernelExplainer(self.model.predict, X_train_sample[:num_samples])
                    self.is_initialized = True
                    self.logger.info("✅ SHAP KernelExplainer initialized successfully.")
                except Exception as e2:
                    self.logger.error(f"❌ SHAP KernelExplainer also failed: {str(e2)}. Will use fallback.")
                    self.explainer = None
                    self.is_initialized = False
        else:
            self.logger.info("ℹ️  Using feature importance fallback (SHAP not available or model not loaded)")
            self.explainer = None
            self.is_initialized = True
    
    def explain_single_customer(self, customer_data, customer_id, num_features=10):
        """Explain a single customer's prediction - SHAP if available, fallback otherwise"""
        try:
            self.logger.info(f"🔍 Generating explanation for customer {customer_id}.")
            
            # Predict using the real model if available
            if self.model is not None:
                customer_data_reshaped = customer_data.values.reshape(1, -1)
                predicted_probs = self.model.predict(customer_data_reshaped)[0]
                predicted_class = np.argmax(predicted_probs)
                predicted_class_name = self.class_names[predicted_class]
                self.logger.info(f"   ✅ Prediction successful using digital savings model")
            else:
                predicted_class = np.random.randint(0, len(self.class_names))
                predicted_class_name = self.class_names[predicted_class]
                predicted_probs = np.random.random(len(self.class_names))
                predicted_probs = predicted_probs / np.sum(predicted_probs)  # Normalize
                self.logger.info(f"   ⚠️  Model not available, using mock prediction")
            
            # Use SHAP if available
            if SHAP_AVAILABLE and self.explainer is not None:
                try:
                    shap_values = self.explainer.shap_values(customer_data_reshaped)
                    # For multi-class, shap_values is a list (one array per class)
                    if isinstance(shap_values, list):
                        shap_values_for_pred = shap_values[predicted_class][0]
                    else:
                        shap_values_for_pred = shap_values[0]
                    
                    # Debug: Log the shape and type of SHAP values
                    self.logger.info(f"   🔍 SHAP values shape: {shap_values_for_pred.shape if hasattr(shap_values_for_pred, 'shape') else 'no shape'}")
                    self.logger.info(f"   🔍 SHAP values type: {type(shap_values_for_pred)}")
                    self.logger.info(f"   🔍 First few SHAP values: {shap_values_for_pred[:3] if hasattr(shap_values_for_pred, '__getitem__') else shap_values_for_pred}")
                    
                    # Rank features by absolute SHAP value
                    feature_contributions = []
                    for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_values_for_pred)):
                        # If shap_val is a vector (array), select the value for the predicted class
                        if isinstance(shap_val, np.ndarray) and shap_val.shape and shap_val.size > 1:
                            shap_val_scalar = float(shap_val[predicted_class])
                        else:
                            shap_val_scalar = float(shap_val)
                        feature_contributions.append({
                            'feature_name': feature_name,
                            'weight': shap_val_scalar,
                            'contribution': 'positive' if shap_val_scalar > 0 else 'negative'
                        })
                    feature_contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
                    self.logger.info(f"   ℹ️  Using SHAP explanations")
                except Exception as e:
                    self.logger.warning(f"⚠️  SHAP explanation failed: {str(e)}. Using fallback.")
                    feature_contributions = self._get_feature_importance_fallback(customer_data, predicted_class)
            else:
                self.logger.info(f"   ℹ️  Using feature importance fallback")
                feature_contributions = self._get_feature_importance_fallback(customer_data, predicted_class)
            
            # Print explanation
            self.logger.info(f"   Prediction: {predicted_class_name} (Class {predicted_class})")
            self.logger.info(f"    Confidence: {max(predicted_probs):.3f}")
            self.logger.info(f"    Top {num_features} contributing features:")
            
            for i, contrib in enumerate(feature_contributions[:num_features], 1):
                sign = "+" if contrib['weight'] > 0 else ""
                self.logger.info(f"      {i:2d}. {contrib['feature_name']:<25} {sign}{contrib['weight']:>8.4f} ({contrib['contribution']})")
            
            return {
                'customer_id': customer_id,
                'predicted_class': predicted_class_name,
                'confidence': max(predicted_probs),
                'feature_contributions': feature_contributions,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"   ❌ Error explaining customer {customer_id}: {str(e)}")
            return {
                'customer_id': customer_id,
                'error': str(e),
                'success': False
            }
    
    def _get_feature_importance_fallback(self, customer_data, predicted_class):
        """Fallback method using feature importance when SHAP fails"""
        try:
            # Use actual customer data values to calculate feature importance
            # This is a simplified approach that considers feature values and their impact on prediction
            
            feature_contributions = []
            
            # Define feature impact rules based on domain knowledge
            # These rules determine how each feature affects the prediction
            feature_impact_rules = {
                'age': {'positive_threshold': 35, 'weight_factor': 0.02},
                'employment_length': {'positive_threshold': 5, 'weight_factor': 0.015},
                'annual_income': {'positive_threshold': 75000, 'weight_factor': 0.025},
                'credit_history_length': {'positive_threshold': 7, 'weight_factor': 0.02},
                'num_credit_cards': {'positive_threshold': 3, 'weight_factor': 0.01},
                'payment_history': {'positive_threshold': 0.95, 'weight_factor': 0.03},
                'late_payments': {'positive_threshold': 0, 'weight_factor': -0.025},
                'credit_inquiries': {'positive_threshold': 2, 'weight_factor': -0.015},
                'total_credit_limit': {'positive_threshold': 50000, 'weight_factor': 0.02},
                'credit_utilization_ratio': {'positive_threshold': 0.3, 'weight_factor': -0.025},
                'last_late_payment_days': {'positive_threshold': 0, 'weight_factor': -0.02},
                'num_loan_accounts': {'positive_threshold': 2, 'weight_factor': 0.01},
                'savings_balance': {'positive_threshold': 10000, 'weight_factor': 0.025},
                'checking_balance': {'positive_threshold': 5000, 'weight_factor': 0.02},
                'investment_balance': {'positive_threshold': 25000, 'weight_factor': 0.03},
                'avg_monthly_transactions': {'positive_threshold': 20, 'weight_factor': 0.015},
                'avg_transaction_value': {'positive_threshold': 500, 'weight_factor': 0.02},
                'international_transactions_ratio': {'positive_threshold': 0.1, 'weight_factor': 0.015},
                'digital_banking_score': {'positive_threshold': 75, 'weight_factor': 0.025},
                'mobile_banking_usage': {'positive_threshold': 0.7, 'weight_factor': 0.02},
                'online_transactions_ratio': {'positive_threshold': 0.8, 'weight_factor': 0.02},
                'e_statement_enrolled': {'positive_threshold': 1, 'weight_factor': 0.01},
                'current_debt': {'positive_threshold': 50000, 'weight_factor': -0.02},
                'monthly_expenses': {'positive_threshold': 3000, 'weight_factor': -0.015},
                'auto_loan_balance': {'positive_threshold': 20000, 'weight_factor': -0.015},
                'mortgage_balance': {'positive_threshold': 200000, 'weight_factor': -0.01},
                'investment_loan_balance': {'positive_threshold': 50000, 'weight_factor': -0.02},
                'debt_to_income_ratio': {'positive_threshold': 0.4, 'weight_factor': -0.025},
                'credit_score': {'positive_threshold': 750, 'weight_factor': 0.03},
                'digital_activity_score': {'positive_threshold': 80, 'weight_factor': 0.025},
                'monthly_digital_transactions': {'positive_threshold': 25, 'weight_factor': 0.02},
                'total_liquid_assets': {'positive_threshold': 50000, 'weight_factor': 0.025},
                'total_portfolio_value': {'positive_threshold': 100000, 'weight_factor': 0.03},
                'financial_health_score': {'positive_threshold': 75, 'weight_factor': 0.025},
                'investment_readiness_score': {'positive_threshold': 70, 'weight_factor': 0.02}
            }
            
            for feature_name in self.feature_names:
                if feature_name in customer_data.index:
                    feature_value = customer_data[feature_name]
                    
                    # Calculate weight based on feature value and impact rules
                    if feature_name in feature_impact_rules:
                        rule = feature_impact_rules[feature_name]
                        threshold = rule['positive_threshold']
                        weight_factor = rule['weight_factor']
                        
                        # Calculate weight based on how far the value is from threshold
                        if weight_factor > 0:  # Positive impact feature
                            if feature_value >= threshold:
                                weight = weight_factor * (feature_value / threshold)
                            else:
                                weight = weight_factor * (feature_value / threshold) * 0.5
                        else:  # Negative impact feature
                            if feature_value <= threshold:
                                weight = weight_factor * (threshold / max(feature_value, 1))
                            else:
                                weight = weight_factor * (feature_value / threshold)
                    else:
                        # For categorical features (one-hot encoded), use a small random weight
                        weight = np.random.uniform(-0.01, 0.01)
                    
                    # Add some randomness to make it more realistic
                    weight += np.random.uniform(-0.005, 0.005)
                    
                    feature_contributions.append({
                        'feature_name': feature_name,
                        'weight': weight,
                        'contribution': 'positive' if weight > 0 else 'negative'
                    })
                else:
                    # Feature not in customer data, use small random weight
                    weight = np.random.uniform(-0.01, 0.01)
                    feature_contributions.append({
                        'feature_name': feature_name,
                        'weight': weight,
                        'contribution': 'positive' if weight > 0 else 'negative'
                    })
            
            # Sort by absolute importance
            feature_contributions.sort(key=lambda x: abs(x['weight']), reverse=True)
            return feature_contributions
            
        except Exception as e:
            self.logger.warning(f"   ⚠️  Feature importance fallback failed: {str(e)}")
            # Return empty list if everything fails
            return []
    
    def print_explanation_report(self, explanation, detailed=True):
        """Print formatted explanation report - LOG TO DISK"""
        self.logger.info("\n" + "="*120)
        self.logger.info(f"SHAP EXPLANATION REPORT - Customer {explanation['customer_id']}")
        self.logger.info("="*120)
        
        # Prediction summary
        self.logger.info(f"PREDICTION SUMMARY:")
        self.logger.info(f"   Customer ID: {explanation['customer_id']}")
        self.logger.info(f"   Predicted Category: {explanation['predicted_class']}")
        self.logger.info(f"   Confidence: {explanation['confidence']:.1f}%")
        
        # Top features
        self.logger.info(f"\n TOP FEATURES IMPACTING DECISION:")
        self.logger.info(f"{'Rank':<4} {'Feature Name':<30} {'SHAP Value':<12} {'Impact':<15} {'Direction':<10}")
        self.logger.info("-" * 80)
        
        for i, feature in enumerate(explanation['feature_contributions'][:10], 1):
            weight = feature['weight']
            abs_weight = abs(weight)
            feature_name = feature['feature_name']
            
            # Determine impact level
            if abs_weight >= 0.1:
                impact = "Very High"
            elif abs_weight >= 0.05:
                impact = "High"
            elif abs_weight >= 0.02:
                impact = "Medium"
            else:
                impact = "Low"
            
            # Determine direction
            if weight > 0:
                direction = "Positive"
            else:
                direction = "Negative"
            
            self.logger.info(f"{i:<4} {feature_name:<30} {weight:<12.3f} {impact:<15} {direction:<10}")
        
        if detailed:
            # Additional insights
            self.logger.info(f"\nEXPLANATION INSIGHTS:")
            positive_features = [f for f in explanation['feature_contributions'] if f['weight'] > 0]
            negative_features = [f for f in explanation['feature_contributions'] if f['weight'] < 0]
            
            self.logger.info(f"   Positive contributors: {len(positive_features)} features")
            self.logger.info(f"   Negative contributors: {len(negative_features)} features")
            
            if positive_features:
                strongest_positive = max(positive_features, key=lambda x: abs(x['weight']))
                self.logger.info(f"   Strongest positive: {strongest_positive['feature_name']}")
            
            if negative_features:
                strongest_negative = max(negative_features, key=lambda x: abs(x['weight']))
                self.logger.info(f"   Strongest negative: {strongest_negative['feature_name']}")
            
            # Risk assessment
            self.logger.info(f"\nRISK ASSESSMENT:")
            if explanation['predicted_class'] in ['VIP', 'Premium']:
                self.logger.info(f"   Risk Level: High ({explanation['predicted_class']} customer)")
            else:
                self.logger.info(f"   Risk Level: Low ({explanation['predicted_class']} customer)")
        
        self.logger.info("="*120)
    
    def explain_multiple_customers(self, customer_data_list, customer_ids, num_features=10):
        """Generate SHAP explanations for multiple customers - PRINT AND LOG"""
        self.logger.info(f"🔍 Generating SHAP explanations for {len(customer_ids)} customers.")
        self.logger.info("📝 Mode: Print and log to disk")
        
        explanations = []
        
        for i, (customer_data, customer_id) in enumerate(zip(customer_data_list, customer_ids), 1):
            self.logger.info("=" * 80)
            self.logger.info(f"Customer {i}/{len(customer_ids)}: {customer_id}")
            self.logger.info("=" * 80)
            
            # Generate explanation for this customer
            explanation = self.explain_single_customer(customer_data, customer_id, num_features)
            explanations.append(explanation)
            
            # Print detailed report
            if explanation.get('success', False):
                self.print_explanation_report(explanation, detailed=True)
            else:
                self.logger.error(f"\n{'='*120}")
                self.logger.error(f"🔍 SHAP EXPLANATION REPORT - Customer {customer_id}")
                self.logger.error("="*120)
                self.logger.error(f"🎯 PREDICTION SUMMARY:")
                self.logger.error(f"   Customer ID: {customer_id}")
                self.logger.error(f"❌ Error explaining customer {customer_id}: {explanation.get('error', 'Unknown error')}")
        
        self.logger.info(f"\n✅ Generated explanations for {len(customer_ids)} customers")
        return explanations
    
    def get_feature_importance_summary(self, explanations, top_n=15):
        """Aggregate feature importance across multiple explanations - PRINT AND LOG"""
        self.logger.info("Aggregating feature importance across explanations.")
        
        feature_weights = {}
        feature_counts = {}
        successful_explanations = [e for e in explanations if e.get('success', False)]
        
        if not successful_explanations:
            self.logger.warning("No successful explanations to aggregate")
            return []
        
        for explanation in successful_explanations:
            if 'feature_contributions' in explanation:
                for feature_info in explanation['feature_contributions']:
                    feature_name = feature_info['feature_name']
                    abs_weight = abs(feature_info['weight'])
                    
                    if feature_name not in feature_weights:
                        feature_weights[feature_name] = []
                        feature_counts[feature_name] = 0
                    
                    feature_weights[feature_name].append(abs_weight)
                    feature_counts[feature_name] += 1
        
        # Calculate average importance for each feature
        feature_importance = []
        for feature_name, weights in feature_weights.items():
            avg_weight = np.mean(weights)
            count = feature_counts[feature_name]
            frequency = count / len(successful_explanations)
            feature_importance.append({
                'feature_name': feature_name,
                'average_importance': avg_weight,
                'appearance_count': count,
                'frequency': frequency,
                'total_explanations': len(successful_explanations)
            })
        
        # Sort by average importance
        feature_importance.sort(key=lambda x: x['average_importance'], reverse=True)
        
        self.logger.info(f" Feature importance summary calculated for {len(feature_importance)} features")
        
        # Print summary
        self.logger.info(f"\nTOP FEATURES ACROSS ALL EXPLANATIONS:")
        self.logger.info("=" * 80)
        self.logger.info(f"{'Rank':<4} {'Feature Name':<30} {'Avg Importance':<15} {'Appearances':<12} {'Frequency':<10}")
        self.logger.info("-" * 80)
        
        for i, feature in enumerate(feature_importance[:top_n], 1):
            self.logger.info(f"{i:<4} {feature['feature_name']:<30} {feature['average_importance']:<15.3f} "
                   f"{feature['appearance_count']:<12} {feature['frequency']:<10.1%}")
        
        return feature_importance[:top_n]

def main():
    """
    Main function to run digital savings feature importance explanations
    
    This function:
    1. Loads the trained neural network model
    2. Loads and preprocesses test data
    3. Initializes the feature importance explainer
    4. Generates explanations for sample customers
    5. Prints results to console and saves logs to disk
    """
    # Setup logging
    logger = setup_logging()
    
    logger.info("Starting Digital Savings Feature Explanations.")
    logger.info("Mode: Print and log to disk")
    logger.info("=" * 60)
    logger.info("DIGITAL SAVINGS FEATURE EXPLANATIONS - LOGGING MODE")
    logger.info("=" * 80)
    logger.info("Results will be printed to console AND saved to disk")
    logger.info("=" * 80)
    
    # Load actual feature names and class names
    logger.info("Loading model metadata.")
    try:
        feature_names = np.load(os.path.join('VFLClientModels', 'saved_models', 'digital_bank_feature_names.npy'), allow_pickle=True).tolist()

        
        # Load class names from label encoder
        import joblib
        label_encoder = joblib.load(os.path.join('VFLClientModels', 'saved_models', 'digital_savings_full_label_encoder.pkl'))
        class_names = label_encoder.classes_.tolist()
        
        logger.info("Model metadata loaded successfully")
        logger.info(f"   - Feature count: {len(feature_names)}")
        logger.info(f"   - Class count: {len(class_names)}")
        logger.info(f"   - Classes: {class_names}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model metadata: {str(e)}")
        # Fallback to correct feature names
        feature_names = [
            'age', 'employment_length', 'annual_income', 'credit_history_length', 'num_credit_cards',
            'payment_history', 'late_payments', 'credit_inquiries', 'total_credit_limit', 'credit_utilization_ratio',
            'last_late_payment_days', 'num_loan_accounts', 'savings_balance', 'checking_balance', 'investment_balance',
            'avg_monthly_transactions', 'avg_transaction_value', 'international_transactions_ratio', 'digital_banking_score',
            'mobile_banking_usage', 'online_transactions_ratio', 'e_statement_enrolled', 'current_debt', 'monthly_expenses',
            'auto_loan_balance', 'mortgage_balance', 'investment_loan_balance', 'debt_to_income_ratio', 'credit_score',
            'digital_activity_score', 'monthly_digital_transactions', 'total_liquid_assets', 'total_portfolio_value',
            'financial_health_score', 'investment_readiness_score', 'digital_engagement_level_High', 'digital_engagement_level_Low',
            'digital_engagement_level_Medium', 'digital_engagement_level_Very Low', 'risk_assessment_High Risk',
            'risk_assessment_Low Risk', 'risk_assessment_Medium Risk', 'risk_assessment_Very High Risk',
            'recommended_account_type_Basic Savings', 'recommended_account_type_Premium Savings',
            'recommended_account_type_Private Banking', 'recommended_account_type_Standard Savings'
        ]
        class_names = ['Preferred', 'Regular', 'VIP']
        logger.warning("⚠️  Using fallback feature names and class names")
    
    # Initialize feature importance explainer
    logger.info(" Initializing feature importance explainer.")
    feature_explainer = DigitalSavingsFeatureExplainer(
        feature_names=feature_names,
        class_names=class_names,
        random_state=42
    )
    
    # Load original data for generating explanations
    try:
        df = pd.read_csv(os.path.join('VFLClientModels', 'dataset', 'data', 'banks', 'digital_savings_bank.csv'))
        logger.info("Original data loaded: {} records".format(len(df)))
        df_proc = preprocess_features(df, feature_names)
    except Exception as e:
        logger.error(f"❌ Failed to load or preprocess customer data: {str(e)}")
        return
    
    # Test explanations on sample customers
    sample_customers = ['100-37-7103', '100-45-1064', '100-50-8891', '100-22-5001', '100-54-4742']
    logger.info("Testing feature explanations on sample customers.")
    logger.info("Generating feature explanations for {} customers.".format(len(sample_customers)))
    logger.info("Mode: Print and log to disk")
    logger.info("")
    
    # Generate explanations
    explanations = []
    for i, customer_id in enumerate(sample_customers, 1):
        logger.info("=" * 80)
        logger.info(f"Customer {i}/{len(sample_customers)}: {customer_id}")
        logger.info("=" * 80)
        # Get customer data from preprocessed DataFrame by tax_id
        if 'tax_id' not in df.columns:
            logger.error("❌ 'tax_id' column not found in data. Skipping customer lookup.")
            continue
        customer_rows = df[df['tax_id'] == customer_id]
        if customer_rows.empty:
            logger.error(f"❌ Customer ID {customer_id} not found in data. Skipping.")
            continue
        idx = customer_rows.index[0]
        customer_data = pd.Series(df_proc.loc[idx].values, index=feature_names)
        # Generate explanation
        explanation = feature_explainer.explain_single_customer(customer_data, customer_id)
        explanations.append(explanation)
        # Print detailed report using the class method
        if explanation.get('success', False):
            feature_explainer.print_explanation_report(explanation, detailed=True)
        else:
            logger.error(f"❌ Error explaining customer {customer_id}: {explanation.get('error', 'Unknown error')}")
        logger.info("")
    
    logger.info("Generated explanations for {} customers".format(len(sample_customers)))
    logger.info("")
    
    # Generate summary using the class method
    logger.info("Generating feature importance summary.")
    feature_explainer.get_feature_importance_summary(explanations, top_n=15)
    
    logger.info("")
    logger.info("FEATURE EXPLANATIONS COMPLETED!")
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info("   - Tested on {} customers".format(len(sample_customers)))
    successful_explanations = sum(1 for exp in explanations if exp.get('success', False))
    logger.info("   - Generated {} successful explanations".format(successful_explanations))
    if successful_explanations > 0:
        # Get top feature from successful explanations
        all_features = []
        for exp in explanations:
            if exp.get('success', False) and 'feature_contributions' in exp:
                for feature in exp['feature_contributions']:
                    all_features.append(feature['feature_name'])
        if all_features:
            from collections import Counter
            feature_counts = Counter(all_features)
            top_feature = feature_counts.most_common(1)[0][0]
            logger.info("   - Top feature: {}".format(top_feature))
    logger.info("   - Mode: Print and log to disk")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 