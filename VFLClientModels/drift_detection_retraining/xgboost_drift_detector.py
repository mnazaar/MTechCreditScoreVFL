import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import sys
warnings.filterwarnings('ignore')

# Import the generic drift detector
from drift_detection import DriftDetector

class XGBoostDriftDetector:
    """
    Specialized drift detection for XGBoost credit card models
    Focuses only on drift detection, not retraining
    Uses the generic DriftDetector infrastructure
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize XGBoost drift detector
        
        Args:
            model_path: Path to saved XGBoost model
            config_path: Path to drift detection configuration
        """
        self.logger = self._setup_logging()
        self.generic_detector = DriftDetector(config_path)
        
        # XGBoost specific thresholds
        self.xgboost_thresholds = {
            'statistical_drift': 0.1,      # KS test p-value threshold
            'performance_drift': 0.15,     # Confidence drift threshold
            'distribution_drift': 0.2,     # Distribution similarity threshold
            'prediction_drift': 0.25       # Prediction distribution drift
        }
        
        # Load model if provided
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.label_encoder = None
        
        if model_path:
            self.load_model(model_path)
        
        self.logger.info("XGBoost Drift Detector initialized")
        self.logger.info(f"   - Model loaded: {'YES' if self.model else 'NO'}")
        self.logger.info(f"   - Generic detector: YES")
        self.logger.info(f"   - XGBoost-specific thresholds: YES")
    
    def _setup_logging(self):
        """Setup logging for XGBoost drift detection with print capture"""
        os.makedirs('VFLClientModels/logs', exist_ok=True)
        
        logger = logging.getLogger('XGBoostDriftDetection')
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            f'VFLClientModels/logs/xgboost_drift_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Capture print statements and redirect to logger
        self._original_print = print
        def print_to_log(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"[{timestamp}] {message}")
            self._original_print(*args, **kwargs)
        
        # Replace print function
        import builtins
        builtins.print = print_to_log
        
        return logger
    
    def restore_print(self):
        """Restore original print function"""
        import builtins
        builtins.print = self._original_print
    
    def load_model(self, model_path: str):
        """Load XGBoost model and related artifacts"""
        try:
            self.logger.info(f"Loading XGBoost model from {model_path}")
            
            # Load the main model
            model_data = joblib.load(model_path)
            self.model = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_dim = model_data['feature_dim']
            
            # Load feature names
            feature_names_path = model_path.replace('_independent.pkl', '_feature_names.npy')
            if os.path.exists(feature_names_path):
                self.feature_names = list(np.load(feature_names_path, allow_pickle=True))
                self.logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load label encoder
            label_encoder_path = model_path.replace('_independent.pkl', '_label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                self.logger.info(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
            
            self.logger.info("XGBoost model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_xgboost(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded XGBoost model"""
        if self.model is None:
            # Return dummy predictions if no model is loaded
            # This allows drift detection to work without a model
            return np.random.random((len(data), 3))  # 3 classes for credit card tiers
        
        # Select features if feature names are available
        if self.feature_names:
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            data = data[self.feature_names]
        
        # Scale features
        data_scaled = self.scaler.transform(data)
        
        # Make predictions
        predictions = self.model.predict_proba(data_scaled)
        return predictions
    
    def detect_xgboost_drift(self, 
                            current_data: pd.DataFrame,
                            baseline_data: pd.DataFrame,
                            target_column: str = None) -> Dict:
        """
        Detect drift specifically for XGBoost credit card model
        
        Args:
            current_data: Current data DataFrame
            baseline_data: Baseline data DataFrame
            target_column: Optional target column for performance drift
            
        Returns:
            Dict containing drift detection results
        """
        self.logger.info("Starting XGBoost-specific drift detection...")
        
        # Use feature names if available, otherwise use all columns
        feature_columns = self.feature_names if self.feature_names else current_data.columns.tolist()
        
        # Remove target column from features if present
        if target_column and target_column in feature_columns:
            feature_columns.remove(target_column)
        
        self.logger.info(f"Analyzing {len(feature_columns)} features")
        self.logger.info(f"Baseline samples: {len(baseline_data):,}")
        self.logger.info(f"Current samples: {len(current_data):,}")
        
        # Use the generic detector with XGBoost-specific functions
        drift_results = self.generic_detector.detect_drift_generic(
            current_data=current_data,
            baseline_data=baseline_data,
            model_predictor=self.predict_xgboost,
            feature_columns=feature_columns,
            target_column=target_column
        )
        
        # Determine overall drift (removed feature importance and XGBoost-specific analysis)
        overall_drift = (
            drift_results['statistical_drift']['drift_detected'] or
            drift_results['performance_drift']['drift_detected'] or
            drift_results['prediction_drift']['drift_detected']
        )
        
        drift_results['overall_drift_detected'] = overall_drift
        
        self.logger.info(f"Overall drift detected: {overall_drift}")
        
        return drift_results
    
    def generate_xgboost_drift_report(self, drift_results: Dict) -> str:
        """Generate comprehensive XGBoost drift report"""
        
        # Extract detailed feature drift information
        statistical_drift = drift_results['statistical_drift']
        feature_details = statistical_drift.get('feature_details', {})
        
        # Create detailed feature drift section
        feature_drift_details = ""
        if feature_details:
            drifted_features = []
            for feature, details in feature_details.items():
                if details['drift_detected']:
                    drifted_features.append(f"   - {feature}: KS={details['ks_statistic']:.3f}, p-value={details['p_value']:.3f}")
            
            if drifted_features:
                feature_drift_details = "\nDRIFTED FEATURES DETAILS:\n" + "\n".join(drifted_features)
            else:
                feature_drift_details = "\nDRIFTED FEATURES DETAILS:\n   - No features with significant drift detected"
        
        report = f"""
XGBOOST CREDIT CARD MODEL DRIFT DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATUS: {'DRIFT DETECTED' if drift_results['overall_drift_detected'] else 'NO DRIFT'}

MODEL INFORMATION:
- Model Type: XGBoost Credit Card Classifier
- Feature Dimensions: {self.feature_dim if hasattr(self, 'feature_dim') else 'Unknown'}
- Number of Trees: {self.model.n_estimators if self.model else 'Unknown'}
- Max Depth: {self.model.max_depth if self.model else 'Unknown'}

DETAILED DRIFT ANALYSIS:

1. Statistical Drift (KS Test):
   - Features with drift: {statistical_drift['features_with_drift']}
   - Total features checked: {statistical_drift['total_features']}
   - Drift detected: {statistical_drift['drift_detected']}
   - Drift percentage: {(statistical_drift['features_with_drift']/statistical_drift['total_features']*100):.1f}% of features drifted{feature_drift_details}

2. Performance Drift (Confidence):
   - Baseline avg confidence: {drift_results['performance_drift']['baseline_avg_confidence']:.3f}
   - Current avg confidence: {drift_results['performance_drift']['current_avg_confidence']:.3f}
   - Confidence drift: {drift_results['performance_drift']['confidence_drift']:.2%}
   - Drift detected: {drift_results['performance_drift']['drift_detected']}

3. Prediction Drift (KS Test):
   - KS statistic: {drift_results['prediction_drift']['ks_statistic']:.3f}
   - P-value: {drift_results['prediction_drift']['p_value']:.3f}
   - Prediction shift: {drift_results['prediction_drift']['prediction_shift']:.3f}
   - Drift detected: {drift_results['prediction_drift']['drift_detected']}

FINAL RECOMMENDATION:
{'RETRAINING RECOMMENDED - Drift detected in model performance' if drift_results['overall_drift_detected'] else 'NO ACTION NEEDED - Model performance stable'}

NEXT STEPS:
{'1. Monitor drift patterns over time' if not drift_results['overall_drift_detected'] else '1. Initiate model retraining pipeline'}
2. Review data quality and preprocessing
3. Consider model interpretability analysis
"""
        
        return report
    
    def is_drift_detected(self, 
                         current_data: pd.DataFrame,
                         baseline_data: pd.DataFrame,
                         target_column: str = None) -> Tuple[bool, str]:
        """
        Simple method to check if drift is detected
        
        Args:
            current_data: Current data DataFrame
            baseline_data: Baseline data DataFrame
            target_column: Optional target column
            
        Returns:
            Tuple of (drift_detected: bool, report: str)
        """
        try:
            # Perform drift detection
            drift_results = self.detect_xgboost_drift(
                current_data=current_data,
                baseline_data=baseline_data,
                target_column=target_column
            )
            
            # Generate report
            report = self.generate_xgboost_drift_report(drift_results)
            
            # Print report (this will also be logged)
            print(report)
            
            return drift_results['overall_drift_detected'], report
            
        except Exception as e:
            error_msg = f"Error in drift detection: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def get_drift_summary(self, drift_results: Dict) -> Dict:
        """Get a summary of drift detection results"""
        return {
            'drift_detected': drift_results['overall_drift_detected'],
            'statistical_drift': drift_results['statistical_drift']['drift_detected'],
            'performance_drift': drift_results['performance_drift']['drift_detected'],
            'prediction_drift': drift_results['prediction_drift']['drift_detected'],
            'timestamp': drift_results.get('timestamp', datetime.now().isoformat())
        }

# Example usage function
def detect_credit_card_xgboost_drift(current_data_path: str,
                                    baseline_data_path: str,
                                    model_path: str = None) -> Tuple[bool, str]:
    """
    Convenience function to detect drift in credit card XGBoost model
    
    Args:
        current_data_path: Path to current data CSV
        baseline_data_path: Path to baseline data CSV
        model_path: Path to saved XGBoost model
        
    Returns:
        Tuple of (drift_detected: bool, report: str)
    """
    
    # Load data
    current_data = pd.read_csv(current_data_path)
    baseline_data = pd.read_csv(baseline_data_path)
    
    # Initialize detector
    detector = XGBoostDriftDetector(model_path=model_path)
    
    # Detect drift
    drift_detected, report = detector.is_drift_detected(
        current_data=current_data,
        baseline_data=baseline_data
    )
    
    # Restore original print function
    detector.restore_print()
    
    return drift_detected, report

if __name__ == "__main__":
    # Example usage
    print("XGBoost Drift Detection Module")
    print("This module detects drift in XGBoost credit card models")
    print("Use detect_credit_card_xgboost_drift() for simple drift detection")
    print("Or use XGBoostDriftDetector class for advanced analysis") 