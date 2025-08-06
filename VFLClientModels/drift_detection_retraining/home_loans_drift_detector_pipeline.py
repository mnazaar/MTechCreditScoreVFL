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

# Try to import TensorFlow, but provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️  TensorFlow not available. Model loading will be skipped.")
    print("   To install TensorFlow: pip install tensorflow")

# Import the generic drift detector
try:
    from drift_detection import DriftDetector
except ImportError as e:
    print(f"❌ Error importing drift_detection: {e}")
    print("   This might be due to missing dependencies like TensorFlow")
    DriftDetector = None

class HomeLoansDriftDetector:
    """
    Specialized drift detection for Home Loans neural network models
    Focuses only on drift detection, not retraining
    Uses the generic DriftDetector infrastructure
    """
    
    def __init__(self, model_path: str = None, config_path: str = None):
        """
        Initialize Home Loans drift detector
        
        Args:
            model_path: Path to saved Home Loans model
            config_path: Path to drift detection configuration
        """
        self.logger = self._setup_logging()
        
        if DriftDetector is None:
            self.logger.error("DriftDetector not available. Please install required dependencies.")
            self.generic_detector = None
        else:
            self.generic_detector = DriftDetector(config_path)
        
        # Home Loans specific thresholds
        self.home_loans_thresholds = {
            'statistical_drift': 0.1,      # KS test p-value threshold
            'performance_drift': 0.15,     # Confidence drift threshold
            'distribution_drift': 0.2,     # Distribution similarity threshold
            'prediction_drift': 0.25       # Prediction distribution drift
        }
        
        # Load model if provided
        self.model = None
        self.feature_names = None
        self.scaler = None
        
        if model_path:
            self.load_model(model_path)
        
        self.logger.info("Home Loans Drift Detector initialized")
        self.logger.info(f"   - Model loaded: {'YES' if self.model else 'NO'}")
        self.logger.info(f"   - Generic detector: YES")
        self.logger.info(f"   - Home Loans-specific thresholds: YES")
    
    def _setup_logging(self):
        """Setup logging for Home Loans drift detection with print capture"""
        os.makedirs('VFLClientModels/logs', exist_ok=True)
        
        logger = logging.getLogger('HomeLoansDriftDetection')
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
            f'VFLClientModels/logs/home_loans_drift_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
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
        """Load Home Loans model and related artifacts"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("Skipping model loading due to missing TensorFlow.")
            return

        try:
            self.logger.info(f"Loading Home Loans model from {model_path}")
            
            # Load the main model
            self.model = load_model(model_path)
            
            # Load scaler
            scaler_path = model_path.replace('.keras', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Loaded scaler")
            
            # Load feature names
            feature_names_path = model_path.replace('.keras', '_feature_names.npy')
            if os.path.exists(feature_names_path):
                self.feature_names = list(np.load(feature_names_path, allow_pickle=True))
                self.logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            self.logger.info("Home Loans model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_home_loans(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the loaded Home Loans model"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            # Return dummy predictions if no model is loaded or TensorFlow is not available
            # This allows drift detection to work without a model
            self.logger.warning("Skipping prediction due to missing TensorFlow or no model loaded.")
            return np.random.random((len(data), 1))  # 1 output for home loan amount
        
        # Select features if feature names are available
        if self.feature_names:
            missing_features = [f for f in self.feature_names if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            data = data[self.feature_names]
        
        # Scale features
        if self.scaler:
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data.values
        
        # Make predictions
        predictions = self.model.predict(data_scaled, verbose=0)
        return predictions
    
    def detect_home_loans_drift(self, 
                               current_data: pd.DataFrame,
                               baseline_data: pd.DataFrame,
                               target_column: str = None) -> Dict:
        """
        Detect drift specifically for Home Loans neural network model
        
        Args:
            current_data: Current data DataFrame
            baseline_data: Baseline data DataFrame
            target_column: Optional target column for performance drift
            
        Returns:
            Dict containing drift detection results
        """
        self.logger.info("Starting Home Loans-specific drift detection...")
        
        # Use feature names if available, otherwise use all columns
        feature_columns = self.feature_names if self.feature_names else current_data.columns.tolist()
        
        # Remove target column from features if present
        if target_column and target_column in feature_columns:
            feature_columns.remove(target_column)
        
        self.logger.info(f"Analyzing {len(feature_columns)} features")
        self.logger.info(f"Baseline samples: {len(baseline_data):,}")
        self.logger.info(f"Current samples: {len(current_data):,}")
        
        # Use the generic detector with Home Loans-specific functions
        if self.generic_detector is None:
            self.logger.error("Generic detector not available. Cannot perform drift detection.")
            return {
                'overall_drift_detected': False,
                'error': 'Generic detector not available'
            }
        
        drift_results = self.generic_detector.detect_drift_generic(
            current_data=current_data,
            baseline_data=baseline_data,
            model_predictor=self.predict_home_loans,
            feature_columns=feature_columns,
            target_column=target_column
        )
        
        # Determine overall drift (removed feature importance and model-specific analysis)
        overall_drift = False
        if 'statistical_drift' in drift_results and 'performance_drift' in drift_results and 'prediction_drift' in drift_results:
            overall_drift = (
                drift_results['statistical_drift']['drift_detected'] or
                drift_results['performance_drift']['drift_detected'] or
                drift_results['prediction_drift']['drift_detected']
            )
        else:
            self.logger.warning("Some drift detection components failed. Using fallback.")
            overall_drift = False
        
        drift_results['overall_drift_detected'] = overall_drift
        
        self.logger.info(f"Overall drift detected: {overall_drift}")
        
        return drift_results
    
    def generate_home_loans_drift_report(self, drift_results: Dict) -> str:
        """Generate comprehensive Home Loans drift report"""
        
        # Extract detailed feature drift information
        statistical_drift = drift_results.get('statistical_drift', {})
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
HOME LOANS NEURAL NETWORK MODEL DRIFT DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATUS: {'DRIFT DETECTED' if drift_results['overall_drift_detected'] else 'NO DRIFT'}

MODEL INFORMATION:
- Model Type: Home Loans Neural Network
- Feature Dimensions: {len(self.feature_names) if self.feature_names else 'Unknown'}
- Model Architecture: {'Loaded' if self.model else 'Not Loaded'}
- Scaler: {'Loaded' if self.scaler else 'Not Loaded'}

DETAILED DRIFT ANALYSIS:

1. Statistical Drift (KS Test):
   - Features with drift: {statistical_drift.get('features_with_drift', 'N/A')}
   - Total features checked: {statistical_drift.get('total_features', 'N/A')}
   - Drift detected: {statistical_drift.get('drift_detected', 'N/A')}
   - Drift percentage: {f"{(statistical_drift.get('features_with_drift', 0)/statistical_drift.get('total_features', 1)*100):.1f}%" if statistical_drift.get('total_features', 0) > 0 else 'N/A'} of features drifted{feature_drift_details}

2. Performance Drift (Confidence):
   - Baseline avg confidence: {drift_results.get('performance_drift', {}).get('baseline_avg_confidence', 'N/A')}
   - Current avg confidence: {drift_results.get('performance_drift', {}).get('current_avg_confidence', 'N/A')}
   - Confidence drift: {drift_results.get('performance_drift', {}).get('confidence_drift', 'N/A')}
   - Drift detected: {drift_results.get('performance_drift', {}).get('drift_detected', 'N/A')}

3. Prediction Drift (KS Test):
   - KS statistic: {drift_results.get('prediction_drift', {}).get('ks_statistic', 'N/A')}
   - P-value: {drift_results.get('prediction_drift', {}).get('p_value', 'N/A')}
   - Prediction shift: {drift_results.get('prediction_drift', {}).get('prediction_shift', 'N/A')}
   - Drift detected: {drift_results.get('prediction_drift', {}).get('drift_detected', 'N/A')}

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
            drift_results = self.detect_home_loans_drift(
                current_data=current_data,
                baseline_data=baseline_data,
                target_column=target_column
            )
            
            # Generate report
            report = self.generate_home_loans_drift_report(drift_results)
            
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
def detect_home_loans_drift(current_data_path: str,
                           baseline_data_path: str,
                           model_path: str = None) -> Tuple[bool, str]:
    """
    Convenience function to detect drift in home loans neural network model
    
    Args:
        current_data_path: Path to current data CSV
        baseline_data_path: Path to baseline data CSV
        model_path: Path to saved Home Loans model
        
    Returns:
        Tuple of (drift_detected: bool, report: str)
    """
    
    # Load data
    current_data = pd.read_csv(current_data_path)
    baseline_data = pd.read_csv(baseline_data_path)
    
    # Initialize detector
    detector = HomeLoansDriftDetector(model_path=model_path)
    
    # Detect drift
    drift_detected, report = detector.is_drift_detected(
        current_data=current_data,
        baseline_data=baseline_data
    )
    
    # Restore original print function
    detector.restore_print()
    
    return drift_detected, report

if __name__ == "__main__":
    # Home Loans Drift Detection Pipeline - Independent Execution
    print("🏠 Home Loans Drift Detection Pipeline Starting...")
    print("=" * 60)
    
    # Configuration - Hardcoded paths for independent execution
    CONFIG = {
        'data_path': 'VFLClientModels/dataset/data/banks/home_loans_bank.csv',
        'baseline_data_path': 'VFLClientModels/dataset/data/banks/home_loans_bank_baseline.csv',
        'model_path': 'VFLClientModels/saved_models/home_loans_model.keras',
        'retraining_script': 'VFLClientModels/models/home_loans_model.py',
        'detector_class': 'HomeLoansDriftDetector'
    }
    
    try:
        print(f"📁 Loading current data from: {CONFIG['data_path']}")
        current_data = pd.read_csv(CONFIG['data_path'])
        print(f"✅ Current data loaded: {len(current_data):,} samples, {len(current_data.columns)} features")
        
        print(f"📁 Loading baseline data from: {CONFIG['baseline_data_path']}")
        baseline_data = pd.read_csv(CONFIG['baseline_data_path'])
        print(f"✅ Baseline data loaded: {len(baseline_data):,} samples, {len(baseline_data.columns)} features")
        
        print(f"🤖 Loading Home Loans model from: {CONFIG['model_path']}")
        detector = HomeLoansDriftDetector(model_path=CONFIG['model_path'])
        if TENSORFLOW_AVAILABLE:
            print("✅ Home Loans Drift Detector initialized successfully")
        else:
            print("⚠️  Home Loans Drift Detector initialized (TensorFlow not available)")
            print("   Drift detection will work with dummy predictions")
        
        print("\n🔍 Starting drift detection analysis...")
        print("-" * 40)
        
        # Perform drift detection
        drift_detected, report = detector.is_drift_detected(
            current_data=current_data,
            baseline_data=baseline_data
        )
        
        # Generate summary
        print("\n" + "=" * 60)
        print("📊 DRIFT DETECTION SUMMARY")
        print("=" * 60)
        
        if drift_detected:
            print("⚠️  DRIFT DETECTED - Model retraining recommended!")
            print("🔧 Next steps:")
            print("   1. Review the detailed report above")
            print("   2. Initiate model retraining pipeline")
            print("   3. Monitor drift patterns over time")
        else:
            print("✅ NO DRIFT DETECTED - Model performance is stable")
            print("📈 Model is performing well on current data")
            print("🔍 Continue monitoring for future drift")
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"VFLClientModels/reports/home_loans_drift_report_{timestamp}.txt"
        os.makedirs('VFLClientModels/reports', exist_ok=True)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Restore original print function
        detector.restore_print()
        
        print("\n🎉 Home Loans Drift Detection Pipeline completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        print("Please ensure all data files and model files exist in the specified paths:")
        for key, path in CONFIG.items():
            if key != 'detector_class':
                print(f"   {key}: {path}")
    except Exception as e:
        print(f"❌ Error during drift detection: {str(e)}")
        print("Please check the logs for detailed error information")
    finally:
        print("\n" + "=" * 60)
        print("🏁 Pipeline execution finished") 
        if drift_detected:
            print("home_loan_drift_detected=true")
        else:
            print("home_loan_drift_detected=false")
