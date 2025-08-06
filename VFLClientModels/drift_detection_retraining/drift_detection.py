import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Callable, Any
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    """
    Comprehensive drift detection system for VFL credit scoring pipeline
    Generic and reusable for different model types
    """
    
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.drift_history = []
        self.baseline_metrics = {}
        self.drift_thresholds = self.config.get('drift_thresholds', {
            'statistical_drift': 0.1,      # KS test p-value threshold
            'performance_drift': 0.15,     # MAE increase threshold
            'distribution_drift': 0.2,     # Distribution similarity threshold
            'prediction_drift': 0.25       # Prediction distribution drift
        })
        
    def _setup_logging(self):
        """Setup logging for drift detection"""
        os.makedirs('VFLClientModels/logs', exist_ok=True)
        
        logger = logging.getLogger('DriftDetection')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            file_handler = logging.FileHandler(
                f'VFLClientModels/logs/drift_detection_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load drift detection configuration"""
        default_config = {
            'drift_thresholds': {
                'statistical_drift': 0.1,
                'performance_drift': 0.15,
                'distribution_drift': 0.2,
                'prediction_drift': 0.25
            },
            'monitoring_frequency': 'daily',
            'retrain_threshold': 3,  # Number of drift detections before retraining
            'baseline_window': 30,   # Days for baseline calculation
            'detection_window': 7    # Days for drift detection
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def calculate_statistical_drift(self, baseline_data: pd.DataFrame, 
                                  current_data: pd.DataFrame, 
                                  features: List[str]) -> Dict:
        """
        Calculate statistical drift using Kolmogorov-Smirnov test
        Generic method that works with any DataFrame
        """
        from scipy.stats import ks_2samp
        
        drift_results = {
            'features_with_drift': 0,
            'total_features': 0,
            'drift_detected': False,
            'feature_details': {}
        }
        
        for feature in features:
            if feature in baseline_data.columns and feature in current_data.columns:
                baseline_values = baseline_data[feature].dropna()
                current_values = current_data[feature].dropna()
                
                if len(baseline_values) > 0 and len(current_values) > 0:
                    ks_stat, p_value = ks_2samp(baseline_values, current_values)
                    drift_detected = p_value < self.drift_thresholds['statistical_drift']
                    
                    drift_results['feature_details'][feature] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'drift_detected': drift_detected
                    }
                    
                    drift_results['total_features'] += 1
                    if drift_detected:
                        drift_results['features_with_drift'] += 1
        
        drift_results['drift_detected'] = drift_results['features_with_drift'] > 0
        return drift_results
    
    def calculate_performance_drift_generic(self, 
                                          baseline_predictions: np.ndarray,
                                          current_predictions: np.ndarray,
                                          baseline_targets: np.ndarray = None,
                                          current_targets: np.ndarray = None) -> Dict:
        """
        Calculate performance drift using predictions and optional targets
        Generic method that works with any model type
        """
        try:
            # Calculate confidence scores (max probability for classification, direct values for regression)
            if len(baseline_predictions.shape) > 1 and baseline_predictions.shape[1] > 1:
                # Classification case - use max probability as confidence
                baseline_confidence = np.max(baseline_predictions, axis=1)
                current_confidence = np.max(current_predictions, axis=1)
            else:
                # Regression case - use prediction values directly
                baseline_confidence = baseline_predictions.flatten()
                current_confidence = current_predictions.flatten()
            
            # Calculate performance metrics
            baseline_avg_confidence = np.mean(baseline_confidence)
            current_avg_confidence = np.mean(current_confidence)
            
            confidence_drift = (baseline_avg_confidence - current_avg_confidence) / baseline_avg_confidence if baseline_avg_confidence != 0 else 0
            
            drift_detected = abs(confidence_drift) > self.drift_thresholds['performance_drift']
            
            result = {
                'baseline_avg_confidence': baseline_avg_confidence,
                'current_avg_confidence': current_avg_confidence,
                'confidence_drift': confidence_drift,
                'drift_detected': drift_detected
            }
            
            # Add target-based metrics if available
            if baseline_targets is not None and current_targets is not None:
                baseline_mae = mean_absolute_error(baseline_targets, baseline_predictions.flatten())
                current_mae = mean_absolute_error(current_targets, current_predictions.flatten())
                mae_drift = (current_mae - baseline_mae) / baseline_mae if baseline_mae != 0 else 0
                
                result.update({
                    'baseline_mae': baseline_mae,
                    'current_mae': current_mae,
                    'mae_drift': mae_drift,
                    'mae_drift_detected': abs(mae_drift) > self.drift_thresholds['performance_drift']
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating performance drift: {str(e)}")
            return {
                'baseline_avg_confidence': 0,
                'current_avg_confidence': 0,
                'confidence_drift': 0,
                'drift_detected': False
            }
    
    def calculate_distribution_drift(self, baseline_data: pd.DataFrame, 
                                   current_data: pd.DataFrame, 
                                   features: List[str]) -> Dict:
        """
        Calculate distribution drift using Wasserstein distance
        Generic method that works with any DataFrame
        """
        from scipy.stats import wasserstein_distance
        
        drift_results = {
            'features_with_drift': 0,
            'total_features': 0,
            'drift_detected': False,
            'feature_details': {}
        }
        
        for feature in features:
            if feature in baseline_data.columns and feature in current_data.columns:
                baseline_values = baseline_data[feature].dropna()
                current_values = current_data[feature].dropna()
                
                if len(baseline_values) > 0 and len(current_values) > 0:
                    w_distance = wasserstein_distance(baseline_values, current_values)
                    # Normalize by feature range
                    feature_range = baseline_values.max() - baseline_values.min()
                    normalized_distance = w_distance / feature_range if feature_range > 0 else 0
                    
                    drift_detected = normalized_distance > self.drift_thresholds['distribution_drift']
                    
                    drift_results['feature_details'][feature] = {
                        'wasserstein_distance': w_distance,
                        'normalized_distance': normalized_distance,
                        'drift_detected': drift_detected
                    }
                    
                    drift_results['total_features'] += 1
                    if drift_detected:
                        drift_results['features_with_drift'] += 1
        
        drift_results['drift_detected'] = drift_results['features_with_drift'] > 0
        return drift_results
    
    def calculate_prediction_drift_generic(self, 
                                         baseline_predictions: np.ndarray,
                                         current_predictions: np.ndarray) -> Dict:
        """
        Calculate drift in model predictions
        Generic method that works with any model type
        """
        try:
            # Flatten predictions for comparison
            baseline_pred_flat = baseline_predictions.flatten()
            current_pred_flat = current_predictions.flatten()
            
            # Calculate prediction distribution drift
            from scipy.stats import ks_2samp
            ks_stat, p_value = ks_2samp(baseline_pred_flat, current_pred_flat)
            
            # Calculate prediction shift
            baseline_pred_mean = np.mean(baseline_pred_flat)
            current_pred_mean = np.mean(current_pred_flat)
            prediction_shift = current_pred_mean - baseline_pred_mean
            
            drift_detected = p_value < self.drift_thresholds['prediction_drift']
            
            return {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'baseline_pred_mean': baseline_pred_mean,
                'current_pred_mean': current_pred_mean,
                'prediction_shift': prediction_shift,
                'drift_detected': drift_detected
            }
        except Exception as e:
            self.logger.error(f"Error calculating prediction drift: {str(e)}")
            return {
                'ks_statistic': 0,
                'p_value': 1,
                'baseline_pred_mean': 0,
                'current_pred_mean': 0,
                'prediction_shift': 0,
                'drift_detected': False
            }
    
    def calculate_feature_importance_drift(self, 
                                         baseline_importance: np.ndarray,
                                         current_importance: np.ndarray) -> Dict:
        """
        Calculate drift in feature importance patterns
        Generic method that works with any feature importance arrays
        """
        try:
            # Ensure arrays have the same length
            min_length = min(len(baseline_importance), len(current_importance))
            baseline_importance = baseline_importance[:min_length]
            current_importance = current_importance[:min_length]
            
            # Calculate importance drift
            importance_drift = np.abs(baseline_importance - current_importance)
            max_drift = np.max(importance_drift)
            avg_drift = np.mean(importance_drift)
            
            drift_detected = max_drift > 0.2  # 20% threshold for feature importance drift
            
            return {
                'max_importance_drift': max_drift,
                'avg_importance_drift': avg_drift,
                'drift_detected': drift_detected,
                'baseline_importance': baseline_importance.tolist(),
                'current_importance': current_importance.tolist()
            }
        except Exception as e:
            self.logger.error(f"Error calculating feature importance drift: {str(e)}")
            return {
                'max_importance_drift': 0,
                'avg_importance_drift': 0,
                'drift_detected': False,
                'baseline_importance': [],
                'current_importance': []
            }
    
    def detect_drift_generic(self, 
                            current_data: pd.DataFrame,
                            baseline_data: pd.DataFrame,
                            model_predictor: Callable,
                            feature_columns: List[str],
                            target_column: str = None,
                            feature_importance_getter: Callable = None) -> Dict:
        """
        Generic drift detection that works with any model type
        
        Args:
            current_data: Current data DataFrame
            baseline_data: Baseline data DataFrame
            model_predictor: Function that takes DataFrame and returns predictions
            feature_columns: List of feature column names
            target_column: Optional target column name for performance drift
            feature_importance_getter: Optional function to get feature importance
        """
        self.logger.info("🔍 Starting generic drift detection...")
        
        # Prepare data
        current_features = current_data[feature_columns].copy()
        baseline_features = baseline_data[feature_columns].copy()
        
        # Get predictions
        try:
            current_predictions = model_predictor(current_features)
            baseline_predictions = model_predictor(baseline_features)
        except Exception as e:
            self.logger.error(f"Error getting predictions: {str(e)}")
            return {'error': f"Prediction error: {str(e)}"}
        
        # Perform drift detection
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'statistical_drift': self.calculate_statistical_drift(baseline_features, current_features, feature_columns),
            'performance_drift': self.calculate_performance_drift_generic(
                baseline_predictions, current_predictions
            ),
            'prediction_drift': self.calculate_prediction_drift_generic(
                baseline_predictions, current_predictions
            ),
            'overall_drift_detected': False
        }
        
        # Add feature importance drift if available
        if feature_importance_getter is not None:
            try:
                baseline_importance = feature_importance_getter(baseline_features)
                current_importance = feature_importance_getter(current_features)
                drift_results['feature_importance_drift'] = self.calculate_feature_importance_drift(
                    baseline_importance, current_importance
                )
            except Exception as e:
                self.logger.warning(f"Feature importance drift calculation failed: {str(e)}")
                drift_results['feature_importance_drift'] = {
                    'drift_detected': False,
                    'error': str(e)
                }
        
        # Determine overall drift
        drift_detected = (
            drift_results['statistical_drift']['drift_detected'] or
            drift_results['performance_drift']['drift_detected'] or
            drift_results['prediction_drift']['drift_detected'] or
            drift_results.get('feature_importance_drift', {}).get('drift_detected', False)
        )
        
        drift_results['overall_drift_detected'] = drift_detected
        
        # Log results
        self.logger.info(f"🎯 Overall drift detected: {drift_detected}")
        if drift_detected:
            self.logger.warning("⚠️  DRIFT DETECTED - Consider retraining models")
        
        return drift_results
    
    def should_retrain(self, drift_history: List[Dict], threshold: int = None) -> Tuple[bool, str]:
        """
        Determine if retraining is needed based on drift history
        Generic method that works with any drift history
        """
        if threshold is None:
            threshold = self.config['retrain_threshold']
        
        if len(drift_history) < threshold:
            return False, f"Insufficient drift history for retraining decision (need {threshold}, have {len(drift_history)})"
        
        # Count recent drift detections
        recent_drifts = [drift for drift in drift_history[-threshold:] if drift.get('overall_drift_detected', False)]
        
        if len(recent_drifts) >= threshold - 1:  # Retrain if most recent checks show drift
            return True, f"Drift detected in {len(recent_drifts)} recent checks"
        
        return False, "No significant drift pattern detected"
    
    def generate_drift_report_generic(self, drift_results: Dict, model_name: str = "Model") -> str:
        """Generate a comprehensive drift report for any model type"""
        report = f"""
{model_name.upper()} DRIFT DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATUS: {'⚠️  DRIFT DETECTED' if drift_results['overall_drift_detected'] else '✅ NO DRIFT'}

DETAILED ANALYSIS:
1. Statistical Drift:
   - Features with drift: {drift_results['statistical_drift']['features_with_drift']}
   - Total features checked: {drift_results['statistical_drift']['total_features']}
   - Drift detected: {drift_results['statistical_drift']['drift_detected']}

2. Performance Drift:
   - Baseline avg confidence: {drift_results['performance_drift']['baseline_avg_confidence']:.3f}
   - Current avg confidence: {drift_results['performance_drift']['current_avg_confidence']:.3f}
   - Confidence drift: {drift_results['performance_drift']['confidence_drift']:.2%}
   - Drift detected: {drift_results['performance_drift']['drift_detected']}

3. Prediction Drift:
   - KS statistic: {drift_results['prediction_drift']['ks_statistic']:.3f}
   - P-value: {drift_results['prediction_drift']['p_value']:.3f}
   - Prediction shift: {drift_results['prediction_drift']['prediction_shift']:.3f}
   - Drift detected: {drift_results['prediction_drift']['drift_detected']}
"""
        
        if 'feature_importance_drift' in drift_results:
            report += f"""
4. Feature Importance Drift:
   - Max importance drift: {drift_results['feature_importance_drift']['max_importance_drift']:.3f}
   - Avg importance drift: {drift_results['feature_importance_drift']['avg_importance_drift']:.3f}
   - Drift detected: {drift_results['feature_importance_drift']['drift_detected']}
"""
        
        report += f"""
RECOMMENDATION: {'🔄 RETRAINING RECOMMENDED' if drift_results['overall_drift_detected'] else '✅ NO ACTION NEEDED'}
        """
        
        return report 