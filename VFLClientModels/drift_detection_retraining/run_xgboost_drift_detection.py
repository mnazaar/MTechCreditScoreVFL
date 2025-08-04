#!/usr/bin/env python3
"""
Practical XGBoost Drift Detection Script
Uses real dataset paths for credit card data
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_drift_detector import XGBoostDriftDetector, detect_credit_card_xgboost_drift

def setup_logging():
    """Setup logging to capture all output"""
    os.makedirs('VFLClientModels/logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('DriftDetectionRun')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
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
        f'VFLClientModels/logs/drift_detection_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Capture print statements
    original_print = print
    def print_to_log(*args, **kwargs):
        message = ' '.join(str(arg) for arg in args)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"[{timestamp}] {message}")
        original_print(*args, **kwargs)
    
    # Replace print function
    import builtins
    builtins.print = print_to_log
    
    return logger, original_print

def load_and_prepare_data():
    """Load and prepare the real credit card datasets"""
    print("Loading real credit card datasets...")
    
    # Define data paths
    current_data_path = 'VFLClientModels/dataset/data/banks/credit_card_bank.csv'
    baseline_data_path = 'VFLClientModels/dataset/data/banks/credit_card_bank_baseline.csv'
    
    # Check if files exist
    if not os.path.exists(current_data_path):
        raise FileNotFoundError(f"Current data file not found: {current_data_path}")
    
    if not os.path.exists(baseline_data_path):
        raise FileNotFoundError(f"Baseline data file not found: {baseline_data_path}")
    
    # Load datasets
    current_data = pd.read_csv(current_data_path)
    baseline_data = pd.read_csv(baseline_data_path)
    
    print(f"Data loaded successfully:")
    print(f"   - Current data: {len(current_data):,} samples, {len(current_data.columns)} features")
    print(f"   - Baseline data: {len(baseline_data):,} samples, {len(baseline_data.columns)} features")
    
    # Check for common features
    common_features = set(current_data.columns) & set(baseline_data.columns)
    print(f"   - Common features: {len(common_features)}")
    
    # Show feature differences
    current_only = set(current_data.columns) - set(baseline_data.columns)
    baseline_only = set(baseline_data.columns) - set(current_data.columns)
    
    if current_only:
        print(f"   - Features only in current: {list(current_only)}")
    if baseline_only:
        print(f"   - Features only in baseline: {list(baseline_only)}")
    
    return current_data, baseline_data, list(common_features)

def create_credit_card_derived_features(data, verbose=True):
    """Create derived features for credit card analysis (same as in training)"""
    if verbose:
        print("Creating derived features...")
    
    data = data.copy()  # Ensure we don't modify the original data
    
    # Credit capacity utilization
    data['credit_capacity_ratio'] = data['total_credit_limit'] / data['total_credit_limit'].replace(0, 1)
    
    # Income to limit ratio
    data['income_to_limit_ratio'] = data['annual_income'] / data['total_credit_limit'].replace(0, 1)
    
    # Debt service ratio
    data['debt_service_ratio'] = (data['current_debt'] * 0.03) / (data['annual_income'] / 12)
    
    # Risk-adjusted income
    data['risk_adjusted_income'] = data['annual_income'] * (1 - data['debt_to_income_ratio'])
    
    if verbose:
        print("Derived features created")
    return data

# Legacy function for backward compatibility
def create_derived_features(data):
    """Legacy function - use create_credit_card_derived_features instead"""
    return create_credit_card_derived_features(data, verbose=True)

def select_credit_card_features(data, verbose=True):
    """Select features for credit card analysis (same as in training)"""
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
        # Derived features
        'credit_capacity_ratio', 'income_to_limit_ratio', 'debt_service_ratio', 'risk_adjusted_income'
    ]
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in data.columns]
    missing_features = [f for f in feature_columns if f not in data.columns]
    
    if missing_features and verbose:
        print(f"Missing features: {missing_features}")
    
    if verbose:
        print(f"Using {len(available_features)} features for drift detection")
    return available_features

def preprocess_credit_card_data_for_drift_detection(current_data, baseline_data, verbose=True):
    """
    Complete preprocessing pipeline for credit card drift detection
    
    Args:
        current_data: Current data DataFrame
        baseline_data: Baseline data DataFrame
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (processed_current_data, processed_baseline_data)
    """
    if verbose:
        print("Starting credit card data preprocessing for drift detection...")
    
    # Create derived features for both datasets
    current_data_processed = create_credit_card_derived_features(current_data, verbose=verbose)
    baseline_data_processed = create_credit_card_derived_features(baseline_data, verbose=verbose)
    
    # Select features for analysis
    feature_columns = select_credit_card_features(current_data_processed, verbose=verbose)
    
    # Ensure both datasets have the same features
    available_features = list(set(feature_columns) & set(current_data_processed.columns) & set(baseline_data_processed.columns))
    
    if verbose:
        print(f"Final feature set: {len(available_features)} features")
    
    # Prepare data for drift detection
    current_features = current_data_processed[available_features].copy()
    baseline_features = baseline_data_processed[available_features].copy()
    
    # Handle any infinite or missing values
    current_features = current_features.replace([np.inf, -np.inf], np.nan)
    baseline_features = baseline_features.replace([np.inf, -np.inf], np.nan)
    
    current_features = current_features.fillna(current_features.median())
    baseline_features = baseline_features.fillna(baseline_features.median())
    
    if verbose:
        print(f"Preprocessing complete:")
        print(f"   - Current features: {current_features.shape}")
        print(f"   - Baseline features: {baseline_features.shape}")
    
    return current_features, baseline_features

def run_drift_detection_with_real_data():
    """Run drift detection on real credit card data"""
    print("\n" + "="*80)
    print("XGBOOST DRIFT DETECTION ON REAL CREDIT CARD DATA")
    print("="*80)
    
    try:
        # Load and prepare data
        current_data, baseline_data, common_features = load_and_prepare_data()
        
        # Use the unified preprocessing pipeline
        current_features, baseline_features = preprocess_credit_card_data_for_drift_detection(
            current_data, baseline_data, verbose=True
        )
        
        # Initialize drift detector
        print("\nInitializing XGBoost Drift Detector...")
        detector = XGBoostDriftDetector()
        
        # Run drift detection
        print("\nRunning drift detection...")
        drift_results = detector.detect_xgboost_drift(
            current_data=current_features,
            baseline_data=baseline_features
        )
        
        # Generate and print report
        report = detector.generate_xgboost_drift_report(drift_results)
        print("\n" + "="*80)
        print("DRIFT DETECTION REPORT")
        print("="*80)
        print(report)
        
        # Get summary
        summary = detector.get_drift_summary(drift_results)
        print(f"\nQUICK SUMMARY:")
        print(f"   - Overall drift detected: {'YES' if summary['drift_detected'] else 'NO'}")
        print(f"   - Statistical drift: {'YES' if summary['statistical_drift'] else 'NO'}")
        print(f"   - Performance drift: {'YES' if summary['performance_drift'] else 'NO'}")
        print(f"   - Prediction drift: {'YES' if summary['prediction_drift'] else 'NO'}")
        
        return drift_results, summary
        
    except Exception as e:
        print(f"Error in drift detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def run_convenience_function():
    """Run drift detection using the convenience function"""
    print("\n" + "="*80)
    print("CONVENIENCE FUNCTION DRIFT DETECTION")
    print("="*80)
    
    try:
        # Define paths
        current_data_path = 'VFLClientModels/dataset/data/banks/credit_card_bank.csv'
        baseline_data_path = 'VFLClientModels/dataset/data/banks/credit_card_bank_baseline.csv'
        model_path = 'VFLClientModels/saved_models/credit_card_xgboost_independent.pkl'
        
        print(f"Using data paths:")
        print(f"   - Current: {current_data_path}")
        print(f"   - Baseline: {baseline_data_path}")
        print(f"   - Model: {model_path}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("   Drift detection will run without model-specific analysis")
        
        # Run drift detection
        print("\nRunning drift detection with convenience function...")
        drift_detected, report = detect_credit_card_xgboost_drift(
            current_data_path=current_data_path,
            baseline_data_path=baseline_data_path,
            model_path=model_path if os.path.exists(model_path) else None
        )
        
        print(f"\nResult: Drift {'DETECTED' if drift_detected else 'NOT DETECTED'}")
        
        return drift_detected, report
        
    except Exception as e:
        print(f"Error in convenience function: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)

def main():
    """Main function to run drift detection on real data"""
    # Setup logging
    logger, original_print = setup_logging()
    
    print("XGBoost Drift Detection on Real Credit Card Data")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run comprehensive drift detection
    drift_results, summary = run_drift_detection_with_real_data()
    
    if drift_results is not None:
        print(f"\nDrift detection completed successfully")
        
        # Run convenience function for comparison
        convenience_drift, convenience_report = run_convenience_function()
        
        print(f"\nCOMPARISON:")
        print(f"   - Comprehensive detection: {'Drift detected' if summary['drift_detected'] else 'No drift'}")
        print(f"   - Convenience function: {'Drift detected' if convenience_drift else 'No drift'}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if summary['drift_detected']:
            print("   Consider retraining the XGBoost model")
            print("   Monitor drift patterns over time")
            print("   Investigate data quality issues")
        else:
            print("   Model appears stable")
            print("   Continue regular monitoring")
            print("   Consider performance optimization")
        
        print(f"\nResults saved to logs:")
        print(f"   - Log file: VFLClientModels/logs/xgboost_drift_detection_*.log")
        print(f"   - Run log: VFLClientModels/logs/drift_detection_run_*.log")
        
    else:
        print(f"\nDrift detection failed")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Restore original print function
    import builtins
    builtins.print = original_print

if __name__ == "__main__":
    main() 