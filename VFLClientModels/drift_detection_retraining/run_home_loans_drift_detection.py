#!/usr/bin/env python3
"""
Practical Home Loans Drift Detection Script
Uses real dataset paths for home loans data
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from home_loans_drift_detector import HomeLoansDriftDetector, detect_home_loans_drift

def setup_logging():
    """Setup logging to capture all output"""
    os.makedirs('VFLClientModels/logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('HomeLoansDriftDetectionRun')
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
        f'VFLClientModels/logs/home_loans_drift_detection_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
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
    """Load and prepare the real home loans datasets"""
    print("Loading real home loans datasets...")
    
    # Define data paths
    current_data_path = 'VFLClientModels/dataset/data/banks/home_loans_bank.csv'
    baseline_data_path = 'VFLClientModels/dataset/data/banks/home_loans_bank_baseline.csv'
    
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

def select_home_loans_features(data, verbose=True):
    """Select features for home loans analysis (same as in training)"""
    feature_columns = [
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
    
    # Filter to available features
    available_features = [f for f in feature_columns if f in data.columns]
    missing_features = [f for f in feature_columns if f not in data.columns]
    
    if missing_features and verbose:
        print(f"Missing features: {missing_features}")
    
    if verbose:
        print(f"Using {len(available_features)} features for drift detection")
    return available_features

def preprocess_home_loans_data_for_drift_detection(current_data, baseline_data, verbose=True):
    """
    Complete preprocessing pipeline for home loans drift detection
    
    Args:
        current_data: Current data DataFrame
        baseline_data: Baseline data DataFrame
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (processed_current_data, processed_baseline_data)
    """
    if verbose:
        print("Starting home loans data preprocessing for drift detection...")
    
    # Select features for analysis
    feature_columns = select_home_loans_features(current_data, verbose=verbose)
    
    # Ensure both datasets have the same features
    available_features = list(set(feature_columns) & set(current_data.columns) & set(baseline_data.columns))
    
    if verbose:
        print(f"Final feature set: {len(available_features)} features")
    
    # Prepare data for drift detection
    current_features = current_data[available_features].copy()
    baseline_features = baseline_data[available_features].copy()
    
    # Handle any infinite or missing values
    current_features = current_features.replace([np.inf, -np.inf], np.nan)
    baseline_features = baseline_features.replace([np.inf, -np.inf], np.nan)
    
    current_features = current_features.fillna(current_features.median())
    baseline_features = baseline_features.fillna(baseline_features.median())
    
    if verbose:
        print(f"Home loans preprocessing complete:")
        print(f"   - Current features: {current_features.shape}")
        print(f"   - Baseline features: {baseline_features.shape}")
    
    return current_features, baseline_features

def run_drift_detection_with_real_data():
    """Run drift detection on real home loans data"""
    print("\n" + "="*80)
    print("HOME LOANS DRIFT DETECTION ON REAL DATA")
    print("="*80)
    
    try:
        # Load and prepare data
        current_data, baseline_data, common_features = load_and_prepare_data()
        
        # Use the unified preprocessing pipeline
        current_features, baseline_features = preprocess_home_loans_data_for_drift_detection(
            current_data, baseline_data, verbose=True
        )
        
        # Initialize drift detector
        print("\nInitializing Home Loans Drift Detector...")
        detector = HomeLoansDriftDetector()
        
        # Run drift detection
        print("\nRunning drift detection...")
        drift_results = detector.detect_home_loans_drift(
            current_data=current_features,
            baseline_data=baseline_features
        )
        
        # Generate and print report
        report = detector.generate_home_loans_drift_report(drift_results)
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
        current_data_path = 'VFLClientModels/dataset/data/banks/home_loans_bank.csv'
        baseline_data_path = 'VFLClientModels/dataset/data/banks/home_loans_bank_baseline.csv'
        model_path = 'VFLClientModels/saved_models/home_loans_model.keras'
        
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
        drift_detected, report = detect_home_loans_drift(
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
    
    print("Home Loans Drift Detection on Real Data")
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
            print("   Consider retraining the Home Loans model")
            print("   Monitor drift patterns over time")
            print("   Investigate data quality issues")
        else:
            print("   Model appears stable")
            print("   Continue regular monitoring")
            print("   Consider performance optimization")
        
        print(f"\nResults saved to logs:")
        print(f"   - Log file: VFLClientModels/logs/home_loans_drift_detection_*.log")
        print(f"   - Run log: VFLClientModels/logs/home_loans_drift_detection_run_*.log")
        
    else:
        print(f"\nDrift detection failed")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Restore original print function
    import builtins
    builtins.print = original_print

if __name__ == "__main__":
    main() 