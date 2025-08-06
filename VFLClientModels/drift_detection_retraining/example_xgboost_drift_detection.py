#!/usr/bin/env python3
"""
Example script demonstrating XGBoost drift detection
Shows how to use the XGBoostDriftDetector class for drift detection
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add the current directory to Python path to import the drift detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xgboost_drift_detector import XGBoostDriftDetector, detect_credit_card_xgboost_drift

def create_sample_data_for_drift_detection():
    """
    Create sample baseline and current data for drift detection testing
    This simulates data drift scenarios
    """
    print("📊 Creating sample data for drift detection...")
    
    # Create baseline data (good quality)
    np.random.seed(42)
    n_samples = 1000
    
    baseline_data = pd.DataFrame({
        'annual_income': np.random.normal(75000, 20000, n_samples),
        'credit_score': np.random.normal(720, 80, n_samples),
        'payment_history': np.random.normal(0.95, 0.05, n_samples),
        'employment_length': np.random.normal(8, 3, n_samples),
        'debt_to_income_ratio': np.random.normal(0.35, 0.1, n_samples),
        'age': np.random.normal(45, 12, n_samples),
        'credit_history_length': np.random.normal(15, 5, n_samples),
        'num_credit_cards': np.random.poisson(3, n_samples),
        'num_loan_accounts': np.random.poisson(2, n_samples),
        'total_credit_limit': np.random.normal(50000, 15000, n_samples),
        'credit_utilization_ratio': np.random.normal(0.3, 0.15, n_samples),
        'late_payments': np.random.poisson(1, n_samples),
        'credit_inquiries': np.random.poisson(2, n_samples),
        'last_late_payment_days': np.random.exponential(365, n_samples),
        'current_debt': np.random.normal(25000, 10000, n_samples),
        'monthly_expenses': np.random.normal(4000, 1000, n_samples),
        'savings_balance': np.random.normal(15000, 8000, n_samples),
        'checking_balance': np.random.normal(5000, 3000, n_samples),
        'investment_balance': np.random.normal(25000, 15000, n_samples),
        'auto_loan_balance': np.random.normal(15000, 8000, n_samples),
        'mortgage_balance': np.random.normal(200000, 80000, n_samples),
        'credit_capacity_ratio': np.random.normal(0.8, 0.2, n_samples),
        'income_to_limit_ratio': np.random.normal(1.5, 0.5, n_samples),
        'debt_service_ratio': np.random.normal(0.25, 0.1, n_samples),
        'risk_adjusted_income': np.random.normal(65000, 18000, n_samples)
    })
    
    # Create current data with some drift (simulating data quality issues)
    current_data = baseline_data.copy()
    
    # Introduce drift scenarios
    drift_scenarios = [
        "income_drift",      # Income distribution shift
        "credit_score_drift", # Credit score degradation
        "utilization_drift",  # Higher credit utilization
        "no_drift"           # No drift (for testing)
    ]
    
    scenario = np.random.choice(drift_scenarios)
    print(f"🎲 Selected drift scenario: {scenario}")
    
    if scenario == "income_drift":
        # Simulate economic downturn - lower incomes
        current_data['annual_income'] = current_data['annual_income'] * 0.85 + np.random.normal(0, 5000, n_samples)
        current_data['risk_adjusted_income'] = current_data['risk_adjusted_income'] * 0.8
        print("📉 Applied income drift: 15% reduction in average income")
        
    elif scenario == "credit_score_drift":
        # Simulate credit score degradation
        current_data['credit_score'] = current_data['credit_score'] * 0.9 + np.random.normal(0, 20, n_samples)
        current_data['payment_history'] = current_data['payment_history'] * 0.95
        current_data['late_payments'] = current_data['late_payments'] * 1.5
        print("📉 Applied credit score drift: 10% reduction in average credit score")
        
    elif scenario == "utilization_drift":
        # Simulate higher credit utilization
        current_data['credit_utilization_ratio'] = current_data['credit_utilization_ratio'] * 1.3
        current_data['credit_utilization_ratio'] = np.clip(current_data['credit_utilization_ratio'], 0, 1)
        current_data['debt_to_income_ratio'] = current_data['debt_to_income_ratio'] * 1.2
        print("📉 Applied utilization drift: 30% increase in credit utilization")
        
    else:  # no_drift
        print("✅ No drift applied - testing stable scenario")
    
    # Ensure all values are within reasonable bounds
    current_data['credit_score'] = np.clip(current_data['credit_score'], 300, 850)
    current_data['payment_history'] = np.clip(current_data['payment_history'], 0, 1)
    current_data['debt_to_income_ratio'] = np.clip(current_data['debt_to_income_ratio'], 0, 1)
    current_data['credit_utilization_ratio'] = np.clip(current_data['credit_utilization_ratio'], 0, 1)
    
    # Save sample data
    os.makedirs('sample_data', exist_ok=True)
    baseline_data.to_csv('sample_data/baseline_credit_card_data.csv', index=False)
    current_data.to_csv('sample_data/current_credit_card_data.csv', index=False)
    
    print(f"✅ Sample data created:")
    print(f"   - Baseline: {len(baseline_data):,} samples")
    print(f"   - Current: {len(current_data):,} samples")
    print(f"   - Features: {len(baseline_data.columns)}")
    print(f"   - Saved to: sample_data/")
    
    return baseline_data, current_data

def example_basic_drift_detection():
    """Example of basic drift detection using the convenience function"""
    print("\n" + "="*80)
    print("🎯 EXAMPLE 1: Basic Drift Detection")
    print("="*80)
    
    # Create sample data
    baseline_data, current_data = create_sample_data_for_drift_detection()
    
    # Save data for the convenience function
    baseline_data.to_csv('sample_data/baseline_data.csv', index=False)
    current_data.to_csv('sample_data/current_data.csv', index=False)
    
    # Use the convenience function
    print("\n🔍 Running drift detection with convenience function...")
    
    try:
        drift_detected, report = detect_credit_card_xgboost_drift(
            current_data_path='sample_data/current_data.csv',
            baseline_data_path='sample_data/baseline_data.csv',
            model_path='VFLClientModels/saved_models/credit_card_xgboost_independent.pkl'
        )
        
        print(f"\n🎯 Result: Drift {'DETECTED' if drift_detected else 'NOT DETECTED'}")
        
    except Exception as e:
        print(f"❌ Error in basic drift detection: {str(e)}")
        print("This is expected if the model file doesn't exist yet")

def example_advanced_drift_detection():
    """Example of advanced drift detection using the XGBoostDriftDetector class"""
    print("\n" + "="*80)
    print("🎯 EXAMPLE 2: Advanced Drift Detection")
    print("="*80)
    
    # Create sample data
    baseline_data, current_data = create_sample_data_for_drift_detection()
    
    # Initialize the detector
    print("\n🔧 Initializing XGBoost Drift Detector...")
    
    try:
        detector = XGBoostDriftDetector(
            model_path='VFLClientModels/saved_models/credit_card_xgboost_independent.pkl'
        )
        
        # Perform drift detection
        print("\n🔍 Running comprehensive drift detection...")
        drift_results = detector.detect_xgboost_drift(
            current_data=current_data,
            baseline_data=baseline_data
        )
        
        # Generate detailed report
        report = detector.generate_xgboost_drift_report(drift_results)
        print(report)
        
        # Get summary
        summary = detector.get_drift_summary(drift_results)
        print(f"\n📋 DRIFT SUMMARY:")
        print(f"   - Overall drift: {summary['drift_detected']}")
        print(f"   - Statistical drift: {summary['statistical_drift']}")
        print(f"   - Performance drift: {summary['performance_drift']}")
        print(f"   - Prediction drift: {summary['prediction_drift']}")
        print(f"   - Feature importance drift: {summary['feature_importance_drift']}")
        print(f"   - XGBoost-specific drift: {summary['xgboost_specific_drift']}")
        
    except Exception as e:
        print(f"❌ Error in advanced drift detection: {str(e)}")
        print("This is expected if the model file doesn't exist yet")

def example_integration_with_retraining_pipeline():
    """Example of how to integrate drift detection with retraining pipeline"""
    print("\n" + "="*80)
    print("🎯 EXAMPLE 3: Integration with Retraining Pipeline")
    print("="*80)
    
    print("""
This example shows how to integrate drift detection with a retraining pipeline:

1. Monitor data regularly
2. Detect drift using XGBoostDriftDetector
3. Trigger retraining if drift is detected
4. Update baseline data after retraining

Example integration code:

```python
from xgboost_drift_detector import XGBoostDriftDetector

# Initialize detector
detector = XGBoostDriftDetector(
    model_path='VFLClientModels/saved_models/credit_card_xgboost_independent.pkl'
)

# Load current and baseline data
current_data = pd.read_csv('current_data.csv')
baseline_data = pd.read_csv('baseline_data.csv')

# Check for drift
drift_detected, report = detector.is_drift_detected(
    current_data=current_data,
    baseline_data=baseline_data
)

if drift_detected:
    print("🔄 Drift detected! Initiating retraining pipeline...")
    # Call your retraining function here
    # retrain_credit_card_xgboost_model()
else:
    print("✅ No drift detected. Model is stable.")
```

Key benefits of this approach:
- Separation of concerns: Drift detection is separate from retraining
- Reusable: Can be used with different retraining pipelines
- Comprehensive: Detects multiple types of drift
- XGBoost-specific: Optimized for XGBoost model characteristics
""")

def main():
    """Main function to run all examples"""
    print("🎯 XGBoost Drift Detection Examples")
    print("="*80)
    print("This script demonstrates the XGBoost drift detection module")
    print("It shows different ways to use the drift detection functionality")
    
    # Run examples
    example_basic_drift_detection()
    example_advanced_drift_detection()
    example_integration_with_retraining_pipeline()
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80)
    print("\n📋 Key Features Demonstrated:")
    print("   ✅ Basic drift detection with convenience function")
    print("   ✅ Advanced drift detection with detailed analysis")
    print("   ✅ XGBoost-specific drift patterns (leaf distribution)")
    print("   ✅ Comprehensive drift reporting")
    print("   ✅ Integration with retraining pipelines")
    print("\n🔗 Next Steps:")
    print("   1. Train your XGBoost model using credit_card_xgboost_model.py")
    print("   2. Use this drift detection module in your monitoring pipeline")
    print("   3. Integrate with your retraining pipeline when drift is detected")

if __name__ == "__main__":
    main() 