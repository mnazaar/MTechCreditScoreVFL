#!/usr/bin/env python3
"""
Test script for single customer credit score prediction using VFL AutoML XGBoost model.

This script demonstrates how to use the predict_credit_score_by_tax_id function
from the vfl_automl_xgboost_model.py file to get predictions for any customer.
"""

import sys
import os

# Add the current directory to path to import the VFL model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the prediction function
from vfl_automl_xgboost_model import predict_credit_score_by_tax_id, example_single_customer_prediction

def test_single_customer_prediction():
    """Test the single customer prediction functionality."""
    
    print("🎯 VFL AutoML XGBoost - Single Customer Prediction Test")
    print("=" * 60)
    
    # Test 1: Use the example function
    print("\n📋 Test 1: Using example function")
    print("-" * 40)
    result1 = example_single_customer_prediction()
    
    # Test 2: Direct function call with custom tax ID
    print("\n📋 Test 2: Direct function call")
    print("-" * 40)
    
    # Replace this with an actual tax ID from your dataset
    custom_tax_id = "TAX002"  # Change this to a real tax ID
    
    try:
        result2 = predict_credit_score_by_tax_id(
            tax_id=custom_tax_id,
            phase_name="Direct Test"
        )
        
        print(f"✅ Successfully predicted for {custom_tax_id}")
        print(f"📊 Prediction Results:")
        print(f"   Credit Score: {result2['predicted_credit_score']} points")
        print(f"   Confidence: {result2['confidence_level']} ({result2['confidence_score']})")
        print(f"   68% CI: {result2['confidence_intervals']['68_percent']['lower']} - {result2['confidence_intervals']['68_percent']['upper']}")
        print(f"   95% CI: {result2['confidence_intervals']['95_percent']['lower']} - {result2['confidence_intervals']['95_percent']['upper']}")
        
        # Show services
        services = [k for k, v in result2['services_available'].items() if v]
        print(f"   Services: {services}")
        
    except Exception as e:
        print(f"❌ Error predicting for {custom_tax_id}: {str(e)}")
        print("💡 Make sure to use a valid tax ID from your dataset")
    
    # Test 3: Show how to use in a loop for multiple customers
    print("\n📋 Test 3: Multiple customers example")
    print("-" * 40)
    
    # Example tax IDs (replace with actual ones from your dataset)
    test_tax_ids = ["TAX001", "TAX002", "TAX003"]  # Replace with real tax IDs
    
    print(f"Testing predictions for {len(test_tax_ids)} customers...")
    
    for i, tax_id in enumerate(test_tax_ids, 1):
        try:
            result = predict_credit_score_by_tax_id(
                tax_id=tax_id,
                phase_name=f"Batch Test {i}"
            )
            
            print(f"   {i}. {tax_id}: {result['predicted_credit_score']} points (Confidence: {result['confidence_level']})")
            
        except Exception as e:
            print(f"   {i}. {tax_id}: Error - {str(e)}")
    
    print("\n✅ Test completed!")
    print("💡 To use this in your own code:")
    print("   from vfl_automl_xgboost_model import predict_credit_score_by_tax_id")
    print("   result = predict_credit_score_by_tax_id('YOUR_TAX_ID')")
    print("   print(f'Credit Score: {result[\"predicted_credit_score\"]}')")


if __name__ == "__main__":
    test_single_customer_prediction() 