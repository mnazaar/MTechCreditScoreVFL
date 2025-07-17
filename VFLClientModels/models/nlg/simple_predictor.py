"""
Simple Credit Score Predictor Usage
==================================

A minimal script to predict credit score for a single customer.

Usage:
    python simple_predictor.py
    # Then enter a customer tax ID when prompted
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from credit_score_predictor import CreditScorePredictor

def predict_single_customer(tax_id):
    """Predict credit score for a single customer and print results."""
    
    print(f"🎯 Predicting credit score for customer: {tax_id}")
    print("=" * 50)
    
    try:
        # Initialize predictor and get prediction
        predictor = CreditScorePredictor()
        result = predictor.predict_credit_score(tax_id)
        
        # Print results in a clean format
        print(f"📊 RESULTS:")
        print(f"   Predicted Credit Score: {result['predicted_credit_score']} points")
        print(f"   Confidence Level: {result['confidence_level']}")
        print(f"   Confidence Score: {result['confidence_score']}")
        print(f"   Uncertainty: ±{result['prediction_uncertainty']} points")
        
        print(f"\n📈 Confidence Intervals:")
        print(f"   68% Range: {result['confidence_intervals']['68_percent']['lower']} - {result['confidence_intervals']['68_percent']['upper']} points")
        print(f"   95% Range: {result['confidence_intervals']['95_percent']['lower']} - {result['confidence_intervals']['95_percent']['upper']} points")
        
        print(f"\n🏦 Available Services:")
        for service, available in result['services_available'].items():
            status = "✅" if available else "❌"
            print(f"   {status} {service.replace('_', ' ').title()}")
        
        if result['actual_credit_score'] is not None:
            print(f"\n📋 Actual vs Predicted:")
            print(f"   Actual Score: {result['actual_credit_score']} points")
            print(f"   Error: {result['prediction_error']['absolute_error']} points ({result['prediction_error']['percentage_error']}%)")
        
        print(f"\n✅ Prediction completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def main():
    """Main function to get customer ID and predict credit score."""
    
    print("🚀 VFL AutoML Credit Score Predictor")
    print("=" * 40)
    print("Enter a customer tax ID to predict their credit score.")
    print("Example: 100-13-3553")
    print("=" * 40)
    
    # Get customer ID from user
    tax_id = input("\nEnter customer tax ID: ").strip()
    
    if not tax_id:
        print("❌ No tax ID provided. Exiting.")
        return
    
    # Make prediction
    result = predict_single_customer(tax_id)
    
    if result:
        print(f"\n💡 Tip: You can also use this in your code:")
        print(f"   from credit_score_predictor import CreditScorePredictor")
        print(f"   predictor = CreditScorePredictor()")
        print(f"   result = predictor.predict_credit_score('{tax_id}')")

if __name__ == "__main__":
    main() 