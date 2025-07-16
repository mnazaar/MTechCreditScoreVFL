"""
Example Usage of Credit Score Predictor Controller
==================================================

This file demonstrates how to use the CreditScorePredictor class to predict
credit scores for customers using the VFL AutoML XGBoost model.

Usage:
    python example_usage.py
"""

import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from credit_score_predictor import CreditScorePredictor

def example_single_prediction():
    """Example of predicting credit score for a single customer."""
    print("🎯 Example 1: Single Customer Prediction")
    print("=" * 50)
    predictor = CreditScorePredictor()
    tax_id = "100-13-3553"  # Replace with actual tax ID
    try:
        results = predictor.predict_credit_score(tax_id)
        print(f"\n📊 Detailed Results for {tax_id}:")
        print(f"   Predicted Credit Score: {results['predicted_credit_score']} points")
        print(f"   Confidence Level: {results['confidence_level']}")
        print(f"   Confidence Score: {results['confidence_score']}")
        print(f"   Uncertainty: ±{results['prediction_uncertainty']} points")
        print(f"\n📈 Confidence Intervals:")
        print(f"   68% Range: {results['confidence_intervals']['68_percent']['lower']} - {results['confidence_intervals']['68_percent']['upper']} points")
        print(f"   95% Range: {results['confidence_intervals']['95_percent']['lower']} - {results['confidence_intervals']['95_percent']['upper']} points")
        print(f"\n🏦 Available Services:")
        for service, available in results['services_available'].items():
            status = "✅" if available else "❌"
            print(f"   {status} {service.replace('_', ' ').title()}")
        if results['actual_credit_score'] is not None:
            print(f"\n📋 Actual vs Predicted:")
            print(f"   Actual Score: {results['actual_credit_score']} points")
            print(f"   Error: {results['prediction_error']['absolute_error']} points ({results['prediction_error']['percentage_error']}%)")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_summary_prediction():
    """Example of getting a human-readable summary."""
    print("\n🎯 Example 2: Human-Readable Summary")
    print("=" * 50)
    predictor = CreditScorePredictor()
    tax_id = "TAX002"  # Replace with actual tax ID
    try:
        results = predictor.predict_credit_score(tax_id)
        print(f"\nCredit Score Prediction for {tax_id}")
        print("=" * 40)
        print(f"Predicted Score: {results['predicted_credit_score']} points")
        print(f"Confidence: {results['confidence_level']} ({results['confidence_score']})")
        print(f"68% Range: {results['confidence_intervals']['68_percent']['lower']} - {results['confidence_intervals']['68_percent']['upper']} points")
        print(f"95% Range: {results['confidence_intervals']['95_percent']['lower']} - {results['confidence_intervals']['95_percent']['upper']} points")
        print(f"Services Available: {[k for k, v in results['services_available'].items() if v]}")
        if results['actual_credit_score'] is not None:
            print(f"Actual Score: {results['actual_credit_score']} points")
            print(f"Error: {results['prediction_error']['absolute_error']} points ({results['prediction_error']['percentage_error']}%)")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_batch_predictions():
    """Example of predicting credit scores for multiple customers."""
    print("\n🎯 Example 3: Batch Predictions")
    print("=" * 50)
    predictor = CreditScorePredictor()
    tax_ids = ["100-58-9666", "100-24-5923", "100-13-3553"]  # Replace with actual tax IDs
    print(f"📊 Predicting credit scores for {len(tax_ids)} customers...")
    for i, tax_id in enumerate(tax_ids, 1):
        try:
            print(f"\n{i}. Customer: {tax_id}")
            print("-" * 30)
            results = predictor.predict_credit_score(tax_id)
            print(f"   Score: {results['predicted_credit_score']} points")
            print(f"   Confidence: {results['confidence_level']} ({results['confidence_score']})")
            print(f"   68% CI: {results['confidence_intervals']['68_percent']['lower']} - {results['confidence_intervals']['68_percent']['upper']}")
            services = [k for k, v in results['services_available'].items() if v]
            print(f"   Services: {', '.join(services) if services else 'None'}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def example_custom_model_path():
    """Example of using a custom model path (not needed in minimal version)."""
    print("\n🎯 Example 4: Custom Model Path")
    print("=" * 50)
    print("(Custom model path is not needed in the minimal version; using default.)")
    predictor = CreditScorePredictor()
    tax_id = "TAX001"  # Replace with actual tax ID
    try:
        results = predictor.predict_credit_score(tax_id)
        print(f"✅ Prediction successful using default model path")
        print(f"   Score: {results['predicted_credit_score']} points")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Run all examples."""
    print("🚀 VFL AutoML XGBoost Credit Score Predictor - Examples")
    print("=" * 60)
    print("This demonstrates how to use the CreditScorePredictor controller")
    print("to predict credit scores with confidence intervals.")
    print("\nNote: Replace the example tax IDs with actual tax IDs from your dataset.")
    print("=" * 60)
    example_single_prediction()
    example_summary_prediction()
    example_batch_predictions()
    example_custom_model_path()
    print("\n✅ All examples completed!")
    print("\n💡 Tips:")
    print("   - Replace 'TAX001', 'TAX002', etc. with actual tax IDs from your dataset")
    print("   - Ensure all required model files are in the saved_models/ directory")
    print("   - Ensure all dataset files are in the ../dataset/data/ directory")
    print("   - The predictor automatically handles missing bank services")

if __name__ == "__main__":
    main() 