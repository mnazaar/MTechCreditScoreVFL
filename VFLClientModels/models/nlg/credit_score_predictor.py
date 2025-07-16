"""
Simple Credit Score Predictor
============================

A minimal controller to predict credit scores for a single customer using the VFL AutoML XGBoost model.

This predictor simply calls the predict_credit_score_by_tax_id function from vfl_automl_xgboost_model.py.

Usage:
    predictor = CreditScorePredictor()
    result = predictor.predict_credit_score("TAX001")
    print(f"Credit Score: {result['predicted_credit_score']}")
"""

import sys
import os
from datetime import datetime
import logging

# Setup logger
logger = logging.getLogger("CreditScorePredictor")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Add the parent directory to path to import the main VFL model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vfl_automl_xgboost_simple import predict_with_confidence_by_tax_id

class CreditScorePredictor:
    """
    Minimal credit score predictor for a single customer using the new VFL XGBoost Simple pipeline.
    """
    def predict_credit_score(self, tax_id):
        logger.info(f"[predict_credit_score] Starting prediction for tax_id: {tax_id}")
        result = predict_with_confidence_by_tax_id(tax_id)
        logger.info(f"[predict_credit_score] Prediction complete for tax_id: {tax_id}")
        return result
    
    def predict_customer_insights(self, customer_id):
        logger.info(f"[predict_customer_insights] Start for customer_id: {customer_id}")
        start_time = datetime.now()
        try:
            prediction_result = self.predict_credit_score(customer_id)
            logger.info(f"[predict_customer_insights] Prediction result received for {customer_id}")
            
            # Check if prediction_result is None (customer not in cache)
            if prediction_result is None:
                logger.error(f"[predict_customer_insights] No prediction result for {customer_id} - customer not in cache")
                return {
                    'customer_id': customer_id,
                    'error': f'Customer {customer_id} not found in prediction cache. Please ensure the customer exists in the training data.',
                    'timestamp': datetime.now().isoformat(),
                    'status': 'failed'
                }
            
            predicted_score = prediction_result['predicted']
            ci68 = prediction_result['68_CI']
            ci95 = prediction_result['95_CI']
            actual_score = prediction_result.get('actual', None)
            adjusted_score = predicted_score
            adjustment_made = False
            ci_95_lower, ci_95_upper = ci95
            if predicted_score < ci_95_lower or predicted_score > ci_95_upper:
                adjusted_score = (ci_95_lower + ci_95_upper) / 2
                adjustment_made = True
            else:
                ci_68_lower, ci_68_upper = ci68
                if predicted_score < ci_68_lower or predicted_score > ci_68_upper:
                    adjusted_score = (ci_95_lower + ci_95_upper) / 2
                    adjustment_made = True
            logger.info(f"[predict_customer_insights] CI adjustment done for {customer_id}")
            insights = {
                'customer_id': customer_id,
                'predicted_credit_score': adjusted_score,
                'score_adjusted': adjustment_made,
                'confidence_intervals': {
                    '68_percent': {'lower': ci68[0], 'upper': ci68[1]},
                    '95_percent': {'lower': ci95[0], 'upper': ci95[1]}
                },
                'timestamp': datetime.now().isoformat(),
            }
            if actual_score is not None:
                insights['actual_credit_score'] = actual_score
                absolute_error = abs(adjusted_score - actual_score)
                percentage_error = (absolute_error / actual_score) * 100 if actual_score > 0 else 0
                insights['prediction_error'] = {
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error
                }
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"[predict_customer_insights] Finished for {customer_id} in {elapsed:.2f}s")
            return insights
        except Exception as e:
            logger.error(f"[predict_customer_insights] Exception for {customer_id}: {e}")
            return {
                'customer_id': customer_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }

def main():
    """Test the minimal credit score predictor."""
    print("🎯 VFL AutoML Credit Score Predictor (Minimal)")
    print("=" * 40)
    try:
        predictor = CreditScorePredictor()
        tax_id = "TAX001"  # Replace with actual tax ID from your dataset
        print(f"\n🔍 Predicting credit score for: {tax_id}")
        result = predictor.predict_credit_score(tax_id)
        print(f"\n📊 RESULTS:")
        print(f"   Predicted Score: {result['predicted']}")
        print(f"   68% CI: {result['68_CI'][0]} - {result['68_CI'][1]}")
        print(f"   95% CI: {result['95_CI'][0]} - {result['95_CI'][1]}")
        if result.get('actual') is not None:
            print(f"   Actual Score: {result['actual']}")
            abs_err = abs(result['predicted'] - result['actual'])
            pct_err = (abs_err / result['actual']) * 100 if result['actual'] > 0 else 0
            print(f"   Error: {abs_err} points ({pct_err:.2f}%)")
        print(f"\n✅ AutoML model prediction completed successfully!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 