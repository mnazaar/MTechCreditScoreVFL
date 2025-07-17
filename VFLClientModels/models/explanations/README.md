# VFL Model Explanations

This directory contains explanation methods for the VFL (Vertical Federated Learning) models.

## 📁 Files

- `__init__.py` - Package initialization
- `credit_card_lime_explainer.py` - LIME explanations for Credit Card XGBoost model
- `run_lime_explanations.py` - Simple script to run LIME explanations
- `README.md` - This file

## 🔍 LIME Explanations

### What is LIME?
LIME (Local Interpretable Model-agnostic Explanations) provides local explanations for individual predictions by approximating the model's behavior around a specific instance.

### Features
- **Print Only Mode**: No files saved to disk - all results printed to console
- **Privacy Preserving**: Works with VFL framework without exposing raw data
- **Individual Customer Analysis**: Shows top features impacting each customer's decision
- **Confidence Assessment**: Provides risk levels and confidence scores

### Installation
```bash
pip install lime
```

### Usage

#### Method 1: Direct Python Script
```bash
cd VFLClientModels/models/explanations
python credit_card_lime_explainer.py
```

#### Method 2: Using the Runner Script
```bash
cd VFLClientModels/models/explanations
python run_lime_explanations.py
```

### Example Output
```
🔍 LIME EXPLANATION REPORT - Customer 12345
========================================
🎯 PREDICTION SUMMARY:
   Customer ID: 12345
   Predicted Card Tier: Premium
   Confidence: 87.3%
   LIME Samples Used: 1,000

🎯 TOP FEATURES IMPACTING DECISION:
Rank Feature Name              Weight     Impact         Direction
1    credit_score              0.125      Very High      Positive
2    annual_income             0.098      High           Positive
3    credit_utilization_ratio  -0.076     High           Negative
```

### Key Features Explained

#### Impact Levels
- **Very High**: Weight ≥ 0.1 (dominant features)
- **High**: Weight ≥ 0.05 (important features)
- **Medium**: Weight ≥ 0.02 (moderate features)
- **Low**: Weight < 0.02 (minor features)

#### Direction
- **Positive**: Feature increases predicted class probability
- **Negative**: Feature decreases predicted class probability

#### Risk Assessment
- **Very Low**: Confidence ≥ 90%
- **Low**: Confidence ≥ 80%
- **Medium**: Confidence ≥ 70%
- **High**: Confidence < 70%

### Privacy Features
- ✅ No raw feature values exposed
- ✅ Only feature names and relative importance shown
- ✅ Compatible with VFL privacy requirements
- ✅ Local model predictions only

### VFL Integration
- Works with existing credit card XGBoost model
- Can be extended to other banks in the VFL system
- Maintains federated learning privacy constraints
- No cross-bank data sharing

## 🔧 Customization

### Modify Number of Features
Change `num_features` parameter in the explainer:
```python
explanation = lime_explainer.explain_prediction(
    customer_data, customer_id, num_features=15  # Show top 15 features
)
```

### Modify Number of LIME Samples
Change `num_samples` parameter for accuracy vs speed trade-off:
```python
explanation = lime_explainer.explain_prediction(
    customer_data, customer_id, num_samples=2000  # More samples = more accurate
)
```

### Test Different Customers
Modify the test customer selection in the main function:
```python
test_customers = 10  # Test 10 customers instead of 5
```

## 🚨 Troubleshooting

### LIME Not Installed
```
⚠️  LIME not available. Install with: pip install lime
```

### Model Not Found
Ensure the trained model exists at:
```
saved_models/credit_card_xgboost_independent.pkl
```

### Memory Issues
Reduce the number of LIME samples:
```python
num_samples=500  # Instead of 1000
```

## 📊 Performance Notes

- **Speed**: ~2-5 seconds per customer explanation
- **Memory**: ~100MB for explainer initialization
- **Accuracy**: Improves with more LIME samples
- **Scalability**: Can handle thousands of customers

## 🔄 Future Enhancements

- [ ] SHAP explanations for deeper insights
- [ ] Multi-bank explanation aggregation
- [ ] Interactive visualization (if needed)
- [ ] Batch processing for large datasets 