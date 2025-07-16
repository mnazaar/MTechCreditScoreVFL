# Credit Score Predictor Controller

This directory contains a high-level controller for predicting credit scores using the trained VFL AutoML XGBoost model.

## Overview

The `CreditScorePredictor` class provides a simple interface to predict credit scores for customers by leveraging all the existing infrastructure from the main VFL AutoML XGBoost model. It acts as a controller that:

- Takes a tax ID as input
- Uses existing data loading and preprocessing methods
- Extracts representations from all four banks (3 Neural Networks + 1 XGBoost)
- Predicts credit score with confidence intervals
- Returns structured results

## Files

- `credit_score_predictor.py` - Main controller class
- `example_usage.py` - Example usage demonstrations
- `README.md` - This documentation file

## Features

### 🎯 Core Functionality
- **Single Customer Prediction**: Predict credit score for any customer by tax ID
- **Confidence Intervals**: 68% and 95% confidence ranges for predictions
- **Service Availability**: Shows which bank services the customer has
- **Error Analysis**: Compares predicted vs actual scores when available

### 🏦 Bank Integration
- **Auto Loans Bank**: Neural network (16D representations)
- **Digital Banking Bank**: Neural network (8D representations)  
- **Home Loans Bank**: Neural network (16D representations)
- **Credit Card Bank**: XGBoost (8D representations)

### 📊 Confidence Scoring
- **Monte Carlo Dropout**: Uncertainty estimation using MC dropout
- **Confidence Levels**: Very High, High, Medium, Low, Very Low
- **Prediction Uncertainty**: Standard deviation of predictions
- **Calibrated Intervals**: 68% and 95% confidence intervals

## Usage

### Basic Usage

```python
from credit_score_predictor import CreditScorePredictor

# Initialize the predictor
predictor = CreditScorePredictor()

# Predict credit score for a customer
tax_id = "TAX001"  # Replace with actual tax ID
results = predictor.predict_credit_score(tax_id)

# Access results
print(f"Predicted Score: {results['predicted_credit_score']}")
print(f"Confidence: {results['confidence_level']}")
print(f"68% CI: {results['confidence_intervals']['68_percent']['lower']} - {results['confidence_intervals']['68_percent']['upper']}")
```

### Getting a Summary

```python
# Get human-readable summary
summary = predictor.get_prediction_summary(tax_id)
print(summary)
```

### Custom Model Path

```python
# Use a different model file
predictor = CreditScorePredictor(model_path='path/to/your/model.keras')
```

## Output Format

The `predict_credit_score()` method returns a dictionary with the following structure:

```python
{
    'tax_id': 'TAX001',
    'predicted_credit_score': 725.3,
    'confidence_score': 0.847,
    'confidence_level': 'High',
    'prediction_uncertainty': 12.5,
    'confidence_intervals': {
        '68_percent': {
            'lower': 712.8,
            'upper': 737.8,
            'width': 25.0
        },
        '95_percent': {
            'lower': 700.3,
            'upper': 750.3,
            'width': 50.0
        }
    },
    'services_available': {
        'auto_loans': True,
        'digital_banking': False,
        'home_loans': True,
        'credit_card': True
    },
    'actual_credit_score': 730,  # If available
    'prediction_error': {         # If actual score available
        'absolute_error': 4.7,
        'percentage_error': 0.6
    },
    'prediction_timestamp': '2024-01-15T10:30:00',
    'model_info': {
        'model_path': 'saved_models/vfl_automl_xgboost_final_model.keras',
        'model_type': 'VFL AutoML XGBoost (3 NN + 1 XGBoost)',
        'representation_sizes': {
            'auto': 16,
            'digital': 8,
            'home': 16,
            'credit_card': 8
        }
    }
}
```

## Requirements

### Model Files
The predictor requires the following trained models in the `saved_models/` directory:

- `vfl_automl_xgboost_final_model.keras` - Main VFL model
- `auto_loans_model.keras` - Auto loans neural network
- `digital_bank_model.keras` - Digital banking neural network
- `home_loans_model.keras` - Home loans neural network
- `credit_card_xgboost_independent.pkl` - Credit card XGBoost model
- `credit_card_scaler.pkl` - Credit card scaler
- `credit_card_feature_names.npy` - Credit card feature names

### Dataset Files
The predictor requires the following dataset files in the `../dataset/data/` directory:

- `banks/auto_loans_bank.csv`
- `banks/digital_savings_bank.csv`
- `banks/home_loans_bank.csv`
- `banks/credit_card_bank.csv`
- `credit_scoring_dataset.csv`

### Python Dependencies
- tensorflow
- keras
- pandas
- numpy
- joblib
- scikit-learn

## Examples

### Example 1: Single Prediction
```python
from credit_score_predictor import CreditScorePredictor

predictor = CreditScorePredictor()
results = predictor.predict_credit_score("TAX001")

print(f"Credit Score: {results['predicted_credit_score']} points")
print(f"Confidence: {results['confidence_level']}")
print(f"68% Range: {results['confidence_intervals']['68_percent']['lower']} - {results['confidence_intervals']['68_percent']['upper']}")
```

### Example 2: Batch Predictions
```python
tax_ids = ["TAX001", "TAX002", "TAX003"]
predictor = CreditScorePredictor()

for tax_id in tax_ids:
    try:
        results = predictor.predict_credit_score(tax_id)
        print(f"{tax_id}: {results['predicted_credit_score']} points")
    except Exception as e:
        print(f"{tax_id}: Error - {e}")
```

### Example 3: Summary Report
```python
predictor = CreditScorePredictor()
summary = predictor.get_prediction_summary("TAX001")
print(summary)
```

## Error Handling

The predictor handles various error conditions:

- **Customer Not Found**: Raises `ValueError` if tax ID not found in any dataset
- **Missing Model Files**: Raises `FileNotFoundError` if required models are missing
- **Missing Dataset Files**: Raises `FileNotFoundError` if required datasets are missing
- **Invalid Tax ID**: Provides clear error messages for invalid inputs

## Performance Notes

- **Initialization**: Takes time to load all models and infrastructure (~30-60 seconds)
- **Prediction**: Fast once initialized (~1-2 seconds per prediction)
- **Memory Usage**: Moderate due to loading multiple neural networks
- **Scalability**: Designed for single customer predictions, not batch processing

## Integration

The controller is designed to be easily integrated into:

- **Web APIs**: Use as a service endpoint
- **Batch Processing**: Loop through multiple tax IDs
- **Reporting Systems**: Generate credit score reports
- **Decision Systems**: Integrate into loan approval workflows

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure all model files are in `saved_models/` directory
2. **Dataset Not Found**: Ensure all CSV files are in `../dataset/data/` directory
3. **Import Errors**: Check that all dependencies are installed
4. **Memory Issues**: Close other applications if running out of memory

### Debug Mode

Enable detailed logging by modifying the logging level in the main VFL model:

```python
# In vfl_automl_xgboost_model.py
logger.setLevel(logging.DEBUG)
```

## License

This controller is part of the VFL AutoML XGBoost credit scoring system and follows the same license terms as the main project. 