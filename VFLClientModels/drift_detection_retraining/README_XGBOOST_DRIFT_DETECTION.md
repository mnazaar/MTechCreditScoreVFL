# XGBoost Drift Detection Module

## Overview

The XGBoost Drift Detection Module is a specialized component for detecting data drift in XGBoost credit card models. It leverages the existing generic drift detection infrastructure while providing XGBoost-specific analysis capabilities.

## Key Features

- **Separation of Concerns**: Drift detection is separate from retraining logic
- **XGBoost-Specific Analysis**: Includes leaf distribution drift detection
- **Comprehensive Reporting**: Detailed drift analysis reports
- **Easy Integration**: Simple API for integration with monitoring pipelines
- **Reusable**: Can be used with different retraining pipelines

## Architecture

```
XGBoostDriftDetector
├── Generic Drift Detection (from drift_detection.py)
│   ├── Statistical Drift (KS Test)
│   ├── Performance Drift (Confidence)
│   ├── Distribution Drift (Wasserstein)
│   └── Prediction Drift (KS Test)
└── XGBoost-Specific Analysis
    └── Leaf Distribution Drift (Chi-square Test)
```

## Files

- `xgboost_drift_detector.py` - Main drift detection module
- `example_xgboost_drift_detection.py` - Usage examples
- `test_xgboost_drift_detection.py` - Test suite
- `README_XGBOOST_DRIFT_DETECTION.md` - This documentation

## Quick Start

### Basic Usage

```python
from xgboost_drift_detector import XGBoostDriftDetector

# Initialize detector
detector = XGBoostDriftDetector(
    model_path='VFLClientModels/saved_models/credit_card_xgboost_independent.pkl'
)

# Load your data
current_data = pd.read_csv('current_data.csv')
baseline_data = pd.read_csv('baseline_data.csv')

# Detect drift
drift_detected, report = detector.is_drift_detected(
    current_data=current_data,
    baseline_data=baseline_data
)

if drift_detected:
    print("🔄 Drift detected! Consider retraining.")
else:
    print("✅ No drift detected. Model is stable.")
```

### Convenience Function

```python
from xgboost_drift_detector import detect_credit_card_xgboost_drift

drift_detected, report = detect_credit_card_xgboost_drift(
    current_data_path='current_data.csv',
    baseline_data_path='baseline_data.csv',
    model_path='VFLClientModels/saved_models/credit_card_xgboost_independent.pkl'
)
```

## API Reference

### XGBoostDriftDetector Class

#### Constructor
```python
XGBoostDriftDetector(model_path=None, config_path=None)
```

**Parameters:**
- `model_path`: Path to saved XGBoost model (optional)
- `config_path`: Path to drift detection configuration (optional)

#### Methods

##### `load_model(model_path)`
Load XGBoost model and related artifacts.

##### `detect_xgboost_drift(current_data, baseline_data, target_column=None)`
Perform comprehensive drift detection.

**Returns:** Dictionary with drift detection results

##### `is_drift_detected(current_data, baseline_data, target_column=None)`
Simple drift detection with report generation.

**Returns:** Tuple of (drift_detected: bool, report: str)

##### `generate_xgboost_drift_report(drift_results)`
Generate comprehensive drift report.

**Returns:** Formatted report string

##### `get_drift_summary(drift_results)`
Get summary of drift detection results.

**Returns:** Dictionary with drift summary

## Drift Detection Types

### 1. Statistical Drift (KS Test)
- Compares feature distributions between baseline and current data
- Uses Kolmogorov-Smirnov test
- Threshold: p-value < 0.1

### 2. Performance Drift (Confidence)
- Monitors model confidence changes
- Tracks average prediction confidence
- Threshold: 15% confidence drift

### 3. Distribution Drift (Wasserstein)
- Measures distribution similarity using Wasserstein distance
- Normalized by feature range
- Threshold: 20% normalized distance

### 4. Prediction Drift (KS Test)
- Compares prediction distributions
- Uses KS test on prediction values
- Threshold: p-value < 0.25

### 5. Feature Importance Drift
- Tracks changes in feature importance patterns
- Compares importance scores
- Threshold: 30% importance change

### 6. XGBoost-Specific Drift (Leaf Distribution)
- Analyzes leaf node distribution changes
- Uses Chi-square test on leaf distributions
- XGBoost-specific drift pattern detection

## Configuration

### Default Thresholds
```python
xgboost_thresholds = {
    'statistical_drift': 0.1,      # KS test p-value threshold
    'performance_drift': 0.15,     # Confidence drift threshold
    'distribution_drift': 0.2,     # Distribution similarity threshold
    'prediction_drift': 0.25,      # Prediction distribution drift
    'feature_importance_drift': 0.3 # Feature importance change threshold
}
```

### Custom Configuration
Create a JSON configuration file:
```json
{
    "drift_thresholds": {
        "statistical_drift": 0.05,
        "performance_drift": 0.1,
        "distribution_drift": 0.15,
        "prediction_drift": 0.2,
        "feature_importance_drift": 0.25
    },
    "monitoring_frequency": "daily",
    "retrain_threshold": 3
}
```

## Integration with Retraining Pipeline

### Example Integration
```python
from xgboost_drift_detector import XGBoostDriftDetector

def monitor_and_retrain():
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
        print("🔄 Drift detected! Initiating retraining...")
        # Call your retraining function
        retrain_credit_card_xgboost_model()
        
        # Update baseline data after retraining
        current_data.to_csv('baseline_data.csv', index=False)
        print("✅ Baseline data updated")
    else:
        print("✅ No drift detected. Model is stable.")
    
    return drift_detected, report
```

## Testing

### Run Tests
```bash
cd VFLClientModels/drift_detection_retraining
python test_xgboost_drift_detection.py
```

### Run Examples
```bash
cd VFLClientModels/drift_detection_retraining
python example_xgboost_drift_detection.py
```

## Logging

The module provides comprehensive logging:
- Console output for real-time monitoring
- File logging for historical analysis
- Log files: `VFLClientModels/logs/xgboost_drift_detection_YYYYMMDD.log`

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- joblib
- xgboost (for model loading)

## Error Handling

The module includes robust error handling:
- Graceful handling of missing model files
- Validation of input data
- Comprehensive error messages
- Fallback mechanisms for drift detection

## Performance Considerations

- Statistical tests are computationally efficient
- Leaf distribution analysis scales with tree count
- Memory usage scales with data size
- Recommended batch size: 1000-10000 samples

## Best Practices

1. **Regular Monitoring**: Run drift detection daily/weekly
2. **Baseline Updates**: Update baseline data after successful retraining
3. **Threshold Tuning**: Adjust thresholds based on your domain
4. **Data Quality**: Ensure input data quality before drift detection
5. **Model Validation**: Validate retrained models before deployment

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure model file exists
   - Check file permissions
   - Verify model format

2. **Feature Mismatch**
   - Ensure current data has all required features
   - Check feature names match training data

3. **Memory Issues**
   - Reduce batch size for large datasets
   - Use data sampling for initial testing

### Debug Mode
Enable debug logging:
```python
import logging
logging.getLogger('XGBoostDriftDetection').setLevel(logging.DEBUG)
```

## Contributing

When contributing to the drift detection module:
1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility

## License

This module is part of the VFL Credit Scoring project and follows the same license terms. 