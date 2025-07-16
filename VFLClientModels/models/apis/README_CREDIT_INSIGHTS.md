# Credit Score Customer Insights API

## Overview

The `/credit-score/customer-insights` endpoint provides comprehensive credit score predictions and insights for individual customers using the VFL AutoML XGBoost model. This endpoint leverages the existing `CreditScorePredictor` functionality and provides both production and debug modes.

## Endpoint Details

- **URL**: `/credit-score/customer-insights`
- **Method**: `POST`
- **Content-Type**: `application/json`

## Request Format

```json
{
  "customer_id": "string",
  "debug": boolean
}
```

### Parameters

- `customer_id` (required): The customer's tax ID (e.g., "100-13-3553")
- `debug` (optional): Boolean flag to control debug information output
  - `false` (default): Production mode - excludes actual credit score and prediction error
  - `true`: Debug mode - includes all available information

## Response Format

### Production Mode (debug=false)

```json
{
  "customer_id": "100-13-3553",
  "predicted_credit_score": 725.3,
  "confidence_level": "High",
  "confidence_score": 0.847,
  "prediction_uncertainty": 12.5,
  "confidence_intervals": {
    "68_percent": {
      "lower": 712.8,
      "upper": 737.8,
      "width": 25.0
    },
    "95_percent": {
      "lower": 700.3,
      "upper": 750.3,
      "width": 50.0
    }
  },
  "available_services": {
    "auto_loans": true,
    "digital_banking": false,
    "home_loans": true,
    "credit_card": true
  },
  "timestamp": "2025-01-08T21:48:13.433Z",
  "model_info": {
    "model_type": "VFL AutoML XGBoost (3 NN + 1 XGBoost)",
    "representation_sizes": {
      "auto": 16,
      "digital": 8,
      "home": 16,
      "credit_card": 12
    }
  }
}
```

### Debug Mode (debug=true)

Includes all production information plus:

```json
{
  // ... production fields ...
  "actual_credit_score": 730,
  "prediction_error": {
    "absolute_error": 4.7,
    "percentage_error": 0.6
  }
}
```

## Usage Examples

### Python

```python
import requests

# Production mode
response = requests.post(
    "http://localhost:5000/credit-score/customer-insights",
    json={
        "customer_id": "100-13-3553",
        "debug": False
    },
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    data = response.json()
    print(f"Predicted Credit Score: {data['predicted_credit_score']}")
    print(f"Confidence Level: {data['confidence_level']}")
else:
    print(f"Error: {response.json()['error']}")

# Debug mode
response = requests.post(
    "http://localhost:5000/credit-score/customer-insights",
    json={
        "customer_id": "100-13-3553",
        "debug": True
    },
    headers={"Content-Type": "application/json"}
)
```

### cURL

```bash
# Production mode
curl -X POST http://localhost:5000/credit-score/customer-insights \
     -H 'Content-Type: application/json' \
     -d '{"customer_id": "100-13-3553", "debug": false}'

# Debug mode
curl -X POST http://localhost:5000/credit-score/customer-insights \
     -H 'Content-Type: application/json' \
     -d '{"customer_id": "100-13-3553", "debug": true}'
```

### JavaScript/Fetch

```javascript
// Production mode
fetch('http://localhost:5000/credit-score/customer-insights', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        customer_id: '100-13-3553',
        debug: false
    })
})
.then(response => response.json())
.then(data => {
    console.log('Predicted Credit Score:', data.predicted_credit_score);
    console.log('Confidence Level:', data.confidence_level);
})
.catch(error => console.error('Error:', error));
```

## Error Responses

### Missing Customer ID

```json
{
  "error": "Missing customer_id in request body"
}
```

### Customer Not Found

```json
{
  "error": "Customer with ID INVALID-ID does not have any services",
  "customer_id": "INVALID-ID"
}
```

### Prediction Failed

```json
{
  "error": "Prediction failed",
  "customer_id": "100-13-3553"
}
```

### Internal Server Error

```json
{
  "error": "Internal server error: [error details]"
}
```

## Testing

### Run the Test Suite

```bash
cd VFLClientModels/models/apis
python test_credit_insights_api.py
```

### Manual Testing

1. Start the API server:
   ```bash
   cd VFLClientModels/models/apis
   python app.py
   ```

2. Test with curl:
   ```bash
   # Health check
   curl http://localhost:5000/health
   
   # Production mode
   curl -X POST http://localhost:5000/credit-score/customer-insights \
        -H 'Content-Type: application/json' \
        -d '{"customer_id": "100-13-3553", "debug": false}'
   
   # Debug mode
   curl -X POST http://localhost:5000/credit-score/customer-insights \
        -H 'Content-Type: application/json' \
        -d '{"customer_id": "100-13-3553", "debug": true}'
   ```

## Model Information

The endpoint uses the VFL AutoML XGBoost model which combines:

- **Auto Loans Bank**: Neural Network (16D representations)
- **Digital Banking Bank**: Neural Network (8D representations)  
- **Home Loans Bank**: Neural Network (16D representations)
- **Credit Card Bank**: XGBoost (12D representations)

## Confidence Scoring

The model provides confidence scores based on:
- Monte Carlo Dropout sampling (30 samples by default)
- Prediction uncertainty estimation
- Confidence intervals (68% and 95%)

## Available Services

The response includes information about which banking services the customer has:
- `auto_loans`: Auto loan services
- `digital_banking`: Digital banking services
- `home_loans`: Home loan services
- `credit_card`: Credit card services

## Notes

- The endpoint reuses the existing `CreditScorePredictor` functionality
- Debug mode should only be used in development/testing environments
- Production mode excludes sensitive information like actual credit scores
- All timestamps are in ISO 8601 format
- Confidence scores range from 0.0 to 1.0
- Credit scores are predicted on a 300-850 scale 