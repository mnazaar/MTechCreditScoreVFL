# Auto Loans Feature Prediction API

This Flask API provides endpoints to predict the top 3 features that influence a customer's auto loan credit score prediction.

## Features

- **Health Check**: Verify API status
- **Customer Prediction**: Get intermediate representations and predicted features for a customer ID
- **Customer List**: Get available customer IDs

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the required model files are in the correct locations:
   - `../saved_models/auto_loans_model.keras`
   - `../explanations/privateexplanations/saved_models/auto_loans_feature_predictor.keras`
   - `../saved_models/auto_loans_feature_names.npy`
   - `../saved_models/auto_loans_scaler.pkl` (optional)
   - `../../dataset/data/banks/auto_loans_bank.csv`

## Usage

### Start the API Server

```bash
python app.py
```

The API will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-05T14:30:00.000000",
  "models_loaded": true
}
```

#### 2. Predict Features for Customer
```bash
POST /predict
Content-Type: application/json

{
  "customer_id": "12345"
}
```

Response:
```json
{
  "customer_id": "12345",
  "intermediate_representation": [0.1, 0.2, 0.3, ...],
  "predicted_features": [
    {
      "feature_name": "credit_score",
      "direction": "Positive",
      "impact": "High",
      "confidence": {
        "index": 0.95,
        "direction": 0.88,
        "impact": 0.92
      }
    },
    {
      "feature_name": "annual_income",
      "direction": "Positive",
      "impact": "Medium",
      "confidence": {
        "index": 0.87,
        "direction": 0.91,
        "impact": 0.85
      }
    },
    {
      "feature_name": "payment_history",
      "direction": "Negative",
      "impact": "Low",
      "confidence": {
        "index": 0.78,
        "direction": 0.82,
        "impact": 0.79
      }
    }
  ],
  "timestamp": "2025-07-05T14:30:00.000000"
}
```

#### 3. List Available Customers
```bash
GET /customers
```

Response:
```json
{
  "customer_ids": ["12345", "12346", "12347", ...],
  "total_customers": 400000,
  "sample_size": 100
}
```

## Error Responses

### Customer Not Found
```json
{
  "error": "Customer with ID 99999 not found"
}
```

### Missing Customer ID
```json
{
  "error": "Missing customer_id in request body"
}
```

### Models Not Loaded
```json
{
  "error": "Models not loaded. Please check server logs."
}
```

## Example Usage with curl

```bash
# Health check
curl http://localhost:5000/health

# Get customer list
curl http://localhost:5000/customers

# Predict features for customer
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "12345"}'
```

## Example Usage with Python

```python
import requests
import json

# API base URL
base_url = "http://localhost:5000"

# Health check
response = requests.get(f"{base_url}/health")
print("Health:", response.json())

# Get customer list
response = requests.get(f"{base_url}/customers")
customers = response.json()
print("Available customers:", customers['customer_ids'][:5])

# Predict features for first customer
customer_id = customers['customer_ids'][0]
response = requests.post(
    f"{base_url}/predict",
    json={"customer_id": customer_id}
)
prediction = response.json()
print(f"Prediction for customer {customer_id}:")
for i, feature in enumerate(prediction['predicted_features'], 1):
    print(f"  {i}. {feature['feature_name']} ({feature['direction']}, {feature['impact']})")
```

## Logs

Logs are saved in the `logs/` directory with timestamps for debugging and monitoring.

## Notes

- The API loads all models and data on startup for better performance
- Customer IDs can be either tax_id values or numeric indices
- The intermediate representation is a 16-dimensional vector from the penultimate layer
- Predicted features include confidence scores for each prediction component 