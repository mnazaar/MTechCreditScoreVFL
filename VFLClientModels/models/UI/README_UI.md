# Credit Score Predictor UI

A beautiful Streamlit web application that provides an intuitive interface for the Credit Score Prediction API.

## Features

- 🎯 **Credit Score Prediction**: Get predicted credit scores with confidence intervals
- 📊 **Visual Analytics**: Beautiful charts and metrics display
- 📝 **Natural Language Explanations**: Detailed insights for each financial product
- 🔍 **Feature Impact Analysis**: See which factors most influence the credit score
- 🛠️ **Debug Mode**: Toggle to see actual scores and prediction errors
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

1. **API Server**: Make sure the credit score API is running on `localhost:5000`
2. **Python Dependencies**: Install the required packages

## Installation

1. Install the required packages:
```bash
pip install -r requirements_ui.txt
```

2. Start the API server (if not already running):
```bash
cd VFLClientModels/models/apis
python app.py
```

## Running the UI

1. Start the Streamlit application:
```bash
streamlit run credit_score_ui.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **Enter Customer ID**: Input a customer's tax ID (e.g., "100-16-1590")
2. **Configure Options**:
   - **Debug Mode**: Enable to see actual credit scores and prediction errors
   - **Natural Language Explanations**: Enable for detailed product-specific insights
3. **Click "Predict Credit Score"**: Get comprehensive analysis

## What You'll See

### Main Results
- **Predicted Credit Score**: The primary prediction with visual emphasis
- **Confidence Intervals**: 68% and 95% confidence ranges
- **Timestamp**: When the analysis was performed

### Product Explanations (if enabled)
- **Auto Loan**: Insights specific to auto loan applications
- **Credit Card**: Credit card-specific analysis
- **Digital Savings**: Digital banking insights
- **Home Loan**: Mortgage and home loan considerations

### Feature Impact Analysis
- **Positive Factors**: Features that improve the credit score (green border)
- **Negative Factors**: Features that reduce the credit score (red border)
- **Impact Levels**: Very High, High, Medium, Low

### Debug Information (if enabled)
- **Actual Credit Score**: The true credit score (if available)
- **Prediction Error**: Absolute and percentage error metrics

## API Endpoint

The UI calls the `/credit-score/customer-insights` endpoint with the following request format:

```json
{
    "customer_id": "100-16-1590",
    "debug": true,
    "nl_explanation": true
}
```

## Example Response

The API returns comprehensive data including:
- Predicted and actual credit scores
- Confidence intervals
- Natural language explanations for each product
- Feature-level impact analysis
- Prediction error metrics (in debug mode)

## Troubleshooting

1. **API Connection Error**: Ensure the API server is running on port 5000
2. **Customer Not Found**: Try different customer IDs from the training dataset
3. **Slow Response**: The API may take time for complex predictions

## Customization

You can modify the UI by editing `credit_score_ui.py`:
- Change the API URL in the `call_credit_score_api` function
- Modify the styling in the CSS section
- Add new visualization components
- Customize the product names and icons

## Dependencies

- `streamlit`: Web application framework
- `requests`: HTTP library for API calls
- `pandas`: Data manipulation (for future enhancements)

## License

This UI is part of the MTech Credit Score VFL project. 