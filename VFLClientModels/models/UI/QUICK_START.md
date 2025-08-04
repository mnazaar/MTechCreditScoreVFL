# 🚀 Quick Start Guide - Credit Score Predictor UI

## What You Get

A beautiful Streamlit web application that provides an intuitive interface for your credit score prediction API. The UI displays:

- 📊 **Predicted Credit Score** with visual emphasis
- 🎯 **Confidence Intervals** (68% and 95%)
- 📝 **Natural Language Explanations** for each financial product
- 🔍 **Feature Impact Analysis** showing which factors influence the score
- 🛠️ **Debug Mode** to see actual scores and prediction errors

## Files Created

1. **`credit_score_ui.py`** - Main Streamlit application
2. **`requirements_ui.txt`** - Python dependencies
3. **`test_api_connection.py`** - API connection test script
4. **`run_ui.bat`** - Windows launcher script
5. **`run_ui.sh`** - Linux/Mac launcher script
6. **`README_UI.md`** - Detailed documentation
7. **`QUICK_START.md`** - This quick start guide

## Quick Start (Windows)

1. **Start the API Server** (in a new command prompt):
   ```cmd
   cd VFLClientModels\models\apis
   python app.py
   ```

2. **Run the UI** (in another command prompt):
   ```cmd
   run_ui.bat
   ```
   Or manually:
   ```cmd
   pip install -r requirements_ui.txt
   streamlit run credit_score_ui.py
   ```

3. **Open your browser** to `http://localhost:8501`

## Quick Start (Linux/Mac)

1. **Start the API Server** (in a new terminal):
   ```bash
   cd VFLClientModels/models/apis
   python3 app.py
   ```

2. **Run the UI** (in another terminal):
   ```bash
   ./run_ui.sh
   ```
   Or manually:
   ```bash
   pip3 install -r requirements_ui.txt
   streamlit run credit_score_ui.py
   ```

3. **Open your browser** to `http://localhost:8501`

## How to Use

1. **Enter Customer ID**: Type a customer's tax ID (e.g., "100-16-1590")
2. **Configure Options**:
   - ✅ **Debug Mode**: Enable to see actual scores and errors
   - ✅ **Natural Language Explanations**: Enable for detailed insights
3. **Click "Predict Credit Score"**: Get comprehensive analysis

## What You'll See

### Main Results
- **Predicted Credit Score**: Large, prominent display
- **Confidence Intervals**: Range of possible scores
- **Timestamp**: When analysis was performed

### Product Explanations (if enabled)
- 🚗 **Auto Loan**: Auto loan-specific insights
- 💳 **Credit Card**: Credit card analysis
- 🏦 **Digital Savings**: Digital banking insights
- 🏠 **Home Loan**: Mortgage considerations

### Feature Impact Analysis
- 📈 **Positive Factors**: Green border, improve score
- 📉 **Negative Factors**: Red border, reduce score
- **Impact Levels**: Very High, High, Medium, Low

### Debug Information (if enabled)
- **Actual Credit Score**: True score (if available)
- **Prediction Error**: Absolute and percentage errors

## API Endpoint

The UI calls: `POST http://localhost:5000/credit-score/customer-insights`

**Request Format:**
```json
{
    "customer_id": "100-16-1590",
    "debug": true,
    "nl_explanation": true
}
```

## Troubleshooting

### API Connection Issues
- Run `python test_api_connection.py` to test the API
- Ensure the API server is running on port 5000
- Check firewall settings

### Package Issues
- Run `pip install -r requirements_ui.txt`
- Make sure you have Python 3.7+ installed

### Customer Not Found
- Try different customer IDs from your training dataset
- Check the API logs for specific error messages

## Customization

You can modify the UI by editing `credit_score_ui.py`:
- Change API URL in `call_credit_score_api()` function
- Modify styling in the CSS section
- Add new visualization components
- Customize product names and icons

## Example Response

The API returns comprehensive data including:
- Predicted and actual credit scores
- Confidence intervals
- Natural language explanations for each product
- Feature-level impact analysis
- Prediction error metrics (in debug mode)

---

**🎉 That's it! You now have a beautiful, functional UI for your credit score prediction API!** 