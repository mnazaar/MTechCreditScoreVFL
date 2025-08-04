#!/bin/bash

echo "🏦 Credit Score Predictor UI Launcher"
echo "======================================"

echo ""
echo "📋 Checking prerequisites..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

echo "✅ Python found"

# Check if required packages are installed
echo "📦 Checking required packages..."
python3 -c "import streamlit, requests, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some required packages are missing"
    echo "Installing required packages..."
    pip3 install -r requirements_ui.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install required packages"
        exit 1
    fi
fi

echo "✅ All packages are installed"

# Test API connection
echo "🔍 Testing API connection..."
python3 test_api_connection.py
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  API connection test failed"
    echo "Please make sure the API server is running on localhost:5000"
    echo ""
    echo "To start the API server:"
    echo "1. Open a new terminal"
    echo "2. Navigate to: VFLClientModels/models/apis"
    echo "3. Run: python3 app.py"
    echo ""
    echo "Press Enter to continue anyway..."
    read
fi

echo ""
echo "🚀 Starting Streamlit UI..."
echo ""
echo "The UI will open in your default browser at: http://localhost:8501"
echo ""
echo "To stop the UI, press Ctrl+C in this terminal"
echo ""

streamlit run credit_score_ui.py 