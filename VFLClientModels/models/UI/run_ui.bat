@echo off
echo 🏦 Credit Score Predictor UI Launcher
echo ======================================

echo.
echo 📋 Checking prerequisites...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python and try again
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if required packages are installed
echo 📦 Checking required packages...
python -c "import streamlit, requests, pandas" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some required packages are missing
    echo Installing required packages...
    pip install -r requirements_ui.txt
    if errorlevel 1 (
        echo ❌ Failed to install required packages
        pause
        exit /b 1
    )
)

echo ✅ All packages are installed

REM Test API connection
echo 🔍 Testing API connection...
python test_api_connection.py
if errorlevel 1 (
    echo.
    echo ⚠️  API connection test failed
    echo Please make sure the API server is running on localhost:5000
    echo.
    echo To start the API server:
    echo 1. Open a new command prompt
    echo 2. Navigate to: VFLClientModels\models\apis
    echo 3. Run: python app.py
    echo.
    echo Press any key to continue anyway...
    pause
)

echo.
echo 🚀 Starting Streamlit UI...
echo.
echo The UI will open in your default browser at: http://localhost:8501
echo.
echo To stop the UI, press Ctrl+C in this window
echo.

streamlit run credit_score_ui.py

pause 