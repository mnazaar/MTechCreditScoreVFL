import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_home_loans_model_and_data():
    """Load the home loans model and prepare data"""
    print("Loading home loans model and data...")
    
    try:
        # Load the trained model
        model = keras.models.load_model(os.path.join('..', '..', 'saved_models', 'home_loans_model.keras'), compile=False)
        print("Home loans model loaded successfully")
        
        # Load feature names
        feature_names = np.load(os.path.join('..', '..', 'saved_models', 'home_loans_feature_names.npy'), allow_pickle=True).tolist()
        print(f"Feature names loaded: {len(feature_names)} features")
        
        # Try to load scaler
        scaler_path = os.path.join('..', '..', 'saved_models', 'home_loans_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            print("Using fallback StandardScaler")
        
        # Load original data
        df = pd.read_csv(os.path.join('..', '..', '..', 'dataset', 'data', 'banks', 'home_loans_bank.csv'))
        print(f"Original data loaded: {len(df)} records")
        
        return model, feature_names, scaler, df
        
    except Exception as e:
        print(f"Error loading model/data: {str(e)}")
        raise

def test_shap_values():
    """Test SHAP values to understand the structure"""
    print("=" * 60)
    print("Testing Home Loans SHAP values structure...")
    
    # Load model and data
    model, feature_names, scaler, df = load_home_loans_model_and_data()
    
    # Prepare data
    X = df[feature_names].values
    X_scaled = scaler.fit_transform(X)
    
    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"Feature names length: {len(feature_names)}")
    
    # Test with just one sample
    test_sample = X_scaled[0:1]
    print(f"Test sample shape: {test_sample.shape}")
    
    # Initialize SHAP explainer
    print("Initializing SHAP explainer...")
    explainer = shap.KernelExplainer(lambda x: model.predict(x), X_scaled[:10])  # Use first 10 samples as background
    
    # Get SHAP values for test sample
    print("Getting SHAP values...")
    shap_values = explainer.shap_values(test_sample)
    
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"SHAP values dtype: {shap_values.dtype}")
    
    # Handle both list and array cases
    if isinstance(shap_values, list):
        print("SHAP values is a list")
        shap_val = shap_values[0].ravel() if hasattr(shap_values[0], 'ravel') else np.array(shap_values[0])
    else:
        print("SHAP values is a numpy array")
        shap_val = shap_values.ravel() if hasattr(shap_values, 'ravel') else np.array(shap_values)
    
    print(f"Processed SHAP values shape: {shap_val.shape}")
    print(f"Processed SHAP values length: {len(shap_val)}")
    
    # Check if we can access all indices
    print(f"Can access index 0: {shap_val[0]}")
    print(f"Can access index -1: {shap_val[-1]}")
    print(f"Can access index len-1: {shap_val[len(shap_val)-1]}")
    
    # Try to get top 3 features
    feature_importance = np.abs(shap_val)
    print(f"Feature importance shape: {feature_importance.shape}")
    
    # Get top 3 indices
    top_indices = np.argsort(feature_importance)[-3:][::-1]
    print(f"Top 3 indices: {top_indices}")
    
    # Check if indices are within bounds
    for idx in top_indices:
        if idx < len(feature_names) and idx < len(shap_val):
            print(f"Index {idx} is valid - Feature: {feature_names[idx]}, Impact: {shap_val[idx]}")
        else:
            print(f"Index {idx} is out of bounds!")
    
    print("=" * 60)

if __name__ == "__main__":
    test_shap_values() 