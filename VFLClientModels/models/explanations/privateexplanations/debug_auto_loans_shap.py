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

def debug_auto_loans_shap():
    """Debug SHAP values for auto loans model"""
    print("🔍 Debugging Auto Loans SHAP Values")
    print("=" * 60)
    
    try:
        # Load model and data
        print("1. Loading model and data...")
        model = keras.models.load_model(os.path.join('..', '..', 'saved_models', 'auto_loans_model.keras'), compile=False)
        feature_names = np.load(os.path.join('..', '..', 'saved_models', 'auto_loans_feature_names.npy'), allow_pickle=True).tolist()
        df = pd.read_csv(os.path.join('..', '..', '..', 'dataset', 'data', 'banks', 'auto_loans_bank.csv'))
        
        print(f"   - Model loaded: {model}")
        print(f"   - Feature names: {len(feature_names)} features")
        print(f"   - Data shape: {df.shape}")
        
        # Prepare data
        print("\n2. Preparing data...")
        X = df[feature_names].values
        print(f"   - X shape: {X.shape}")
        print(f"   - X sample values: {X[0][:5]}")  # First 5 values of first sample
        
        # Test prediction
        print("\n3. Testing model prediction...")
        sample_pred = model.predict(X[:1])
        print(f"   - Sample prediction: {sample_pred}")
        print(f"   - Prediction shape: {sample_pred.shape}")
        
        # Test SHAP with single sample
        print("\n4. Testing SHAP with single sample...")
        background_data = X[:100]  # Use first 100 samples as background
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background_data)
        
        # Test with first sample
        test_sample = X[:1]
        print(f"   - Test sample shape: {test_sample.shape}")
        print(f"   - Test sample values: {test_sample[0][:5]}")
        
        shap_values = explainer.shap_values(test_sample)
        print(f"   - SHAP values type: {type(shap_values)}")
        print(f"   - SHAP values shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'No shape'}")
        
        if isinstance(shap_values, list):
            print(f"   - SHAP values list length: {len(shap_values)}")
            shap_val = shap_values[0]
        else:
            shap_val = shap_values
            
        print(f"   - SHAP values shape: {shap_val.shape}")
        print(f"   - SHAP values sample: {shap_val[0][:5]}")
        print(f"   - SHAP values range: [{shap_val.min():.6f}, {shap_val.max():.6f}]")
        print(f"   - SHAP values mean: {shap_val.mean():.6f}")
        print(f"   - SHAP values std: {shap_val.std():.6f}")
        
        # Check if all values are zero
        if np.allclose(shap_val, 0):
            print("   ⚠️  WARNING: All SHAP values are zero!")
        else:
            print("   ✅ SHAP values are not all zero")
        
        # Test with different background size
        print("\n5. Testing with different background sizes...")
        for bg_size in [10, 50, 100]:
            try:
                background_data = X[:bg_size]
                explainer = shap.KernelExplainer(lambda x: model.predict(x), background_data)
                shap_values = explainer.shap_values(test_sample)
                
                if isinstance(shap_values, list):
                    shap_val = shap_values[0]
                else:
                    shap_val = shap_values
                
                print(f"   - Background size {bg_size}: SHAP range [{shap_val.min():.6f}, {shap_val.max():.6f}]")
            except Exception as e:
                print(f"   - Background size {bg_size}: Error - {str(e)}")
        
        # Test with different sample
        print("\n6. Testing with different sample...")
        test_sample2 = X[100:101]  # Different sample
        shap_values2 = explainer.shap_values(test_sample2)
        
        if isinstance(shap_values2, list):
            shap_val2 = shap_values2[0]
        else:
            shap_val2 = shap_values2
            
        print(f"   - Different sample SHAP range: [{shap_val2.min():.6f}, {shap_val2.max():.6f}]")
        
        # Check model architecture
        print("\n7. Model architecture analysis...")
        print(f"   - Model layers: {len(model.layers)}")
        for i, layer in enumerate(model.layers):
            print(f"   - Layer {i}: {layer.__class__.__name__} - {layer.output_shape}")
        
        # Check if model is regression or classification
        output_shape = model.layers[-1].output_shape
        print(f"   - Output shape: {output_shape}")
        if output_shape[-1] == 1:
            print("   - Model type: Regression")
        else:
            print(f"   - Model type: Classification ({output_shape[-1]} classes)")
        
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_auto_loans_shap() 