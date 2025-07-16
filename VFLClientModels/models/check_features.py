import numpy as np

# Check features for each model
models = {
    'home_loans': 'home_loans_feature_names.npy',
    'auto_loans': 'auto_loans_feature_names.npy', 
    'digital_savings': 'digital_bank_feature_names.npy',
    'credit_cards': 'credit_card_feature_names.npy'
}

for model_name, filename in models.items():
    try:
        features = np.load(f'saved_models/{filename}', allow_pickle=True)
        print(f"\n{model_name.upper()}:")
        print(f"  Feature count: {len(features)}")
        print(f"  Features: {list(features)}")
    except Exception as e:
        print(f"\n{model_name.upper()}: Error - {e}") 