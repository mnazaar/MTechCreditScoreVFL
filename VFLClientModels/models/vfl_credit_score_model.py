import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# ============================================================================
# SIMPLE CONFIGURATION - MODIFY THESE VALUES FOR YOUR EXPERIMENTS
# ============================================================================

# Architecture: Each number represents units in that layer (length = number of layers)
LAYER_UNITS = [1024, 512, 256, 128, 64]  # 4 layers with these units

# Sampling configuration
SAMPLE_SIZE = 5000  # Number of customers to sample (set to None for full dataset)
RANDOM_SEED = 42    # For reproducible results

# ============================================================================

def get_penultimate_layer_model(model):
    """Create a model that outputs the penultimate layer activations"""
    # Find the penultimate layer (the one before the final output layer)
    for i, layer in enumerate(reversed(model.layers)):
        if hasattr(layer, 'activation') and layer.activation is not None:
            if i > 0:  # Skip the output layer
                penultimate_layer = model.layers[-(i+1)]
                break
    else:
        # Fallback to second-to-last layer
        penultimate_layer = model.layers[-2]
    
    # Create new model that outputs the penultimate layer
    feature_extractor = models.Model(
        inputs=model.inputs,
        outputs=penultimate_layer.output,
        name=f"{model.name}_feature_extractor"
    )
    
    print(f"Feature extractor input shape: {feature_extractor.input_shape}")
    print(f"Feature extractor output shape: {feature_extractor.output_shape}")
    
    return feature_extractor

def load_client_models():
    """Load the trained client models"""
    try:
        auto_loans_model = load_model('saved_models/auto_loans_model.keras', compile=False)
        digital_bank_model = load_model('saved_models/digital_bank_model.keras', compile=False)
        
        print("\nAuto Loans Model Summary:")
        auto_loans_model.summary()
        print("\nDigital Bank Model Summary:")
        digital_bank_model.summary()
        
        return auto_loans_model, digital_bank_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("\nPlease ensure both models exist at:")
        print("- saved_models/auto_loans_model.keras")
        print("- saved_models/digital_bank_model.keras")
        raise

def load_and_preprocess_data():
    """Load and preprocess data from all sources, handling limited alignment with optional sampling"""
    use_sampling = SAMPLE_SIZE is not None
    
    # Set random seed for reproducible sampling
    np.random.seed(RANDOM_SEED)
    
    # Load datasets
    auto_loans_df = pd.read_csv('../dataset/data/banks/auto_loans_bank.csv')
    digital_bank_df = pd.read_csv('../dataset/data/banks/digital_savings_bank.csv')
    master_df = pd.read_csv('../dataset/data/credit_scoring_dataset.csv')
    
    print("\nOriginal Dataset Statistics:")
    print(f"Auto Loans customers: {len(auto_loans_df)}")
    print(f"Digital Bank customers: {len(digital_bank_df)}")
    print(f"Master dataset customers: {len(master_df)}")
    
    # Apply sampling if configured
    if use_sampling and SAMPLE_SIZE < len(master_df):
        print(f"\nApplying random sampling:")
        print(f"  Sample size: {SAMPLE_SIZE:,} customers")
        print(f"  Random seed: {RANDOM_SEED}")
        
        # Calculate original service combination ratios and maintain them
        all_customers = set(master_df['tax_id'])
        auto_customers = set(auto_loans_df['tax_id'])
        digital_customers = set(digital_bank_df['tax_id'])
        
        # Create temporary alignment for ratio calculation
        temp_df = pd.DataFrame({'tax_id': list(all_customers)})
        temp_df['has_auto'] = temp_df['tax_id'].isin(auto_customers)
        temp_df['has_digital'] = temp_df['tax_id'].isin(digital_customers)
        temp_df['combo'] = temp_df['has_auto'].astype(str) + temp_df['has_digital'].astype(str)
        
        # Calculate target counts for each combination
        combo_ratios = temp_df['combo'].value_counts(normalize=True)
        target_counts = (combo_ratios * SAMPLE_SIZE).round().astype(int)
        
        # Ensure we don't exceed sample_size due to rounding
        if target_counts.sum() > SAMPLE_SIZE:
            largest_combo = target_counts.idxmax()
            target_counts[largest_combo] -= (target_counts.sum() - SAMPLE_SIZE)
        elif target_counts.sum() < SAMPLE_SIZE:
            largest_combo = target_counts.idxmax()
            target_counts[largest_combo] += (SAMPLE_SIZE - target_counts.sum())
        
        print(f"  Target distribution:")
        combo_names = {'TrueTrue': 'Both Services', 'TrueFalse': 'Auto Only', 
                      'FalseTrue': 'Digital Only', 'FalseFalse': 'Neither'}
        for combo, count in target_counts.items():
            name = combo_names.get(combo, combo)
            ratio = count / SAMPLE_SIZE * 100
            print(f"    {name}: {count:,} ({ratio:.1f}%)")
        
        # Sample from each combination
        sampled_customers = []
        merged_temp = master_df.merge(temp_df, on='tax_id', how='left')
        
        for combo, target_count in target_counts.items():
            combo_customers = merged_temp[merged_temp['combo'] == combo]
            if len(combo_customers) >= target_count:
                sampled = combo_customers.sample(n=target_count, random_state=RANDOM_SEED)
            else:
                sampled = combo_customers
                print(f"    Warning: Only {len(combo_customers)} customers available for {combo_names.get(combo, combo)}, using all")
            
            sampled_customers.append(sampled)
        
        # Combine all sampled customers
        sampled_master = pd.concat(sampled_customers, ignore_index=True)
        
        # Filter bank datasets to match sampled customers
        sampled_customer_ids = set(sampled_master['tax_id'])
        auto_loans_df = auto_loans_df[auto_loans_df['tax_id'].isin(sampled_customer_ids)]
        digital_bank_df = digital_bank_df[digital_bank_df['tax_id'].isin(sampled_customer_ids)]
        master_df = sampled_master
        
        print(f"\nSampled Dataset Statistics:")
        print(f"Auto Loans customers: {len(auto_loans_df)}")
        print(f"Digital Bank customers: {len(digital_bank_df)}")
        print(f"Master dataset customers: {len(master_df)}")
    
    else:
        if use_sampling:
            print(f"\nSampling disabled: sample_size ({SAMPLE_SIZE}) >= dataset size ({len(master_df)})")
        else:
            print(f"\nUsing full dataset (sampling disabled)")
    
    # Get all unique customers from master dataset (this is our universe)
    all_customers = set(master_df['tax_id'])
    auto_customers = set(auto_loans_df['tax_id'])
    digital_customers = set(digital_bank_df['tax_id'])
    
    print(f"\nFinal Alignment Statistics:")
    print(f"Total customers in master: {len(all_customers)}")
    print(f"Customers with auto loans: {len(auto_customers)}")
    print(f"Customers with digital banking: {len(digital_customers)}")
    print(f"Customers in both banks: {len(auto_customers.intersection(digital_customers))}")
    print(f"Customers in auto only: {len(auto_customers - digital_customers)}")
    print(f"Customers in digital only: {len(digital_customers - auto_customers)}")
    print(f"Customers with no bank data: {len(all_customers - auto_customers - digital_customers)}")
    
    # Create alignment matrix for all customers
    customer_df = pd.DataFrame({'tax_id': sorted(list(all_customers))})
    customer_df['has_auto'] = customer_df['tax_id'].isin(auto_customers)
    customer_df['has_digital'] = customer_df['tax_id'].isin(digital_customers)
    
    # Sort datasets by tax_id for consistent indexing
    auto_loans_df = auto_loans_df.sort_values('tax_id').reset_index(drop=True)
    digital_bank_df = digital_bank_df.sort_values('tax_id').reset_index(drop=True)
    master_df = master_df.sort_values('tax_id').reset_index(drop=True)
    
    # Load client models and create feature extractors
    auto_loans_model, digital_bank_model = load_client_models()
    auto_loans_extractor = get_penultimate_layer_model(auto_loans_model)
    digital_bank_extractor = get_penultimate_layer_model(digital_bank_model)
    
    # Define feature sets used by each model (matching the actual models)
    auto_features = [
        # Core financial features
        'annual_income', 'credit_score', 'payment_history', 'employment_length', 
        'debt_to_income_ratio', 'age',
        # Credit history and behavior  
        'credit_history_length', 'num_credit_cards', 'num_loan_accounts', 
        'total_credit_limit', 'credit_utilization_ratio', 'late_payments', 
        'credit_inquiries', 'last_late_payment_days',
        # Financial position
        'current_debt', 'monthly_expenses', 'savings_balance', 
        'checking_balance', 'investment_balance',
        # Existing loans
        'auto_loan_balance', 'mortgage_balance'
    ]
    
    digital_features = [
        # Core financial features
        'annual_income', 'savings_balance', 'checking_balance', 'investment_balance',
        'payment_history', 'credit_score', 'age', 'employment_length',
        # Transaction and banking behavior  
        'avg_monthly_transactions', 'avg_transaction_value', 'digital_banking_score',
        'mobile_banking_usage', 'online_transactions_ratio', 'international_transactions_ratio',
        'e_statement_enrolled',
        # Financial behavior and credit
        'monthly_expenses', 'total_credit_limit', 'credit_utilization_ratio',
        'num_credit_cards', 'credit_history_length',
        # Additional loan and debt information
        'current_debt', 'mortgage_balance',
        # Additional calculated metrics (from the dataset)
        'total_wealth', 'net_worth', 'credit_efficiency', 'financial_stability_score'
    ]
    
    print(f"\nFeature Verification:")
    print(f"Auto model expects {len(auto_features)} features")
    print(f"Digital model expects {len(digital_features)} features")
    print(f"Auto model input shape: {auto_loans_extractor.input_shape}")
    print(f"Digital model input shape: {digital_bank_extractor.input_shape}")
    
    # Verify feature availability
    auto_missing = [f for f in auto_features if f not in auto_loans_df.columns]
    digital_missing = [f for f in digital_features if f not in digital_bank_df.columns]
    
    if auto_missing:
        print(f"Missing auto features: {auto_missing}")
    if digital_missing:
        print(f"Missing digital features: {digital_missing}")
    
    # Initialize scalers
    auto_scaler = StandardScaler()
    digital_scaler = StandardScaler()
    
    def extract_bank_representations(bank_df, features, scaler, extractor, customers_with_service, bank_name):
        """Extract representations for customers with service at this bank"""
        output_size = extractor.output_shape[-1]
        all_representations = np.zeros((len(customer_df), output_size))
        
        if len(bank_df) > 0:
            print(f"\nProcessing {bank_name} Bank:")
            print(f"  Dataset size: {len(bank_df)}")
            print(f"  Feature count: {len(features)}")
            print(f"  Output representation size: {output_size}")
            
            # Scale features
            X_scaled = scaler.fit_transform(bank_df[features])
            
            # Get representations from model
            representations = extractor.predict(X_scaled, verbose=0)
            
            # Map representations to correct customer positions
            bank_customer_ids = bank_df['tax_id'].values
            for i, customer_id in enumerate(bank_customer_ids):
                customer_idx = customer_df[customer_df['tax_id'] == customer_id].index[0]
                all_representations[customer_idx] = representations[i]
        
        # Create service availability mask
        service_mask = customers_with_service.values.astype(np.float32).reshape(-1, 1)
        
        return all_representations, service_mask, scaler
    
    # Extract representations from both banks
    auto_repr, auto_mask, fitted_auto_scaler = extract_bank_representations(
        auto_loans_df, auto_features, auto_scaler, auto_loans_extractor, 
        customer_df['has_auto'], 'Auto Loans'
    )
    
    digital_repr, digital_mask, fitted_digital_scaler = extract_bank_representations(
        digital_bank_df, digital_features, digital_scaler, digital_bank_extractor,
        customer_df['has_digital'], 'Digital Banking'
    )
    
    # Combine all features
    X_combined = np.concatenate([
        auto_repr,      # Auto loan representations
        auto_mask,      # Auto loan availability mask
        digital_repr,   # Digital bank representations  
        digital_mask    # Digital bank availability mask
    ], axis=1)
    
    print(f"\nCombined Feature Matrix:")
    print(f"  Shape: {X_combined.shape}")
    print(f"  Auto representations: {auto_repr.shape[1]} features")
    print(f"  Digital representations: {digital_repr.shape[1]} features")
    print(f"  Total features: {X_combined.shape[1]}")
    
    # Get target variable (credit scores)
    y = master_df.set_index('tax_id').loc[customer_df['tax_id'], 'credit_score'].values
    customer_ids = customer_df['tax_id'].values
    
    # Print service combination statistics
    service_combinations = {
        'both': ((customer_df['has_auto']) & (customer_df['has_digital'])).sum(),
        'auto_only': ((customer_df['has_auto']) & (~customer_df['has_digital'])).sum(),
        'digital_only': ((~customer_df['has_auto']) & (customer_df['has_digital'])).sum(),
        'neither': ((~customer_df['has_auto']) & (~customer_df['has_digital'])).sum()
    }
    
    print(f"\nService Combination Distribution:")
    for combo, count in service_combinations.items():
        percentage = (count / len(customer_df)) * 100
        print(f"  {combo}: {count} customers ({percentage:.1f}%)")
    
    # Split data while maintaining service distribution
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_combined, y, customer_ids, 
        test_size=0.2, 
        random_state=42,
        stratify=customer_df[['has_auto', 'has_digital']].apply(
            lambda x: f"{int(x['has_auto'])}{int(x['has_digital'])}", axis=1
        )
    )
    
    return (X_train, X_test, y_train, y_test, ids_train, ids_test, 
            fitted_auto_scaler, fitted_digital_scaler, auto_repr.shape[1], digital_repr.shape[1])

def build_vfl_model(input_shape, auto_repr_size, digital_repr_size):
    """Build VFL model with simple configuration"""
    
    # Input layer
    inputs = layers.Input(shape=input_shape, name='vfl_input')
    
    # Split combined input into components
    auto_repr = layers.Lambda(lambda x: x[:, :auto_repr_size], name='auto_representations')(inputs)
    auto_mask = layers.Lambda(lambda x: x[:, auto_repr_size:auto_repr_size+1], name='auto_mask')(inputs)
    digital_repr = layers.Lambda(lambda x: x[:, auto_repr_size+1:auto_repr_size+1+digital_repr_size], name='digital_representations')(inputs)
    digital_mask = layers.Lambda(lambda x: x[:, -1:], name='digital_mask')(inputs)
    
    # Expand masks to match representation dimensions
    auto_mask_expanded = layers.RepeatVector(auto_repr_size)(auto_mask)
    auto_mask_expanded = layers.Reshape((auto_repr_size,), name='auto_mask_expanded')(auto_mask_expanded)
    
    digital_mask_expanded = layers.RepeatVector(digital_repr_size)(digital_mask)
    digital_mask_expanded = layers.Reshape((digital_repr_size,), name='digital_mask_expanded')(digital_mask_expanded)
    
    # Apply masks to representations
    auto_masked = layers.Multiply(name='auto_masked')([auto_repr, auto_mask_expanded])
    digital_masked = layers.Multiply(name='digital_masked')([digital_repr, digital_mask_expanded])
    
    # Bank-specific processing
    auto_processed = layers.Dense(128, activation='relu', name='auto_dense')(auto_masked)
    auto_processed = layers.BatchNormalization(name='auto_bn')(auto_processed)
    auto_processed = layers.Dropout(0.2, name='auto_dropout')(auto_processed)
    
    digital_processed = layers.Dense(64, activation='relu', name='digital_dense')(digital_masked)
    digital_processed = layers.BatchNormalization(name='digital_bn')(digital_processed)
    digital_processed = layers.Dropout(0.2, name='digital_dropout')(digital_processed)
    
    # Combine bank features
    combined_features = layers.Concatenate(name='combined_bank_features')([auto_processed, digital_processed])
    
    # Add service availability information
    service_info = layers.Concatenate(name='service_availability')([auto_mask, digital_mask])
    service_processed = layers.Dense(16, activation='relu', name='service_dense')(service_info)
    service_processed = layers.BatchNormalization(name='service_bn')(service_processed)
    
    # Final feature combination
    all_features = layers.Concatenate(name='all_features')([combined_features, service_processed])
    
    # Build main hidden layers using LAYER_UNITS configuration
    x = all_features
    for i, units in enumerate(LAYER_UNITS):
        x = layers.Dense(units, activation='relu', name=f'hidden_dense_{i+1}')(x)
        x = layers.BatchNormalization(name=f'hidden_bn_{i+1}')(x)
        x = layers.Dropout(0.2, name=f'hidden_dropout_{i+1}')(x)
    
    # Final prediction layers
    x = layers.Dense(64, activation='relu', name='pre_output_dense_1')(x)
    x = layers.BatchNormalization(name='pre_output_bn_1')(x)
    x = layers.Dropout(0.1, name='pre_output_dropout_1')(x)
    
    x = layers.Dense(32, activation='relu', name='pre_output_dense_2')(x)
    x = layers.BatchNormalization(name='pre_output_bn_2')(x)
    x = layers.Dropout(0.05, name='pre_output_dropout_2')(x)
    
    # Output layer with credit score scaling
    raw_output = layers.Dense(1, activation='sigmoid', name='raw_output')(x)
    outputs = layers.Lambda(lambda x: x * 550 + 300, name='credit_score_output')(raw_output)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='vfl_model')
    
    print(f"\nVFL Model Architecture:")
    print(f"  Hidden layers: {len(LAYER_UNITS)} layers")
    print(f"  Units per layer: {LAYER_UNITS}")
    print(f"  Total parameters: {model.count_params():,}")
    
    return model

def save_to_excel(predictions_df, X_test, y_test, ids_test, metrics, history):
    """Save all results to a human-readable Excel file with multiple sheets"""
    print("\nSaving results to Excel file...")
    
    # Create Excel writer
    excel_path = 'data/vfl_results.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Summary Sheet
        summary_data = {
            'Metric': [
                'Root Mean Square Error (RMSE)',
                'Mean Absolute Error (MAE)',
                'R-squared Score',
                'Total Samples',
                'Samples with Auto Loans',
                'Samples with Digital Banking',
                'Samples with Both Services'
            ],
            'Value': [
                f"{metrics['rmse']:.2f}",
                f"{metrics['mae']:.2f}",
                f"{metrics['r2']:.4f}",
                len(y_test),
                sum(X_test[:, 16] == 1),
                sum(X_test[:, -1] == 1),
                sum((X_test[:, 16] == 1) & (X_test[:, -1] == 1))
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. All Predictions Sheet with Detailed Features
        all_predictions_df = pd.DataFrame({
            'Tax ID': ids_test,
            'Has Auto Loans': X_test[:, 16] == 1,
            'Has Digital Banking': X_test[:, -1] == 1,
            'Actual Credit Score': y_test,
            'Predicted Credit Score': metrics['predictions'],
            'Absolute Error': abs(y_test - metrics['predictions']),
            'Percent Error': (abs(y_test - metrics['predictions']) / y_test) * 100
        })
        
        # Add Auto Loans Features (first 16 columns)
        for i in range(16):
            all_predictions_df[f'Auto_Feature_{i+1}'] = X_test[:, i]
            
        # Add Digital Banking Features (columns 17-24, excluding masks)
        for i in range(8):
            all_predictions_df[f'Digital_Feature_{i+1}'] = X_test[:, 17+i]
            
        # Add mask values
        all_predictions_df['Auto_Mask'] = X_test[:, 16]
        all_predictions_df['Digital_Mask'] = X_test[:, -1]
        
        all_predictions_df['Has Auto Loans'] = all_predictions_df['Has Auto Loans'].map({True: 'Yes', False: 'No'})
        all_predictions_df['Has Digital Banking'] = all_predictions_df['Has Digital Banking'].map({True: 'Yes', False: 'No'})
        all_predictions_df['Predicted Credit Score'] = all_predictions_df['Predicted Credit Score'].round(2)
        all_predictions_df['Absolute Error'] = all_predictions_df['Absolute Error'].round(2)
        all_predictions_df['Percent Error'] = all_predictions_df['Percent Error'].round(2)
        
        # Sort by percent error to show most interesting cases first
        all_predictions_df = all_predictions_df.sort_values('Percent Error', ascending=False)
        all_predictions_df.to_excel(writer, sheet_name='All Predictions', index=False)
        
        # 3. Feature Statistics Sheet
        feature_stats = pd.DataFrame()
        # Auto loan features
        for i in range(16):
            col = f'Auto_Feature_{i+1}'
            stats = all_predictions_df[all_predictions_df['Has Auto Loans'] == 'Yes'][col].describe()
            feature_stats[col] = stats
        # Digital banking features
        for i in range(8):
            col = f'Digital_Feature_{i+1}'
            stats = all_predictions_df[all_predictions_df['Has Digital Banking'] == 'Yes'][col].describe()
            feature_stats[col] = stats
        feature_stats.to_excel(writer, sheet_name='Feature Statistics', float_format='%.3f')
        
        # 4. Segment Analysis Sheet
        segments = [
            ('Both Services', (X_test[:, 16] == 1) & (X_test[:, -1] == 1)),
            ('Only Auto Loans', (X_test[:, 16] == 1) & (X_test[:, -1] == 0)),
            ('Only Digital Banking', (X_test[:, 16] == 0) & (X_test[:, -1] == 1)),
            ('No Services', (X_test[:, 16] == 0) & (X_test[:, -1] == 0))
        ]
        
        segment_data = []
        for segment_name, mask in segments:
            if sum(mask) > 0:
                segment_metrics = {
                    'Segment': segment_name,
                    'Sample Count': sum(mask),
                    'Average Error': np.mean(abs(y_test[mask] - metrics['predictions'][mask])),
                    'RMSE': np.sqrt(np.mean((y_test[mask] - metrics['predictions'][mask]) ** 2)),
                    'Average % Error': np.mean(abs(y_test[mask] - metrics['predictions'][mask]) / y_test[mask] * 100),
                    'Min Error': np.min(abs(y_test[mask] - metrics['predictions'][mask])),
                    'Max Error': np.max(abs(y_test[mask] - metrics['predictions'][mask])),
                    'Mean Actual Score': np.mean(y_test[mask]),
                    'Mean Predicted Score': np.mean(metrics['predictions'][mask])
                }
                segment_data.append(segment_metrics)
        
        segment_df = pd.DataFrame(segment_data)
        for col in segment_df.columns:
            if col != 'Segment':
                segment_df[col] = segment_df[col].round(2)
        segment_df.to_excel(writer, sheet_name='Segment Analysis', index=False)
        
        # 5. Training History Sheet
        history_df = pd.DataFrame({
            'Epoch': range(1, len(history.history['loss']) + 1),
            'Training Loss': history.history['loss'],
            'Validation Loss': history.history['val_loss'],
            'Training MAE': history.history['mae'],
            'Validation MAE': history.history['val_mae']
        })
        history_df = history_df.round(4)
        history_df.to_excel(writer, sheet_name='Training History', index=False)
        
        # 6. Error Distribution Sheet
        errors = abs(y_test - metrics['predictions'])
        error_bins = [0, 10, 20, 30, 50, 100, float('inf')]
        error_labels = ['0-10', '11-20', '21-30', '31-50', '51-100', '100+']
        error_dist = pd.cut(errors, bins=error_bins, labels=error_labels)
        error_dist_df = pd.DataFrame({
            'Error Range': error_labels,
            'Count': pd.value_counts(error_dist, sort=False),
            'Percentage': pd.value_counts(error_dist, normalize=True, sort=False) * 100
        })
        error_dist_df['Percentage'] = error_dist_df['Percentage'].round(2)
        error_dist_df.to_excel(writer, sheet_name='Error Distribution', index=False)
        
        # 7. Score Distribution Sheet
        score_bins = [300, 500, 600, 650, 700, 750, 800, 850]
        score_labels = ['300-500', '501-600', '601-650', '651-700', '701-750', '751-800', '801-850']
        
        actual_dist = pd.cut(y_test, bins=score_bins, labels=score_labels)
        pred_dist = pd.cut(metrics['predictions'], bins=score_bins, labels=score_labels)
        
        score_dist_df = pd.DataFrame({
            'Score Range': score_labels,
            'Actual Count': pd.value_counts(actual_dist, sort=False),
            'Actual %': pd.value_counts(actual_dist, normalize=True, sort=False) * 100,
            'Predicted Count': pd.value_counts(pred_dist, sort=False),
            'Predicted %': pd.value_counts(pred_dist, normalize=True, sort=False) * 100
        })
        score_dist_df = score_dist_df.round(2)
        score_dist_df.to_excel(writer, sheet_name='Score Distribution', index=False)
    
    print(f"Results saved to '{excel_path}'")

def train_model():
    """Train the VFL credit score prediction model"""
    # Create necessary directories
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    (X_train, X_test, y_train, y_test, ids_train, ids_test, 
     auto_scaler, digital_scaler, auto_repr_size, digital_repr_size) = load_and_preprocess_data()
    
    print(f"\nTraining Configuration:")
    print(f"Architecture: {len(LAYER_UNITS)} layers, {LAYER_UNITS}")
    print(f"Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}")
    
    print(f"\nTraining Data Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature vector size: {X_train.shape[1]}")
    print(f"Auto loans representation size: {auto_repr_size}")
    print(f"Digital bank representation size: {digital_repr_size}")
    
    # Print target variable statistics
    print("\nCredit Score Statistics:")
    print(f"Training Mean: {np.mean(y_train):.2f}")
    print(f"Training Std: {np.std(y_train):.2f}")
    print(f"Training Min: {np.min(y_train):.2f}")
    print(f"Training Max: {np.max(y_train):.2f}")
    
    # Save training and test datasets as Excel files
    save_vfl_data_to_excel(X_train, y_train, ids_train, 'data/vfl_train_data.xlsx')
    save_vfl_data_to_excel(X_test, y_test, ids_test, 'data/vfl_test_data.xlsx')
    
    # Also save NPZ files for model use
    train_data = {'features': X_train, 'labels': y_train, 'customer_ids': ids_train}
    test_data = {'features': X_test, 'labels': y_test, 'customer_ids': ids_test}
    np.savez('data/vfl_train_data.npz', **train_data)
    np.savez('data/vfl_test_data.npz', **test_data)
    
    print("\nSaved VFL datasets:")
    print("- Training data: 'data/vfl_train_data.xlsx'")
    print("- Test data: 'data/vfl_test_data.xlsx'")

    # Build VFL model
    model = build_vfl_model(
        input_shape=(X_train.shape[1],), 
        auto_repr_size=auto_repr_size, 
        digital_repr_size=digital_repr_size
    )
    
    # Print model summary
    print("\nVFL Model Architecture:")
    model.summary()
    print(f"Total Parameters: {model.count_params():,}")
    
    # Configure optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Robust to outliers
        metrics=['mae', 'mse']
    )
    
    # Configure callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-8,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'saved_models/vfl_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print(f"\nTraining VFL model...")
    print(f"Model Parameters: {model.count_params():,}")
    print(f"Training for max 100 epochs with batch size 32")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    r2 = 1 - np.sum((y_test - y_pred.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred.flatten()
    }
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'tax_id': ids_test,
        'has_auto': X_test[:, auto_repr_size] == 1,
        'has_digital': X_test[:, -1] == 1,
        'actual_score': y_test,
        'predicted_score': y_pred.flatten(),
    })
    
    predictions_df['difference'] = predictions_df['actual_score'] - predictions_df['predicted_score']
    predictions_df['percent_error'] = (abs(predictions_df['difference']) / predictions_df['actual_score']) * 100
    predictions_df = predictions_df.sort_values('percent_error', ascending=False)
    
    # Save results to Excel
    save_to_excel(predictions_df, X_test, y_test, ids_test, metrics, history)
    
    print("\nModel Performance:")
    print("=" * 50)
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Print test set sample statistics
    print(f"\nTest Set Sample Breakdown:")
    print("=" * 30)
    service_types = [
        ('both', (X_test[:, auto_repr_size] == 1) & (X_test[:, -1] == 1), 'Both Services'),
        ('auto_only', (X_test[:, auto_repr_size] == 1) & (X_test[:, -1] == 0), 'Auto Loans Only'),
        ('digital_only', (X_test[:, auto_repr_size] == 0) & (X_test[:, -1] == 1), 'Digital Banking Only'),
        ('neither', (X_test[:, auto_repr_size] == 0) & (X_test[:, -1] == 0), 'No Services')
    ]
    for service_key, mask, service_name in service_types:
        count = mask.sum()
        percentage = (count / len(X_test)) * 100
        if count > 0:
            avg_error = np.mean(abs(y_test[mask] - y_pred.flatten()[mask]))
            print(f"{service_name}: {count} samples ({percentage:.1f}%) - Avg Error: {avg_error:.2f}")
        else:
            print(f"{service_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nTotal Test Samples: {len(X_test):,}")
    print(f"Overall Average Error: {mae:.2f}")
    
    # Print sample predictions
    print(f"\n" + "=" * 100)
    print(f"📊 DETAILED SAMPLE PREDICTIONS - MANUAL MODEL")
    print(f"=" * 100)
    print(f"Showing first 10 customers from test set with actual vs predicted credit scores")
    print(f"Credit score range: 300-850 points")
    print()
    
    # Enhanced table header
    print(f"{'#':<3} {'Tax ID':<15} {'Actual':<8} {'Predicted':<10} {'Error':<8} {'Error%':<8} {'Auto':<6} {'Digital':<8} {'Services':<12} {'Score Range':<12}")
    print("-" * 100)
    
    total_error = 0
    for i in range(min(10, len(ids_test))):
        tax_id = ids_test[i]
        actual = y_test[i]
        predicted = y_pred.flatten()[i]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100 if actual > 0 else 0
        total_error += error
        
        # Service information
        has_auto = "Yes" if X_test[i, auto_repr_size] == 1 else "No"
        has_digital = "Yes" if X_test[i, -1] == 1 else "No"
        
        # Determine service combination
        if X_test[i, auto_repr_size] == 1 and X_test[i, -1] == 1:
            services = "Both"
        elif X_test[i, auto_repr_size] == 1:
            services = "Auto Only"
        elif X_test[i, -1] == 1:
            services = "Digital Only"
        else:
            services = "Neither"
        
        # Credit score range classification
        if actual >= 750:
            score_range = "Excellent"
        elif actual >= 700:
            score_range = "Good"
        elif actual >= 650:
            score_range = "Fair"
        elif actual >= 600:
            score_range = "Poor"
        else:
            score_range = "Very Poor"
            
        print(f"{i+1:<3} {tax_id:<15} {actual:<8.0f} {predicted:<10.1f} {error:<8.1f} {error_pct:<8.1f} {has_auto:<6} {has_digital:<8} {services:<12} {score_range:<12}")
    
    # Summary statistics for the 10 samples
    avg_error = total_error / min(10, len(ids_test))
    print("-" * 100)
    print(f"📈 Sample Statistics (10 customers):")
    print(f"   Average Error: {avg_error:.1f} points")
    print(f"   Min Actual Score: {y_test[:10].min():.0f}")
    print(f"   Max Actual Score: {y_test[:10].max():.0f}")
    print(f"   Prediction Range: {y_pred.flatten()[:10].min():.1f} - {y_pred.flatten()[:10].max():.1f}")
    
    # Service distribution in sample
    sample_auto = sum(1 for i in range(min(10, len(ids_test))) if X_test[i, auto_repr_size] == 1)
    sample_digital = sum(1 for i in range(min(10, len(ids_test))) if X_test[i, -1] == 1)
    sample_both = sum(1 for i in range(min(10, len(ids_test))) if X_test[i, auto_repr_size] == 1 and X_test[i, -1] == 1)
    sample_neither = sum(1 for i in range(min(10, len(ids_test))) if X_test[i, auto_repr_size] == 0 and X_test[i, -1] == 0)
    
    print(f"   Service Distribution: {sample_both} Both, {sample_auto-sample_both} Auto Only, {sample_digital-sample_both} Digital Only, {sample_neither} Neither")
    print("=" * 100)
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('VFL Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/vfl_training_history.png')
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 8))
    
    # Color code by service combination
    colors = ['red', 'blue', 'green', 'orange']
    for i, (service_key, mask, service_name) in enumerate(service_types):
        if mask.sum() > 0:
            plt.scatter(y_test[mask], y_pred.flatten()[mask], 
                       alpha=0.6, c=colors[i], label=service_name, s=20)
    
    plt.plot([300, 850], [300, 850], 'k--', alpha=0.8, label='Perfect Prediction')
    plt.xlabel('Actual Credit Score')
    plt.ylabel('Predicted Credit Score')
    plt.title('VFL Model: Predicted vs Actual Credit Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(300, 850)
    plt.ylim(300, 850)
    plt.tight_layout()
    plt.savefig('plots/vfl_predictions.png')
    plt.close()
    
    # Save final model
    model.save('saved_models/vfl_model.keras')
    
    print(f"\nConfiguration Summary:")
    print(f"  Architecture: {len(LAYER_UNITS)} layers")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Final validation loss: {min(history.history['val_loss']):.4f}")
    
    print("\nModel saved as 'saved_models/vfl_model.keras'")
    print("Training plots saved as 'plots/vfl_training_history.png'")
    print("Prediction plot saved as 'plots/vfl_predictions.png'")

def save_vfl_data_to_excel(features, labels, customer_ids, filename):
    """Convert VFL feature data to human-readable Excel format"""
    # Determine feature sizes dynamically
    auto_repr_size = 16  # Default assumption - will be updated
    digital_repr_size = 8  # Default assumption - will be updated
    
    # Adjust based on actual feature matrix
    total_features = features.shape[1]
    if total_features > 26:  # More than expected, recalculate
        # Structure: auto_repr + auto_mask + digital_repr + digital_mask
        # We know there are 2 masks, so: auto_repr + digital_repr = total - 2
        # Need to determine the split - use common sizes
        auto_repr_size = 16 if total_features >= 26 else 8
        digital_repr_size = total_features - auto_repr_size - 2
    
    # Create column names dynamically
    column_names = []
    
    # Auto loan representation features
    for i in range(auto_repr_size):
        column_names.append(f'Auto_Repr_{i+1}')
    column_names.append('Has_Auto_Loan')
    
    # Digital banking representation features  
    for i in range(digital_repr_size):
        column_names.append(f'Digital_Repr_{i+1}')
    column_names.append('Has_Digital_Banking')
    
    # Create DataFrame with features
    df = pd.DataFrame(features, columns=column_names)
    
    # Add customer ID and credit score at the beginning
    df.insert(0, 'Customer_ID', customer_ids)
    df.insert(1, 'Actual_Credit_Score', labels)
    
    # Add service usage summary
    df.insert(2, 'Services_Used', df.apply(lambda row: 
        'Both Services' if row['Has_Auto_Loan'] == 1 and row['Has_Digital_Banking'] == 1
        else 'Auto Loans Only' if row['Has_Auto_Loan'] == 1
        else 'Digital Banking Only' if row['Has_Digital_Banking'] == 1
        else 'No Services', axis=1))
    
    # Add feature statistics
    auto_cols = [f'Auto_Repr_{i}' for i in range(1, auto_repr_size + 1)]
    digital_cols = [f'Digital_Repr_{i}' for i in range(1, digital_repr_size + 1)]
    
    df['Auto_Repr_Sum'] = df[auto_cols].sum(axis=1)
    df['Digital_Repr_Sum'] = df[digital_cols].sum(axis=1)
    df['Total_Feature_Activity'] = df['Auto_Repr_Sum'] + df['Digital_Repr_Sum']
    
    # Round numerical values for readability
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(4)
    
    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main data sheet
        df.to_excel(writer, sheet_name='VFL_Data', index=False)
        
        # Summary statistics sheet
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Customers',
                'Customers with Auto Loans',
                'Customers with Digital Banking', 
                'Customers with Both Services',
                'Customers with No Services',
                'Average Credit Score',
                'Credit Score Std Dev',
                'Min Credit Score',
                'Max Credit Score'
            ],
            'Value': [
                len(df),
                df['Has_Auto_Loan'].sum(),
                df['Has_Digital_Banking'].sum(),
                ((df['Has_Auto_Loan'] == 1) & (df['Has_Digital_Banking'] == 1)).sum(),
                ((df['Has_Auto_Loan'] == 0) & (df['Has_Digital_Banking'] == 0)).sum(),
                df['Actual_Credit_Score'].mean().round(2),
                df['Actual_Credit_Score'].std().round(2),
                df['Actual_Credit_Score'].min(),
                df['Actual_Credit_Score'].max()
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary', index=False)
        
        # Feature correlation with credit score
        correlations = []
        for col in auto_cols + digital_cols:
            if col in df.columns:
                corr = df[col].corr(df['Actual_Credit_Score'])
                correlations.append({'Feature': col, 'Correlation_with_Credit_Score': corr})
        
        if correlations:
            corr_df = pd.DataFrame(correlations).sort_values('Correlation_with_Credit_Score', 
                                                            key=abs, ascending=False)
            corr_df['Correlation_with_Credit_Score'] = corr_df['Correlation_with_Credit_Score'].round(4)
            corr_df.to_excel(writer, sheet_name='Feature_Correlations', index=False)
    
    print(f"Saved VFL data to '{filename}' ({len(df)} customers)")

if __name__ == "__main__":
    print("VFL Credit Score Model - Simple Configuration")
    print("=" * 50)
    print(f"Current Architecture: {len(LAYER_UNITS)} layers with units {LAYER_UNITS}")
    print(f"Current Sample Size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}")
    print(f"Random Seed: {RANDOM_SEED}")
    print()
    print("To modify:")
    print("- Change LAYER_UNITS array at the top of the file")
    print("- Change SAMPLE_SIZE for different data sizes")
    print("- Change RANDOM_SEED for different random samples")
    print()
    
    # Train the model
    train_model()
    
    print("\nTraining completed!")
    print("\nTo experiment:")
    print("- Edit LAYER_UNITS = [512, 256, 128, 64] for different architectures")
    print("- Edit SAMPLE_SIZE = 5000 for different data sizes")
    print("- Edit RANDOM_SEED = 42 for different random samples") 