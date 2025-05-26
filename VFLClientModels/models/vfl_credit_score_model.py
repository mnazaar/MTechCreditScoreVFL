import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

def get_penultimate_layer_model(model):
    """Create a model that outputs the penultimate layer activations"""
    # Get the penultimate layer
    penultimate_layer = model.layers[-2]  # Second to last layer
    
    # Create new model that outputs the penultimate layer
    feature_extractor = models.Model(
        inputs=model.inputs,
        outputs=penultimate_layer.output,
        name=f"{model.name}_feature_extractor"
    )
    
    # Print input shape for debugging
    print(f"Feature extractor input shape: {feature_extractor.input_shape}")
    
    return feature_extractor

def load_client_models():
    """Load the trained client models"""
    try:
        auto_loans_model = load_model('VFLClientModels/models/saved_models/auto_loans_model.keras', compile=False)
        digital_bank_model = load_model('VFLClientModels/models/saved_models/digital_bank_model.keras', compile=False)
        
        # Print model summaries for debugging
        print("\nAuto Loans Model Summary:")
        auto_loans_model.summary()
        print("\nDigital Bank Model Summary:")
        digital_bank_model.summary()
        
        return auto_loans_model, digital_bank_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print("\nPlease ensure both models exist at:")
        print("- VFLClientModels/models/saved_models/auto_loans_model.keras")
        print("- VFLClientModels/models/saved_models/digital_bank_model.keras")
        raise

def load_and_preprocess_data():
    """Load and preprocess data from all sources, handling missing customers"""
    # Load datasets
    auto_loans_df = pd.read_csv('VFLClientModels/dataset/data/banks/auto_loans_bank.csv')
    digital_bank_df = pd.read_csv('VFLClientModels/dataset/data/banks/digital_savings_bank.csv')
    master_df = pd.read_csv('VFLClientModels/dataset/data/credit_scoring_dataset.csv')
    
    # Print available columns for debugging
    print("\nAvailable columns in datasets:")
    print("Auto Loans columns:", auto_loans_df.columns.tolist())
    print("Digital Bank columns:", digital_bank_df.columns.tolist())
    print("Master columns:", master_df.columns.tolist())
    
    # Get all unique customers
    all_customers = set(master_df['tax_id'])  # We need all customers from master for prediction
    auto_customers = set(auto_loans_df['tax_id'])
    digital_customers = set(digital_bank_df['tax_id'])
    
    print(f"\nCustomer Statistics:")
    print(f"Total customers in master: {len(all_customers)}")
    print(f"Customers with auto loans: {len(auto_customers)}")
    print(f"Customers with digital banking: {len(digital_customers)}")
    print(f"Customers in both banks: {len(auto_customers.intersection(digital_customers))}")
    
    # Create a DataFrame with all customers
    customer_df = pd.DataFrame({'tax_id': list(all_customers)})
    customer_df['has_auto'] = customer_df['tax_id'].isin(auto_customers)
    customer_df['has_digital'] = customer_df['tax_id'].isin(digital_customers)
    
    # Sort by tax_id for consistency
    customer_df = customer_df.sort_values('tax_id')
    auto_loans_df = auto_loans_df.sort_values('tax_id')
    digital_bank_df = digital_bank_df.sort_values('tax_id')
    master_df = master_df.sort_values('tax_id')
    
    # Load client models
    auto_loans_model, digital_bank_model = load_client_models()
    
    # Create feature extractors
    auto_loans_extractor = get_penultimate_layer_model(auto_loans_model)
    digital_bank_extractor = get_penultimate_layer_model(digital_bank_model)
    
    # Prepare features for each bank based on model input shapes
    auto_features = [
        'annual_income', 'credit_score', 'payment_history',
        'employment_length', 'debt_to_income_ratio', 'age',
        'num_credit_cards', 'num_loan_accounts', 'credit_utilization_ratio'
    ]
    
    digital_features = [
        'annual_income', 'savings_balance', 'checking_balance',
        'payment_history', 'age', 'avg_monthly_transactions',
        'avg_transaction_value', 'mobile_banking_usage',
        'digital_banking_score', 'online_transactions_ratio', 'e_statement_enrolled'
    ]
    
    # Initialize scalers
    auto_scaler = StandardScaler()
    digital_scaler = StandardScaler()
    
    # Get representations for each bank
    def get_bank_representations(bank_df, features, scaler, extractor, has_bank, bank_name):
        # Initialize zero representations for all customers
        output_size = extractor.output_shape[-1]  # Get size from model
        zero_repr = np.zeros((len(customer_df), output_size))
        
        if len(bank_df) > 0:
            # Verify all features exist
            missing_features = [f for f in features if f not in bank_df.columns]
            if missing_features:
                raise ValueError(f"Missing features in {bank_name} dataset: {missing_features}")
            
            # Verify input shape matches model expectations
            if len(features) != extractor.input_shape[-1]:
                raise ValueError(
                    f"{bank_name} model expects {extractor.input_shape[-1]} features, "
                    f"but got {len(features)} features"
                )
            
            print(f"\nProcessing {bank_name} features:")
            print(f"Input shape: {(len(bank_df), len(features))}")
            print(f"Expected model input shape: {extractor.input_shape}")
            
            # Scale features for available customers
            X_scaled = scaler.fit_transform(bank_df[features])
            
            # Get representations
            representations = extractor.predict(X_scaled, verbose=0)
            
            # Map representations to correct customer indices
            customer_indices = customer_df[customer_df['tax_id'].isin(bank_df['tax_id'])].index
            zero_repr[customer_indices] = representations
        
        # Create availability mask
        mask = customer_df[has_bank].values.astype(np.float32).reshape(-1, 1)
        return zero_repr, mask
    
    # Get representations and masks for each bank
    print("\nExtracting features from Auto Loans model...")
    auto_repr, auto_mask = get_bank_representations(
        auto_loans_df, auto_features, auto_scaler, auto_loans_extractor, 'has_auto', 'Auto Loans'
    )
    
    print("\nExtracting features from Digital Bank model...")
    digital_repr, digital_mask = get_bank_representations(
        digital_bank_df, digital_features, digital_scaler, digital_bank_extractor, 'has_digital', 'Digital Bank'
    )
    
    # Combine representations and masks
    X_combined = np.concatenate([
        auto_repr, auto_mask,
        digital_repr, digital_mask
    ], axis=1)
    
    print(f"\nCombined feature shape: {X_combined.shape}")
    
    # Get target variable
    y = master_df.set_index('tax_id').loc[customer_df['tax_id'], 'credit_score'].values
    customer_ids = customer_df['tax_id'].values
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_combined, y, customer_ids, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, ids_train, ids_test

def build_vfl_model(input_shape):
    """Build VFL model that handles missing representations"""
    # Calculate sizes
    total_features = input_shape[0]
    auto_size = 16  # Size of auto loans representation
    digital_size = 8  # Size of digital bank representation
    
    # Create input layer
    inputs = layers.Input(shape=input_shape)
    
    # Split inputs into representations and masks
    auto_repr = layers.Lambda(lambda x: x[:, :auto_size])(inputs)
    auto_mask = layers.Lambda(lambda x: x[:, auto_size:auto_size+1])(inputs)
    digital_repr = layers.Lambda(lambda x: x[:, auto_size+1:auto_size+1+digital_size])(inputs)
    digital_mask = layers.Lambda(lambda x: x[:, -1:])(inputs)
    
    # Expand masks to match representation sizes
    auto_mask_expanded = layers.RepeatVector(auto_size)(auto_mask)
    auto_mask_expanded = layers.Reshape((auto_size,))(auto_mask_expanded)
    digital_mask_expanded = layers.RepeatVector(digital_size)(digital_mask)
    digital_mask_expanded = layers.Reshape((digital_size,))(digital_mask_expanded)
    
    # Apply masks
    auto_masked = layers.multiply([auto_repr, auto_mask_expanded])
    digital_masked = layers.multiply([digital_repr, digital_mask_expanded])
    
    # Combine masked representations
    combined = layers.Concatenate()([auto_masked, digital_masked])
    
    # Initial dense layer
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # First residual block (maintaining 64 units)
    res1 = x
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res1])
    
    # Second residual block (maintaining 64 units)
    res2 = x
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res2])
    
    # Gradual reduction in layer size
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Final dense layers
    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer (linear activation for direct credit score prediction)
    outputs = layers.Dense(1)(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def save_to_excel(predictions_df, X_test, y_test, ids_test, metrics, history):
    """Save all results to a human-readable Excel file with multiple sheets"""
    print("\nSaving results to Excel file...")
    
    # Create Excel writer
    excel_path = 'VFLClientModels/models/data/vfl_results.xlsx'
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
    os.makedirs('VFLClientModels/models/saved_models', exist_ok=True)
    os.makedirs('VFLClientModels/models/plots', exist_ok=True)
    os.makedirs('VFLClientModels/models/data', exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, ids_train, ids_test = load_and_preprocess_data()
    
    print(f"\nTraining Data Statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Feature vector size: {X_train.shape[1]}")
    print(f"Auto loans features: 16 (+ 1 mask)")
    print(f"Digital bank features: 8 (+ 1 mask)")
    
    # Print target variable statistics
    print("\nCredit Score Statistics:")
    print(f"Training Mean: {np.mean(y_train):.2f}")
    print(f"Training Std: {np.std(y_train):.2f}")
    print(f"Training Min: {np.min(y_train):.2f}")
    print(f"Training Max: {np.max(y_train):.2f}")
    
    # Save training and test datasets as Excel files for human review
    def save_vfl_data_to_excel(features, labels, customer_ids, filename):
        """Convert VFL feature data to human-readable Excel format"""
        # Create DataFrame with features
        df = pd.DataFrame(features, columns=[
            # Auto loan representation features (16 features from penultimate layer)
            'Auto_Repr_1', 'Auto_Repr_2', 'Auto_Repr_3', 'Auto_Repr_4',
            'Auto_Repr_5', 'Auto_Repr_6', 'Auto_Repr_7', 'Auto_Repr_8',
            'Auto_Repr_9', 'Auto_Repr_10', 'Auto_Repr_11', 'Auto_Repr_12',
            'Auto_Repr_13', 'Auto_Repr_14', 'Auto_Repr_15', 'Auto_Repr_16',
            # Auto loan availability mask
            'Has_Auto_Loan',
            # Digital banking representation features (8 features from penultimate layer)
            'Digital_Repr_1', 'Digital_Repr_2', 'Digital_Repr_3', 'Digital_Repr_4',
            'Digital_Repr_5', 'Digital_Repr_6', 'Digital_Repr_7', 'Digital_Repr_8',
            # Digital banking availability mask
            'Has_Digital_Banking'
        ])
        
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
        df['Auto_Repr_Sum'] = df[[f'Auto_Repr_{i}' for i in range(1, 17)]].sum(axis=1)
        df['Digital_Repr_Sum'] = df[[f'Digital_Repr_{i}' for i in range(1, 9)]].sum(axis=1)
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
            for col in [f'Auto_Repr_{i}' for i in range(1, 17)] + [f'Digital_Repr_{i}' for i in range(1, 9)]:
                corr = df[col].corr(df['Actual_Credit_Score'])
                correlations.append({'Feature': col, 'Correlation_with_Credit_Score': corr})
            
            corr_df = pd.DataFrame(correlations).sort_values('Correlation_with_Credit_Score', 
                                                            key=abs, ascending=False)
            corr_df['Correlation_with_Credit_Score'] = corr_df['Correlation_with_Credit_Score'].round(4)
            corr_df.to_excel(writer, sheet_name='Feature_Correlations', index=False)
        
        print(f"Saved VFL data to '{filename}' ({len(df)} customers)")

    # Save both datasets as Excel files
    save_vfl_data_to_excel(X_train, y_train, ids_train, 'VFLClientModels/models/data/vfl_train_data.xlsx')
    save_vfl_data_to_excel(X_test, y_test, ids_test, 'VFLClientModels/models/data/vfl_test_data.xlsx')

    # Also save NPZ files for model use
    train_data = {
        'features': X_train,
        'labels': y_train,
        'customer_ids': ids_train
    }
    test_data = {
        'features': X_test,
        'labels': y_test,
        'customer_ids': ids_test
    }
    np.savez('VFLClientModels/models/data/vfl_train_data.npz', **train_data)
    np.savez('VFLClientModels/models/data/vfl_test_data.npz', **test_data)

    print("\nSaved VFL datasets:")
    print("- Training data: 'VFLClientModels/models/data/vfl_train_data.xlsx'")
    print("- Test data: 'VFLClientModels/models/data/vfl_test_data.xlsx'")

    # Build model with linear output
    model = build_vfl_model(input_shape=(X_train.shape[1],))
    
    # Print model summary
    print("\nVFL Model Architecture:")
    model.summary()
    
    # Compile model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\nTraining VFL model...")
    history = model.fit(
        X_train,
        y_train,  # Use raw credit scores
        epochs=200,  # Increased epochs
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)  # Direct prediction of credit scores
    
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
        'has_auto': X_test[:, 16] == 1,
        'has_digital': X_test[:, -1] == 1,
        'actual_score': y_test,
        'predicted_score': y_pred.flatten(),
    })
    
    predictions_df['difference'] = predictions_df['actual_score'] - predictions_df['predicted_score']
    predictions_df['percent_error'] = (abs(predictions_df['difference']) / predictions_df['actual_score']) * 100
    
    # Sort by absolute difference to show most interesting cases
    predictions_df = predictions_df.sort_values('percent_error', ascending=False)
    
    # Save results to Excel with detailed information
    save_to_excel(predictions_df, X_test, y_test, ids_test, metrics, history)
    
    print("\nModel Performance:")
    print("=" * 50)
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Print detailed predictions for top 10 errors
    print("\nTop 10 Prediction Errors:")
    print("=" * 100)
    print(f"{'Tax ID':<15} {'Has Auto':<10} {'Has Digital':<12} {'Actual':>8} {'Predicted':>10} {'Diff':>8} {'% Error':>8}")
    print("-" * 100)
    
    # Display top 10 predictions by error
    top_10 = predictions_df.head(10)
    for _, row in top_10.iterrows():
        print(f"{row['tax_id']:<15} "
              f"{'Yes' if row['has_auto'] else 'No':<10} "
              f"{'Yes' if row['has_digital'] else 'No':<12} "
              f"{row['actual_score']:>8.0f} "
              f"{row['predicted_score']:>10.0f} "
              f"{row['difference']:>8.0f} "
              f"{row['percent_error']:>8.1f}%")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('VFL Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('VFLClientModels/models/plots/vfl_training_history.png')
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([300, 850], [300, 850], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Credit Score')
    plt.ylabel('Predicted Credit Score')
    plt.title('VFL Model: Predicted vs Actual Credit Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig('VFLClientModels/models/plots/vfl_predictions.png')
    plt.close()
    
    # Save model
    model.save('VFLClientModels/models/saved_models/vfl_credit_score_model.keras')
    print("\nModel saved as 'VFLClientModels/models/saved_models/vfl_credit_score_model.keras'")
    print("Plots saved in 'VFLClientModels/models/plots/' directory")

if __name__ == "__main__":
    train_model() 