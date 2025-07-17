import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def get_penultimate_layer_model(model, model_name):
    """Create a model that outputs the penultimate layer activations"""
    # Find the penultimate layer by name
    penultimate_layer_name = 'penultimate_layer'
    
    # Get the penultimate layer
    try:
        penultimate_layer = model.get_layer(penultimate_layer_name)
        print(f"Found penultimate layer in {model_name}: {penultimate_layer.name}")
    except ValueError:
        # If not found by name, get the second to last layer
        penultimate_layer = model.layers[-3]  # Skip output and batch norm layers
        print(f"Using layer {penultimate_layer.name} as penultimate for {model_name}")
    
    # Create new model that outputs the penultimate layer
    feature_extractor = models.Model(
        inputs=model.inputs,
        outputs=penultimate_layer.output,
        name=f"{model_name}_feature_extractor"
    )
    
    print(f"Feature extractor for {model_name}:")
    print(f"  Input shape: {feature_extractor.input_shape}")
    print(f"  Output shape: {feature_extractor.output_shape}")
    
    return feature_extractor

def load_client_models():
    """Load the trained client models"""
    print("Loading client models...")
    
    # Load digital savings model
    digital_model_path = 'VFLClientModels/models/saved_models/digital_savings_full_model.keras'
    if os.path.exists(digital_model_path):
        digital_model = load_model(digital_model_path)
        print(f"Loaded digital savings model from {digital_model_path}")
    else:
        print(f"Digital savings model not found at {digital_model_path}")
        return None, None
    
    # Load auto loans model  
    auto_model_path = 'VFLClientModels/models/saved_models/auto_loans_full_model.keras'
    if os.path.exists(auto_model_path):
        auto_model = load_model(auto_model_path)
        print(f"Loaded auto loans model from {auto_model_path}")
    else:
        print(f"Auto loans model not found at {auto_model_path}")
        return None, None
    
    return digital_model, auto_model

def load_and_preprocess_data():
    """Load and preprocess data from all sources"""
    print("Loading datasets...")
    
    # Load main credit dataset
    credit_df = pd.read_csv('data/credit_scoring_dataset.csv')
    print(f"Loaded credit dataset: {len(credit_df)} customers")
    
    # Load bank datasets
    digital_df = pd.read_csv('data/banks/digital_savings_bank_full.csv')
    auto_df = pd.read_csv('data/banks/auto_loans_bank_full.csv')
    
    print(f"Loaded digital savings dataset: {len(digital_df)} customers")
    print(f"Loaded auto loans dataset: {len(auto_df)} customers")
    
    # Load client models
    digital_model, auto_model = load_client_models()
    if digital_model is None or auto_model is None:
        raise ValueError("Could not load client models")
    
    # Get feature extractors
    digital_extractor = get_penultimate_layer_model(digital_model, "digital_savings")
    auto_extractor = get_penultimate_layer_model(auto_model, "auto_loans")
    
    # Prepare features for each bank
    digital_features = [
        'annual_income', 'savings_balance', 'checking_balance', 'investment_balance',
        'payment_history', 'age', 'avg_monthly_transactions', 'avg_transaction_value',
        'mobile_banking_usage', 'digital_banking_score', 'online_transactions_ratio',
        'e_statement_enrolled', 'credit_score', 'employment_length', 'debt_to_income_ratio',
        'credit_utilization_ratio', 'num_credit_cards', 'num_loan_accounts',
        'credit_history_length', 'late_payments', 'credit_inquiries'
    ]
    
    auto_features = [
        'annual_income', 'credit_score', 'payment_history', 'employment_length',
        'debt_to_income_ratio', 'age', 'num_credit_cards', 'num_loan_accounts',
        'total_credit_limit', 'credit_utilization_ratio', 'credit_history_length',
        'late_payments', 'credit_inquiries', 'savings_balance', 'checking_balance',
        'investment_balance', 'monthly_expenses', 'num_dependents', 'education_level',
        'marital_status', 'home_ownership'
    ]
    
    # Initialize scalers
    digital_scaler = StandardScaler()
    auto_scaler = StandardScaler()
    
    # Get representations for each bank
    def get_bank_representations(bank_df, features, scaler, extractor, bank_name):
        print(f"\nProcessing {bank_name} representations...")
        
        # Prepare features
        X = bank_df[features].values
        X_scaled = scaler.fit_transform(X)
        
        # Get representations
        representations = extractor.predict(X_scaled, batch_size=1000, verbose=1)
        print(f"{bank_name} representations shape: {representations.shape}")
        
        return representations, scaler
    
    # Get representations
    digital_repr, digital_scaler = get_bank_representations(
        digital_df, digital_features, digital_scaler, digital_extractor, "Digital Savings"
    )
    
    auto_repr, auto_scaler = get_bank_representations(
        auto_df, auto_features, auto_scaler, auto_extractor, "Auto Loans"
    )
    
    # Create masks (all customers have both services in full dataset)
    digital_mask = np.ones((len(credit_df), 1))
    auto_mask = np.ones((len(credit_df), 1))
    
    # Combine all features
    X = np.concatenate([
        digital_repr,  # Digital representations
        digital_mask,  # Digital mask
        auto_repr,     # Auto representations  
        auto_mask      # Auto mask
    ], axis=1)
    
    # Target variable
    y = credit_df['credit_score'].values
    
    print(f"\nFinal dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Digital representations: {digital_repr.shape[1]} features")
    print(f"Auto representations: {auto_repr.shape[1]} features")
    
    return X, y, credit_df['tax_id'].values, digital_scaler, auto_scaler

def build_enhanced_vfl_model(digital_size, auto_size):
    """Build enhanced VFL model with attention mechanisms and deeper architecture"""
    
    # Input layers for representations and masks
    digital_repr_input = layers.Input(shape=(digital_size,), name='digital_representations')
    digital_mask_input = layers.Input(shape=(1,), name='digital_mask')
    auto_repr_input = layers.Input(shape=(auto_size,), name='auto_representations')
    auto_mask_input = layers.Input(shape=(1,), name='auto_mask')
    
    # Apply masks to representations
    digital_mask_expanded = layers.RepeatVector(digital_size)(digital_mask_input)
    digital_mask_expanded = layers.Reshape((digital_size,))(digital_mask_expanded)
    digital_masked = layers.multiply([digital_repr_input, digital_mask_expanded], name='digital_masked')
    
    auto_mask_expanded = layers.RepeatVector(auto_size)(auto_mask_input)
    auto_mask_expanded = layers.Reshape((auto_size,))(auto_mask_expanded)
    auto_masked = layers.multiply([auto_repr_input, auto_mask_expanded], name='auto_masked')
    
    # Enhanced feature processing for each bank
    # Digital banking feature processing
    digital_processed = layers.Dense(64, activation='relu', name='digital_dense_1')(digital_masked)
    digital_processed = layers.BatchNormalization(name='digital_bn_1')(digital_processed)
    digital_processed = layers.Dropout(0.2, name='digital_dropout_1')(digital_processed)
    
    digital_processed = layers.Dense(32, activation='relu', name='digital_dense_2')(digital_processed)
    digital_processed = layers.BatchNormalization(name='digital_bn_2')(digital_processed)
    digital_processed = layers.Dropout(0.2, name='digital_dropout_2')(digital_processed)
    
    # Auto loans feature processing
    auto_processed = layers.Dense(64, activation='relu', name='auto_dense_1')(auto_masked)
    auto_processed = layers.BatchNormalization(name='auto_bn_1')(auto_processed)
    auto_processed = layers.Dropout(0.2, name='auto_dropout_1')(auto_processed)
    
    auto_processed = layers.Dense(32, activation='relu', name='auto_dense_2')(auto_processed)
    auto_processed = layers.BatchNormalization(name='auto_bn_2')(auto_processed)
    auto_processed = layers.Dropout(0.2, name='auto_dropout_2')(auto_processed)
    
    # Multi-head attention mechanism
    def multi_head_attention(query, key, value, num_heads=4, name_prefix=""):
        """Multi-head attention mechanism"""
        head_dim = query.shape[-1] // num_heads
        
        # Split into multiple heads
        query_heads = []
        key_heads = []
        value_heads = []
        
        for i in range(num_heads):
            q_head = layers.Dense(head_dim, name=f'{name_prefix}_query_head_{i}')(query)
            k_head = layers.Dense(head_dim, name=f'{name_prefix}_key_head_{i}')(key)
            v_head = layers.Dense(head_dim, name=f'{name_prefix}_value_head_{i}')(value)
            
            query_heads.append(q_head)
            key_heads.append(k_head)
            value_heads.append(v_head)
        
        # Compute attention for each head
        attended_heads = []
        for i in range(num_heads):
            # Attention scores
            scores = layers.dot([query_heads[i], key_heads[i]], axes=-1, name=f'{name_prefix}_scores_{i}')
            scores = layers.Activation('softmax', name=f'{name_prefix}_softmax_{i}')(scores)
            
            # Apply attention to values
            attended = layers.dot([scores, value_heads[i]], axes=-1, name=f'{name_prefix}_attended_{i}')
            attended_heads.append(attended)
        
        # Concatenate heads
        if num_heads > 1:
            multi_head_output = layers.Concatenate(name=f'{name_prefix}_concat')(attended_heads)
        else:
            multi_head_output = attended_heads[0]
        
        return multi_head_output
    
    # Self-attention for each bank
    digital_self_attended = multi_head_attention(
        digital_processed, digital_processed, digital_processed, 
        num_heads=2, name_prefix="digital_self_attn"
    )
    
    auto_self_attended = multi_head_attention(
        auto_processed, auto_processed, auto_processed,
        num_heads=2, name_prefix="auto_self_attn"
    )
    
    # Cross-attention between banks
    cross_attended_digital = multi_head_attention(
        digital_self_attended, auto_self_attended, auto_self_attended,
        num_heads=2, name_prefix="cross_attn_digital"
    )
    
    cross_attended_auto = multi_head_attention(
        auto_self_attended, digital_self_attended, digital_self_attended,
        num_heads=2, name_prefix="cross_attn_auto"
    )
    
    # Combine all features
    combined = layers.Concatenate(name='combined_features')([
        digital_self_attended, auto_self_attended,
        cross_attended_digital, cross_attended_auto
    ])
    
    # Add service availability information
    service_info = layers.Concatenate(name='service_info')([digital_mask_input, auto_mask_input])
    service_processed = layers.Dense(16, activation='relu', name='service_dense')(service_info)
    service_processed = layers.BatchNormalization(name='service_bn')(service_processed)
    
    # Final feature combination
    all_features = layers.Concatenate(name='all_features')([combined, service_processed])
    
    # Enhanced deep network with residual connections
    # Initial transformation
    x = layers.Dense(512, activation='relu', name='initial_dense')(all_features)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Dropout(0.3, name='initial_dropout')(x)
    
    # Residual Block 1 (512 units)
    res1 = x
    x = layers.Dense(512, activation='relu', name='res1_dense_1')(x)
    x = layers.BatchNormalization(name='res1_bn_1')(x)
    x = layers.Dropout(0.3, name='res1_dropout_1')(x)
    x = layers.Dense(512, activation='relu', name='res1_dense_2')(x)
    x = layers.BatchNormalization(name='res1_bn_2')(x)
    x = layers.Dropout(0.3, name='res1_dropout_2')(x)
    x = layers.Dense(512, activation='relu', name='res1_dense_3')(x)
    x = layers.BatchNormalization(name='res1_bn_3')(x)
    x = layers.Add(name='res1_add')([x, res1])
    
    # Residual Block 2 (512 units)
    res2 = x
    x = layers.Dense(512, activation='relu', name='res2_dense_1')(x)
    x = layers.BatchNormalization(name='res2_bn_1')(x)
    x = layers.Dropout(0.3, name='res2_dropout_1')(x)
    x = layers.Dense(512, activation='relu', name='res2_dense_2')(x)
    x = layers.BatchNormalization(name='res2_bn_2')(x)
    x = layers.Dropout(0.3, name='res2_dropout_2')(x)
    x = layers.Dense(512, activation='relu', name='res2_dense_3')(x)
    x = layers.BatchNormalization(name='res2_bn_3')(x)
    x = layers.Add(name='res2_add')([x, res2])
    
    # Transition to smaller size
    x = layers.Dense(256, activation='relu', name='transition_dense')(x)
    x = layers.BatchNormalization(name='transition_bn')(x)
    x = layers.Dropout(0.25, name='transition_dropout')(x)
    
    # Residual Block 3 (256 units)
    res3 = x
    x = layers.Dense(256, activation='relu', name='res3_dense_1')(x)
    x = layers.BatchNormalization(name='res3_bn_1')(x)
    x = layers.Dropout(0.25, name='res3_dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='res3_dense_2')(x)
    x = layers.BatchNormalization(name='res3_bn_2')(x)
    x = layers.Dropout(0.25, name='res3_dropout_2')(x)
    x = layers.Dense(256, activation='relu', name='res3_dense_3')(x)
    x = layers.BatchNormalization(name='res3_bn_3')(x)
    x = layers.Add(name='res3_add')([x, res3])
    
    # Residual Block 4 (256 units)
    res4 = x
    x = layers.Dense(256, activation='relu', name='res4_dense_1')(x)
    x = layers.BatchNormalization(name='res4_bn_1')(x)
    x = layers.Dropout(0.25, name='res4_dropout_1')(x)
    x = layers.Dense(256, activation='relu', name='res4_dense_2')(x)
    x = layers.BatchNormalization(name='res4_bn_2')(x)
    x = layers.Dropout(0.25, name='res4_dropout_2')(x)
    x = layers.Dense(256, activation='relu', name='res4_dense_3')(x)
    x = layers.BatchNormalization(name='res4_bn_3')(x)
    x = layers.Add(name='res4_add')([x, res4])
    
    # Gradual reduction layers
    x = layers.Dense(128, activation='relu', name='reduce_dense_1')(x)
    x = layers.BatchNormalization(name='reduce_bn_1')(x)
    x = layers.Dropout(0.2, name='reduce_dropout_1')(x)
    
    x = layers.Dense(64, activation='relu', name='reduce_dense_2')(x)
    x = layers.BatchNormalization(name='reduce_bn_2')(x)
    x = layers.Dropout(0.15, name='reduce_dropout_2')(x)
    
    x = layers.Dense(32, activation='relu', name='reduce_dense_3')(x)
    x = layers.BatchNormalization(name='reduce_bn_3')(x)
    x = layers.Dropout(0.1, name='reduce_dropout_3')(x)
    
    # Final prediction layers
    x = layers.Dense(16, activation='relu', name='final_dense')(x)
    x = layers.BatchNormalization(name='final_bn')(x)
    x = layers.Dropout(0.05, name='final_dropout')(x)
    
    # Output layer with credit score range constraint (300-850)
    raw_output = layers.Dense(1, activation='linear', name='raw_output')(x)
    
    # Apply sigmoid and scale to credit score range
    normalized_output = layers.Activation('sigmoid', name='sigmoid_output')(raw_output)
    credit_score_output = layers.Lambda(lambda x: x * 550 + 300, name='credit_score_output')(normalized_output)
    
    # Create model
    model = models.Model(
        inputs=[digital_repr_input, digital_mask_input, auto_repr_input, auto_mask_input],
        outputs=credit_score_output,
        name='enhanced_vfl_credit_score_model'
    )
    
    return model

def train_enhanced_vfl_model():
    """Train the enhanced VFL credit score prediction model"""
    print("="*80)
    print("ENHANCED VFL CREDIT SCORE PREDICTION MODEL")
    print("="*80)
    
    # Create directories
    os.makedirs('VFLClientModels/models/saved_models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    X, y, customer_ids, digital_scaler, auto_scaler = load_and_preprocess_data()
    
    # Split data
    train_idx, test_idx = train_test_split(
        range(len(X)), test_size=0.2, random_state=42, stratify=None
    )
    
    # Extract representations and masks
    digital_size = 16  # From digital savings model
    auto_size = 32     # From auto loans model
    
    digital_repr = X[:, :digital_size]
    digital_mask = X[:, digital_size:digital_size+1]
    auto_repr = X[:, digital_size+1:digital_size+1+auto_size]
    auto_mask = X[:, -1:]
    
    # Prepare training data
    X_train = {
        'digital_representations': digital_repr[train_idx],
        'digital_mask': digital_mask[train_idx],
        'auto_representations': auto_repr[train_idx],
        'auto_mask': auto_mask[train_idx]
    }
    
    X_test = {
        'digital_representations': digital_repr[test_idx],
        'digital_mask': digital_mask[test_idx],
        'auto_representations': auto_repr[test_idx],
        'auto_mask': auto_mask[test_idx]
    }
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"\nData split:")
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    
    # Build enhanced VFL model
    model = build_enhanced_vfl_model(digital_size, auto_size)
    
    # Print model summary
    print("\n" + "="*80)
    print("ENHANCED VFL MODEL ARCHITECTURE")
    print("="*80)
    model.summary()
    print(f"Total parameters: {model.count_params():,}")
    
    # Enhanced optimizer with learning rate scheduling
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='huber',  # More robust to outliers
        metrics=['mae', 'mse']
    )
    
    # Enhanced callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
        verbose=1,
        cooldown=5
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'VFLClientModels/models/saved_models/enhanced_vfl_model.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    
    # Custom callback for detailed monitoring
    class DetailedMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 20 == 0:
                val_loss = logs.get('val_loss', 0)
                val_mae = logs.get('val_mae', 0)
                val_mse = logs.get('val_mse', 0)
                lr = float(self.model.optimizer.learning_rate)
                print(f"\nEpoch {epoch}: val_loss={val_loss:.4f}, val_mae={val_mae:.2f}, val_mse={val_mse:.2f}, lr={lr:.2e}")
    
    monitor = DetailedMonitor()
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING ENHANCED VFL MODEL")
    print("="*80)
    
    history = model.fit(
        X_train, y_train,
        epochs=500,  # Increased epochs for larger model
        batch_size=128,  # Larger batch size
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, checkpoint, monitor],
        verbose=1,
        shuffle=True
    )
    
    # Evaluate model
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Predictions
    y_pred = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"MSE: {mse:.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'customer_id': customer_ids[test_idx],
        'actual_score': y_test,
        'predicted_score': y_pred,
        'absolute_error': np.abs(y_test - y_pred),
        'percent_error': np.abs(y_test - y_pred) / y_test * 100,
        'has_digital': X_test['digital_mask'].flatten() == 1,
        'has_auto': X_test['auto_mask'].flatten() == 1
    })
    
    # Save results
    results_df.to_csv('data/enhanced_vfl_results.csv', index=False)
    
    # Save scalers
    joblib.dump(digital_scaler, 'VFLClientModels/models/saved_models/digital_scaler_enhanced.pkl')
    joblib.dump(auto_scaler, 'VFLClientModels/models/saved_models/auto_scaler_enhanced.pkl')
    
    print(f"\nResults saved to 'data/enhanced_vfl_results.csv'")
    print(f"Model saved to 'VFLClientModels/models/saved_models/enhanced_vfl_model.keras'")
    
    return model, history, results_df

if __name__ == "__main__":
    model, history, results = train_enhanced_vfl_model() 