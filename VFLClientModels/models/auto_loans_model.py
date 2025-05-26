import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data():
    """Load and preprocess data for auto loans model with robust scaling"""
    # Load data
    df = pd.read_csv('VFLClientModels/dataset/data/banks/auto_loans_bank.csv')
    
    # Select features (excluding total_credit_limit)
    features = [
        'annual_income',           # Strong predictor of loan amount
        'credit_score',           # Credit worthiness
        'payment_history',        # Payment reliability
        'employment_length',      # Job stability
        'debt_to_income_ratio',   # Debt burden
        'age',                    # Age factor
        'num_credit_cards',       # Credit relationship
        'num_loan_accounts',      # Existing loans
        'credit_utilization_ratio' # Credit usage
    ]
    
    # Prepare X and y
    X = df[features]
    y = df['auto_loan_limit']
    
    # Remove zero loan amounts as they represent ineligible loans
    mask = y > 0
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple log transform for target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train, y_test, scaler

def build_model(input_shape):
    """Build neural network model with 16 units in penultimate layer"""
    # L1L2 regularization
    regularizer = tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-5)
    
    model = models.Sequential([
        # Input layer
        layers.Dense(128, activation='relu', input_shape=input_shape,
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Hidden layer 1
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Hidden layer 2
        layers.Dense(32, activation='relu',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Penultimate layer with 16 units
        layers.Dense(16, activation='relu', name='penultimate',
                    kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        
        # Output layer (no activation for regression)
        layers.Dense(1)
    ])
    
    return model

def print_test_predictions(X_test, y_test, model, scaler, n_samples=5):
    """Print detailed test predictions for random samples"""
    # Get random indices
    test_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions
    X_test_samples = X_test[test_indices]
    y_test_samples = y_test.iloc[test_indices]
    y_pred_log = model.predict(X_test_samples)
    y_pred = np.expm1(y_pred_log)
    
    print("\nDetailed Test Predictions:")
    print("=" * 100)
    print(f"{'Actual Amount':>15} {'Predicted Amount':>15} {'Difference':>15} {'% Error':>10} {'Features':<40}")
    print("-" * 100)
    
    # Get feature names
    features = [
        'annual_income', 'credit_score', 'payment_history',
        'employment_length', 'debt_to_income_ratio', 'age',
        'num_credit_cards', 'num_loan_accounts', 'credit_utilization_ratio'
    ]
    
    for idx, (actual, pred) in enumerate(zip(y_test_samples, y_pred)):
        diff = actual - pred[0]
        pct_error = (abs(diff) / actual * 100) if actual != 0 else float('inf')
        
        # Get key features for this sample
        sample_features = X_test_samples[idx]
        # Inverse transform to get original scale
        sample_features_orig = scaler.inverse_transform(sample_features.reshape(1, -1))[0]
        key_info = f"Income: ${sample_features_orig[0]:,.0f}, Score: {sample_features_orig[1]:.0f}"
        
        print(f"${actual:>14,.0f} ${pred[0]:>14,.0f} ${diff:>14,.0f} {pct_error:>9.1f}% {key_info:<40}")
    print("-" * 100)

def train_model():
    """Train the auto loans prediction model with enhanced training process"""
    # Create necessary directories
    os.makedirs('VFLClientModels/models/saved_models', exist_ok=True)
    os.makedirs('VFLClientModels/models/plots', exist_ok=True)
    
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train_log, y_test_log, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Build model
    model = build_model(input_shape=(X_train_scaled.shape[1],))
    
    # Compile model with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # Back to original learning rate
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        mode='min'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        mode='min',
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train_scaled,
        y_train_log,  # Using log-transformed target
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)  # Convert back from log scale
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred.flatten()) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred.flatten()))
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
    r2 = 1 - np.sum((y_test - y_pred.flatten()) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    
    print("\nModel Performance:")
    print("=" * 50)
    print(f"Root Mean Square Error: ${rmse:,.2f}")
    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R-squared Score: {r2:.4f}")
    
    # Print detailed test predictions
    print_test_predictions(X_test_scaled, y_test, model, scaler)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    
    # Plot loss and validation metrics
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.plot(history.history['mae'], label='Training MAE', linestyle='--')
    plt.plot(history.history['val_mae'], label='Validation MAE', linestyle='--')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('VFLClientModels/models/plots/auto_loans_training_history.png')
    plt.close()
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Auto Loan Limit')
    plt.ylabel('Predicted Auto Loan Limit')
    plt.title('Predicted vs Actual Auto Loan Limits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('VFLClientModels/models/plots/auto_loans_predictions.png')
    plt.close()
    
    # Save model and features
    model.save('VFLClientModels/models/saved_models/auto_loans_model.keras')
    feature_names = [
        'annual_income',
        'credit_score',
        'payment_history',
        'employment_length',
        'debt_to_income_ratio',
        'age',
        'num_credit_cards',
        'num_loan_accounts',
        'credit_utilization_ratio'
    ]
    np.save('VFLClientModels/models/saved_models/auto_loans_feature_names.npy', feature_names)
    
    print("\nModel saved as 'VFLClientModels/models/saved_models/auto_loans_model.keras'")
    print("Feature names saved as 'VFLClientModels/models/saved_models/auto_loans_feature_names.npy'")
    print("Plots saved in 'VFLClientModels/models/plots/' directory")

def analyze_feature_importance(model, X_train_scaled, scaler):
    """Analyze feature importance using permutation importance"""
    feature_names = [
        'annual_income', 'credit_score', 'payment_history',
        'employment_length', 'debt_to_income_ratio', 'age',
        'num_credit_cards', 'num_loan_accounts', 'credit_utilization_ratio'
    ]
    
    # Get baseline predictions
    baseline_pred = model.predict(X_train_scaled)
    baseline_mse = np.mean(baseline_pred ** 2)
    
    # Calculate importance for each feature
    importance_scores = []
    for i in range(X_train_scaled.shape[1]):
        # Create a copy and permute one feature
        X_permuted = X_train_scaled.copy()
        X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
        
        # Get predictions with permuted feature
        permuted_pred = model.predict(X_permuted)
        permuted_mse = np.mean(permuted_pred ** 2)
        
        # Calculate importance
        importance = (permuted_mse - baseline_mse) / baseline_mse
        importance_scores.append((feature_names[i], importance))
    
    # Sort by importance
    return sorted(importance_scores, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    train_model() 