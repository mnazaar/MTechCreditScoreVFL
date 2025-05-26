import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def create_model(input_dim, num_classes):
    """Create a simple neural network model with 8-unit penultimate layer"""
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Penultimate layer with 8 units
        Dense(8, activation='relu', name='penultimate_layer'),
        BatchNormalization(),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data():
    """Load and preprocess the digital savings bank data"""
    # Load the data
    df = pd.read_csv('dataset/data/banks/digital_savings_bank.csv')
    
    # Basic feature engineering
    df['transaction_volume'] = df['avg_monthly_transactions'] * df['avg_transaction_value']
    df['digital_engagement_score'] = (df['digital_banking_score'] + df['mobile_banking_usage']) / 2
    
    # Core features only
    feature_columns = [
        'annual_income',
        'savings_balance',
        'checking_balance',
        'payment_history',
        'age',
        'avg_monthly_transactions',
        'avg_transaction_value',
        'transaction_volume',
        'digital_banking_score',
        'mobile_banking_usage',
        'digital_engagement_score'
    ]
    
    X = df[feature_columns]
    y = df['customer_category']
    customer_ids = df['tax_id']
    
    # Handle any infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Log transform monetary values
    monetary_columns = ['annual_income', 'savings_balance', 'checking_balance', 
                       'avg_transaction_value', 'transaction_volume']
    for col in monetary_columns:
        X[col] = np.log1p(X[col])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_scaled, y_encoded, customer_ids,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_encoded
    )
    
    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train_balanced)
    y_test_cat = to_categorical(y_test)
    
    return (X_train_balanced, X_test, y_train_cat, y_test_cat,
            label_encoder.classes_, feature_columns, ids_test, scaler, label_encoder)

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Digital Bank Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Digital Bank Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./models/plots/digital_bank_training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Digital Bank Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./models/plots/digital_bank_confusion_matrix.png')
    plt.close()

def test_model_samples(X_test, y_test, customer_ids, classes, model, n_samples=5):
    """Test the model with sample records"""
    print("\nTesting model with random sample records:")
    print("-" * 100)
    print(f"{'Customer ID':<15} {'Original Category':<20} {'Predicted Category':<20} {'Confidence':<10} {'Correct':<10}")
    print("-" * 100)
    
    # Get random indices for sampling
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    # Get predictions
    y_pred_prob = model.predict(X_test[indices])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test[indices], axis=1)
    
    # Print results
    for idx, (true, pred, prob) in enumerate(zip(y_true, y_pred, y_pred_prob)):
        customer_id = customer_ids.iloc[indices[idx]]
        confidence = prob[pred] * 100
        is_correct = "✓" if true == pred else "✗"
        print(f"{customer_id:<15} {classes[true]:<20} {classes[pred]:<20} {confidence:>6.2f}% {is_correct:^10}")
    
    print("-" * 100)

def main():
    # Create necessary directories
    os.makedirs('./models/saved_models', exist_ok=True)
    os.makedirs('./models/plots', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (X_train, X_test, y_train, y_test, classes, feature_names,
     customer_ids, scaler, label_encoder) = load_and_preprocess_data()
    
    # Create and train model
    print("\nCreating and training model...")
    model = create_model(input_dim=len(feature_names), num_classes=len(classes))
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            './models/saved_models/digital_bank_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred, target_names=classes))
    
    plot_confusion_matrix(y_test_labels, y_pred, classes)
    
    np.save('./models/saved_models/feature_names.npy', feature_names)
    
    print("\nModel saved as './models/saved_models/digital_bank_model.h5'")
    print("Feature names saved as './models/saved_models/feature_names.npy'")
    print("Plots saved in './models/plots/' directory")
    
    test_model_samples(X_test, y_test, customer_ids, classes, model)

if __name__ == "__main__":
    main() 