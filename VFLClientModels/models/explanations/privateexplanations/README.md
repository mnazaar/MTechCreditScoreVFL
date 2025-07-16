# Auto Loans Feature Predictor System

This directory contains a neural network system that predicts the top 3 features and their impacts/directions from the 16D intermediate representations of the auto loans model.

## Overview

The system consists of four main components:

1. **Dataset Generator** (`auto_loans_feature_predictor_dataset.py`)
2. **Neural Network Model** (`auto_loans_feature_predictor_model.py`)
3. **Training Pipeline** (`train_auto_loans_feature_predictor.py`)
4. **Usage Script** (`use_auto_loans_feature_predictor.py`)

## Architecture

### Input
- **16D Intermediate Representations**: Extracted from the penultimate layer of the auto loans neural network model

### Output
For each of the top 3 features:
- **Feature Index**: Which feature from the original feature set
- **Feature Weight**: SHAP-like importance weight
- **Feature Direction**: Positive or negative impact
- **Feature Impact**: High, medium, or low impact level

### Model Structure
- **Input Layer**: 16 units (intermediate representations)
- **Shared Layers**: 128 → 64 → 32 units with batch normalization and dropout
- **Feature-Specific Heads**: 16 units per feature
- **Output Heads**: 12 total outputs (3 features × 4 outputs each)

## Files Description

### 1. `auto_loans_feature_predictor_dataset.py`
Generates the training dataset by:
- Loading the auto loans model and extracting intermediate representations
- Using the auto loans explainer to get top 3 features for each sample
- Creating a structured dataset with input representations and target features

**Usage:**
```bash
python auto_loans_feature_predictor_dataset.py
```

### 2. `auto_loans_feature_predictor_model.py`
Contains the neural network architecture and training logic:
- Multi-output neural network with 12 output heads
- Handles classification (feature indices, directions, impacts) and regression (weights)
- Includes comprehensive evaluation and visualization

**Usage:**
```bash
python auto_loans_feature_predictor_model.py
```

### 3. `train_auto_loans_feature_predictor.py`
Orchestrates the complete training pipeline:
- Runs dataset generation
- Trains the feature predictor model
- Creates example predictions

**Usage:**
```bash
python train_auto_loans_feature_predictor.py
```

### 4. `use_auto_loans_feature_predictor.py`
Demonstrates how to use the trained model:
- Loads the trained feature predictor
- Makes predictions on sample data
- Shows integration with the auto loans model

**Usage:**
```bash
python use_auto_loans_feature_predictor.py
```

## Training Process

1. **Dataset Generation**:
   - Extract 16D intermediate representations from auto loans model
   - Use explainer to get top 3 features for each sample
   - Create structured dataset with input-output pairs

2. **Model Training**:
   - Train multi-output neural network
   - Use appropriate loss functions for each output type
   - Monitor performance on validation set

3. **Evaluation**:
   - Evaluate accuracy for feature indices, directions, and impacts
   - Calculate MAE for feature weights
   - Generate training history plots

## Output Files

After training, the following files are generated:

- `saved_models/auto_loans_feature_predictor.keras` - Trained model
- `saved_models/auto_loans_feature_predictor_feature_names.json` - Feature names
- `saved_models/auto_loans_feature_predictor_info.json` - Model metadata
- `data/auto_loans_feature_prediction_dataset.csv` - Training dataset
- `plots/auto_loans_feature_predictor_training_history.png` - Training plots

## Integration with Auto Loans Explainer

The feature predictor can be integrated with the auto loans explainer to provide:
- Fast feature importance predictions without running SHAP
- Privacy-preserving explanations using only intermediate representations
- Real-time feature analysis for new customers

## Requirements

- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SHAP (for dataset generation)

## Notes

- The system uses the same intermediate representation extraction technique as the VFL system
- All paths are configured to work from the `privateexplanations` subdirectory
- The model is designed to be privacy-preserving by only using intermediate representations 