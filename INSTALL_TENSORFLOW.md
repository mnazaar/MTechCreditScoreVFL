# TensorFlow Installation Guide

To enable full functionality of the Auto Loans Drift Detector Pipeline, install TensorFlow:

## Option 1: Using pip
```bash
pip install tensorflow
```

## Option 2: Using conda
```bash
conda install tensorflow
```

## Option 3: GPU Support (if you have NVIDIA GPU)
```bash
pip install tensorflow[gpu]
```

## Verification
After installation, you can verify TensorFlow is working:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Current Status
The pipeline currently works without TensorFlow but uses dummy predictions for drift detection.
With TensorFlow installed, it will:
- Load the actual Auto Loans neural network model
- Make real predictions for drift detection
- Provide more accurate drift analysis

## Running the Pipeline
Once TensorFlow is installed, run:
```bash
python VFLClientModels/drift_detection_retraining/auto_loans_drift_detector_pipeline.py
```
