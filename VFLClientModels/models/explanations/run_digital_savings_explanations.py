#!/usr/bin/env python3
"""
Digital Savings Feature Explanations Runner
==========================================

This script runs feature importance explanations for the digital savings neural network model.
It provides privacy-preserving insights into which features contributed to customer category decisions.

Usage:
    python run_digital_savings_explanations.py

Features:
    - Loads trained neural network model
    - Generates explanations for sample customers
    - Uses feature importance fallback (no SHAP/LIME)
    - Prints results to console (no disk saving)
    - Privacy-preserving explanations
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main explainer
from digital_savings_feature_explainer import main

if __name__ == "__main__":
    main() 