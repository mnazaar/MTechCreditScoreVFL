#!/usr/bin/env python3
"""
Credit Card Feature Explanations Runner
=======================================

This script runs feature importance explanations for the credit card XGBoost model.
It provides privacy-preserving insights into which features contributed to credit card tier decisions.

Usage:
    python run_credit_card_explanations.py

Features:
    - Loads trained XGBoost model
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
from credit_card_feature_explainer import main

if __name__ == "__main__":
    main() 