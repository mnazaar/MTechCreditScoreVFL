#!/usr/bin/env python3
"""
Home Loans Feature Explanations Runner Script

This script runs the home loans feature importance explanations
and provides a simple interface for generating SHAP explanations.

Usage:
    python run_home_loans_explanations.py

Features:
    - Loads the home loans model and feature names
    - Generates explanations for sample customers
    - Prints results to console and logs to disk
    - Provides detailed feature importance analysis
"""

import os
import sys
import logging
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging for the runner script"""
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/home_loans_runner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function to run home loans explanations"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("HOME LOANS FEATURE EXPLANATIONS RUNNER")
    logger.info("=" * 80)
    logger.info("Starting home loans feature importance explanations...")
    logger.info("Mode: Print and log to disk")
    logger.info("=" * 80)
    
    try:
        # Import and run the home loans explainer
        from home_loans_feature_explainer import main as run_explanations
        
        logger.info("Importing home loans feature explainer...")
        run_explanations()
        
        logger.info("=" * 80)
        logger.info("HOME LOANS EXPLANATIONS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("Results have been printed to console and logged to disk.")
        logger.info("Check the logs directory for detailed log files.")
        logger.info("=" * 80)
        
    except ImportError as e:
        logger.error(f"Failed to import home loans feature explainer: {str(e)}")
        logger.error("Make sure home_loans_feature_explainer.py is in the same directory.")
        return 1
    except Exception as e:
        logger.error(f"Error running home loans explanations: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 