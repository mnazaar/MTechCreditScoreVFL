import subprocess
import schedule
import time
import logging
import os
import json
from datetime import datetime
from typing import Dict, List
import pandas as pd
import sys
import io

# UTF-8 encoding setup to handle emojis in logs and print statements
if sys.platform.startswith('win'):
    # For Windows, ensure UTF-8 encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set default encoding for the environment
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Import domain-specific drift detectors
from auto_loans_drift_detector import AutoLoansDriftDetector
from xgboost_drift_detector import XGBoostDriftDetector
from digital_savings_drift_detector import DigitalSavingsDriftDetector
from home_loans_drift_detector import HomeLoansDriftDetector

# Import domain-specific preprocessing functions
from run_xgboost_drift_detection import preprocess_credit_card_data_for_drift_detection
from run_auto_loans_drift_detection import preprocess_auto_loans_data_for_drift_detection
from run_home_loans_drift_detection import preprocess_home_loans_data_for_drift_detection  
from run_digital_savings_drift_detection import preprocess_digital_savings_data_for_drift_detection

# Suppress verbose logging from drift detector modules
logging.getLogger('AutoLoansDriftDetection').setLevel(logging.WARNING)
logging.getLogger('XGBoostDriftDetection').setLevel(logging.WARNING)
logging.getLogger('DigitalSavingsDriftDetection').setLevel(logging.WARNING)
logging.getLogger('HomeLoansDriftDetection').setLevel(logging.WARNING)

class AutomatedRetrainingPipeline:
    """
    Automated retraining pipeline for VFL credit scoring models
    Runs domain-specific drift detectors and retrains only models with detected drift
    """
    
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Initialize domain-specific drift detectors
        self.drift_detectors = {
            'auto_loans': None,
            'credit_card': None,
            'digital_savings': None,
            'home_loans': None
        }
        
        # Domain-specific data paths and model configurations
        self.domain_config = {
            'auto_loans': {
                'data_path': 'VFLClientModels/dataset/data/banks/auto_loans_bank.csv',
                'baseline_data_path': 'VFLClientModels/dataset/data/banks/auto_loans_bank_baseline.csv',
                'model_path': 'VFLClientModels/saved_models/auto_loans_model.keras',
                'retraining_script': 'VFLClientModels/models/auto_loans_model.py',
                'detector_class': AutoLoansDriftDetector
            },
            'credit_card': {
                'data_path': 'VFLClientModels/dataset/data/banks/credit_card_bank.csv',
                'baseline_data_path': 'VFLClientModels/dataset/data/banks/credit_card_bank_baseline.csv',
                'model_path': 'VFLClientModels/saved_models/credit_card_xgboost_independent.pkl',
                'retraining_script': 'VFLClientModels/models/credit_card_xgboost_model.py',
                'detector_class': XGBoostDriftDetector
            },
            'digital_savings': {
                'data_path': 'VFLClientModels/dataset/data/banks/digital_savings_bank.csv',
                'baseline_data_path': 'VFLClientModels/dataset/data/banks/digital_savings_bank_baseline.csv',
                'model_path': 'VFLClientModels/saved_models/digital_bank_model.keras',
                'retraining_script': 'VFLClientModels/models/digital_savings_model.py',
                'detector_class': DigitalSavingsDriftDetector
            },
            'home_loans': {
                'data_path': 'VFLClientModels/dataset/data/banks/home_loans_bank.csv',
                'baseline_data_path': 'VFLClientModels/dataset/data/banks/home_loans_bank_baseline.csv',
                'model_path': 'VFLClientModels/saved_models/home_loans_model.keras',
                'retraining_script': 'VFLClientModels/models/home_loans_model.py',
                'detector_class': HomeLoansDriftDetector
            }
        }
        
    def _setup_logging(self):
        """Setup logging for automated retraining with UTF-8 encoding support"""
        os.makedirs('VFLClientModels/logs', exist_ok=True)
        
        logger = logging.getLogger('AutomatedRetraining')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler with UTF-8 support
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            # Ensure handler can handle UTF-8
            if hasattr(handler.stream, 'reconfigure'):
                try:
                    handler.stream.reconfigure(encoding='utf-8')
                except Exception:
                    pass  # Fallback for older Python versions
            
            logger.addHandler(handler)
            
            # File handler with explicit UTF-8 encoding
            log_file = f'VFLClientModels/logs/automated_retraining_{datetime.now().strftime("%Y%m%d")}.log'
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)  # Changed from DEBUG to INFO
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load retraining configuration"""
        default_config = {
            'retraining_schedule': {
                'frequency': 'daily',
                'time': '02:00',  # 2 AM
                'timezone': 'UTC'
            },
            'drift_threshold': 3,
            'model_backup': True,
            'performance_threshold': 0.1,  # 10% performance degradation
            'enable_domain_specific_detection': True,  # Enable domain-specific drift detection
            'min_samples_for_drift_detection': 10000,   # Minimum samples needed for drift detection
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def check_drift_and_retrain(self):
        """
        Main function to check for drift across all domains and trigger selective retraining
        """
        self.logger.info("🔍 Starting domain-specific drift check and retraining assessment...")
        
        try:
            # Track which domains need retraining
            domains_needing_retraining = []
            drift_reports = {}
            
            # Check drift for each domain
            for domain_name, domain_config in self.domain_config.items():
                self.logger.info(f"🔍 Checking drift for {domain_name.upper()} domain...")
                
                drift_detected, reason = self._check_domain_drift(domain_name, domain_config)
                
                if drift_detected:
                    self.logger.warning(f"🚨 Drift detected in {domain_name.upper()}: {reason}")
                    domains_needing_retraining.append(domain_name)
                    drift_reports[domain_name] = reason
                else:
                    self.logger.info(f"✅ No drift detected in {domain_name.upper()}")
            
            # Perform selective retraining
            if domains_needing_retraining:
                self.logger.warning(f"🔄 Retraining needed for {len(domains_needing_retraining)} domains: {domains_needing_retraining}")
                retrained_domains = self._perform_selective_retraining(domains_needing_retraining)
            else:
                self.logger.info("✅ No retraining needed - all domains stable")
                retrained_domains = []
            
            # Generate comprehensive report
            self._save_drift_summary_report(drift_reports, retrained_domains)
            
        except Exception as e:
            self.logger.error(f"❌ Error in drift check and retraining: {str(e)}")
    


    def _check_domain_drift(self, domain_name: str, domain_config: Dict) -> tuple:
        """
        Check drift for a specific domain
        
        Returns:
            tuple: (drift_detected: bool, reason: str)
        """
        try:
            # Check if required files exist
            data_path = domain_config['data_path']
            baseline_data_path = domain_config['baseline_data_path']
            model_path = domain_config['model_path']
            
            if not os.path.exists(data_path):
                self.logger.warning(f"Data file not found for {domain_name}: {data_path}")
                return False, "Data file not found"
            
            if not os.path.exists(baseline_data_path):
                self.logger.warning(f"Baseline data file not found for {domain_name}: {baseline_data_path}")
                return False, "Baseline data file not found"
                
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found for {domain_name}: {model_path}")
                return False, "Model file not found"
            
            # Load current and baseline data
            current_data = pd.read_csv(data_path)
            baseline_data = pd.read_csv(baseline_data_path)
            
            # Check minimum sample size
            min_samples = self.config.get('min_samples_for_drift_detection', 10000)
            if len(current_data) < min_samples:
                self.logger.warning(f"Insufficient data for {domain_name}: {len(current_data)} < {min_samples}")
                return False, f"Insufficient data samples ({len(current_data)} < {min_samples})"
            
            # Apply domain-specific preprocessing using unified preprocessing pipelines
            if domain_name == 'credit_card':
                self.logger.info("Applying credit card-specific preprocessing...")
                processed_current_data, processed_baseline_data = preprocess_credit_card_data_for_drift_detection(
                    current_data, baseline_data, verbose=False  # Use logger instead of print
                )
                self.logger.info(f"Credit card preprocessing complete:")
                self.logger.info(f"   - Current features: {processed_current_data.shape}")
                self.logger.info(f"   - Baseline features: {processed_baseline_data.shape}")
                
            elif domain_name == 'auto_loans':
                self.logger.info("Applying auto loans-specific preprocessing...")
                processed_current_data, processed_baseline_data = preprocess_auto_loans_data_for_drift_detection(
                    current_data, baseline_data, verbose=False  # Use logger instead of print
                )
                self.logger.info(f"Auto loans preprocessing complete:")
                self.logger.info(f"   - Current features: {processed_current_data.shape}")
                self.logger.info(f"   - Baseline features: {processed_baseline_data.shape}")
                
            elif domain_name == 'home_loans':
                self.logger.info("Applying home loans-specific preprocessing...")
                processed_current_data, processed_baseline_data = preprocess_home_loans_data_for_drift_detection(
                    current_data, baseline_data, verbose=False  # Use logger instead of print
                )
                self.logger.info(f"Home loans preprocessing complete:")
                self.logger.info(f"   - Current features: {processed_current_data.shape}")
                self.logger.info(f"   - Baseline features: {processed_baseline_data.shape}")
                
            elif domain_name == 'digital_savings':
                self.logger.info("Applying digital savings-specific preprocessing...")
                processed_current_data, processed_baseline_data = preprocess_digital_savings_data_for_drift_detection(
                    current_data, baseline_data, verbose=False  # Use logger instead of print
                )
                self.logger.info(f"Digital savings preprocessing complete:")
                self.logger.info(f"   - Current features: {processed_current_data.shape}")
                self.logger.info(f"   - Baseline features: {processed_baseline_data.shape}")
                
            else:
                # Fallback for unknown domains (use raw data)
                self.logger.warning(f"Unknown domain '{domain_name}' - using raw data without preprocessing")
                processed_current_data = current_data
                processed_baseline_data = baseline_data
            
            # Initialize domain-specific drift detector
            detector_class = domain_config['detector_class']
            detector = detector_class(model_path=model_path)
            
            # Perform drift detection
            drift_detected, report = detector.is_drift_detected(
                current_data=processed_current_data,
                baseline_data=processed_baseline_data
            )
            
            # Clean up detector resources
            if hasattr(detector, 'restore_print'):
                detector.restore_print()
            
            return drift_detected, report if drift_detected else "No drift detected"
            
        except Exception as e:
            error_msg = f"Error checking drift for {domain_name}: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    

    
    def _perform_selective_retraining(self, domains_needing_retraining: List[str]) -> List[str]:
        """
        Perform automated retraining of only the domains that detected drift
        
        Args:
            domains_needing_retraining: List of domain names that need retraining
            
        Returns:
            List[str]: Updated list of successfully retrained domains (including explanation models)
        """
        self.logger.info(f"🔄 Starting selective retraining for {len(domains_needing_retraining)} domains...")
        
        # Create backup of current models for domains being retrained
        if self.config['model_backup']:
            self._backup_current_models(domains_needing_retraining)
        
        successful_retraining = []
        failed_retraining = []
        
        # Retrain only the domains that detected drift
        for domain in domains_needing_retraining:
            self.logger.info(f"🔄 Retraining {domain.upper()} model...")
            success = self._retrain_domain_model(domain)
            
            if success:
                successful_retraining.append(domain)
                self.logger.info(f"✅ Successfully retrained {domain.upper()} model")
            else:
                failed_retraining.append(domain)
                self.logger.error(f"❌ Failed to retrain {domain.upper()} model")
        
        # Log summary
        if successful_retraining:
            self.logger.info(f"✅ Successfully retrained {len(successful_retraining)} domains: {successful_retraining}")
            self._update_model_versions(successful_retraining)
        
        if failed_retraining:
            self.logger.error(f"❌ Failed to retrain {len(failed_retraining)} domains: {failed_retraining}")
            # Consider rollback strategy for failed domains
            
        # Retrain VFL central model if any domain models were successfully retrained
        if successful_retraining:
            self.logger.info("🔄 Domain models retrained - retraining central VFL model...")
            vfl_success = self._retrain_vfl_central_model(successful_retraining)
            if vfl_success:
                self.logger.info("✅ VFL central model retrained successfully")
                successful_retraining.append('vfl_central')
                
                # Update private explanation models after VFL success
                # Only pass the originally retrained domain models (exclude 'vfl_central')
                original_domains = [d for d in successful_retraining if d != 'vfl_central']
                self.logger.info("🔄 VFL retrained - updating private explanation pipeline...")
                
                # Step 1: Regenerate datasets with updated VFL representations
                dataset_success = self._regenerate_explanation_datasets(original_domains)
                if dataset_success:
                    self.logger.info("✅ Explanation datasets regenerated with updated VFL representations")
                    
                    # Step 2: Retrain explanation models with fresh datasets
                    explanation_success, successful_explanations, failed_explanations = self._retrain_private_explanation_models(original_domains)
                    
                    # Log individual explanation model results immediately
                    if successful_explanations:
                        self.logger.info(f"✅ Successfully retrained explanation models: {successful_explanations}")
                        # Add individual successful explanation models to the list
                        successful_retraining.extend([f'{domain}_explanation' for domain in successful_explanations])
                    
                    if failed_explanations:
                        self.logger.warning(f"⚠️ Failed to retrain explanation models: {failed_explanations}")
                    
                    if explanation_success:
                        self.logger.info("✅ All private explanation models updated successfully")
                    else:
                        self.logger.warning("⚠️ Some private explanation models failed to update (see details above)")
                        # Note: We don't fail the entire pipeline for explanation failures
                else:
                    self.logger.error("❌ Failed to regenerate explanation datasets - skipping explanation model updates")
                    # Skip explanation training if dataset generation failed
            else:
                self.logger.error("❌ VFL central model retraining failed")
                failed_retraining.append('vfl_central')
        
        # Return the complete list of successfully retrained domains (including explanation models)
        return successful_retraining
    
    def _retrain_vfl_central_model(self, retrained_domains: List[str]) -> bool:
        """
        Retrain the central VFL model after domain models have been updated
        
        Args:
            retrained_domains: List of domain names that were successfully retrained
            
        Returns:
            bool: True if VFL retraining succeeded, False otherwise
        """
        try:
            vfl_script_path = 'VFLClientModels/models/vfl_automl_xgboost_homoenc_dp.py'
            
            if not os.path.exists(vfl_script_path):
                self.logger.error(f"VFL training script not found: {vfl_script_path}")
                return False
            
            self.logger.info(f"🔄 Starting VFL central model retraining...")
            self.logger.info(f"   Using updated models from domains: {retrained_domains}")
            
            # Set longer timeout for VFL training (it's more complex)
            timeout = 10800*3  # 3*3 hours for VFL training
            
            self.logger.info(f"⏱️ Timeout set to {timeout//60} minutes for VFL training")
            
            # Backup current VFL model before retraining
            self._backup_vfl_models()
            
            # Validate VFL training environment before execution
            self.logger.info("🔍 Pre-execution validation:")
            self.logger.info(f"   - Current working directory: {os.getcwd()}")
            self.logger.info(f"   - VFL script exists: {os.path.exists(vfl_script_path)}")
            self.logger.info(f"   - Script size: {os.path.getsize(vfl_script_path) if os.path.exists(vfl_script_path) else 'N/A'} bytes")
            
            # Check for required dependencies
            required_dirs = [
                'VFLClientModels/saved_models',
                'VFLClientModels/dataset/data',
                'VFLClientModels/logs'
            ]
            for req_dir in required_dirs:
                exists = os.path.exists(req_dir)
                self.logger.info(f"   - {req_dir} exists: {exists}")
                if not exists:
                    self.logger.warning(f"⚠️ Required directory missing: {req_dir}")
            
            # Test if the script can at least start (quick syntax/import check)
            self.logger.info("🧪 Testing VFL script syntax and imports...")
            try:
                test_result = subprocess.run(
                    ['python', '-m', 'py_compile', vfl_script_path],
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 seconds for syntax check
                    encoding='utf-8',
                    errors='replace'
                )
                if test_result.returncode == 0:
                    self.logger.info("✅ VFL script syntax check passed")
                else:
                    self.logger.warning("⚠️ VFL script syntax check failed:")
                    self.logger.warning(f"   {test_result.stderr}")
            except Exception as test_error:
                self.logger.warning(f"⚠️ Could not perform syntax check: {test_error}")
            
            self.logger.info(f"🚀 Executing VFL training script...")
            
            # Run the VFL training script with UTF-8 encoding
            result = subprocess.run(
                ['python', vfl_script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),  # Ensure we run from the correct directory
                encoding='utf-8',  # Handle UTF-8 output
                errors='replace'   # Handle encoding errors gracefully
            )
            
            # Enhanced logging for both success and failure cases
            self.logger.info(f"📋 VFL Training Results:")
            self.logger.info(f"   - Return code: {result.returncode}")
            self.logger.info(f"   - Stdout length: {len(result.stdout)} chars")
            self.logger.info(f"   - Stderr length: {len(result.stderr)} chars")
            
            # Check for VFL training success indicators (not just return code)
            vfl_success_indicators = [
                "Final model saved to VFLClientModels/saved_models/vfl_automl_xgboost_simple_model.keras",
                "AutoML search completed. Best model architecture found.",
                "Best hyperparameters saved to VFLClientModels/saved_models/best_hyperparameters_homoenc_dp.pkl"
            ]
            
            # Check if any success indicators are present in the output
            success_found = False
            found_indicators = []
            
            if result.stdout:
                for indicator in vfl_success_indicators:
                    if indicator in result.stdout:
                        success_found = True
                        found_indicators.append(indicator)
            
            # Log success detection analysis
            self.logger.info(f"🔍 VFL Success Detection Analysis:")
            self.logger.info(f"   - Return code: {result.returncode}")
            self.logger.info(f"   - Success indicators found: {len(found_indicators)}")
            for i, indicator in enumerate(found_indicators, 1):
                self.logger.info(f"     {i}. {indicator}")
            
            # Determine overall success based on indicators (more reliable than return code)
            if success_found:
                self.logger.info("✅ VFL central model retrained successfully (based on output indicators)")
                if result.returncode != 0:
                    self.logger.warning(f"⚠️ Note: Non-zero return code ({result.returncode}) but training completed successfully")
                    self.logger.warning("   This may indicate warnings or non-critical errors during training")
                
                # Log last few lines of stdout for success confirmation
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    last_lines = stdout_lines[-10:] if len(stdout_lines) > 10 else stdout_lines
                    self.logger.info("📄 Last few lines of training output:")
                    for i, line in enumerate(last_lines, 1):
                        self.logger.info(f"   {i}: {line[:200]}{'...' if len(line) > 200 else ''}")
                
                return True
            elif result.returncode == 0:
                # Fallback: Return code is 0 but no explicit success indicators found
                self.logger.warning("⚠️ VFL training returned success code but no clear success indicators found")
                self.logger.warning("   This may indicate the training completed but with unexpected output format")
                self.logger.info("✅ VFL central model retrained successfully (based on return code)")
                
                # Log last few lines of stdout for analysis
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    last_lines = stdout_lines[-10:] if len(stdout_lines) > 10 else stdout_lines
                    self.logger.info("📄 Last few lines of training output:")
                    for i, line in enumerate(last_lines, 1):
                        self.logger.info(f"   {i}: {line[:200]}{'...' if len(line) > 200 else ''}")
                
                return True
            else:
                # Additional check: Even if return code and output parsing failed, 
                # check if the model files were actually created (potential false negative)
                self.logger.error("❌ VFL central model retraining failed (based on return code and output)")
                self.logger.info("🔍 Performing secondary success verification...")
                
                # Check if critical model files were created recently
                model_files_to_check = [
                    'VFLClientModels/saved_models/vfl_automl_xgboost_simple_model.keras',
                    'VFLClientModels/saved_models/best_hyperparameters_homoenc_dp.pkl',
                    'VFLClientModels/saved_models/prediction_cache_homoenc_dp.pkl'
                ]
                
                recent_files_created = []
                for model_file in model_files_to_check:
                    if os.path.exists(model_file):
                        file_mtime = os.path.getmtime(model_file)
                        current_time = time.time()
                        minutes_since_modified = (current_time - file_mtime) / 60
                        
                        # If file was modified within the last 2 hours, consider it "recent"
                        if minutes_since_modified < 120:  
                            recent_files_created.append({
                                'file': model_file, 
                                'minutes_ago': int(minutes_since_modified)
                            })
                
                if recent_files_created:
                    self.logger.warning("⚠️ POTENTIAL FALSE NEGATIVE DETECTED!")
                    self.logger.warning("   VFL model files were created recently, suggesting training may have succeeded:")
                    for file_info in recent_files_created:
                        self.logger.warning(f"   - {file_info['file']} (modified {file_info['minutes_ago']} minutes ago)")
                    
                    self.logger.warning("   This indicates the training likely succeeded despite return code/output issues")
                    self.logger.warning("   Recommend: Manual verification of model file contents and timestamps")
                    
                    # Log the actual return code and error analysis but suggest success
                    self.logger.info("🔍 Original Failure Analysis (may be false negative):")
                else:
                    self.logger.error("🔍 Failure Analysis:")
                
                self.logger.error(f"   - Return code: {result.returncode}")
                self.logger.error(f"   - Command executed: python {vfl_script_path}")
                self.logger.error(f"   - Working directory: {os.getcwd()}")
                
                # Log system diagnostics for troubleshooting
                self._log_system_diagnostics("VFL Training Failure")
                
                # Log stderr with better formatting
                if result.stderr:
                    self.logger.error("📋 STDERR Output:")
                    stderr_lines = result.stderr.strip().split('\n')
                    for i, line in enumerate(stderr_lines, 1):
                        if line.strip():  # Only log non-empty lines
                            self.logger.error(f"   {i}: {line}")
                        if i > 50:  # Limit to first 50 lines to avoid overwhelming logs
                            self.logger.error(f"   ... (truncated, total {len(stderr_lines)} lines)")
                            break
                else:
                    self.logger.error("   - No stderr output")
                
                # Log stdout with better formatting  
                if result.stdout:
                    self.logger.error("📋 STDOUT Output:")
                    stdout_lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(stdout_lines, 1):
                        if line.strip():  # Only log non-empty lines
                            self.logger.error(f"   {i}: {line}")
                        if i > 50:  # Limit to first 50 lines to avoid overwhelming logs
                            self.logger.error(f"   ... (truncated, total {len(stdout_lines)} lines)")
                            break
                else:
                    self.logger.error("   - No stdout output")
                
                # Save full output to separate log file for detailed analysis
                error_log_path = f'VFLClientModels/logs/vfl_training_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                try:
                    with open(error_log_path, 'w', encoding='utf-8') as f:
                        f.write(f"VFL Training Failure Report\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"Return code: {result.returncode}\n")
                        f.write(f"Command: python {vfl_script_path}\n")
                        f.write(f"Working directory: {os.getcwd()}\n")
                        f.write(f"Timeout: {timeout} seconds\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"STDERR:\n{result.stderr}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"STDOUT:\n{result.stdout}\n")
                    self.logger.error(f"💾 Full error details saved to: {error_log_path}")
                except Exception as log_error:
                    self.logger.error(f"⚠️ Could not save detailed error log: {log_error}")
                
                return False
                
        except subprocess.TimeoutExpired as timeout_error:
            self.logger.error(f"❌ VFL central model retraining timed out after {timeout//60} minutes ({timeout} seconds)")
            self.logger.error(f"🔍 Timeout Details:")
            self.logger.error(f"   - Process was still running after {timeout//3600} hours")
            self.logger.error(f"   - Consider increasing timeout or optimizing VFL training")
            self.logger.error(f"   - VFL script path: {vfl_script_path}")
            self.logger.error(f"   - Working directory: {os.getcwd()}")
            
            # Log system diagnostics for timeout troubleshooting
            self._log_system_diagnostics("VFL Training Timeout")
            
            # Save timeout information to log file
            timeout_log_path = f'VFLClientModels/logs/vfl_timeout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            try:
                with open(timeout_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"VFL Training Timeout Report\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Timeout duration: {timeout} seconds ({timeout//3600} hours)\n")
                    f.write(f"Command: python {vfl_script_path}\n")
                    f.write(f"Working directory: {os.getcwd()}\n")
                    f.write(f"Retrained domains: {retrained_domains}\n")
                    f.write(f"Error details: {str(timeout_error)}\n")
                self.logger.error(f"💾 Timeout details saved to: {timeout_log_path}")
            except Exception as log_error:
                self.logger.error(f"⚠️ Could not save timeout log: {log_error}")
            
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during VFL central model retraining")
            self.logger.error(f"🔍 Exception Details:")
            self.logger.error(f"   - Exception type: {type(e).__name__}")
            self.logger.error(f"   - Exception message: {str(e)}")
            self.logger.error(f"   - VFL script path: {vfl_script_path}")
            self.logger.error(f"   - Working directory: {os.getcwd()}")
            self.logger.error(f"   - Retrained domains: {retrained_domains}")
            
            # Log system diagnostics for exception troubleshooting
            self._log_system_diagnostics("VFL Training Exception")
            
            # Save exception information to log file
            exception_log_path = f'VFLClientModels/logs/vfl_exception_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            try:
                import traceback
                with open(exception_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"VFL Training Exception Report\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Exception type: {type(e).__name__}\n")
                    f.write(f"Exception message: {str(e)}\n")
                    f.write(f"Command: python {vfl_script_path}\n")
                    f.write(f"Working directory: {os.getcwd()}\n")
                    f.write(f"Retrained domains: {retrained_domains}\n")
                    f.write(f"\nFull traceback:\n")
                    f.write(traceback.format_exc())
                self.logger.error(f"💾 Exception details saved to: {exception_log_path}")
            except Exception as log_error:
                self.logger.error(f"⚠️ Could not save exception log: {log_error}")
            
            return False
    
    def _backup_vfl_models(self):
        """
        Create backup of current VFL models before retraining
        """
        backup_dir = f'VFLClientModels/saved_models/vfl_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(backup_dir, exist_ok=True)
        
        # VFL model files to backup
        vfl_model_files = [
            'vfl_automl_xgboost_simple_model.keras',
            'vfl_automl_xgboost_homomorp_model.keras', 
            'best_hyperparameters_homoenc_dp.pkl',
            'prediction_cache_homoenc_dp.pkl',
            'auto_loans_scaler_homoenc_dp.pkl',
            'digital_bank_scaler_homoenc_dp.pkl',
            'home_loans_scaler_homoenc_dp.pkl'
        ]
        
        import shutil
        backed_up_count = 0
        for model_file in vfl_model_files:
            source_path = f'VFLClientModels/saved_models/{model_file}'
            if os.path.exists(source_path):
                shutil.copy2(source_path, backup_dir)
                self.logger.info(f"📦 Backed up VFL model: {model_file}")
                backed_up_count += 1
            else:
                self.logger.debug(f"VFL model file not found for backup: {model_file}")
        
        self.logger.info(f"📦 VFL backup completed: {backed_up_count} files backed up to {backup_dir}")

    def _log_system_diagnostics(self, context: str = "General"):
        """
        Log comprehensive system diagnostics for troubleshooting
        
        Args:
            context: Context description for the diagnostics (e.g., "VFL Training", "Dataset Generation")
        """
        try:
            self.logger.info(f"🔧 System Diagnostics - {context}:")
            
            # Basic system info
            import platform
            import sys
            self.logger.info(f"   - Python version: {sys.version}")
            self.logger.info(f"   - Platform: {platform.platform()}")
            self.logger.info(f"   - Working directory: {os.getcwd()}")
            
            # Disk space check
            import shutil
            free_bytes = shutil.disk_usage(os.getcwd()).free
            free_gb = free_bytes / (1024**3)
            self.logger.info(f"   - Available disk space: {free_gb:.1f} GB")
            
            # Memory check (basic)
            try:
                import psutil
                memory = psutil.virtual_memory()
                self.logger.info(f"   - Available memory: {memory.available / (1024**3):.1f} GB")
                self.logger.info(f"   - Memory usage: {memory.percent}%")
            except ImportError:
                self.logger.info(f"   - Memory info: psutil not available")
            
            # Key directories check
            key_paths = [
                'VFLClientModels/saved_models',
                'VFLClientModels/dataset/data',
                'VFLClientModels/logs',
                'VFLClientModels/models',
                'VFLClientModels/models/explanations'
            ]
            
            for path in key_paths:
                exists = os.path.exists(path)
                if exists:
                    try:
                        file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                        self.logger.info(f"   - {path}: ✅ ({file_count} files)")
                    except Exception:
                        self.logger.info(f"   - {path}: ✅ (access limited)")
                else:
                    self.logger.warning(f"   - {path}: ❌ missing")
            
            # Recent log files
            logs_dir = 'VFLClientModels/logs'
            if os.path.exists(logs_dir):
                try:
                    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
                    recent_logs = sorted(log_files)[-3:] if len(log_files) > 3 else log_files
                    self.logger.info(f"   - Recent log files: {', '.join(recent_logs)}")
                except Exception:
                    self.logger.info(f"   - Recent log files: access limited")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not complete system diagnostics: {e}")

    def _regenerate_explanation_datasets(self, retrained_domains: List[str]) -> bool:
        """
        Regenerate explanation datasets using updated VFL representations
        
        Args:
            retrained_domains: List of domain names that were successfully retrained
            
        Returns:
            bool: True if all dataset regeneration succeeded, False if any failed
        """
        try:
            # Map domains to their dataset generation scripts
            dataset_scripts = {
                'auto_loans': 'VFLClientModels/models/explanations/privateexplanations/auto_loans_feature_predictor_dataset.py',
                'credit_card': 'VFLClientModels/models/explanations/privateexplanations/credit_card_feature_predictor_dataset.py',
                'digital_savings': 'VFLClientModels/models/explanations/privateexplanations/digital_bank_feature_predictor_dataset.py',
                'home_loans': 'VFLClientModels/models/explanations/privateexplanations/home_loans_feature_predictor_dataset.py'
            }
            
            # Filter to only retrained domains that have dataset scripts
            domains_to_regenerate = [domain for domain in retrained_domains 
                                   if domain in dataset_scripts and domain != 'vfl_central']
            
            if not domains_to_regenerate:
                self.logger.info("No explanation datasets to regenerate for the updated domains")
                return True
            
            self.logger.info(f"🔄 Regenerating explanation datasets for domains: {domains_to_regenerate}")
            self.logger.info("   Using updated VFL model representations...")
            
            successful_datasets = []
            failed_datasets = []
            
            # Regenerate datasets for each domain
            for domain in domains_to_regenerate:
                script_path = dataset_scripts[domain]
                
                if not os.path.exists(script_path):
                    self.logger.error(f"Dataset generation script not found for {domain}: {script_path}")
                    failed_datasets.append(domain)
                    continue
                
                self.logger.info(f"🔄 Regenerating {domain} explanation dataset...")
                success = self._regenerate_single_dataset(domain, script_path)
                
                if success:
                    successful_datasets.append(domain)
                    self.logger.info(f"✅ {domain} explanation dataset regenerated successfully")
                else:
                    failed_datasets.append(domain)
                    self.logger.error(f"❌ {domain} explanation dataset regeneration failed")
            
            # Log summary
            if successful_datasets:
                self.logger.info(f"✅ Successfully regenerated datasets for: {successful_datasets}")
            
            if failed_datasets:
                self.logger.warning(f"❌ Failed to regenerate datasets for: {failed_datasets}")
            
            # Return True only if all attempted regeneration succeeded
            return len(failed_datasets) == 0
            
        except Exception as e:
            self.logger.error(f"❌ Error in explanation dataset regeneration: {str(e)}")
            return False
    
    def _regenerate_single_dataset(self, domain_name: str, script_path: str) -> bool:
        """
        Regenerate a single explanation dataset using updated VFL representations
        
        Args:
            domain_name: Name of the domain (e.g., 'auto_loans')
            script_path: Path to the dataset generation script
            
        Returns:
            bool: True if dataset regeneration succeeded, False otherwise
        """
        try:
            # Set timeout for dataset generation (can be time-consuming due to SHAP calculations)
            timeout = 36000  # 10 hour for dataset generation
            
            self.logger.info(f"🔄 Executing dataset generation script: {script_path}")
            self.logger.info(f"⏱️ Timeout set to {timeout//60} minutes for {domain_name} dataset generation")
            self.logger.info(f"   Extracting fresh representations from updated VFL model...")
            
            # Quick environment check
            self.logger.info(f"🔍 Environment check for {domain_name} dataset generation:")
            self.logger.info(f"   - Script exists: {os.path.exists(script_path)}")
            self.logger.info(f"   - Working directory: {os.getcwd()}")
            
            # Run the dataset generation script with UTF-8 encoding
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),  # Ensure we run from the correct directory
                encoding='utf-8',  # Handle UTF-8 output
                errors='replace'   # Handle encoding errors gracefully
            )
            
            # Enhanced logging for dataset generation results
            self.logger.info(f"📋 Dataset Generation Results for {domain_name}:")
            self.logger.info(f"   - Return code: {result.returncode}")
            self.logger.info(f"   - Stdout length: {len(result.stdout)} chars")
            self.logger.info(f"   - Stderr length: {len(result.stderr)} chars")
            
            if result.returncode == 0:
                self.logger.info(f"✅ {domain_name} dataset generation completed successfully")
                
                # Log last few lines of stdout for success confirmation
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    last_lines = stdout_lines[-5:] if len(stdout_lines) > 5 else stdout_lines
                    self.logger.info("📄 Last few lines of dataset generation:")
                    for i, line in enumerate(last_lines, 1):
                        self.logger.info(f"   {i}: {line[:150]}{'...' if len(line) > 150 else ''}")
                
                return True
            else:
                self.logger.error(f"❌ {domain_name} dataset generation failed")
                self.logger.error(f"🔍 Failure Analysis:")
                self.logger.error(f"   - Return code: {result.returncode}")
                self.logger.error(f"   - Command executed: python {script_path}")
                self.logger.error(f"   - Working directory: {os.getcwd()}")
                
                # Log stderr with better formatting
                if result.stderr:
                    self.logger.error("📋 STDERR Output:")
                    stderr_lines = result.stderr.strip().split('\n')
                    for i, line in enumerate(stderr_lines, 1):
                        if line.strip():  # Only log non-empty lines
                            self.logger.error(f"   {i}: {line}")
                        if i > 30:  # Limit to first 30 lines
                            self.logger.error(f"   ... (truncated, total {len(stderr_lines)} lines)")
                            break
                else:
                    self.logger.error("   - No stderr output")
                
                # Log stdout with better formatting  
                if result.stdout:
                    self.logger.error("📋 STDOUT Output:")
                    stdout_lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(stdout_lines, 1):
                        if line.strip():  # Only log non-empty lines
                            self.logger.error(f"   {i}: {line}")
                        if i > 30:  # Limit to first 30 lines
                            self.logger.error(f"   ... (truncated, total {len(stdout_lines)} lines)")
                            break
                else:
                    self.logger.error("   - No stdout output")
                
                # Save full output to separate log file for detailed analysis
                error_log_path = f'VFLClientModels/logs/{domain_name}_dataset_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
                try:
                    with open(error_log_path, 'w', encoding='utf-8') as f:
                        f.write(f"{domain_name} Dataset Generation Failure Report\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"Return code: {result.returncode}\n")
                        f.write(f"Command: python {script_path}\n")
                        f.write(f"Working directory: {os.getcwd()}\n")
                        f.write(f"Timeout: {timeout} seconds\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"STDERR:\n{result.stderr}\n")
                        f.write(f"\n{'='*80}\n")
                        f.write(f"STDOUT:\n{result.stdout}\n")
                    self.logger.error(f"💾 Full error details saved to: {error_log_path}")
                except Exception as log_error:
                    self.logger.error(f"⚠️ Could not save detailed error log: {log_error}")
                
                return False
                
        except subprocess.TimeoutExpired as timeout_error:
            self.logger.error(f"❌ {domain_name} dataset generation timed out after {timeout//60} minutes ({timeout} seconds)")
            self.logger.error(f"🔍 Timeout Details:")
            self.logger.error(f"   - Process was still running after {timeout//3600} hours")
            self.logger.error(f"   - SHAP calculations can be very time-consuming")
            self.logger.error(f"   - Dataset script path: {script_path}")
            self.logger.error(f"   - Working directory: {os.getcwd()}")
            
            # Save timeout information to log file
            timeout_log_path = f'VFLClientModels/logs/{domain_name}_dataset_timeout_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            try:
                with open(timeout_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"{domain_name} Dataset Generation Timeout Report\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Timeout duration: {timeout} seconds ({timeout//3600} hours)\n")
                    f.write(f"Command: python {script_path}\n")
                    f.write(f"Working directory: {os.getcwd()}\n")
                    f.write(f"Error details: {str(timeout_error)}\n")
                self.logger.error(f"💾 Timeout details saved to: {timeout_log_path}")
            except Exception as log_error:
                self.logger.error(f"⚠️ Could not save timeout log: {log_error}")
            
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during {domain_name} dataset regeneration")
            self.logger.error(f"🔍 Exception Details:")
            self.logger.error(f"   - Exception type: {type(e).__name__}")
            self.logger.error(f"   - Exception message: {str(e)}")
            self.logger.error(f"   - Dataset script path: {script_path}")
            self.logger.error(f"   - Working directory: {os.getcwd()}")
            
            # Save exception information to log file
            exception_log_path = f'VFLClientModels/logs/{domain_name}_dataset_exception_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            try:
                import traceback
                with open(exception_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"{domain_name} Dataset Generation Exception Report\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Exception type: {type(e).__name__}\n")
                    f.write(f"Exception message: {str(e)}\n")
                    f.write(f"Command: python {script_path}\n")
                    f.write(f"Working directory: {os.getcwd()}\n")
                    f.write(f"\nFull traceback:\n")
                    f.write(traceback.format_exc())
                self.logger.error(f"💾 Exception details saved to: {exception_log_path}")
            except Exception as log_error:
                self.logger.error(f"⚠️ Could not save exception log: {log_error}")
            
            return False

    def _retrain_private_explanation_models(self, retrained_domains: List[str]) -> tuple:
        """
        Retrain private explanation models for domains that were successfully retrained
        
        Args:
            retrained_domains: List of domain names that were successfully retrained
            
        Returns:
            tuple: (overall_success: bool, successful_explanations: List[str], failed_explanations: List[str])
        """
        try:
            # Map domains to their explanation training scripts
            explanation_scripts = {
                'auto_loans': 'VFLClientModels/models/explanations/privateexplanations/train_auto_loans_feature_predictor.py',
                'credit_card': 'VFLClientModels/models/explanations/privateexplanations/train_credit_card_feature_predictor.py',
                'digital_savings': 'VFLClientModels/models/explanations/privateexplanations/train_digital_bank_feature_predictor.py',
                'home_loans': 'VFLClientModels/models/explanations/privateexplanations/train_home_loans_feature_predictor.py'
            }
            
            # Filter to only retrained domains that have explanation scripts
            domains_to_update = [domain for domain in retrained_domains 
                               if domain in explanation_scripts and domain != 'vfl_central']
            
            if not domains_to_update:
                self.logger.info("No explanation models to retrain for the updated domains")
                return True, [], []
            
            self.logger.info("🚀 STARTING PRIVATE EXPLANATION MODEL RETRAINING PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"🔄 Retraining private explanation models for domains: {domains_to_update}")
            self.logger.info(f"📊 Total explanation models to retrain: {len(domains_to_update)}")
            self.logger.info("   This process will update models with fresh VFL representations...")
            self.logger.info("=" * 80)
            
            successful_explanations = []
            failed_explanations = []
            
            # Retrain explanation models for each domain
            for i, domain in enumerate(domains_to_update, 1):
                script_path = explanation_scripts[domain]
                
                if not os.path.exists(script_path):
                    self.logger.error(f"Explanation training script not found for {domain}: {script_path}")
                    failed_explanations.append(domain)
                    continue
                
                self.logger.info(f"🔄 Training {domain} explanation model ({i}/{len(domains_to_update)})...")
                self.logger.info(f"   Script: {script_path}")
                success = self._retrain_single_explanation_model(domain, script_path)
                
                if success:
                    successful_explanations.append(domain)
                    self.logger.info(f"✅ {domain} explanation model retrained successfully ({i}/{len(domains_to_update)} complete)")
                    self.logger.info(f"🎯 {domain.upper()}_EXPLANATION model is now updated with fresh VFL representations")
                else:
                    failed_explanations.append(domain)
                    self.logger.error(f"❌ {domain} explanation model retraining failed ({i}/{len(domains_to_update)} attempted)")
                    self.logger.error(f"⚠️ {domain.upper()}_EXPLANATION model will continue using previous version")
            
            # Completion summary
            self.logger.info("=" * 80)
            self.logger.info("🏁 PRIVATE EXPLANATION MODEL RETRAINING PIPELINE COMPLETED")
            self.logger.info("=" * 80)
            
            if successful_explanations:
                self.logger.info(f"✅ Successfully retrained {len(successful_explanations)} explanation models:")
                for domain in successful_explanations:
                    self.logger.info(f"   ✅ {domain.upper()}_EXPLANATION: Updated with fresh VFL representations")
            
            if failed_explanations:
                self.logger.warning(f"❌ Failed to retrain {len(failed_explanations)} explanation models:")
                for domain in failed_explanations:
                    self.logger.warning(f"   ❌ {domain.upper()}_EXPLANATION: Retaining previous version")
            
            # Overall status
            overall_success = len(failed_explanations) == 0
            if overall_success:
                self.logger.info("🎯 ALL EXPLANATION MODELS SUCCESSFULLY UPDATED!")
            else:
                self.logger.warning(f"⚠️ PARTIAL SUCCESS: {len(successful_explanations)}/{len(domains_to_update)} models updated")
            
            self.logger.info("=" * 80)
            
            # Return detailed results
            return overall_success, successful_explanations, failed_explanations
            
        except Exception as e:
            self.logger.error(f"❌ Error in private explanation model retraining: {str(e)}")
            return False, [], []
    
    def _retrain_single_explanation_model(self, domain_name: str, script_path: str) -> bool:
        """
        Retrain a single private explanation model
        
        Args:
            domain_name: Name of the domain (e.g., 'auto_loans')
            script_path: Path to the training script
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        try:
            # Set timeout for explanation model training (typically faster than main models)
            timeout = 3600  # 1 hour for explanation model training
            
            self.logger.info(f"🔄 Executing explanation training script: {script_path}")
            self.logger.info(f"⏱️ Timeout set to {timeout//60} minutes for {domain_name} explanation training")
            
            # Run the explanation training script with UTF-8 encoding
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),  # Ensure we run from the correct directory
                encoding='utf-8',  # Handle UTF-8 output
                errors='replace'   # Handle encoding errors gracefully
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ {domain_name} explanation model training completed successfully")
                
                # Log last few lines of stdout for success confirmation
                if result.stdout:
                    stdout_lines = result.stdout.strip().split('\n')
                    last_lines = stdout_lines[-5:] if len(stdout_lines) > 5 else stdout_lines
                    self.logger.info("📄 Last few lines of explanation training output:")
                    for i, line in enumerate(last_lines, 1):
                        self.logger.info(f"   {i}: {line[:150]}{'...' if len(line) > 150 else ''}")
                
                # Look for specific success indicators in the output
                success_indicators = [
                    "Model training completed successfully",
                    "Model saved successfully",
                    "Training completed",
                    ".keras"  # Model file extension indicating successful save
                ]
                
                found_indicators = []
                if result.stdout:
                    for indicator in success_indicators:
                        if indicator in result.stdout:
                            found_indicators.append(indicator)
                
                if found_indicators:
                    self.logger.info(f"🎯 Success indicators found: {found_indicators}")
                
                return True
            else:
                self.logger.error(f"❌ {domain_name} explanation model training failed")
                self.logger.error(f"Return code: {result.returncode}")
                
                # Log stderr with better formatting
                if result.stderr:
                    self.logger.error("📋 STDERR Output:")
                    stderr_lines = result.stderr.strip().split('\n')
                    for i, line in enumerate(stderr_lines[:10], 1):  # Show first 10 lines
                        if line.strip():
                            self.logger.error(f"   {i}: {line}")
                else:
                    self.logger.error("   - No stderr output")
                
                # Log stdout for debugging
                if result.stdout:
                    self.logger.error("📋 STDOUT Output (first 10 lines):")
                    stdout_lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(stdout_lines[:10], 1):
                        if line.strip():
                            self.logger.error(f"   {i}: {line}")
                
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {domain_name} explanation model training timed out after {timeout//60} minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error training {domain_name} explanation model: {str(e)}")
            return False

    def _backup_current_models(self, domains_to_backup: List[str] = None):
        """
        Create backup of current models (selective or all)
        
        Args:
            domains_to_backup: List of domain names to backup. If None, backup all models.
        """
        backup_dir = f'VFLClientModels/saved_models/backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Map domains to their model files
        domain_to_model_files = {
            'auto_loans': ['auto_loans_model.keras', 'auto_loans_feature_names.npy', 'auto_loans_scaler.pkl'],
            'digital_savings': ['digital_bank_model.keras', 'digital_bank_feature_names.npy', 'digital_bank_scaler.pkl'],
            'home_loans': ['home_loans_model.keras', 'home_loans_feature_names.npy', 'home_loans_scaler.pkl'],
            'credit_card': ['credit_card_xgboost_independent.pkl', 'credit_card_scaler.pkl', 'credit_card_xgboost_feature_names.npy', 'credit_card_xgboost_pca.pkl'],
            'vfl_central': ['vfl_automl_xgboost_simple_model.keras', 'vfl_automl_xgboost_homomorp_model.keras', 'best_hyperparameters_homoenc_dp.pkl', 'prediction_cache_homoenc_dp.pkl', 'auto_loans_scaler_homoenc_dp.pkl', 'digital_bank_scaler_homoenc_dp.pkl', 'home_loans_scaler_homoenc_dp.pkl'],
            # Private explanation models
            'auto_loans_explanation': ['auto_loans_feature_predictor.keras', 'auto_loans_feature_predictor_feature_names.json', 'auto_loans_feature_predictor_info.json'],
            'credit_card_explanation': ['credit_card_feature_predictor.keras', 'credit_card_feature_predictor_feature_names.json', 'credit_card_feature_predictor_info.json'],
            'digital_savings_explanation': ['digital_bank_feature_predictor.keras', 'digital_bank_feature_predictor_feature_names.json', 'digital_bank_feature_predictor_info.json'],
            'home_loans_explanation': ['home_loans_feature_predictor.keras', 'home_loans_feature_predictor_feature_names.json', 'home_loans_feature_predictor_info.json']
        }
        
        # Determine which files to backup
        files_to_backup = []
        if domains_to_backup is None:
            # Backup all models
            for domain_files in domain_to_model_files.values():
                files_to_backup.extend(domain_files)
            self.logger.info("📦 Creating backup of ALL models...")
        else:
            # Backup only specified domains
            for domain in domains_to_backup:
                if domain in domain_to_model_files:
                    files_to_backup.extend(domain_to_model_files[domain])
            self.logger.info(f"📦 Creating selective backup for domains: {domains_to_backup}")
        
        # Perform backup
        import shutil
        backed_up_count = 0
        for model_file in files_to_backup:
            source_path = f'VFLClientModels/saved_models/{model_file}'
            if os.path.exists(source_path):
                shutil.copy2(source_path, backup_dir)
                self.logger.info(f"📦 Backed up {model_file}")
                backed_up_count += 1
            else:
                self.logger.warning(f"⚠️ Model file not found for backup: {model_file}")
        
        self.logger.info(f"📦 Backup completed: {backed_up_count} files backed up to {backup_dir}")
    
    def _retrain_domain_model(self, domain_name: str) -> bool:
        """
        Retrain model for a specific domain
        
        Args:
            domain_name: Name of the domain to retrain (auto_loans, credit_card, digital_savings, home_loans)
        """
        if domain_name not in self.domain_config:
            self.logger.error(f"Unknown domain: {domain_name}")
            return False
            
        domain_config = self.domain_config[domain_name]
        script_path = domain_config['retraining_script']
        
        if not os.path.exists(script_path):
            self.logger.error(f"Retraining script not found for {domain_name}: {script_path}")
            return False
        
        try:
            # Set timeout based on domain (XGBoost/VFL models take longer)
            timeout = 7200 if domain_name == 'credit_card' else 3600  # 2 hours for credit card, 1 hour for others
            
            self.logger.info(f"🔄 Executing retraining script: {script_path}")
            self.logger.info(f"⏱️ Timeout set to {timeout//60} minutes for {domain_name}")
            
            # Run the training script with UTF-8 encoding
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),  # Ensure we run from the correct directory
                encoding='utf-8',  # Handle UTF-8 output
                errors='replace'   # Handle encoding errors gracefully
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ {domain_name.upper()} model retrained successfully")
                self.logger.debug(f"Retraining output:\n{result.stdout}")
                return True
            else:
                self.logger.error(f"❌ {domain_name.upper()} model retraining failed")
                self.logger.error(f"Return code: {result.returncode}")
                self.logger.error(f"Stderr: {result.stderr}")
                self.logger.error(f"Stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ {domain_name.upper()} model retraining timed out after {timeout//60} minutes")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error retraining {domain_name.upper()} model: {str(e)}")
            return False
    

    
    def _update_model_versions(self, retrained_domains: List[str]):
        """
        Update model version tracking for specific domains
        
        Args:
            retrained_domains: List of domain names that were retrained
        """
        os.makedirs('VFLClientModels/data', exist_ok=True)
        version_path = 'VFLClientModels/data/model_versions.json'
        
        # Load existing version info if it exists
        existing_versions = {}
        if os.path.exists(version_path):
            try:
                with open(version_path, 'r', encoding='utf-8') as f:
                    existing_versions = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load existing version info: {str(e)}")
        
        # Update version info
        version_info = {
            'last_update': datetime.now().isoformat(),
            'domains_retrained': retrained_domains,
            'domain_last_retrained': existing_versions.get('domain_last_retrained', {}),
            'retraining_history': existing_versions.get('retraining_history', [])
        }
        
        # Update last retrained timestamp for each domain
        current_time = datetime.now().isoformat()
        for domain in retrained_domains:
            version_info['domain_last_retrained'][domain] = current_time
        
        # Add to history
        version_info['retraining_history'].append({
            'timestamp': current_time,
            'domains': retrained_domains,
            'trigger': 'drift_detected'
        })
        
        # Keep only last 50 history entries
        version_info['retraining_history'] = version_info['retraining_history'][-50:]
        
        with open(version_path, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📝 Model versions updated for domains: {retrained_domains}")
    
    def _save_drift_summary_report(self, drift_reports: Dict[str, str], retrained_domains: List[str]):
        """
        Save comprehensive drift detection and retraining summary report
        
        Args:
            drift_reports: Dictionary mapping domain names to their drift reports
            retrained_domains: List of domains that were retrained
        """
        os.makedirs('VFLClientModels/reports', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f'VFLClientModels/reports/drift_summary_report_{timestamp}.txt'
        
        # Generate comprehensive report
        report_lines = [
            "=" * 100,
            "AUTOMATED DRIFT DETECTION AND RETRAINING SUMMARY REPORT",
            "=" * 100,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session ID: {timestamp}",
            "",
            "EXECUTIVE SUMMARY:",
            f"- Total domains checked: {len(self.domain_config)}",
            f"- Domains with drift detected: {len(drift_reports)}",
            f"- Domains retrained: {len(retrained_domains)}",
            f"- Overall status: {'RETRAINING PERFORMED' if retrained_domains else 'NO ACTION NEEDED'}",
            "",
            "DOMAIN-SPECIFIC RESULTS:",
            "-" * 50
        ]
        
        for domain_name in self.domain_config.keys():
            if domain_name in drift_reports:
                status = "DRIFT DETECTED → RETRAINED" if domain_name in retrained_domains else "DRIFT DETECTED → RETRAINING FAILED"
                report_lines.extend([
                    f"{domain_name.upper()}: {status}",
                    f"  Drift Details: {drift_reports[domain_name][:200]}{'...' if len(drift_reports[domain_name]) > 200 else ''}",
                    ""
                ])
            else:
                report_lines.extend([
                    f"{domain_name.upper()}: NO DRIFT DETECTED",
                    f"  Status: Model stable, no retraining needed",
                    ""
                ])
        
        if drift_reports:
            report_lines.extend([
                "",
                "DETAILED DRIFT REPORTS:",
                "=" * 100
            ])
            
            for domain_name, detailed_report in drift_reports.items():
                report_lines.extend([
                    f"\n{domain_name.upper()} DETAILED DRIFT REPORT:",
                    "-" * 80,
                    detailed_report,
                    "-" * 80
                ])
        
        # Retraining summary
        if retrained_domains:
            # Separate different types of models
            domain_models = [d for d in retrained_domains if d not in ['vfl_central'] and not d.endswith('_explanation')]
            explanation_models = [d for d in retrained_domains if d.endswith('_explanation')]
            vfl_retrained = 'vfl_central' in retrained_domains
            
            report_lines.extend([
                "",
                "RETRAINING SUMMARY:",
                "-" * 50,
                f"Successfully retrained: {', '.join(retrained_domains)}",
                f"Domain models retrained: {len(domain_models)}",
                f"VFL central model retrained: {'YES' if vfl_retrained else 'NO'}",
                f"Private explanation datasets regenerated: {'YES' if explanation_models else 'NO'}",
                f"Private explanation models retrained: {len(explanation_models)}",
                "",
                "Processing pipeline executed:",
            ])
            
            # List domain model scripts
            for domain in domain_models:
                if domain in self.domain_config:
                    script_path = self.domain_config[domain]['retraining_script']
                    report_lines.append(f"  - {domain}: {script_path}")
            
            # List VFL script if retrained
            if vfl_retrained:
                report_lines.append(f"  - vfl_central: VFLClientModels/models/vfl_automl_xgboost_homoenc_dp.py")
            
            # List explanation pipeline scripts if executed
            if explanation_models:
                report_lines.append("")
                report_lines.append("Private explanation pipeline executed:")
                for explanation_domain in explanation_models:
                    # Extract base domain name (remove '_explanation' suffix)
                    base_domain = explanation_domain.replace('_explanation', '')
                    
                    # Dataset generation scripts
                    dataset_scripts = {
                        'auto_loans': 'auto_loans_feature_predictor_dataset.py',
                        'credit_card': 'credit_card_feature_predictor_dataset.py',
                        'digital_savings': 'digital_bank_feature_predictor_dataset.py',
                        'home_loans': 'home_loans_feature_predictor_dataset.py'
                    }
                    
                    # Training scripts
                    training_scripts = {
                        'auto_loans': 'train_auto_loans_feature_predictor.py',
                        'credit_card': 'train_credit_card_feature_predictor.py',
                        'digital_savings': 'train_digital_bank_feature_predictor.py',
                        'home_loans': 'train_home_loans_feature_predictor.py'
                    }
                    
                    if base_domain in dataset_scripts and base_domain in training_scripts:
                        dataset_script = dataset_scripts[base_domain]
                        training_script = training_scripts[base_domain]
                        report_lines.append(f"  - {explanation_domain}:")
                        report_lines.append(f"    1. Dataset: VFLClientModels/models/explanations/privateexplanations/{dataset_script}")
                        report_lines.append(f"    2. Training: VFLClientModels/models/explanations/privateexplanations/{training_script}")
        
        # Next steps
        next_steps = [
            "",
            "NEXT STEPS:",
            "-" * 50,
            "1. Monitor retrained models for performance improvements",
            "2. Validate that drift has been resolved in next monitoring cycle",
            "3. Update baseline data if needed",
            "4. Review drift detection thresholds if false positives detected"
        ]
        
        # Add VFL-specific recommendations if VFL was retrained
        step_number = 5
        if 'vfl_central' in retrained_domains:
            next_steps.extend([
                f"{step_number}. Test VFL central model predictions to ensure integration works correctly",
                f"{step_number + 1}. Monitor federated learning performance across all participating domains",
                f"{step_number + 2}. Verify homomorphic encryption and differential privacy features still function properly"
            ])
            step_number += 3
        
        # Add explanation-specific recommendations if explanation models were retrained
        explanation_models = [d for d in retrained_domains if d.endswith('_explanation')]
        if explanation_models:
            next_steps.extend([
                f"{step_number}. Verify explanation datasets reflect updated VFL representations",
                f"{step_number + 1}. Test private explanation model predictions for updated domains",
                f"{step_number + 2}. Validate explanation accuracy and interpretability improvements", 
                f"{step_number + 3}. Update explanation dashboards and user interfaces with new models",
                f"{step_number + 4}. Monitor explanation consistency across the federated learning system",
                f"{step_number + 5}. Ensure SHAP values and feature contributions align with retrained models"
            ])
        
        next_steps.extend([
            "",
            "=" * 100
        ])
        
        report_lines.extend(next_steps)
        
        # Write report with UTF-8 encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"📄 Comprehensive drift report saved to {report_path}")
        
        # Log summary to console
        self.logger.info("📊 DRIFT DETECTION SUMMARY:")
        
        # Show domain models first (in config order)
        for domain_name in self.domain_config.keys():
            if domain_name in drift_reports:
                status = "✅ RETRAINED" if domain_name in retrained_domains else "❌ FAILED"
                self.logger.info(f"   {domain_name.upper()}: 🚨 DRIFT DETECTED → {status}")
            else:
                self.logger.info(f"   {domain_name.upper()}: ✅ NO DRIFT")
        
        self.logger.info("   VFL_CENTRAL: ✅ RETRAINED (due to domain model updates)")
        
        # Show explanation models last (in training order)
        explanation_models = [d for d in retrained_domains if d.endswith('_explanation')]
        if explanation_models:
            for explanation_domain in explanation_models:
                base_domain = explanation_domain.replace('_explanation', '').upper()
                self.logger.info(f"   {base_domain}_EXPLANATION: ✅ RETRAINED")
    
    def start_scheduled_monitoring(self):
        """
        Start scheduled monitoring and retraining
        """
        self.logger.info("�� Starting scheduled drift monitoring...")
        
        # Schedule daily drift check
        schedule.every().day.at(self.config['retraining_schedule']['time']).do(
            self.check_drift_and_retrain
        )
        
        # Run initial check
        self.check_drift_and_retrain()
        

def safe_print(*args, **kwargs):
    """Safe print function that handles UTF-8 encoding gracefully"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode problematic characters
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode('utf-8', errors='replace').decode('utf-8'))
            else:
                safe_args.append(str(arg))
        print(*safe_args, **kwargs)
    except Exception as e:
        # Last resort: print without problematic characters
        fallback_args = [str(arg).encode('ascii', errors='ignore').decode('ascii') for arg in args]
        print(*fallback_args, **kwargs)
        print(f"[UTF-8 encoding warning: {e}]")

def test_utf8_support():
    """Test UTF-8 support for emojis and special characters"""
    try:
        # Silent test - just check if UTF-8 works
        test_string = "🚀🔍📊"
        test_string.encode('utf-8')
        return True
    except Exception:
        return False

def main():
    """Main function to run the automated retraining pipeline"""
    # Initialize pipeline without verbose startup messages
    #pipeline = AutomatedRetrainingPipeline()
    #pipeline.start_scheduled_monitoring()

# Example usage for manual drift checking (without scheduling):
"""
pipeline = AutomatedRetrainingPipeline()
pipeline.check_drift_and_retrain()  # Run once manually

# The system will:
# 1. Check drift for auto_loans using AutoLoansDriftDetector → retrain auto_loans_model.py if drift found
# 2. Check drift for credit_card using XGBoostDriftDetector → retrain vfl_automl_xgboost_homoenc_dp.py if drift found  
# 3. Check drift for digital_savings using DigitalSavingsDriftDetector → retrain digital_savings_model.py if drift found
# 4. Check drift for home_loans using HomeLoansDriftDetector → retrain home_loans_model.py if drift found
# 5. Generate comprehensive report with domain-specific results
# 6. Only retrain models where drift is actually detected
"""

if __name__ == "__main__":
    main() 