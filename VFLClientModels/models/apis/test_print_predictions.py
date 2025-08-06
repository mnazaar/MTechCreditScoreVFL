import os
import joblib
import logging
import time
from datetime import datetime

# Set up logging
log_dir = 'VFLClientModels/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'test_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'  # Add UTF-8 encoding
)
logger = logging.getLogger(__name__)

_CACHE_PATH = 'VFLClientModels/saved_models/prediction_cache.pkl'

# Load prediction cache if it exists
if os.path.exists(_CACHE_PATH):
    _prediction_cache = joblib.load(_CACHE_PATH)
    msg = "🔄 Loaded prediction cache from {} ({} entries)".format(_CACHE_PATH, len(_prediction_cache)).encode('utf-8').decode('utf-8')
    print(msg)
    logger.info(msg)

    # Iterate and print cache values
    for tax_id, prediction in _prediction_cache.items():
        msg = "📊 Tax ID: {}".format(tax_id).encode('utf-8').decode('utf-8')
        print(msg)
        logger.info(msg)
        msg = "   Predicted Score: {}".format(prediction['predicted']).encode('utf-8').decode('utf-8')
        print(msg)
        logger.info(msg)
        msg = "   68% CI: {}".format(prediction['68_CI']).encode('utf-8').decode('utf-8')
        print(msg)
        logger.info(msg)
        msg = "   95% CI: {}".format(prediction['95_CI']).encode('utf-8').decode('utf-8')
        print(msg)
        logger.info(msg)
        if 'actual' in prediction:
            msg = "   Actual Score: {}".format(prediction['actual']).encode('utf-8').decode('utf-8')
            print(msg)
            logger.info(msg)
        print("---")
        logger.info("---")
else:
    _prediction_cache = {}
    msg = "⚠️ Initialized empty prediction cache".encode('utf-8').decode('utf-8')
    print(msg)
    logger.warning(msg)
    msg = "📝 Logs will be written to {}".format(log_file).encode('utf-8').decode('utf-8')
    print(msg)
    logger.info(msg)
