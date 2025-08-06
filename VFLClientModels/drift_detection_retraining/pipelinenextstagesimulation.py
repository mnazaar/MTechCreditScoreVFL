import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Generate filename with datetime
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"pipeline_next_stage_simulation_{current_datetime}.log"
log_filepath = os.path.join(log_dir, log_filename)

# Configure logger
logger = logging.getLogger('PipelineNextStageSimulation')
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log the execution with fun icon
logger.info("🚀 Pipeline Next Stage Simulation Started!")
logger.info(f"📁 Log file created: {log_filepath}")
logger.info("✨ This is executed")