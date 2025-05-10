#!/bin/bash

# Set up paths
BASE_DIR="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889"
CONFIG_PATH="\${BASE_DIR}/huber_config.json"
LOG_PATH="\${BASE_DIR}/huber_test_log.txt"
PYTHON="\${BASE_DIR}/venv/bin/python"
MODEL_SCRIPT="\${BASE_DIR}/model_training.py"

# Create log file
echo "HUBER LOSS FUNCTION TEST" > \$LOG_PATH
echo "Date: \$(date)" >> \$LOG_PATH
echo "Config file: \${CONFIG_PATH}" >> \$LOG_PATH
echo "=======================================================" >> \$LOG_PATH

echo "Running model training with huber loss function..."
echo "Parameters: huber_delta=1.0"

# Run the model training directly
python \$MODEL_SCRIPT --config \$CONFIG_PATH 2>&1 | tee -a \$LOG_PATH

echo "=======================================================" >> \$LOG_PATH
echo "Test completed."
