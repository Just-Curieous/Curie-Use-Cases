#!/bin/bash

WORKSPACE_DIR="/workspace/quant_code_3178e48d-9a28-4034-b3a4-438eb2192048"
RESULTS_FILE="${WORKSPACE_DIR}/results_3178e48d-9a28-4034-b3a4-438eb2192048_control_group_partition_1.txt"

echo "# Starting experiment with control group parameters" | tee "${RESULTS_FILE}"
echo "# n_estimators=100, subsample=0.8, colsample_bytree=0.8" | tee -a "${RESULTS_FILE}"

# Use python to modify the sample config
python -c "
import json
import sys
import os
import time

# Get the workspace directory
workspace_dir = '${WORKSPACE_DIR}'
timestamp = time.strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join(workspace_dir, f'results_{timestamp}')

# Create the results directory
os.makedirs(results_dir, exist_ok=True)

# Load the sample config
with open(os.path.join(workspace_dir, 'sample_config.json'), 'r') as f:
    config = json.load(f)

# Update the hyperparameters
config['lgbm_params']['n_estimators'] = 100
config['lgbm_params']['subsample'] = 0.8
config['lgbm_params']['colsample_bytree'] = 0.8
config['results_path'] = results_dir

# Save to a temporary file
tmp_config = '/tmp/config_{}.json'.format(timestamp)
with open(tmp_config, 'w') as f:
    json.dump(config, f, indent=4)

# Print the config path
print(tmp_config)
" > /tmp/config_path.txt

CONFIG_PATH=$(cat /tmp/config_path.txt)
echo "Created config file at $CONFIG_PATH" | tee -a "${RESULTS_FILE}"

echo "Running model training..." | tee -a "${RESULTS_FILE}"
"${WORKSPACE_DIR}/venv/bin/python" "${WORKSPACE_DIR}/model_training.py" --config "$CONFIG_PATH" 2>&1 | tee -a "${RESULTS_FILE}"

echo "Experiment completed, results saved to ${RESULTS_FILE}" | tee -a "${RESULTS_FILE}"

# Clean up
rm -f "$CONFIG_PATH" /tmp/config_path.txt

exit 0
