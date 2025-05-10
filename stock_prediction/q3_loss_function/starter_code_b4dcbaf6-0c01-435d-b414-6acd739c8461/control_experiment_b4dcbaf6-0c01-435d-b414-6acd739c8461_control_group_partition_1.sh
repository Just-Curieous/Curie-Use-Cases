#!/bin/bash

# Control experiment script for stock return prediction using LightGBM with regression_l2 (MSE) loss
# This script runs the entire workflow end-to-end for the control group

# Define paths
WORKSPACE_DIR="/workspace/starter_code_b4dcbaf6-0c01-435d-b414-6acd739c8461"
CONFIG_FILE="${WORKSPACE_DIR}/control_group_config.json"
RESULTS_FILE="${WORKSPACE_DIR}/results_b4dcbaf6-0c01-435d-b414-6acd739c8461_control_group_partition_1.txt"
VENV_PATH="${WORKSPACE_DIR}/venv"

# Redirect all output to the results file
exec > "${RESULTS_FILE}" 2>&1

echo "==================================================================="
echo "Starting control experiment workflow for stock return prediction"
echo "Using LightGBM with regression_l2 (MSE) loss function"
echo "==================================================================="
echo "Start time: $(date)"
echo ""

# Setup environment
echo "Setting up environment..."
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
export VIRTUAL_ENV="${VENV_PATH}"
micromamba activate ${VENV_PATH}/

# Setup GPU support for LightGBM
echo "Setting up GPU support for LightGBM..."
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Create results directory
echo "Creating results directory..."
mkdir -p "${WORKSPACE_DIR}/results"

# Run the model training with the control group configuration
echo "Starting model training with regression_l2 loss function..."
cd "${WORKSPACE_DIR}"
if ! ${VENV_PATH}/bin/python model_training.py --config "${CONFIG_FILE}"; then
    echo "ERROR: Model training failed. Check the logs above for details."
    echo "==================================================================="
    echo "End time: $(date)"
    echo "Control experiment workflow failed"
    echo "==================================================================="
    exit 1
fi

echo "Model training completed successfully."

# Extract and summarize key metrics from the latest results file
echo ""
echo "==================================================================="
echo "Experiment Results Summary"
echo "==================================================================="

# Find the latest metrics file
LATEST_METRICS=$(ls -t "${WORKSPACE_DIR}/results/metrics_"*.json 2>/dev/null | head -n 1)

if [ -f "${LATEST_METRICS}" ]; then
    echo "Results file: ${LATEST_METRICS}"
    echo ""
    echo "Key Metrics:"
    
    # Extract metrics using Python for more robust JSON parsing
    ${VENV_PATH}/bin/python -c "
import json
import sys

try:
    with open('${LATEST_METRICS}', 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    
    # Print overall rank correlation
    print(f\"Overall Rank Correlation: {metrics.get('overall', 'N/A')}\")
    
    # Print yearly metrics
    print(\"Yearly Rank Correlations:\")
    years = sorted([k for k in metrics.keys() if k != 'overall'])
    for year in years:
        print(f\"  {year}: {metrics.get(year, 'N/A')}\")
        
except Exception as e:
    print(f\"Error parsing metrics file: {e}\")
    sys.exit(1)
"
else
    echo "No metrics file found. Check for errors in the training process."
fi

echo ""
echo "==================================================================="
echo "End time: $(date)"
echo "Control experiment workflow completed"
echo "==================================================================="