#!/bin/bash

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
export VIRTUAL_ENV="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv"
export VENV_PATH="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv"

# Activate the micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $VENV_PATH/

# Set up OpenCL for GPU support
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Create required directories
mkdir -p /workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/results

# Define paths
CONFIG_PATH="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/control_group_config.json"
RESULTS_FILE="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/results_ac20158a-ee6d-48ad-a2a6-9f6cb203d889_control_group_partition_1.txt"

# Run the model training with the control group configuration
echo "Starting control group experiment with regression_l2 loss function..." | tee -a $RESULTS_FILE
echo "Configuration: Using control_group_config.json" | tee -a $RESULTS_FILE
echo "=======================================================" | tee -a $RESULTS_FILE

# Run the model training script
$VENV_PATH/bin/python /workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/model_training.py --config $CONFIG_PATH 2>&1 | tee -a $RESULTS_FILE

# Extract and display key metrics from the most recent results file
echo "=======================================================" | tee -a $RESULTS_FILE
echo "Experiment completed. Extracting key metrics:" | tee -a $RESULTS_FILE

# Find the most recent metrics file
LATEST_METRICS=$(ls -t /workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/results/metrics_*.json | head -n 1)

if [ -f "$LATEST_METRICS" ]; then
    echo "Results from: $LATEST_METRICS" | tee -a $RESULTS_FILE
    echo "Rank Correlation Metrics:" | tee -a $RESULTS_FILE
    
    # Extract and display overall rank correlation
    OVERALL_CORR=$(grep -o '"overall": [0-9.-]*' $LATEST_METRICS | cut -d' ' -f2)
    echo "Overall Rank Correlation: $OVERALL_CORR" | tee -a $RESULTS_FILE
    
    # Extract and display yearly rank correlations
    for YEAR in {2020..2023}; do
        YEAR_CORR=$(grep -o "\"$YEAR\": [0-9.-]*" $LATEST_METRICS | cut -d' ' -f2)
        if [ ! -z "$YEAR_CORR" ]; then
            echo "$YEAR Rank Correlation: $YEAR_CORR" | tee -a $RESULTS_FILE
        fi
    done
else
    echo "No metrics file found. Check for errors in the experiment." | tee -a $RESULTS_FILE
fi

echo "=======================================================" | tee -a $RESULTS_FILE
echo "Control group experiment completed." | tee -a $RESULTS_FILE