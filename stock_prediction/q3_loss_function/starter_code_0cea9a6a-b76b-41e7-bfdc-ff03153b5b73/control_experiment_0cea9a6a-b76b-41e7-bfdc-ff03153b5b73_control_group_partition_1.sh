#!/bin/bash

# Simple Control Experiment Script for LightGBM Model
# Experiment ID: 0cea9a6a-b76b-41e7-bfdc-ff03153b5b73
# Group: control_group_partition_1

# Define paths
WORKSPACE="/workspace/starter_code_0cea9a6a-b76b-41e7-bfdc-ff03153b5b73"
CONFIG_PATH="$WORKSPACE/control_group_config.json"
OUTPUT_FILE="$WORKSPACE/results_0cea9a6a-b76b-41e7-bfdc-ff03153b5b73_control_group_partition_1.txt"
VENV_PATH="$WORKSPACE/venv"

# Set up environment
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VIRTUAL_ENV="$VENV_PATH"

# Create required directories
mkdir -p "$WORKSPACE/results"

# Set up GPU support for LightGBM
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Start experiment and redirect all output to the results file
{
    echo "Starting simple control experiment for stock return prediction using LightGBM..."
    echo "Configuration: Using control_group_config.json"
    echo "Timestamp: $(date)"
    echo "----------------------------------------"

    # Activate environment and run the model training script
    micromamba activate "$VENV_PATH/"
    cd "$WORKSPACE"
    "$VENV_PATH/bin/python" "$WORKSPACE/model_training.py" --config "$CONFIG_PATH"

    echo "----------------------------------------"
    echo "Control experiment completed at: $(date)"

    # Find and display the latest metrics file
    LATEST_METRICS=$(ls -t "$WORKSPACE/results/metrics_"*.json 2>/dev/null | head -n 1)
    if [ -n "$LATEST_METRICS" ]; then
        echo "Results saved to: $LATEST_METRICS"
    else
        echo "No metrics file found. Check for errors in the experiment run."
    fi
} > "$OUTPUT_FILE" 2>&1