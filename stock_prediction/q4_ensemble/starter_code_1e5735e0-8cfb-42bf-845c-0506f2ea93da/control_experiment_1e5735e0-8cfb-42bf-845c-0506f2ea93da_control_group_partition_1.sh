#!/bin/bash

# Control Group Experiment for Stock Return Prediction
# This script runs the control group experiment using raw factors only with default hyperparameters

# Define output file for logs
OUTPUT_FILE="/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/results_1e5735e0-8cfb-42bf-845c-0506f2ea93da_control_group_partition_1.txt"

# Create results directory if it doesn't exist
mkdir -p "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/results"

# Start timestamp
echo "==================================================" > "$OUTPUT_FILE"
echo "CONTROL GROUP EXPERIMENT - STARTED: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"

# Setup OpenCL environment for GPU acceleration
echo "Setting up OpenCL environment..." >> "$OUTPUT_FILE"
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Activate the environment
echo "Activating micromamba environment..." >> "$OUTPUT_FILE"
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/venv"
micromamba activate "$VENV_PATH/" >> "$OUTPUT_FILE" 2>&1

# Check GPU availability
echo "Checking GPU availability..." >> "$OUTPUT_FILE"
nvidia-smi >> "$OUTPUT_FILE" 2>&1

# Run the experiment using the control group configuration
echo "Starting model training with control group configuration..." >> "$OUTPUT_FILE"
echo "Using configuration: control_group_config.json" >> "$OUTPUT_FILE"

# Execute the model training script
cd /workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/
"$VENV_PATH/bin/python" /workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/model_training.py --config /workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/control_group_config.json >> "$OUTPUT_FILE" 2>&1

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "==================================================" >> "$OUTPUT_FILE"
    echo "CONTROL GROUP EXPERIMENT - COMPLETED SUCCESSFULLY: $(date)" >> "$OUTPUT_FILE"
    echo "==================================================" >> "$OUTPUT_FILE"
    exit 0
else
    echo "==================================================" >> "$OUTPUT_FILE"
    echo "CONTROL GROUP EXPERIMENT - FAILED: $(date)" >> "$OUTPUT_FILE"
    echo "==================================================" >> "$OUTPUT_FILE"
    exit 1
fi