#!/bin/bash

# Control experiment script for stock return prediction using ensemble methods
# Control group (partition_1): 
# - Ensemble architecture: averaging
# - Base models: LightGBM only (single model)
# - Feature selection: all features

# Define paths
WORKSPACE_DIR="/workspace/starter_code_c0338e9a-0531-43c2-9a5d-5a6ba5156e3b"
VENV_PATH="/workspace/starter_code_c0338e9a-0531-43c2-9a5d-5a6ba5156e3b/venv"
CONFIG_PATH="${WORKSPACE_DIR}/control_group_config.json"
RESULTS_FILE="${WORKSPACE_DIR}/results_c0338e9a-0531-43c2-9a5d-5a6ba5156e3b_control_group_partition_1.txt"

# Ensure results directory exists
mkdir -p "${WORKSPACE_DIR}/results"

# Redirect all output to the results file
exec > "${RESULTS_FILE}" 2>&1

echo "Starting control experiment at $(date)"
echo "Control group (partition_1): LightGBM only, all features, averaging ensemble"

# Setup environment
echo "Setting up environment..."
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate ${VENV_PATH}/

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Setup OpenCL for efficient model training
echo "Setting up OpenCL..."
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Run the model training with the control group configuration
echo "Starting model training with control group configuration..."
${VENV_PATH}/bin/python ${WORKSPACE_DIR}/model_training.py --config ${CONFIG_PATH}

echo "Control experiment completed at $(date)"