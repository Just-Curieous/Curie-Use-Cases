#!/bin/bash

# Control Group Experiment Script
# This script runs the LightGBM model with standard regression loss and no ensemble method

# Define paths
WORKSPACE_DIR="/workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f"
CONFIG_FILE="${WORKSPACE_DIR}/control_group_config.json"
OUTPUT_FILE="${WORKSPACE_DIR}/results_9edf2157-19fd-40d4-a07e-50e075a5e58f_control_group_partition_1.txt"
PYTHON_PATH="/workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f/venv/bin/python"

# Set up environment
echo "Setting up environment..." | tee -a "${OUTPUT_FILE}"
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f/venv"
micromamba activate $VENV_PATH/

# Set up OpenCL for GPU acceleration
echo "Setting up OpenCL for GPU acceleration..." | tee -a "${OUTPUT_FILE}"
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Check GPU availability
echo "Checking GPU availability..." | tee -a "${OUTPUT_FILE}"
nvidia-smi >> "${OUTPUT_FILE}" 2>&1 || echo "No NVIDIA GPU detected" | tee -a "${OUTPUT_FILE}"

# Print experiment configuration
echo "Running control group experiment with standard LightGBM model:" | tee -a "${OUTPUT_FILE}"
echo "- Model type: LightGBM" | tee -a "${OUTPUT_FILE}"
echo "- Loss function: standard regression loss (default objective)" | tee -a "${OUTPUT_FILE}"
echo "- Ensemble method: none (single model)" | tee -a "${OUTPUT_FILE}"
echo "- Configuration file: ${CONFIG_FILE}" | tee -a "${OUTPUT_FILE}"
echo "- Dataset path: /workspace/starter_code_dataset" | tee -a "${OUTPUT_FILE}"

# Print configuration details
echo "Configuration details:" | tee -a "${OUTPUT_FILE}"
cat "${CONFIG_FILE}" | tee -a "${OUTPUT_FILE}"

# Run the model training
echo "Starting model training..." | tee -a "${OUTPUT_FILE}"
cd "${WORKSPACE_DIR}"
"${PYTHON_PATH}" "${WORKSPACE_DIR}/model_training.py" --config "${CONFIG_FILE}" 2>&1 | tee -a "${OUTPUT_FILE}"

# Check if the training was successful
if [ $? -eq 0 ]; then
    echo "Model training completed successfully." | tee -a "${OUTPUT_FILE}"
else
    echo "Model training failed." | tee -a "${OUTPUT_FILE}"
    exit 1
fi

# Print summary
echo "Experiment completed. Results saved to ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"
echo "Control group experiment finished at $(date)" | tee -a "${OUTPUT_FILE}"