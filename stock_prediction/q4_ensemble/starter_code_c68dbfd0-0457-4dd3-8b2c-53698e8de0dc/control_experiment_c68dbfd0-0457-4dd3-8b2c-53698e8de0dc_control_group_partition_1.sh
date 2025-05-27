#!/bin/bash

# Control Group Experiment Script for Stock Returns Prediction
# Configuration: Baseline LightGBM with raw factors only and default parameters

# Set up error handling
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Record start time
START_TIME=$(date +%s)
echo "Starting experiment at $(date)"

# Define paths
WORKSPACE="/workspace/starter_code_c68dbfd0-0457-4dd3-8b2c-53698e8de0dc"
CONFIG_FILE="${WORKSPACE}/control_group_config.json"
RESULTS_DIR="${WORKSPACE}/results"
OUTPUT_FILE="${WORKSPACE}/results_c68dbfd0-0457-4dd3-8b2c-53698e8de0dc_control_group_partition_1.txt"
PYTHON_PATH="/opt/micromamba/envs/curie/bin/python"

# Set up OpenCL environment for GPU
echo "Setting up OpenCL environment for GPU..."
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install required packages
echo "Installing required Python packages..."
${PYTHON_PATH} -m pip install pandas numpy scikit-learn lightgbm pyarrow

# Create results directory if it doesn't exist
echo "Creating results directory if it doesn't exist..."
mkdir -p ${RESULTS_DIR}

# Run the model training with the control group configuration
echo "Running model training with control group configuration..."
cd ${WORKSPACE}

# Clear previous output file if it exists
> ${OUTPUT_FILE}

${PYTHON_PATH} ${WORKSPACE}/model_training.py --config ${CONFIG_FILE} >> ${OUTPUT_FILE} 2>&1

# Calculate and display execution time
END_TIME=$(date +%s)
EXECUTION_TIME=$((END_TIME - START_TIME))
echo "Experiment completed at $(date)" >> ${OUTPUT_FILE}
echo "Total execution time: $((EXECUTION_TIME / 60)) minutes and $((EXECUTION_TIME % 60)) seconds" >> ${OUTPUT_FILE}

echo "Experiment completed. Results saved to ${OUTPUT_FILE}"