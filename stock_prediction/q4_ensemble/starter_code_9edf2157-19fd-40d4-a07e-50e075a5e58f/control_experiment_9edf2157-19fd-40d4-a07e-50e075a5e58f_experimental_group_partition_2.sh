#!/bin/bash

# Experimental Group Partition 2 Script
# This script runs the LightGBM ensemble model with MSE+RankCorrelation loss functions and stacking ensemble method

# Define paths
WORKSPACE_DIR="/workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f"
OUTPUT_FILE="${WORKSPACE_DIR}/results_9edf2157-19fd-40d4-a07e-50e075a5e58f_experimental_group_partition_2.txt"
PYTHON_PATH="${WORKSPACE_DIR}/venv/bin/python"

# Configuration file
CONFIG_MSE_RANKCORR_STACK="${WORKSPACE_DIR}/config_mse_rankcorr_stacking.json"

# Clear previous output file if it exists
> "${OUTPUT_FILE}"

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

# Print experiment header
echo "=========================================================" | tee -a "${OUTPUT_FILE}"
echo "EXPERIMENTAL GROUP PARTITION 2: ENSEMBLE METHODS" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"
echo "Starting experiments at $(date)" | tee -a "${OUTPUT_FILE}"
echo "Dataset path: /workspace/starter_code_dataset" | tee -a "${OUTPUT_FILE}"
echo "Running ensemble configuration:" | tee -a "${OUTPUT_FILE}"
echo "1. MSE+RankCorrelation with stacking" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"

# Change to workspace directory
cd "${WORKSPACE_DIR}"

# Function to run an experiment with a specific configuration
run_experiment() {
    local config_file=$1
    local config_name=$2
    
    echo "" | tee -a "${OUTPUT_FILE}"
    echo "=========================================================" | tee -a "${OUTPUT_FILE}"
    echo "RUNNING EXPERIMENT: ${config_name}" | tee -a "${OUTPUT_FILE}"
    echo "Configuration file: ${config_file}" | tee -a "${OUTPUT_FILE}"
    echo "Starting at $(date)" | tee -a "${OUTPUT_FILE}"
    echo "=========================================================" | tee -a "${OUTPUT_FILE}"
    
    # Print configuration details
    echo "Configuration details:" | tee -a "${OUTPUT_FILE}"
    cat "${config_file}" | tee -a "${OUTPUT_FILE}"
    echo "" | tee -a "${OUTPUT_FILE}"
    
    # Run the ensemble model training
    echo "Starting model training..." | tee -a "${OUTPUT_FILE}"
    "${PYTHON_PATH}" "${WORKSPACE_DIR}/ensemble_model_training.py" --config "${config_file}" 2>&1 | tee -a "${OUTPUT_FILE}"
    
    # Check if the training was successful
    if [ $? -eq 0 ]; then
        echo "Model training completed successfully." | tee -a "${OUTPUT_FILE}"
    else
        echo "Model training failed." | tee -a "${OUTPUT_FILE}"
        # Continue with next experiment instead of exiting
    fi
    
    # Get the latest metrics file
    local latest_metrics=$(ls -t ${WORKSPACE_DIR}/results/metrics_*.json | head -1)
    
    # Extract and show metrics if the file exists
    if [ -f "$latest_metrics" ]; then
        echo "Results from $latest_metrics:" | tee -a "${OUTPUT_FILE}"
        cat $latest_metrics | tee -a "${OUTPUT_FILE}"
    else
        echo "No metrics file found for this experiment" | tee -a "${OUTPUT_FILE}"
    fi
    
    echo "Experiment ${config_name} completed at $(date)" | tee -a "${OUTPUT_FILE}"
    echo "=========================================================" | tee -a "${OUTPUT_FILE}"
}

# Run the experiment
run_experiment "${CONFIG_MSE_RANKCORR_STACK}" "MSE+RankCorrelation with stacking ensemble"

# Print summary
echo "" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"
echo "EXPERIMENT COMPLETED" | tee -a "${OUTPUT_FILE}"
echo "Results saved to ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"
echo "Experimental group partition 2 finished at $(date)" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"