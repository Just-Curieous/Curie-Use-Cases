#!/bin/bash

# Experimental Group Partition 1 Script
# This script runs the LightGBM ensemble models with different loss functions and ensemble methods

# Define paths
WORKSPACE_DIR="/workspace/starter_code_9edf2157-19fd-40d4-a07e-50e075a5e58f"
OUTPUT_FILE="${WORKSPACE_DIR}/results_9edf2157-19fd-40d4-a07e-50e075a5e58f_experimental_group_partition_1.txt"
PYTHON_PATH="${WORKSPACE_DIR}/venv/bin/python"

# Configuration files
CONFIG_MSE_MAE_HUBER_AVG="${WORKSPACE_DIR}/config_mse_mae_huber_averaging.json"
CONFIG_MSE_MAE_HUBER_STACK="${WORKSPACE_DIR}/config_mse_mae_huber_stacking.json"
CONFIG_MSE_QUANTILE_AVG="${WORKSPACE_DIR}/config_mse_quantile_averaging.json"
CONFIG_MSE_QUANTILE_STACK="${WORKSPACE_DIR}/config_mse_quantile_stacking.json"
CONFIG_MSE_RANKCORR_AVG="${WORKSPACE_DIR}/config_mse_rankcorr_averaging.json"

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
echo "EXPERIMENTAL GROUP PARTITION 1: ENSEMBLE METHODS" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"
echo "Starting experiments at $(date)" | tee -a "${OUTPUT_FILE}"
echo "Dataset path: /workspace/starter_code_dataset" | tee -a "${OUTPUT_FILE}"
echo "Running 5 different ensemble configurations:" | tee -a "${OUTPUT_FILE}"
echo "1. MSE+MAE+Huber with averaging" | tee -a "${OUTPUT_FILE}"
echo "2. MSE+MAE+Huber with stacking" | tee -a "${OUTPUT_FILE}"
echo "3. MSE+Quantile(0.1,0.5,0.9) with averaging" | tee -a "${OUTPUT_FILE}"
echo "4. MSE+Quantile(0.1,0.5,0.9) with stacking" | tee -a "${OUTPUT_FILE}"
echo "5. MSE+RankCorrelation with averaging" | tee -a "${OUTPUT_FILE}"
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

# Run all experiments

# Experiment 1: MSE+MAE+Huber with averaging
run_experiment "${CONFIG_MSE_MAE_HUBER_AVG}" "MSE+MAE+Huber with averaging ensemble"

# Experiment 2: MSE+MAE+Huber with stacking
run_experiment "${CONFIG_MSE_MAE_HUBER_STACK}" "MSE+MAE+Huber with stacking ensemble"

# Experiment 3: MSE+Quantile(0.1,0.5,0.9) with averaging
run_experiment "${CONFIG_MSE_QUANTILE_AVG}" "MSE+Quantile(0.1,0.5,0.9) with averaging ensemble"

# Experiment 4: MSE+Quantile(0.1,0.5,0.9) with stacking
run_experiment "${CONFIG_MSE_QUANTILE_STACK}" "MSE+Quantile(0.1,0.5,0.9) with stacking ensemble"

# Experiment 5: MSE+RankCorrelation with averaging
run_experiment "${CONFIG_MSE_RANKCORR_AVG}" "MSE+RankCorrelation with averaging ensemble"

# Print summary
echo "" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"
echo "ALL EXPERIMENTS COMPLETED" | tee -a "${OUTPUT_FILE}"
echo "Results saved to ${OUTPUT_FILE}" | tee -a "${OUTPUT_FILE}"
echo "Experimental group partition 1 finished at $(date)" | tee -a "${OUTPUT_FILE}"
echo "=========================================================" | tee -a "${OUTPUT_FILE}"
