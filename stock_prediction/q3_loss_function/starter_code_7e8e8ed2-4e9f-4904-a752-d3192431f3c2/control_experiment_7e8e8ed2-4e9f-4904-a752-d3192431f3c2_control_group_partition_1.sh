#!/bin/bash

# Control Experiment Script for LightGBM Loss Function Comparison
# This script runs the control group experiment with regression_l2 (MSE) loss function

# Define paths
WORKSPACE_DIR="/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2"
RESULTS_FILE="${WORKSPACE_DIR}/results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_control_group_partition_1.txt"
CONFIG_FILE="${WORKSPACE_DIR}/control_group_config.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="${WORKSPACE_DIR}/experiment_results/${TIMESTAMP}_control_group"

# Function to log messages
log_message() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1" | tee -a "$RESULTS_FILE"
}

# Function to handle errors
handle_error() {
    log_message "ERROR: $1"
    log_message "Experiment failed with exit code $2"
    exit $2
}

# Start experiment
{
    log_message "Starting control group experiment (regression_l2 loss function)"
    log_message "Timestamp: ${TIMESTAMP}"
    
    # Create results directory
    log_message "Creating results directory: ${RESULTS_DIR}"
    mkdir -p "${RESULTS_DIR}" || handle_error "Failed to create results directory" $?
    
    # Copy configuration file to results directory for reproducibility
    log_message "Saving configuration file for reproducibility"
    cp "${CONFIG_FILE}" "${RESULTS_DIR}/control_group_config.json" || handle_error "Failed to copy configuration file" $?
    
    # Setup OpenCL for GPU support
    log_message "Setting up OpenCL for GPU support"
    mkdir -p /etc/OpenCL/vendors || handle_error "Failed to create OpenCL directory" $?
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd || handle_error "Failed to create nvidia.icd file" $?
    
    # Activate environment
    log_message "Activating micromamba environment"
    export PATH="/openhands/micromamba/bin:$PATH"
    eval "$(micromamba shell hook --shell bash)"
    export VIRTUAL_ENV="/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2/venv"
    micromamba activate $VIRTUAL_ENV
    
    # Run the control experiment script
    log_message "Running control experiment script"
    cd "${WORKSPACE_DIR}" || handle_error "Failed to change directory" $?
    
    "${VIRTUAL_ENV}/bin/python" "${WORKSPACE_DIR}/run_control_experiment.py" || handle_error "Control experiment failed" $?
    
    # Find the latest metrics file
    LATEST_METRICS=$(find "${WORKSPACE_DIR}/results" -name "metrics_*.json" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -z "$LATEST_METRICS" ]; then
        handle_error "No metrics file found" 1
    fi
    
    # Copy results to experiment directory
    log_message "Copying results to experiment directory"
    cp "$LATEST_METRICS" "${RESULTS_DIR}/" || handle_error "Failed to copy metrics file" $?
    
    # Extract and display key metrics
    log_message "Extracting key metrics from: $(basename "$LATEST_METRICS")"
    
    # Extract overall rank correlation
    OVERALL_CORR=$(grep -o '"overall": [0-9.-]*' "$LATEST_METRICS" | cut -d' ' -f2)
    log_message "Overall Rank Correlation: ${OVERALL_CORR}"
    
    # Extract yearly metrics
    log_message "Yearly Rank Correlations:"
    for YEAR in {2020..2023}; do
        YEAR_CORR=$(grep -o "\"$YEAR\": [0-9.-]*" "$LATEST_METRICS" | cut -d' ' -f2)
        if [ ! -z "$YEAR_CORR" ]; then
            log_message "  $YEAR: ${YEAR_CORR}"
        fi
    done
    
    # Save summary to results directory
    log_message "Saving summary to results directory"
    {
        echo "Control Group Experiment Summary"
        echo "================================"
        echo "Date: $(date)"
        echo "Configuration: regression_l2 loss function"
        echo ""
        echo "Overall Rank Correlation: ${OVERALL_CORR}"
        echo ""
        echo "Yearly Rank Correlations:"
        for YEAR in {2020..2023}; do
            YEAR_CORR=$(grep -o "\"$YEAR\": [0-9.-]*" "$LATEST_METRICS" | cut -d' ' -f2)
            if [ ! -z "$YEAR_CORR" ]; then
                echo "  $YEAR: ${YEAR_CORR}"
            fi
        done
    } > "${RESULTS_DIR}/summary.txt"
    
    log_message "Experiment completed successfully"
    log_message "Results saved to: ${RESULTS_DIR}"
    log_message "Full logs available at: ${RESULTS_FILE}"

} 2>&1 | tee -a "$RESULTS_FILE"

exit 0