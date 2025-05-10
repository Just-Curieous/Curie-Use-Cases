#!/bin/bash

# Experimental Group Partition 1 Script for LightGBM Loss Function Fix Verification
# This script verifies the fix for the LightGBM loss function issues,
# specifically focusing on the huber loss function with huber_delta=1.0.

# Define paths
WORKSPACE_DIR="/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2"
RESULTS_FILE="${WORKSPACE_DIR}/results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_experimental_group_partition_1.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="${WORKSPACE_DIR}/experiment_results/${TIMESTAMP}_experimental_group_partition_1"

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
    # Initialize results file
    echo "Experimental Group Partition 1 - LightGBM Loss Function Fix Verification - $(date)" > "$RESULTS_FILE"
    echo "=========================================================================" >> "$RESULTS_FILE"
    echo "Verifying the fix for LightGBM loss function issues:" >> "$RESULTS_FILE"
    echo "- Creating a configuration file for huber loss function" >> "$RESULTS_FILE"
    echo "- Setting huber_delta=1.0 parameter" >> "$RESULTS_FILE"
    echo "- Verifying that LightGBM uses this loss function as the objective" >> "$RESULTS_FILE"
    echo "- Outputting results to the experiment results file" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    log_message "Starting LightGBM loss function fix verification experiment"
    log_message "Timestamp: ${TIMESTAMP}"
    
    # Create experiment directory
    log_message "Creating experiment directory: ${EXPERIMENT_DIR}"
    mkdir -p "${EXPERIMENT_DIR}" || handle_error "Failed to create experiment directory" $?
    
    # Setup OpenCL for GPU support - skip if not possible
    log_message "Setting up OpenCL for GPU support (skipping if not possible)"
    mkdir -p /etc/OpenCL/vendors 2>/dev/null || log_message "Note: Could not create OpenCL directory, continuing without GPU support"
    if [ -d "/etc/OpenCL/vendors" ]; then
        echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd 2>/dev/null || log_message "Note: Could not create nvidia.icd file, continuing without GPU support"
    fi
    
    # Activate environment
    log_message "Activating micromamba environment"
    export PATH="/openhands/micromamba/bin:$PATH"
    eval "$(micromamba shell hook --shell bash)"
    export VIRTUAL_ENV="/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2/venv"
    micromamba activate $VIRTUAL_ENV
    
    # Run the verification script
    log_message "Running LightGBM loss function fix verification script"
    cd "${WORKSPACE_DIR}" || handle_error "Failed to change directory" $?
    
    "${VIRTUAL_ENV}/bin/python" "${WORKSPACE_DIR}/verify_loss_function_fix.py" || handle_error "Verification script failed" $?
    
    # Copy verification results to the experiment directory
    log_message "Copying verification results to experiment directory"
    mkdir -p "${EXPERIMENT_DIR}/verification_results" || handle_error "Failed to create verification results directory" $?
    
    # Find the most recent huber verification file
    VERIFICATION_RESULTS=$(find "${WORKSPACE_DIR}/results" -name "huber_verification_*.json" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "$VERIFICATION_RESULTS" ]; then
        cp "$VERIFICATION_RESULTS" "${EXPERIMENT_DIR}/verification_results/" || log_message "Warning: Could not copy verification results"
        log_message "Copied verification results: $(basename "$VERIFICATION_RESULTS")"
    else
        log_message "Warning: No verification results file found"
    fi
    
    # Create a final summary file
    FINAL_SUMMARY="${EXPERIMENT_DIR}/final_summary.txt"
    {
        echo "LightGBM Loss Function Fix Verification - Final Summary"
        echo "======================================================"
        echo "Date: $(date)"
        echo ""
        echo "This experiment verified the fix for LightGBM loss function issues:"
        echo "- Created a configuration file for huber loss function"
        echo "- Set huber_delta=1.0 parameter"
        echo "- Verified that LightGBM uses this loss function as the objective"
        echo "- Output results to the experiment results file"
        echo ""
        echo "Verification Results:"
        echo "-------------------"
        if [ -n "$VERIFICATION_RESULTS" ]; then
            cat "$VERIFICATION_RESULTS"
        else
            echo "No verification results file found. Check the main log for details."
        fi
    } > "$FINAL_SUMMARY"
    
    log_message "Final summary saved to: ${FINAL_SUMMARY}"
    log_message "Experiment completed successfully"
    log_message "Full logs available at: ${RESULTS_FILE}"

} 2>&1 | tee -a "$RESULTS_FILE"

exit 0