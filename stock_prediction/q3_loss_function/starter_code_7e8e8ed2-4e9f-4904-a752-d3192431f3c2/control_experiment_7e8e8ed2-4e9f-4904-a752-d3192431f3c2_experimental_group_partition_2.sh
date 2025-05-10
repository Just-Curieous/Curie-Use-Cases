#!/bin/bash

# Experimental Group Partition 2 Script for LightGBM Loss Function Verification
# This script verifies two LightGBM loss functions:
# - mape (Mean Absolute Percentage Error)
# - tweedie (Tweedie regression with tweedie_variance_power=1.5)

# Define paths
WORKSPACE_DIR="/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2"
RESULTS_FILE="${WORKSPACE_DIR}/results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_experimental_group_partition_2.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="${WORKSPACE_DIR}/experiment_results/${TIMESTAMP}_experimental_group_partition_2"
VENV_PATH="/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2/venv"

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
    echo "Experimental Group Partition 2 - LightGBM Loss Function Verification - $(date)" > "$RESULTS_FILE"
    echo "=========================================================================" >> "$RESULTS_FILE"
    echo "Verifying two LightGBM loss functions:" >> "$RESULTS_FILE"
    echo "- mape (Mean Absolute Percentage Error)" >> "$RESULTS_FILE"
    echo "- tweedie (Tweedie regression with tweedie_variance_power=1.5)" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    log_message "Starting LightGBM loss function verification experiment for partition 2"
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
    export VIRTUAL_ENV="${VENV_PATH}"
    micromamba activate $VIRTUAL_ENV
    
    # Run the verification script
    log_message "Running LightGBM loss function verification script for partition 2"
    cd "${WORKSPACE_DIR}" || handle_error "Failed to change directory" $?
    
    "${VIRTUAL_ENV}/bin/python" "${WORKSPACE_DIR}/verify_loss_functions_partition2.py" || handle_error "Verification script failed" $?
    
    # Copy verification results to the experiment directory
    log_message "Copying verification results to experiment directory"
    mkdir -p "${EXPERIMENT_DIR}/verification_results" || handle_error "Failed to create verification results directory" $?
    
    # Find the most recent verification files for both loss functions
    MAPE_VERIFICATION=$(find "${WORKSPACE_DIR}/results/mape" -name "mape_verification_*.json" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    TWEEDIE_VERIFICATION=$(find "${WORKSPACE_DIR}/results/tweedie" -name "tweedie_verification_*.json" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    COMPARISON_RESULTS=$(find "${WORKSPACE_DIR}/results" -name "loss_function_comparison_*.json" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    # Copy the verification files if they exist
    if [ -n "$MAPE_VERIFICATION" ]; then
        cp "$MAPE_VERIFICATION" "${EXPERIMENT_DIR}/verification_results/" || log_message "Warning: Could not copy MAPE verification results"
        log_message "Copied MAPE verification results: $(basename "$MAPE_VERIFICATION")"
    else
        log_message "Warning: No MAPE verification results file found"
    fi
    
    if [ -n "$TWEEDIE_VERIFICATION" ]; then
        cp "$TWEEDIE_VERIFICATION" "${EXPERIMENT_DIR}/verification_results/" || log_message "Warning: Could not copy Tweedie verification results"
        log_message "Copied Tweedie verification results: $(basename "$TWEEDIE_VERIFICATION")"
    else
        log_message "Warning: No Tweedie verification results file found"
    fi
    
    if [ -n "$COMPARISON_RESULTS" ]; then
        cp "$COMPARISON_RESULTS" "${EXPERIMENT_DIR}/verification_results/" || log_message "Warning: Could not copy comparison results"
        log_message "Copied comparison results: $(basename "$COMPARISON_RESULTS")"
    else
        log_message "Warning: No comparison results file found"
    fi
    
    # Create a final summary file
    FINAL_SUMMARY="${EXPERIMENT_DIR}/final_summary.txt"
    {
        echo "LightGBM Loss Function Verification - Final Summary"
        echo "======================================================"
        echo "Date: $(date)"
        echo ""
        echo "This experiment verified two LightGBM loss functions:"
        echo "- mape (Mean Absolute Percentage Error)"
        echo "- tweedie (Tweedie regression with tweedie_variance_power=1.5)"
        echo ""
        echo "Verification Results:"
        echo "-------------------"
        
        if [ -n "$MAPE_VERIFICATION" ] && [ -n "$TWEEDIE_VERIFICATION" ]; then
            echo "MAPE Verification:"
            cat "$MAPE_VERIFICATION"
            echo ""
            echo "Tweedie Verification:"
            cat "$TWEEDIE_VERIFICATION"
            echo ""
            
            if [ -n "$COMPARISON_RESULTS" ]; then
                echo "Comparison Results:"
                cat "$COMPARISON_RESULTS"
            else
                echo "No comparison results file found. Check the main log for details."
            fi
        else
            echo "Verification results files not found. Check the main log for details."
        fi
    } > "$FINAL_SUMMARY"
    
    log_message "Final summary saved to: ${FINAL_SUMMARY}"
    log_message "Experiment completed successfully"
    log_message "Full logs available at: ${RESULTS_FILE}"

} 2>&1 | tee -a "$RESULTS_FILE"

exit 0