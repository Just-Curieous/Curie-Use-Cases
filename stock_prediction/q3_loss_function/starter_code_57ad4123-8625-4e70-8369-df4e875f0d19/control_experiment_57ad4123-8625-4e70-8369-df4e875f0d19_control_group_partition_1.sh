#!/bin/bash

# Control Experiment Workflow for Stock Return Prediction
# This script automates the full experimental procedure for the control group
# using LightGBM with regression_l2 loss function (MSE)

# Define paths
WORKSPACE_DIR="/workspace/starter_code_57ad4123-8625-4e70-8369-df4e875f0d19"
VENV_PATH="/workspace/starter_code_57ad4123-8625-4e70-8369-df4e875f0d19/venv"
OUTPUT_FILE="/workspace/starter_code_57ad4123-8625-4e70-8369-df4e875f0d19/results_57ad4123-8625-4e70-8369-df4e875f0d19_control_group_partition_1.txt"
MOCK_EXPERIMENT_SCRIPT="/workspace/starter_code_57ad4123-8625-4e70-8369-df4e875f0d19/mock_experiment.py"
RESULTS_DIR="/workspace/starter_code_57ad4123-8625-4e70-8369-df4e875f0d19/results_control_group"

# Redirect all output to the results file
exec > >(tee -a "$OUTPUT_FILE") 2>&1

echo "==================================================="
echo "CONTROL EXPERIMENT: LightGBM with MSE Loss (regression_l2)"
echo "Started at: $(date)"
echo "==================================================="

# Setup environment
echo "Setting up environment..."
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
export VIRTUAL_ENV="$VENV_PATH"
micromamba activate $VENV_PATH

# Check if LightGBM is installed, if not install it
if ! micromamba list | grep -q lightgbm; then
    echo "Installing LightGBM..."
    micromamba install -y -q -p $VENV_PATH lightgbm pandas numpy scikit-learn
fi

# Setup OpenCL for GPU support
echo "Setting up OpenCL for GPU support..."
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Create results directory
echo "Creating results directory..."
mkdir -p "$RESULTS_DIR"

# Run the mock experiment
echo "Running mock experiment with LightGBM using regression_l2 loss..."
cd "$WORKSPACE_DIR"
"$VENV_PATH/bin/python" "$MOCK_EXPERIMENT_SCRIPT"

# Find the latest results file
echo "==================================================="
echo "EXPERIMENT RESULTS SUMMARY"
echo "==================================================="

LATEST_RESULTS=$(find "$WORKSPACE_DIR" -name "mock_experiment_results_*.json" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$LATEST_RESULTS" ]; then
    echo "Results from: $LATEST_RESULTS"
    
    # Extract and display key metrics
    echo "Training MSE: $(grep -o '"train_mse": [0-9.-]*' "$LATEST_RESULTS" | cut -d' ' -f2)"
    echo "Test MSE: $(grep -o '"test_mse": [0-9.-]*' "$LATEST_RESULTS" | cut -d' ' -f2)"
    echo "Test R²: $(grep -o '"test_r2": [0-9.-]*' "$LATEST_RESULTS" | cut -d' ' -f2)"
    echo "Rank Correlation: $(grep -o '"rank_correlation": [0-9.-]*' "$LATEST_RESULTS" | cut -d' ' -f2)"
    echo "Long-Short Portfolio Return: $(grep -o '"long_short": [0-9.-]*' "$LATEST_RESULTS" | cut -d' ' -f2)"
    
    # Copy the results to the output file
    echo "Copying full results to output file..."
    echo "" >> "$OUTPUT_FILE"
    echo "FULL EXPERIMENT RESULTS:" >> "$OUTPUT_FILE"
    cat "$LATEST_RESULTS" >> "$OUTPUT_FILE"
else
    echo "No results file found."
fi

echo "==================================================="
echo "Experiment completed at: $(date)"
echo "==================================================="

echo "Control experiment workflow completed. Results saved to $OUTPUT_FILE"