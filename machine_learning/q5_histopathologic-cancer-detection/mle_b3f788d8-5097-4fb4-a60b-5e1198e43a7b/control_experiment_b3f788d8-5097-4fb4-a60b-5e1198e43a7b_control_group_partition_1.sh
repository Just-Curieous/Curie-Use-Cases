#!/bin/bash

# Control experiment script for PatchCamelyon dataset
# Experiment ID: b3f788d8-5097-4fb4-a60b-5e1198e43a7b
# Control Group Partition: 1

# Set error handling
set -e

# Define paths
WORKSPACE_DIR="/workspace/mle_b3f788d8-5097-4fb4-a60b-5e1198e43a7b"
DATASET_DIR="/workspace/mle_dataset"
OUTPUT_DIR="${WORKSPACE_DIR}/output"
VENV_PATH="${WORKSPACE_DIR}/venv"
RESULTS_FILE="${WORKSPACE_DIR}/results_b3f788d8-5097-4fb4-a60b-5e1198e43a7b_control_group_partition_1.txt"
PYTHON_SCRIPT="${WORKSPACE_DIR}/pcam_experiment.py"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Print experiment information
echo "=== PatchCamelyon Cancer Detection Experiment ===" | tee -a "${RESULTS_FILE}"
echo "Experiment ID: b3f788d8-5097-4fb4-a60b-5e1198e43a7b" | tee -a "${RESULTS_FILE}"
echo "Control Group Partition: 1" | tee -a "${RESULTS_FILE}"
echo "Date: $(date)" | tee -a "${RESULTS_FILE}"
echo "====================================================" | tee -a "${RESULTS_FILE}"

# Setup environment
echo "Setting up environment..." | tee -a "${RESULTS_FILE}"
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${VENV_PATH}/"

# Check GPU availability
echo "Checking GPU availability..." | tee -a "${RESULTS_FILE}"
nvidia-smi | tee -a "${RESULTS_FILE}"

# Print experiment configuration
echo -e "\n=== Experiment Configuration ===" | tee -a "${RESULTS_FILE}"
echo "Model: EfficientNetB0" | tee -a "${RESULTS_FILE}"
echo "Optimizer: Adam" | tee -a "${RESULTS_FILE}"
echo "Learning rate: 0.001" | tee -a "${RESULTS_FILE}"
echo "Loss function: Binary Cross-Entropy" | tee -a "${RESULTS_FILE}"
echo "Batch size: 32" | tee -a "${RESULTS_FILE}"
echo "Transfer learning: ImageNet pretrained, fine-tune all layers" | tee -a "${RESULTS_FILE}"
echo "Cross-validation: 5-fold" | tee -a "${RESULTS_FILE}"
echo "====================================================" | tee -a "${RESULTS_FILE}"

# Check dataset
echo -e "\n=== Dataset Information ===" | tee -a "${RESULTS_FILE}"
echo "Dataset location: ${DATASET_DIR}" | tee -a "${RESULTS_FILE}"
echo "Train images count: $(ls ${DATASET_DIR}/train | wc -l)" | tee -a "${RESULTS_FILE}"
echo "Test images count: $(ls ${DATASET_DIR}/test | wc -l)" | tee -a "${RESULTS_FILE}"
echo "====================================================" | tee -a "${RESULTS_FILE}"

# Run the experiment
echo -e "\n=== Running Experiment ===" | tee -a "${RESULTS_FILE}"
echo "Start time: $(date)" | tee -a "${RESULTS_FILE}"

# Execute the Python script
"${VENV_PATH}/bin/python" "${PYTHON_SCRIPT}" "${DATASET_DIR}" "${OUTPUT_DIR}" 2>&1 | tee -a "${RESULTS_FILE}"

echo "End time: $(date)" | tee -a "${RESULTS_FILE}"
echo "====================================================" | tee -a "${RESULTS_FILE}"

# Summarize results
echo -e "\n=== Results Summary ===" | tee -a "${RESULTS_FILE}"
if [ -f "${OUTPUT_DIR}/average_metrics.csv" ]; then
    echo "Average metrics across 5-fold cross-validation:" | tee -a "${RESULTS_FILE}"
    cat "${OUTPUT_DIR}/average_metrics.csv" | tee -a "${RESULTS_FILE}"
else
    echo "Error: Results file not found!" | tee -a "${RESULTS_FILE}"
fi
echo "====================================================" | tee -a "${RESULTS_FILE}"

# List output files
echo -e "\n=== Output Files ===" | tee -a "${RESULTS_FILE}"
ls -la "${OUTPUT_DIR}" | tee -a "${RESULTS_FILE}"
echo "====================================================" | tee -a "${RESULTS_FILE}"

echo -e "\nExperiment completed successfully!" | tee -a "${RESULTS_FILE}"

# Exit successfully
exit 0