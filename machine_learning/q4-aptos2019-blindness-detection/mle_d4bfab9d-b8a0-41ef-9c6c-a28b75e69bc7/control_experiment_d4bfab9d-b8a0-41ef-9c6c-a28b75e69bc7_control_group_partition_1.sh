#!/bin/bash

# Set up error handling
set -e

# Define paths
WORKSPACE_DIR="/workspace/mle_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7"
VENV_PATH="${WORKSPACE_DIR}/venv"
OUTPUT_DIR="${WORKSPACE_DIR}/output"
RESULTS_FILE="${WORKSPACE_DIR}/results_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7_control_group_partition_1.txt"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Initialize the log file
echo "Starting diabetic retinopathy detection experiment at $(date)" > "${RESULTS_FILE}"
echo "=======================================================" >> "${RESULTS_FILE}"

# Set up environment
echo "Setting up environment..." >> "${RESULTS_FILE}"
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${VENV_PATH}/"

# Check if CUDA is available
echo "Checking CUDA availability..." >> "${RESULTS_FILE}"
"${VENV_PATH}/bin/python" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')" >> "${RESULTS_FILE}" 2>&1

# Run the main script
echo "Starting the main workflow..." >> "${RESULTS_FILE}"
echo "=======================================================" >> "${RESULTS_FILE}"

"${VENV_PATH}/bin/python" "${WORKSPACE_DIR}/main.py" \
    --train_csv "/workspace/mle_dataset/train.csv" \
    --test_csv "/workspace/mle_dataset/test.csv" \
    --train_img_dir "/workspace/mle_dataset/train_images" \
    --test_img_dir "/workspace/mle_dataset/test_images" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size 16 \
    --num_epochs 10 >> "${RESULTS_FILE}" 2>&1

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "=======================================================" >> "${RESULTS_FILE}"
    echo "Experiment completed successfully at $(date)" >> "${RESULTS_FILE}"
    
    # Copy the submission file to the workspace directory
    cp "${OUTPUT_DIR}/submission.csv" "${WORKSPACE_DIR}/submission_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7_control_group_partition_1.csv"
    
    # Summarize results
    echo "=======================================================" >> "${RESULTS_FILE}"
    echo "Results Summary:" >> "${RESULTS_FILE}"
    cat "${OUTPUT_DIR}/metrics.txt" >> "${RESULTS_FILE}"
else
    echo "=======================================================" >> "${RESULTS_FILE}"
    echo "Experiment failed at $(date)" >> "${RESULTS_FILE}"
fi

echo "=======================================================" >> "${RESULTS_FILE}"
echo "End of experiment log" >> "${RESULTS_FILE}"