#!/bin/bash

# Test experiment script with reduced epochs
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_e2e145a8-02ff-470e-8297-4c9446c8f955/venv"
micromamba activate $VENV_PATH

# Define paths
WORKSPACE_DIR="/workspace/mle_e2e145a8-02ff-470e-8297-4c9446c8f955"
DATASET_DIR="/workspace/mle_dataset"
RESULTS_DIR="${WORKSPACE_DIR}/test_results"
OUTPUT_FILE="${WORKSPACE_DIR}/test_results.txt"

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Run the experiment with reduced epochs
echo "Running test experiment with reduced epochs..." | tee -a ${OUTPUT_FILE}
${VENV_PATH}/bin/python ${WORKSPACE_DIR}/main.py \
    --train_csv ${DATASET_DIR}/train.csv \
    --test_csv ${DATASET_DIR}/test.csv \
    --train_img_dir ${DATASET_DIR}/train_images \
    --test_img_dir ${DATASET_DIR}/test_images \
    --results_dir ${RESULTS_DIR} \
    --num_epochs 2 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_workers 4 \
    --num_classes 5 \
    --num_folds 2 \
    --seed 42 \
    2>&1 | tee -a ${OUTPUT_FILE}

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Test experiment completed successfully!" | tee -a ${OUTPUT_FILE}
else
    echo "Test experiment failed! Please check the logs for errors." | tee -a ${OUTPUT_FILE}
    exit 1
fi