#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# Experiment ID: e2e145a8-02ff-470e-8297-4c9446c8f955
# Group: control_group
# Partition: 1

# Set environment variables
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
export VIRTUAL_ENV="/workspace/mle_e2e145a8-02ff-470e-8297-4c9446c8f955/venv"
export VENV_PATH=$VIRTUAL_ENV
micromamba activate $VENV_PATH

# Define paths
WORKSPACE_DIR="/workspace/mle_e2e145a8-02ff-470e-8297-4c9446c8f955"
DATASET_DIR="/workspace/mle_dataset"
RESULTS_DIR="${WORKSPACE_DIR}/results"
OUTPUT_FILE="${WORKSPACE_DIR}/results_e2e145a8-02ff-470e-8297-4c9446c8f955_control_group_partition_1.txt"

# Create results directory if it doesn't exist
mkdir -p ${RESULTS_DIR}

# Print experiment information
echo "Starting Diabetic Retinopathy Detection Experiment" | tee -a ${OUTPUT_FILE}
echo "Experiment ID: e2e145a8-02ff-470e-8297-4c9446c8f955" | tee -a ${OUTPUT_FILE}
echo "Group: control_group" | tee -a ${OUTPUT_FILE}
echo "Partition: 1" | tee -a ${OUTPUT_FILE}
echo "Date: $(date)" | tee -a ${OUTPUT_FILE}
echo "----------------------------------------" | tee -a ${OUTPUT_FILE}

# Print system information
echo "System Information:" | tee -a ${OUTPUT_FILE}
echo "Python version: $(python --version 2>&1)" | tee -a ${OUTPUT_FILE}
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')" | tee -a ${OUTPUT_FILE}
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')" | tee -a ${OUTPUT_FILE}
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q 'True'; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')" | tee -a ${OUTPUT_FILE}
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')" | tee -a ${OUTPUT_FILE}
fi
echo "----------------------------------------" | tee -a ${OUTPUT_FILE}

# Run the experiment
echo "Running experiment..." | tee -a ${OUTPUT_FILE}
${VENV_PATH}/bin/python ${WORKSPACE_DIR}/main.py \
    --train_csv ${DATASET_DIR}/train.csv \
    --test_csv ${DATASET_DIR}/test.csv \
    --train_img_dir ${DATASET_DIR}/train_images \
    --test_img_dir ${DATASET_DIR}/test_images \
    --results_dir ${RESULTS_DIR} \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --num_workers 4 \
    --num_classes 5 \
    --num_folds 5 \
    --seed 42 \
    2>&1 | tee -a ${OUTPUT_FILE}

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo "----------------------------------------" | tee -a ${OUTPUT_FILE}
    echo "Experiment completed successfully!" | tee -a ${OUTPUT_FILE}
    echo "Results saved to: ${RESULTS_DIR}" | tee -a ${OUTPUT_FILE}
    echo "Full log saved to: ${OUTPUT_FILE}" | tee -a ${OUTPUT_FILE}
else
    echo "----------------------------------------" | tee -a ${OUTPUT_FILE}
    echo "Experiment failed! Please check the logs for errors." | tee -a ${OUTPUT_FILE}
    exit 1
fi

# Print summary of results
echo "----------------------------------------" | tee -a ${OUTPUT_FILE}
echo "Experiment Summary:" | tee -a ${OUTPUT_FILE}
if [ -f "${RESULTS_DIR}/experiment_results.json" ]; then
    echo "Overall metrics:" | tee -a ${OUTPUT_FILE}
    python -c "import json; data = json.load(open('${RESULTS_DIR}/experiment_results.json')); print(f\"Accuracy: {data['overall_metrics']['accuracy']:.4f}\"); print(f\"Quadratic Weighted Kappa: {data['overall_metrics']['quadratic_weighted_kappa']:.4f}\"); print(f\"Training time: {data['training_time']:.2f} seconds\"); print(f\"Inference time: {data['inference_time']:.2f} ms\"); print(f\"Model size: {data['model_size_mb']:.2f} MB\")" | tee -a ${OUTPUT_FILE}
else
    echo "No results file found!" | tee -a ${OUTPUT_FILE}
fi

echo "----------------------------------------" | tee -a ${OUTPUT_FILE}
echo "Experiment completed at: $(date)" | tee -a ${OUTPUT_FILE}

exit 0