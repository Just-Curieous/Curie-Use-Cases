#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# This script runs the experiment with basic normalization (control group)
# Fixed for NumPy 2.x compatibility

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_1af68a6e-679d-488a-974b-92e131400d32/venv"
micromamba activate $VENV_PATH/

# Make sure all dependencies are installed
micromamba install -y -q -p $VENV_PATH opencv scikit-learn pandas matplotlib tqdm

# Define paths
WORKSPACE_DIR="/workspace/mle_1af68a6e-679d-488a-974b-92e131400d32"
DATASET_DIR="/workspace/mle_dataset"
OUTPUT_DIR="${WORKSPACE_DIR}/output_control_group"
RESULTS_FILE="${WORKSPACE_DIR}/results_1af68a6e-679d-488a-974b-92e131400d32_control_group_partition_1.txt"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Print environment information
{
  echo "=== Environment Information ==="
  echo "Date: $(date)"
  echo "Python: $(python --version)"
  echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
  echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)')"
  echo "OpenCV: $(python -c 'import cv2; print(cv2.__version__)')"
  echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
  if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
  fi
  echo "============================="
  echo ""
} | tee -a $RESULTS_FILE

# Print experiment configuration
{
  echo "=== Experiment Configuration ==="
  echo "Experiment: Diabetic Retinopathy Detection"
  echo "Group: Control Group (Basic Normalization)"
  echo "Model: EfficientNetB4"
  echo "Dataset: APTOS 2019"
  echo "NumPy 2.x Compatibility: Fixed"
  echo "============================="
  echo ""
} | tee -a $RESULTS_FILE

# Run the experiment
{
  echo "=== Starting Experiment ==="
  echo "Command: python ${WORKSPACE_DIR}/dr_detection.py --train_csv ${DATASET_DIR}/train.csv --test_csv ${DATASET_DIR}/test.csv --train_img_dir ${DATASET_DIR}/train_images --test_img_dir ${DATASET_DIR}/test_images --output_dir ${OUTPUT_DIR} --preprocessing_method basic --batch_size 4 --num_epochs 10 --patience 5 --mode train_eval"
  echo ""
  
  python ${WORKSPACE_DIR}/dr_detection.py \
    --train_csv ${DATASET_DIR}/train.csv \
    --test_csv ${DATASET_DIR}/test.csv \
    --train_img_dir ${DATASET_DIR}/train_images \
    --test_img_dir ${DATASET_DIR}/test_images \
    --output_dir ${OUTPUT_DIR} \
    --preprocessing_method basic \
    --batch_size 4 \
    --num_epochs 10 \
    --patience 5 \
    --mode train_eval
  
  echo ""
  echo "=== Experiment Completed ==="
} | tee -a $RESULTS_FILE

# Summarize results
{
  echo "=== Results Summary ==="
  if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "Validation Metrics:"
    cat ${OUTPUT_DIR}/metrics.json
  else
    echo "No metrics file found."
  fi
  
  if [ -f "${OUTPUT_DIR}/submission.csv" ]; then
    echo ""
    echo "Submission file created: ${OUTPUT_DIR}/submission.csv"
    echo "Submission file preview:"
    head -n 5 ${OUTPUT_DIR}/submission.csv
  else
    echo "No submission file found."
  fi
  echo "============================="
} | tee -a $RESULTS_FILE

echo "All experiment output has been saved to: $RESULTS_FILE"