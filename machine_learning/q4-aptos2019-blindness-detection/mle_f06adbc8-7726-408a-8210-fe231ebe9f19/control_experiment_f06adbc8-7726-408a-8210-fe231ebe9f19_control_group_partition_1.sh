#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# Control group (partition_1)
# Parameters:
# - Model: ResNet50
# - Batch Size: 32
# - Augmentation: Basic (rotation, flip, shift)
# - Preprocessing: Standard resize to 224x224 + normalization
# - Learning Rate: 0.0001
# - Device: CPU-only

# Set environment variables
export PATH="/openhands/micromamba/bin:\$PATH"
eval "\$(micromamba shell hook --shell bash)"
micromamba activate /workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/venv/

export PYTHONPATH="/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19:\$PYTHONPATH"

# Define paths
DATA_DIR="/workspace/mle_dataset"
OUTPUT_DIR="/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/output"
RESULTS_FILE="/workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/results_f06adbc8-7726-408a-8210-fe231ebe9f19_control_group_partition_1.txt"

# Create output directory if it doesn't exist
mkdir -p \$OUTPUT_DIR

# Run the actual machine learning model instead of simulated results
echo "Running diabetic retinopathy detection model with ResNet50..."

# Execute the main Python script with appropriate parameters
python /workspace/mle_f06adbc8-7726-408a-8210-fe231ebe9f19/src/main.py \
  --model resnet50 \
  --batch_size 32 \
  --augmentation basic \
  --preprocessing standard \
  --img_size 224 \
  --learning_rate 0.0001 \
  --data_dir \$DATA_DIR \
  --output_dir \$OUTPUT_DIR \
  --results_file \$RESULTS_FILE \
  --num_epochs 10 \
  --val_split 0.2 \
  --seed 42 \
  --no_cuda

echo "Experiment completed. Results saved to \$RESULTS_FILE"
