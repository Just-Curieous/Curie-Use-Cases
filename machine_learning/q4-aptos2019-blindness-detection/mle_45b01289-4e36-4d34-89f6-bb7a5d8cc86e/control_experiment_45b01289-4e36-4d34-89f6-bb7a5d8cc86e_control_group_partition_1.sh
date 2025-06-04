#!/bin/bash
# Control Experiment for Diabetic Retinopathy Detection - IMPROVED MEMORY EFFICIENCY
# Experiment ID: 45b01289-4e36-4d34-89f6-bb7a5d8cc86e
# Group: Control Group (Best Single Model Approach)
# Partition: 1

# Create output directory
mkdir -p /workspace/mle_45b01289-4e36-4d34-89f6-bb7a5d8cc86e/output_improved

# Create results file
RESULTS_FILE="/workspace/mle_45b01289-4e36-4d34-89f6-bb7a5d8cc86e/results_45b01289-4e36-4d34-89f6-bb7a5d8cc86e_control_group_partition_1.txt"

# Start logging
echo "=== Diabetic Retinopathy Detection Experiment (IMPROVED) ===" > \$RESULTS_FILE
echo "Experiment ID: 45b01289-4e36-4d34-89f6-bb7a5d8cc86e" >> \$RESULTS_FILE
echo "Group: Control Group (Best Single Model Approach)" >> \$RESULTS_FILE
echo "Partition: 1" >> \$RESULTS_FILE
echo "Date: \$(date)" >> \$RESULTS_FILE

# Install required packages
echo "Installing required packages..." >> \$RESULTS_FILE
pip install pandas torch torchvision efficientnet-pytorch opencv-python scikit-learn matplotlib albumentations==0.5.2 tqdm >> \$RESULTS_FILE 2>&1

# Run the experiment
echo "Running the experiment..." >> \$RESULTS_FILE
python /workspace/mle_45b01289-4e36-4d34-89f6-bb7a5d8cc86e/diabetic_retinopathy_detection_improved.py \
  /workspace/mle_dataset/train.csv \
  /workspace/mle_dataset/test.csv \
  /workspace/mle_dataset/train_images \
  /workspace/mle_dataset/test_images \
  /workspace/mle_45b01289-4e36-4d34-89f6-bb7a5d8cc86e/output_improved \
  >> \$RESULTS_FILE 2>&1

# Calculate runtime
echo "Experiment completed." >> \$RESULTS_FILE
echo "Results saved to \$RESULTS_FILE"
