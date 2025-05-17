#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# Control Group (Partition 1)

# Set up environment
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/venv"
micromamba activate $VENV_PATH/

# Define output file
OUTPUT_FILE="/workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/results_ca8b2f40-eef3-409b-a194-81d2bf9608b8_control_group_partition_1.txt"

# Create checkpoint directory if it doesn't exist
CHECKPOINT_DIR="/workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/checkpoints"
mkdir -p $CHECKPOINT_DIR

# Print experiment information
echo "=== Diabetic Retinopathy Detection Experiment ===" | tee -a "$OUTPUT_FILE"
echo "Control Group (Partition 1)" | tee -a "$OUTPUT_FILE"
echo "Date: $(date)" | tee -a "$OUTPUT_FILE"
echo "Parameters:" | tee -a "$OUTPUT_FILE"
echo "- Base Model: EfficientNet-B3" | tee -a "$OUTPUT_FILE"
echo "- Preprocessing: Standard resize and normalization" | tee -a "$OUTPUT_FILE"
echo "- Denoising: None" | tee -a "$OUTPUT_FILE"
echo "- Class balancing: None" | tee -a "$OUTPUT_FILE"
echo "- Quality assessment: None" | tee -a "$OUTPUT_FILE"
echo "- Batch size: 8 (reduced from 16)" | tee -a "$OUTPUT_FILE"
echo "- Number of epochs: 3 (reduced from 10)" | tee -a "$OUTPUT_FILE"
echo "- DataLoader workers: 0 (disabled multiprocessing)" | tee -a "$OUTPUT_FILE"
echo "- Early stopping: Enabled with patience of 2 epochs" | tee -a "$OUTPUT_FILE"
echo "- Checkpointing: Enabled for each epoch" | tee -a "$OUTPUT_FILE"
echo "=======================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Run the experiment
echo "Starting experiment workflow..." | tee -a "$OUTPUT_FILE"
$VENV_PATH/bin/python /workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/diabetic_retinopathy_detection.py 2>&1 | tee -a "$OUTPUT_FILE"

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "" | tee -a "$OUTPUT_FILE"
    echo "Experiment completed successfully!" | tee -a "$OUTPUT_FILE"
    echo "Results saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
else
    echo "" | tee -a "$OUTPUT_FILE"
    echo "Error: Experiment failed!" | tee -a "$OUTPUT_FILE"
    exit 1
fi

# Print summary
echo "" | tee -a "$OUTPUT_FILE"
echo "=== Experiment Summary ===" | tee -a "$OUTPUT_FILE"
echo "Model: EfficientNet-B3" | tee -a "$OUTPUT_FILE"
echo "Dataset: APTOS 2019 Diabetic Retinopathy Detection" | tee -a "$OUTPUT_FILE"
echo "Model saved at: /workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/efficientnet_b3_model.pth" | tee -a "$OUTPUT_FILE"
echo "Checkpoints saved at: /workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/checkpoints/" | tee -a "$OUTPUT_FILE"
echo "Confusion matrix saved at: /workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/confusion_matrix.png" | tee -a "$OUTPUT_FILE"
echo "Log file saved at: /workspace/mle_ca8b2f40-eef3-409b-a194-81d2bf9608b8/experiment.log" | tee -a "$OUTPUT_FILE"
echo "=======================================" | tee -a "$OUTPUT_FILE"

exit 0