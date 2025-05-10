#!/bin/bash

# Script for training LightGBM model with regression_l1 loss function
echo "Starting training with regression_l1 loss function..."

# Set paths
CONFIG_PATH="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/regression_l1_config.json"
LOG_FILE="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/results/regression_l1_training.log"
METRICS_DIR="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/results/regression_l1"

# Create results directory if it doesn't exist
mkdir -p "$METRICS_DIR"

# Run the training
echo "Running model_training.py with regression_l1 configuration..."
/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv/bin/python /workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/model_training.py --config "$CONFIG_PATH" > "$LOG_FILE" 2>&1

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training with regression_l1 loss function failed. Check $LOG_FILE for details."
    exit 1
fi

# Find the most recent metrics file
METRICS_FILE=$(find "$METRICS_DIR" -name "metrics_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -f2- -d" ")

# Check if metrics file exists
if [ -z "$METRICS_FILE" ]; then
    echo "Error: No metrics file found for regression_l1 training."
    exit 1
fi

echo "Training with regression_l1 loss function completed successfully."
echo "Metrics file: $METRICS_FILE"

# Extract and display key metrics
echo "Key metrics for regression_l1 loss function:"
grep -o '"overall": [0-9.-]*' "$METRICS_FILE" | head -1

# Return success
exit 0