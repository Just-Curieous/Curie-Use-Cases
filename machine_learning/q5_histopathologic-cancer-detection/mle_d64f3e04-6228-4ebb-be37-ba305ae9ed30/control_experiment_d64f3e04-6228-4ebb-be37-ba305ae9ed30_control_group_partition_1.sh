#!/bin/bash

# Control experiment script for histopathologic cancer detection
# Experiment ID: d64f3e04-6228-4ebb-be37-ba305ae9ed30
# Group: control_group_partition_1

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
export VENV_PATH="/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/venv"

# Initialize and activate micromamba environment
echo "Initializing micromamba environment..."
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate $VENV_PATH/

# Set output file
OUTPUT_FILE="/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/results_d64f3e04-6228-4ebb-be37-ba305ae9ed30_control_group_partition_1.txt"

# Create output directory
mkdir -p /workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/output

# Print system information
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Python version: $(python --version)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
    if python -c 'import torch; print(torch.cuda.is_available())' | grep -q 'True'; then
        echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    fi
    echo ""
    
    echo "=== Experiment Configuration ==="
    echo "Experiment ID: d64f3e04-6228-4ebb-be37-ba305ae9ed30"
    echo "Group: control_group_partition_1"
    echo "Model: ResNet18"
    echo "Preprocessing: Standard RGB normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
    echo "Augmentation: Basic (horizontal flip, vertical flip, rotation)"
    echo "Optimizer: Adam with learning rate 0.001"
    echo "Batch size: 64"
    echo "Early stopping patience: 5 epochs"
    echo "Maximum epochs: 20"
    echo ""
    
    echo "=== Starting Experiment ==="
    
    # Run the experiment
    python /workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/src/experiment.py \
        --data_dir /workspace/mle_dataset \
        --output_dir /workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/output \
        --batch_size 64 \
        --learning_rate 0.001 \
        --num_epochs 20 \
        --patience 5 \
        --seed 42
    
    echo ""
    echo "=== Experiment Completed ==="
    
    # Display summary of results
    if [ -f "/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/output/results.json" ]; then
        echo "=== Results Summary ==="
        python -c "
import json
with open('/workspace/mle_d64f3e04-6228-4ebb-be37-ba305ae9ed30/output/results.json', 'r') as f:
    results = json.load(f)
    
print(f\"Best validation AUC: {results['training_history']['best_val_auc']:.4f} (Epoch {results['training_history']['best_epoch']})\"
      f\"\\nTest AUC: {results['test_metrics']['test_auc']:.4f}\"
      f\"\\nTraining time: {results['training_history']['training_time']:.2f} seconds\"
      f\"\\nInference time: {results['test_metrics']['inference_time']:.2f} seconds\")
"
    else
        echo "Results file not found."
    fi
    
} 2>&1 | tee "$OUTPUT_FILE"

echo "Experiment completed. Results saved to $OUTPUT_FILE"