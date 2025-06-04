#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# Experiment ID: 0083c3b8-243d-4eda-a884-57fddb81c9ce
# Group: control_group
# Partition: 1

# Redirect all output to the results file
exec > /workspace/mle_0083c3b8-243d-4eda-a884-57fddb81c9ce/results_0083c3b8-243d-4eda-a884-57fddb81c9ce_control_group_partition_1.txt 2>&1

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_0083c3b8-243d-4eda-a884-57fddb81c9ce/venv"
micromamba activate $VENV_PATH/

# Set working directory
cd /workspace/mle_0083c3b8-243d-4eda-a884-57fddb81c9ce/

# Print system information
echo "=== System Information ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Python: $(which $VENV_PATH/bin/python)"
$VENV_PATH/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
$VENV_PATH/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if $VENV_PATH/bin/python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"; then
    $VENV_PATH/bin/python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
    $VENV_PATH/bin/python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
    $VENV_PATH/bin/python -c "import torch; print(f'CUDA current device: {torch.cuda.current_device()}')"
fi
echo "=========================="

# Create output directory if it doesn't exist
mkdir -p /workspace/mle_0083c3b8-243d-4eda-a884-57fddb81c9ce/output

# Run the experiment
echo "=== Starting Experiment ==="
echo "Experiment ID: 0083c3b8-243d-4eda-a884-57fddb81c9ce"
echo "Group: control_group"
echo "Partition: 1"
echo "=========================="

# Execute the main Python script
$VENV_PATH/bin/python -m src.main

echo "=== Experiment Completed ==="