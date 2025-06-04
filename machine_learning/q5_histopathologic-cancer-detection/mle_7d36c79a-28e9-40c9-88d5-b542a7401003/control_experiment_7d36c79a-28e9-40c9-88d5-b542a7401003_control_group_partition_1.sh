#!/bin/bash

# Control Experiment Script for PatchCamelyon Cancer Detection
# Experiment ID: 7d36c79a-28e9-40c9-88d5-b542a7401003
# Group: Control Group
# Partition: 1

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
export VENV_PATH="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/venv"
export PYTHONPATH="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003:$PYTHONPATH"
export RESULTS_FILE="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/results_7d36c79a-28e9-40c9-88d5-b542a7401003_control_group_partition_1.txt"

# Initialize and activate the micromamba environment
echo "Initializing micromamba environment..."
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate $VENV_PATH/

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate micromamba environment. Exiting."
    exit 1
fi

# Check for GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Ensure all required packages are installed
echo "Ensuring all required packages are installed..."
micromamba install -y -q -p $VENV_PATH pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia || true
micromamba install -y -q -p $VENV_PATH pandas scikit-learn pillow tqdm || true

# Make the Python script executable
chmod +x /workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/pcam_experiment.py

# Run the experiment
echo "Starting PatchCamelyon cancer detection experiment (Control Group)..."
echo "Results will be saved to: $RESULTS_FILE"

# Execute the Python script and redirect all output to the results file
$VENV_PATH/bin/python /workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/pcam_experiment.py 2>&1 | tee $RESULTS_FILE

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully." | tee -a $RESULTS_FILE
else
    echo "Experiment failed with an error." | tee -a $RESULTS_FILE
fi

# Print summary
echo "Experiment summary:" | tee -a $RESULTS_FILE
echo "- Model: ResNet18" | tee -a $RESULTS_FILE
echo "- Optimizer: Adam" | tee -a $RESULTS_FILE
echo "- Preprocessing: Standard normalization" | tee -a $RESULTS_FILE
echo "- Augmentation: None" | tee -a $RESULTS_FILE
echo "- Learning rate: 0.001" | tee -a $RESULTS_FILE
echo "- Batch size: 64" | tee -a $RESULTS_FILE
echo "- Epochs: 20" | tee -a $RESULTS_FILE

echo "Experiment completed at: $(date)" | tee -a $RESULTS_FILE