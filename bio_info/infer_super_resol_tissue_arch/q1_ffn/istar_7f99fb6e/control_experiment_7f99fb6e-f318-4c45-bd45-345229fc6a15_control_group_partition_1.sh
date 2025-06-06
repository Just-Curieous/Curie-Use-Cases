#!/bin/bash

# Control experiment script for neural network imputation tasks
# UUID: 7f99fb6e-f318-4c45-bd45-345229fc6a15
# Group: control_group_partition_1

# Set up error handling
set -e
set -o pipefail

# Define paths
WORKSPACE_DIR="/workspace/istar_7f99fb6e-f318-4c45-bd45-345229fc6a15"
VENV_PATH="/workspace/istar_7f99fb6e-f318-4c45-bd45-345229fc6a15/venv"
RESULTS_FILE="/workspace/istar_7f99fb6e-f318-4c45-bd45-345229fc6a15/results_7f99fb6e-f318-4c45-bd45-345229fc6a15_control_group_partition_1.txt"
DATA_DIR="/workspace/istar_7f99fb6e-f318-4c45-bd45-345229fc6a15/data/demo/"

# Redirect all output to the results file
exec > "$RESULTS_FILE" 2>&1

# Initialize log file
echo "Starting control experiment at $(date)"
echo "UUID: 7f99fb6e-f318-4c45-bd45-345229fc6a15"
echo "Group: control_group_partition_1"
echo "----------------------------------------"

# Set up environment
echo "Setting up environment..."
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate "$VENV_PATH/"

# Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Print baseline configuration
echo "----------------------------------------"
echo "Running control experiment with baseline configuration:"
echo "- Architecture: 4 layers of 256 units each (hidden_layers='256,256,256,256')"
echo "- Activation: LeakyReLU"
echo "- Learning rate: 0.0001"
echo "- Batch size: 27"
echo "- Optimizer: Adam"
echo "- No regularization"
echo "----------------------------------------"

# Clean up any existing model states to ensure fresh training
if [ -d "$WORKSPACE_DIR/data/demo/states" ]; then
    echo "Removing existing model states..."
    rm -rf "$WORKSPACE_DIR/data/demo/states"
fi

# Clean up any existing rmse_results.txt file
if [ -f "$WORKSPACE_DIR/rmse_results.txt" ]; then
    echo "Removing existing rmse_results.txt..."
    rm -f "$WORKSPACE_DIR/rmse_results.txt"
fi

# Create necessary directories
mkdir -p "$WORKSPACE_DIR/data/demo/states/00"
mkdir -p "$WORKSPACE_DIR/data/demo/training-data-plots"
mkdir -p "$WORKSPACE_DIR/data/demo/cnts-super"

# Create a modified version of impute.py with the required changes
echo "Creating modified impute.py for the experiment..."
cp "$WORKSPACE_DIR/impute.py" "$WORKSPACE_DIR/impute_modified.py"

# Modify the batch size in the impute_modified.py file
sed -i 's/batch_size = min(128, n_train\/\/16)/batch_size = 27  # Fixed batch size for control experiment/g' "$WORKSPACE_DIR/impute_modified.py"

# Modify the ForwardSumModel class to use LeakyReLU with 0.01 negative_slope
sed -i 's/activation = nn.LeakyReLU(0.1, inplace=True)/activation = nn.LeakyReLU(0.01, inplace=True)/g' "$WORKSPACE_DIR/impute_modified.py"

# Record start time
START_TIME=$(date +%s)

# Run the modified impute.py script with the baseline configuration
echo "Running impute.py with baseline configuration..."
cd "$WORKSPACE_DIR"
"$VENV_PATH/bin/python" "$WORKSPACE_DIR/impute_modified.py" "$DATA_DIR" --epochs=400 --device='cuda' --n-states=1 --n-jobs=1

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo "----------------------------------------"
    echo "Experiment completed successfully at $(date)"
    echo "Total execution time: ${MINUTES}m ${SECONDS}s"
    
    # Copy the detailed metrics to the results file
    if [ -f "$WORKSPACE_DIR/rmse_results.txt" ]; then
        echo "----------------------------------------"
        echo "Detailed metrics:"
        cat "$WORKSPACE_DIR/rmse_results.txt"
    fi
    
    # Calculate final RMSE from the results file
    FINAL_RMSE=$(tail -n 1 "$WORKSPACE_DIR/rmse_results.txt" | cut -d',' -f2)
    
    # Summarize the results
    echo "----------------------------------------"
    echo "EXPERIMENT SUMMARY"
    echo "----------------------------------------"
    echo "Baseline configuration:"
    echo "- Architecture: 4 layers of 256 units each (hidden_layers='256,256,256,256')"
    echo "- Activation: LeakyReLU"
    echo "- Learning rate: 0.0001"
    echo "- Batch size: 27"
    echo "- Optimizer: Adam"
    echo "- No regularization"
    echo "----------------------------------------"
    echo "Final RMSE: $FINAL_RMSE"
    echo "Model saved to: $DATA_DIR/states/00/model.pt"
    echo "Execution time: ${MINUTES}m ${SECONDS}s"
    echo "Results saved to: $RESULTS_FILE"
else
    echo "----------------------------------------"
    echo "Experiment failed at $(date)"
    echo "Check the logs above for error details."
    exit 1
fi

# Clean up the modified file
rm "$WORKSPACE_DIR/impute_modified.py"

echo "----------------------------------------"
echo "Control experiment completed. Results saved to $RESULTS_FILE"