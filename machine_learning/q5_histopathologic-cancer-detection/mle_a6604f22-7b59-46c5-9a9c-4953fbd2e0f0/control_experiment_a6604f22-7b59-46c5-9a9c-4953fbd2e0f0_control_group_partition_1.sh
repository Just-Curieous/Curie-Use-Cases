#!/bin/bash

# Define paths
WORKSPACE_DIR="/workspace/mle_a6604f22-7b59-46c5-9a9c-4953fbd2e0f0"
DATASET_DIR="/workspace/mle_dataset"
RESULTS_FILE="${WORKSPACE_DIR}/results_a6604f22-7b59-46c5-9a9c-4953fbd2e0f0_control_group_partition_1.txt"
VENV_PATH="${WORKSPACE_DIR}/venv"

# Create output file and start logging
echo "Starting experiment: EfficientNetB0 for PCam dataset (Control Group)" > $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE

# Set up environment
echo "Setting up environment..." >> $RESULTS_FILE
export PATH="/openhands/micromamba/bin:$PATH"

# Initialize micromamba if not already initialized
if ! command -v micromamba &> /dev/null; then
    echo "Initializing micromamba..." >> $RESULTS_FILE
    micromamba shell init --shell bash --root-prefix=~/.local/share/mamba >> $RESULTS_FILE 2>&1
    eval "$(micromamba shell hook --shell bash)" >> $RESULTS_FILE 2>&1
fi

# Check if venv exists, if not create it
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at $VENV_PATH..." >> $RESULTS_FILE
    micromamba create -y -p $VENV_PATH python=3.10 >> $RESULTS_FILE 2>&1
fi

# Activate the environment
echo "Activating environment..." >> $RESULTS_FILE
eval "$(micromamba shell hook --shell bash)" >> $RESULTS_FILE 2>&1
micromamba activate $VENV_PATH >> $RESULTS_FILE 2>&1

# Install required packages if not already installed
echo "Installing required packages..." >> $RESULTS_FILE
micromamba install -y -q -p $VENV_PATH pytorch torchvision cpuonly -c pytorch >> $RESULTS_FILE 2>&1
micromamba install -y -q -p $VENV_PATH numpy pandas scikit-learn pillow matplotlib tqdm -c conda-forge >> $RESULTS_FILE 2>&1
micromamba install -y -q -p $VENV_PATH timm -c conda-forge >> $RESULTS_FILE 2>&1
micromamba install -y -q -p $VENV_PATH pip -c conda-forge >> $RESULTS_FILE 2>&1

# Install thop for FLOPs calculation using pip
$VENV_PATH/bin/pip install thop seaborn psutil >> $RESULTS_FILE 2>&1

# Check GPU availability
echo "Checking GPU availability..." >> $RESULTS_FILE
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi >> $RESULTS_FILE 2>&1
    GPU_AVAILABLE="true"
else
    echo "NVIDIA SMI not found. No GPU available or drivers not installed." >> $RESULTS_FILE 2>&1
    GPU_AVAILABLE="false"
fi
echo "=======================================================" >> $RESULTS_FILE

# Print Python and PyTorch versions
echo "Python version:" >> $RESULTS_FILE
$VENV_PATH/bin/python --version >> $RESULTS_FILE 2>&1
echo "PyTorch version:" >> $RESULTS_FILE
$VENV_PATH/bin/python -c "import torch; print(torch.__version__); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"None\"}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \"None\"}')" >> $RESULTS_FILE 2>&1
echo "=======================================================" >> $RESULTS_FILE

# Create necessary directories
mkdir -p $WORKSPACE_DIR/models >> $RESULTS_FILE 2>&1
mkdir -p $WORKSPACE_DIR/logs >> $RESULTS_FILE 2>&1

# Print dataset information
echo "Dataset information:" >> $RESULTS_FILE
echo "Train directory: $(ls -la $DATASET_DIR/train | wc -l) files" >> $RESULTS_FILE
echo "Test directory: $(ls -la $DATASET_DIR/test | wc -l) files" >> $RESULTS_FILE
echo "Labels file: $(wc -l $DATASET_DIR/train_labels.csv) lines" >> $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE

# Run the experiment
echo "Running experiment..." >> $RESULTS_FILE
cd $WORKSPACE_DIR

# Execute the main script with all parameters
$VENV_PATH/bin/python $WORKSPACE_DIR/main.py \
    --data_dir $DATASET_DIR \
    --model_dir $WORKSPACE_DIR/models \
    --log_dir $WORKSPACE_DIR/logs \
    --batch_size 32 \
    --num_epochs 30 \
    --learning_rate 0.001 \
    --patience 7 \
    --threshold 0.5 \
    --num_workers 4 \
    --seed 42 \
    >> $RESULTS_FILE 2>&1

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!" >> $RESULTS_FILE
else
    echo "Experiment failed with error code $?" >> $RESULTS_FILE
fi

# Append the latest log file to the results
echo "=======================================================" >> $RESULTS_FILE
echo "Detailed logs:" >> $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE

# Find the most recent log file and append its contents
LATEST_LOG=$(ls -t $WORKSPACE_DIR/logs/experiment_log_*.txt 2>/dev/null | head -n 1)
if [ -n "$LATEST_LOG" ]; then
    cat $LATEST_LOG >> $RESULTS_FILE
else
    echo "No log file found." >> $RESULTS_FILE
fi

# Summarize the results
echo "=======================================================" >> $RESULTS_FILE
echo "EXPERIMENT SUMMARY" >> $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE

# Extract key metrics from the results
LATEST_METRICS=$(ls -t $WORKSPACE_DIR/logs/metrics_*.csv 2>/dev/null | head -n 1)
if [ -n "$LATEST_METRICS" ]; then
    echo "Key metrics from $LATEST_METRICS:" >> $RESULTS_FILE
    cat $LATEST_METRICS >> $RESULTS_FILE
else
    echo "No metrics file found." >> $RESULTS_FILE
fi

echo "=======================================================" >> $RESULTS_FILE
echo "Experiment completed at $(date)" >> $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE

# Print completion message
echo "Experiment completed. Results saved to $RESULTS_FILE"