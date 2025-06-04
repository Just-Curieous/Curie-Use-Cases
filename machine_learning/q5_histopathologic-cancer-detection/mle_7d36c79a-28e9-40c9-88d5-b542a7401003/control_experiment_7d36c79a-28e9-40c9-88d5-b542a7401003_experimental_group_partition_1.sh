#!/bin/bash

# Experimental Script for PatchCamelyon Cancer Detection
# Experiment ID: 7d36c79a-28e9-40c9-88d5-b542a7401003
# Group: Experimental Group
# Partition: 1
#
# This script runs a comprehensive experiment on the PatchCamelyon dataset for cancer detection.
# It implements and evaluates 5 different model configurations:
# 1. ResNet50 - Batch size: 32, Optimizer: Adam with cosine annealing, Learning rate: 0.0005, Epochs: 3
#    Preprocessing: Color normalization + standardization
#    Augmentation: Rotation, flipping
#
# 2. DenseNet121 - Batch size: 32, Optimizer: Adam with cosine annealing, Learning rate: 0.0005, Epochs: 3
#    Preprocessing: Color normalization + standardization
#    Augmentation: Rotation, flipping
#
# 3. EfficientNetB0 - Batch size: 32, Optimizer: Adam with cosine annealing, Learning rate: 0.0005, Epochs: 3
#    Preprocessing: Color normalization + standardization
#    Augmentation: Rotation, flipping
#
# 4. SEResNeXt50 - Batch size: 32, Optimizer: Adam with cosine annealing, Learning rate: 0.0005, Epochs: 3
#    Preprocessing: Color normalization + standardization
#    Augmentation: Rotation, flipping
#
# 5. Custom model with attention mechanisms - Batch size: 32, Optimizer: AdamW with OneCycleLR, Learning rate: 0.0003, Epochs: 5
#    Preprocessing: Color normalization + standardization + contrast enhancement
#    Augmentation: Rotation, flipping, color jitter
#
# Each model is trained with specific hyperparameters, preprocessing techniques, and augmentation methods.
# The script measures and reports AUC-ROC, training time, inference time, and model size for each configuration.

# Set error handling
set -e

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
export VENV_PATH="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/venv"
export PYTHONPATH="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003:$PYTHONPATH"
export RESULTS_FILE="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/results_7d36c79a-28e9-40c9-88d5-b542a7401003_experimental_group_partition_1.txt"
export MODEL_RESULTS_DIR="/workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/model_results"

# Create results directory if it doesn't exist
mkdir -p "${MODEL_RESULTS_DIR}"

# Initialize and activate the micromamba environment
echo "Initializing micromamba environment..." > $RESULTS_FILE
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate $VENV_PATH/

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Failed to activate micromamba environment. Exiting." | tee -a $RESULTS_FILE
    exit 1
fi

# Check for GPU availability
echo "Checking GPU availability..." | tee -a $RESULTS_FILE
nvidia-smi | tee -a $RESULTS_FILE

# Ensure all required packages are installed
echo "Ensuring all required packages are installed..." | tee -a $RESULTS_FILE

# Check if required packages are installed
REQUIRED_PACKAGES=("torch" "torchvision" "timm" "pandas" "numpy" "scikit-learn" "pillow" "tqdm")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! $VENV_PATH/bin/python -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

# Install missing packages if any
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "Installing missing packages: ${MISSING_PACKAGES[*]}" | tee -a $RESULTS_FILE
    if [[ " ${MISSING_PACKAGES[*]} " =~ " torch " ]] || [[ " ${MISSING_PACKAGES[*]} " =~ " torchvision " ]]; then
        micromamba install -y -q -p $VENV_PATH pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia | tee -a $RESULTS_FILE
        # Remove torch and torchvision from MISSING_PACKAGES
        MISSING_PACKAGES=("${MISSING_PACKAGES[@]/torch}")
        MISSING_PACKAGES=("${MISSING_PACKAGES[@]/torchvision}")
    fi
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        for package in "${MISSING_PACKAGES[@]}"; do
            micromamba install -y -q -p $VENV_PATH $package | tee -a $RESULTS_FILE
        done
    fi
fi

# Make the Python script executable
chmod +x /workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/pcam_actual_experiment.py

# Run the experiment and save output to results file
echo "Running PatchCamelyon cancer detection experiment..." | tee -a $RESULTS_FILE
echo "Experiment started at: $(date)" | tee -a $RESULTS_FILE
echo "Experimental Group - Partition 1" | tee -a $RESULTS_FILE
echo "" | tee -a $RESULTS_FILE

# Execute the Python script and redirect all output to the results file
$VENV_PATH/bin/python /workspace/mle_7d36c79a-28e9-40c9-88d5-b542a7401003/pcam_actual_experiment.py 2>&1 | tee -a $RESULTS_FILE

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully." | tee -a $RESULTS_FILE
else
    echo "Experiment failed with an error." | tee -a $RESULTS_FILE
fi

# Print summary
echo "Experiment summary:" | tee -a $RESULTS_FILE
echo "- Models evaluated:" | tee -a $RESULTS_FILE
echo "  1. ResNet50" | tee -a $RESULTS_FILE
echo "  2. DenseNet121" | tee -a $RESULTS_FILE
echo "  3. EfficientNetB0" | tee -a $RESULTS_FILE
echo "  4. SEResNeXt50" | tee -a $RESULTS_FILE
echo "  5. Custom model with attention mechanisms" | tee -a $RESULTS_FILE
echo "- Configurations as per experiment plan:" | tee -a $RESULTS_FILE
echo "  - Batch size: 32" | tee -a $RESULTS_FILE
echo "  - Learning rates: 0.0005 (models 1-4), 0.0003 (model 5)" | tee -a $RESULTS_FILE
echo "  - Epochs: 3 (models 1-4), 5 (model 5)" | tee -a $RESULTS_FILE
echo "  - Optimizers: Adam with cosine annealing (models 1-4), AdamW with OneCycleLR (model 5)" | tee -a $RESULTS_FILE

# Summarize results
echo "Results summary:" | tee -a $RESULTS_FILE
echo "=======================================================================" | tee -a $RESULTS_FILE
echo "Experiment completed at: $(date)" | tee -a $RESULTS_FILE
