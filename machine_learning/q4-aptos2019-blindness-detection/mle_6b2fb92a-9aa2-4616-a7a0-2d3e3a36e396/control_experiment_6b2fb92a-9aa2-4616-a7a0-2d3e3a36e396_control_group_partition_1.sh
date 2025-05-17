#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# Experiment ID: 6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396
# Control Group: control_group_partition_1

# Define output file
OUTPUT_FILE="/workspace/mle_6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396/results_6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396_control_group_partition_1.txt"

# Redirect all output to the results file
{
    # Set environment variables
    export PATH="/openhands/micromamba/bin:$PATH"
    eval "$(micromamba shell hook --shell bash)"
    export VIRTUAL_ENV="/workspace/mle_6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396/venv"
    export VENV_PATH=$VIRTUAL_ENV
    micromamba activate $VENV_PATH/

    # Set working directory
    cd /workspace/mle_6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396

    # Install required dependencies
    echo "=== Installing Required Dependencies ==="
    echo "Installing seaborn..."
    micromamba install -y -q -p $VENV_PATH seaborn || echo "Failed to install seaborn, continuing anyway..."
    
    echo "Installing torch and torchvision..."
    micromamba install -y -q -p $VENV_PATH pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia || echo "Failed to install PyTorch, continuing anyway..."
    
    echo "Installing pandas..."
    micromamba install -y -q -p $VENV_PATH pandas || echo "Failed to install pandas, continuing anyway..."
    
    echo "Installing numpy..."
    micromamba install -y -q -p $VENV_PATH numpy || echo "Failed to install numpy, continuing anyway..."
    
    echo "Installing scikit-learn..."
    micromamba install -y -q -p $VENV_PATH scikit-learn || echo "Failed to install scikit-learn, continuing anyway..."
    
    echo "Installing matplotlib..."
    micromamba install -y -q -p $VENV_PATH matplotlib || echo "Failed to install matplotlib, continuing anyway..."
    
    echo "Installing albumentations..."
    $VENV_PATH/bin/pip install -q albumentations || echo "Failed to install albumentations, continuing anyway..."
    
    echo "Installing efficientnet-pytorch..."
    $VENV_PATH/bin/pip install -q timm || echo "Failed to install timm, continuing anyway..."
    
    echo "Installing tqdm..."
    micromamba install -y -q -p $VENV_PATH tqdm || echo "Failed to install tqdm, continuing anyway..."
    
    echo "Installing opencv..."
    micromamba install -y -q -p $VENV_PATH opencv || echo "Failed to install opencv, continuing anyway..."
    
    echo "Dependencies installation completed."
    echo "==============================="

    # Print environment information
    echo "=== Environment Information ==="
    echo "Python version: $(python --version 2>&1)"
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
    echo "GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")' 2>&1)"
    echo "==============================="

    # Run the experiment
    echo "Starting diabetic retinopathy detection experiment..."
    echo "Experiment ID: 6b2fb92a-9aa2-4616-a7a0-2d3e3a36e396"
    echo "Control Group: control_group_partition_1"
    echo "==============================="

    # Execute the main Python script
    $VENV_PATH/bin/python -m src.main

    # Capture exit status
    EXIT_STATUS=$?

    echo "==============================="
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "Experiment completed successfully."
    else
        echo "Experiment failed with exit code $EXIT_STATUS."
    fi

    # Return the exit status
    exit $EXIT_STATUS
} > "$OUTPUT_FILE" 2>&1