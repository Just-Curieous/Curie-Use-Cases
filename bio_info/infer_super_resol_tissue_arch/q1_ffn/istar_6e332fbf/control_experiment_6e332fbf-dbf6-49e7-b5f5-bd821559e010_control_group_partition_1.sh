#!/bin/bash

# Control experiment script for neural network imputation task
# UUID: 6e332fbf-dbf6-49e7-b5f5-bd821559e010
# Group: control_group_partition_1

# Set paths
WORKSPACE="/workspace/istar_6e332fbf-dbf6-49e7-b5f5-bd821559e010"
VENV_PATH="/workspace/istar_6e332fbf-dbf6-49e7-b5f5-bd821559e010/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
RESULTS_FILE="$WORKSPACE/results_6e332fbf-dbf6-49e7-b5f5-bd821559e010_control_group_partition_1.txt"
DATA_DIR="$WORKSPACE/data/demo/"

# Ensure output directory exists
mkdir -p "$WORKSPACE/states"
mkdir -p "$WORKSPACE/cnts-super"

# Start logging
{
    echo "=== Control Experiment: Neural Network Imputation ==="
    echo "UUID: 6e332fbf-dbf6-49e7-b5f5-bd821559e010"
    echo "Group: control_group_partition_1"
    echo "Start time: $(date)"
    echo ""

    # Check GPU availability
    echo "=== GPU Information ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "GPU is available for computation"
        DEVICE="cuda"
    else
        echo "No GPU found, using CPU"
        DEVICE="cpu"
    fi
    echo ""

    # Check Python environment
    echo "=== Python Environment ==="
    $PYTHON_PATH --version
    echo ""

    # Print experiment configuration
    echo "=== Experiment Configuration ==="
    echo "Data directory: $DATA_DIR"
    echo "Model: Feedforward Neural Network with baseline configuration"
    echo "Architecture: 4 layers of 256 units each"
    echo "Activation: LeakyReLU(0.1) for hidden layers, ELU(alpha=0.01, beta=0.01) for output"
    echo "Optimizer: Adam with default learning rate"
    echo "No regularization applied"
    echo "Training for 400 epochs"
    echo ""

    # Run the experiment
    echo "=== Starting Experiment ==="
    echo "Command: $PYTHON_PATH $WORKSPACE/flexible_nn.py $DATA_DIR --epochs=400 --device=$DEVICE"
    
    # Execute the experiment
    cd $WORKSPACE
    $PYTHON_PATH $WORKSPACE/flexible_nn.py $DATA_DIR --epochs=400 --device=$DEVICE
    
    # Check if experiment completed successfully
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== Experiment Completed Successfully ==="
    else
        echo ""
        echo "=== Experiment Failed ==="
    fi
    
    # Copy results file to the expected location
    if [ -f "$WORKSPACE/rmse_results.txt" ]; then
        echo ""
        echo "=== Results Summary ==="
        cat "$WORKSPACE/rmse_results.txt" | grep -v "^#" | head -n 5
        echo "..."
        cat "$WORKSPACE/rmse_results.txt" | grep -v "^#" | tail -n 5
        
        # Copy the results file to the expected location
        cp "$WORKSPACE/rmse_results.txt" "$RESULTS_FILE"
        echo ""
        echo "Results saved to: $RESULTS_FILE"
    else
        echo ""
        echo "No results file found at $WORKSPACE/rmse_results.txt"
    fi
    
    echo ""
    echo "End time: $(date)"
    echo "=== Experiment Complete ==="

} 2>&1 | tee "$RESULTS_FILE"