#!/bin/bash

# Control experiment script for histopathologic cancer detection
# Experiment ID: edb230ee-5cee-4c3d-a0e5-d92317116daf
# Group: control_group_partition_1

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
export VENV_PATH="/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/venv"
export PYTHONPATH="/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf:$PYTHONPATH"

# Output file for logs
OUTPUT_FILE="/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results_edb230ee-5cee-4c3d-a0e5-d92317116daf_control_group_partition_1.txt"

# Create output directory
mkdir -p "/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results"

# Start logging
{
    echo "=== Starting Cancer Detection Experiment ==="
    echo "Experiment ID: edb230ee-5cee-4c3d-a0e5-d92317116daf"
    echo "Group: control_group_partition_1"
    echo "Date: $(date)"
    echo ""

    # Initialize and activate the environment
    echo "=== Setting up environment ==="
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate $VENV_PATH/
    
    # Check if required packages are installed, install if needed
    echo "=== Checking required packages ==="
    python -c "import torch, torchvision, sklearn, matplotlib, seaborn, pandas, numpy, PIL" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing missing packages..."
        micromamba install -y -q -p $VENV_PATH pytorch torchvision cudatoolkit matplotlib seaborn scikit-learn pandas pillow tqdm
    fi
    
    # Check GPU availability
    echo "=== Checking GPU availability ==="
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    
    # Run the experiment
    echo ""
    echo "=== Running experiment ==="
    $VENV_PATH/bin/python /workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/main.py \
        --data_dir "/workspace/mle_dataset" \
        --output_dir "/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results" \
        --model_name "resnet50" \
        --pretrained \
        --batch_size 32 \
        --epochs 20 \
        --learning_rate 0.001 \
        --num_workers 4 \
        --seed 42
    
    # Check if experiment completed successfully
    if [ $? -eq 0 ]; then
        echo ""
        echo "=== Experiment completed successfully ==="
    else
        echo ""
        echo "=== Experiment failed ==="
        exit 1
    fi
    
    # Summarize results
    echo ""
    echo "=== Results Summary ==="
    if [ -f "/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results/metrics.csv" ]; then
        echo "Performance Metrics:"
        cat "/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results/metrics.csv"
    else
        echo "No metrics file found."
    fi
    
    echo ""
    echo "=== Experiment Configuration ==="
    if [ -f "/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results/experiment_config.json" ]; then
        cat "/workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/results/experiment_config.json"
    else
        echo "No configuration file found."
    fi
    
    echo ""
    echo "=== End of Experiment ==="
    echo "Date: $(date)"

} 2>&1 | tee "$OUTPUT_FILE"

# Make the script executable
chmod +x /workspace/mle_edb230ee-5cee-4c3d-a0e5-d92317116daf/control_experiment_edb230ee-5cee-4c3d-a0e5-d92317116daf_control_group_partition_1.sh