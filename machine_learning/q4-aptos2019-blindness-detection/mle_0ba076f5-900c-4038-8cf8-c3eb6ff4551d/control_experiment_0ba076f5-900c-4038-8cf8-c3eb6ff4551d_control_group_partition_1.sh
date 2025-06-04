#!/bin/bash

# Set up environment
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_0ba076f5-900c-4038-8cf8-c3eb6ff4551d/venv"
micromamba activate $VENV_PATH/

# Define paths
WORKSPACE_DIR="/workspace/mle_0ba076f5-900c-4038-8cf8-c3eb6ff4551d"
DATASET_DIR="/workspace/mle_dataset"
OUTPUT_DIR="${WORKSPACE_DIR}/output"
RESULTS_FILE="${WORKSPACE_DIR}/results_0ba076f5-900c-4038-8cf8-c3eb6ff4551d_control_group_partition_1.txt"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Start logging
{
    echo "=== Diabetic Retinopathy Detection Experiment ==="
    echo "Starting experiment at: $(date)"
    echo ""

    # Check GPU availability
    echo "=== Checking GPU availability ==="
    $VENV_PATH/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}'); print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}'); if torch.cuda.is_available(): print(f'CUDA device: {torch.cuda.get_device_name(0)}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"
    echo ""

    # Display dataset information
    echo "=== Dataset Information ==="
    echo "Training data:"
    $VENV_PATH/bin/python -c "import pandas as pd; df = pd.read_csv('${DATASET_DIR}/train.csv'); print(f'Total samples: {len(df)}'); print('Class distribution:'); print(df['diagnosis'].value_counts())"
    echo ""
    echo "Test data:"
    $VENV_PATH/bin/python -c "import pandas as pd; df = pd.read_csv('${DATASET_DIR}/test.csv'); print(f'Total samples: {len(df)}')"
    echo ""

    # Run the main script
    echo "=== Running Main Script ==="
    $VENV_PATH/bin/python ${WORKSPACE_DIR}/main.py \
        --train_csv ${DATASET_DIR}/train.csv \
        --test_csv ${DATASET_DIR}/test.csv \
        --train_img_dir ${DATASET_DIR}/train_images \
        --test_img_dir ${DATASET_DIR}/test_images \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 8 \
        --num_epochs 50 \
        --learning_rate 0.0001 \
        --patience 10 \
        --val_split 0.2
    
    # Display results
    echo ""
    echo "=== Final Results ==="
    if [ -f "${OUTPUT_DIR}/metrics.txt" ]; then
        cat ${OUTPUT_DIR}/metrics.txt
    else
        echo "No metrics file found."
    fi
    
    echo ""
    echo "=== Predictions Sample ==="
    if [ -f "${OUTPUT_DIR}/predictions.csv" ]; then
        head -n 10 ${OUTPUT_DIR}/predictions.csv
    else
        echo "No predictions file found."
    fi
    
    echo ""
    echo "Experiment completed at: $(date)"

} 2>&1 | tee ${RESULTS_FILE}

echo "Results saved to: ${RESULTS_FILE}"