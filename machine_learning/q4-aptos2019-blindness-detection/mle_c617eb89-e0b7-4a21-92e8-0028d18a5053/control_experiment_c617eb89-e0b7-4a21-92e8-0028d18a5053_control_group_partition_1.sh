#!/bin/bash

# Control experiment script for diabetic retinopathy detection
# Experiment ID: c617eb89-e0b7-4a21-92e8-0028d18a5053
# Group: control_group_partition_1
# Using ResNet50 with PyTorch for improved performance

# Set up environment
export PATH="/openhands/micromamba/bin:$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_c617eb89-e0b7-4a21-92e8-0028d18a5053/venv"
micromamba activate $VENV_PATH/

# Define paths
WORKSPACE_DIR="/workspace/mle_c617eb89-e0b7-4a21-92e8-0028d18a5053"
DATASET_DIR="/workspace/mle_dataset"
TRAIN_CSV="${DATASET_DIR}/train.csv"
TEST_CSV="${DATASET_DIR}/test.csv"
TRAIN_IMAGES_DIR="${DATASET_DIR}/train_images"
TEST_IMAGES_DIR="${DATASET_DIR}/test_images"
OUTPUT_DIR="${WORKSPACE_DIR}/output"
RESULTS_FILE="${WORKSPACE_DIR}/results_c617eb89-e0b7-4a21-92e8-0028d18a5053_control_group_partition_1.txt"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Clean up any previous output files
rm -f ${RESULTS_FILE}

# Print system information
{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "Python version: $($VENV_PATH/bin/python --version)"
    echo "PyTorch version: $($VENV_PATH/bin/python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}\")')"
    echo "GPU information:"
    nvidia-smi
    echo ""

    echo "=== Dataset Information ==="
    echo "Train samples: $(wc -l < ${TRAIN_CSV}) (including header)"
    echo "Test samples: $(wc -l < ${TEST_CSV}) (including header)"
    echo "Train images: $(ls ${TRAIN_IMAGES_DIR} | wc -l)"
    echo "Test images: $(ls ${TEST_IMAGES_DIR} | wc -l)"
    echo ""

    echo "=== Experiment Settings ==="
    echo "- Using PyTorch with GPU acceleration"
    echo "- Using batch size of 16"
    echo "- Using standard image size (224x224) for ResNet50"
    echo "- Using ResNet50 model with pretrained weights"
    echo "- Using all training samples (full dataset)"
    echo "- Training for 30 epochs with early stopping"
    echo "- Advanced preprocessing with circular crop"
    echo "- Data augmentation (flips, rotations, color jitter)"
    echo "- Learning rate scheduling"
    echo "- DataLoader with num_workers=0 to avoid multiprocessing issues"
    echo ""

    echo "=== Starting Experiment ==="
    echo "Model: ResNet50 (PyTorch)"
    echo "Experiment ID: c617eb89-e0b7-4a21-92e8-0028d18a5053"
    echo "Group: control_group_partition_1"
    echo ""
    
    # Run the experiment with ResNet50
    $VENV_PATH/bin/python ${WORKSPACE_DIR}/diabetic_retinopathy_detection_resnet50.py \
        --train_csv ${TRAIN_CSV} \
        --test_csv ${TEST_CSV} \
        --train_images_dir ${TRAIN_IMAGES_DIR} \
        --test_images_dir ${TEST_IMAGES_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --epochs 30 \
        --img_size 224 \
        --patience 5

    EXPERIMENT_EXIT_CODE=$?
    echo ""
    echo "=== Experiment Completed with Exit Code: ${EXPERIMENT_EXIT_CODE} ==="
    
    # Display results
    echo "=== Results ==="
    if [ -f "${OUTPUT_DIR}/metrics.csv" ]; then
        echo "Metrics:"
        cat ${OUTPUT_DIR}/metrics.csv
    else
        echo "No metrics file found."
    fi
    
    echo ""
    echo "Submission file preview:"
    if [ -f "${OUTPUT_DIR}/submission.csv" ]; then
        head -n 10 ${OUTPUT_DIR}/submission.csv
    else
        echo "No submission file found."
    fi
    
    # Display quadratic weighted kappa score if available
    if [ -f "${OUTPUT_DIR}/metrics.csv" ] && grep -q "kappa" "${OUTPUT_DIR}/metrics.csv"; then
        echo ""
        echo "Quadratic Weighted Kappa Score:"
        grep "kappa" "${OUTPUT_DIR}/metrics.csv"
    fi
    
    echo ""
    echo "=== End of Experiment ==="
} > "${RESULTS_FILE}" 2>&1

echo "Results saved to ${RESULTS_FILE}"