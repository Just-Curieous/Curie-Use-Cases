#!/bin/bash

# Experimental group script for diabetic retinopathy detection
# Experiment ID: c617eb89-e0b7-4a21-92e8-0028d18a5053
# Group: experimental_group_partition_1
# Using multiple CNN models (EfficientNetB4, DenseNet121, InceptionV3) with PyTorch

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
OUTPUT_DIR="${WORKSPACE_DIR}/output_experimental"
RESULTS_FILE="${WORKSPACE_DIR}/results_c617eb89-e0b7-4a21-92e8-0028d18a5053_experimental_group_partition_1.txt"

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
    echo "- Using model-specific image sizes (224x224 for EfficientNetB4/DenseNet121, 299x299 for InceptionV3)"
    echo "- Using multiple models: EfficientNetB4, DenseNet121, InceptionV3"
    echo "- Using all training samples (full dataset)"
    echo "- Training for 30 epochs with early stopping"
    echo "- Advanced preprocessing with circular crop"
    echo "- Data augmentation (flips, rotations, color jitter)"
    echo "- Learning rate scheduling"
    echo "- DataLoader with num_workers=0 to avoid multiprocessing issues"
    echo ""

    echo "=== Starting Experiment ==="
    echo "Models: EfficientNetB4, DenseNet121, InceptionV3 (PyTorch)"
    echo "Experiment ID: c617eb89-e0b7-4a21-92e8-0028d18a5053"
    echo "Group: experimental_group_partition_1"
    echo ""
    
    # Run the experiment with multiple models
    $VENV_PATH/bin/python ${WORKSPACE_DIR}/diabetic_retinopathy_detection_multi_model.py \
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
    if [ -f "${OUTPUT_DIR}/all_models_metrics.csv" ]; then
        echo "Combined Metrics for All Models:"
        cat ${OUTPUT_DIR}/all_models_metrics.csv
    else
        echo "No combined metrics file found."
    fi
    
    # Display individual model metrics
    for MODEL in "efficientnet_b4" "densenet121" "inception_v3"; do
        echo ""
        echo "=== ${MODEL} Results ==="
        
        # Display submission file preview
        echo "Submission file preview for ${MODEL}:"
        if [ -f "${OUTPUT_DIR}/submission_${MODEL}.csv" ]; then
            head -n 5 ${OUTPUT_DIR}/submission_${MODEL}.csv
        else
            echo "No submission file found for ${MODEL}."
        fi
        
        # Display quadratic weighted kappa score if available
        if grep -q "${MODEL}" "${OUTPUT_DIR}/all_models_metrics.csv"; then
            echo ""
            echo "Metrics for ${MODEL}:"
            grep "${MODEL}" "${OUTPUT_DIR}/all_models_metrics.csv"
        fi
    done
    
    echo ""
    echo "=== Model Comparison ==="
    echo "See the following files for visual comparisons:"
    echo "- ${OUTPUT_DIR}/model_comparison.png"
    echo "- ${OUTPUT_DIR}/inference_time_comparison.png"
    
    echo ""
    echo "=== End of Experiment ==="
} > "${RESULTS_FILE}" 2>&1

echo "Results saved to ${RESULTS_FILE}"