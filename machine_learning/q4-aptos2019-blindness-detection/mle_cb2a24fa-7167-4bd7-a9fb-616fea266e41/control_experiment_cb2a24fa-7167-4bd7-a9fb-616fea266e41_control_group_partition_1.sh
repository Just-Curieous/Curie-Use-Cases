#!/bin/bash

# Control script for diabetic retinopathy detection experiment
# Control Group - Standard Classification Approach

# Set up environment
export PATH="/openhands/micromamba/bin:\$PATH"
micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
eval "\$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/mle_cb2a24fa-7167-4bd7-a9fb-616fea266e41/venv"
micromamba activate \$VENV_PATH/

# Install required packages
\$VENV_PATH/bin/pip install seaborn scikit-learn matplotlib pandas opencv-python efficientnet_pytorch

# Define paths
WORKSPACE_DIR="/workspace/mle_cb2a24fa-7167-4bd7-a9fb-616fea266e41"
DATASET_DIR="/workspace/mle_dataset"
OUTPUT_DIR="\${WORKSPACE_DIR}/output"
RESULTS_FILE="\${WORKSPACE_DIR}/results_cb2a24fa-7167-4bd7-a9fb-616fea266e41_control_group_partition_1.txt"

# Create output directory if it doesn't exist
mkdir -p \${OUTPUT_DIR}

# Start logging
echo "=== Diabetic Retinopathy Detection Experiment - Control Group ===" > \${RESULTS_FILE}
echo "Started at: \$(date)" >> \${RESULTS_FILE}
echo "" >> \${RESULTS_FILE}

# Check GPU availability
echo "=== GPU Information ===" >> \${RESULTS_FILE}
\${VENV_PATH}/bin/python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}');" >> \${RESULTS_FILE} 2>&1
echo "" >> \${RESULTS_FILE}

# Check dataset information
echo "=== Dataset Information ===" >> \${RESULTS_FILE}
\${VENV_PATH}/bin/python -c "import pandas as pd; train = pd.read_csv('\${DATASET_DIR}/train.csv'); test = pd.read_csv('\${DATASET_DIR}/test.csv'); print(f'Training samples: {len(train)}'); print(f'Test samples: {len(test)}'); print('Class distribution in training data:'); print(train['diagnosis'].value_counts().sort_index())" >> \${RESULTS_FILE} 2>&1
echo "" >> \${RESULTS_FILE}

# Run the training
echo "=== Running Training and Evaluation ===" >> \${RESULTS_FILE}
if \${VENV_PATH}/bin/python \${WORKSPACE_DIR}/main.py \
        --train_csv \${DATASET_DIR}/train.csv \
        --test_csv \${DATASET_DIR}/test.csv \
        --train_img_dir \${DATASET_DIR}/train_images \
        --test_img_dir \${DATASET_DIR}/test_images \
        --output_dir \${OUTPUT_DIR} \
        --batch_size 8 >> \${RESULTS_FILE} 2>&1; then
    
    # Summarize results
    echo "" >> \${RESULTS_FILE}
    echo "=== Results Summary ===" >> \${RESULTS_FILE}
    
    # Extract metrics from the log file
    if [ -f "\${OUTPUT_DIR}/training.log" ]; then
        echo "Best validation metrics:" >> \${RESULTS_FILE}
        grep "Validation Kappa\|Validation Accuracy\|Class" \${OUTPUT_DIR}/training.log | tail -n 10 >> \${RESULTS_FILE}
    fi
    
    # List generated files
    echo "" >> \${RESULTS_FILE}
    echo "=== Generated Files ===" >> \${RESULTS_FILE}
    ls -la \${OUTPUT_DIR} >> \${RESULTS_FILE}
else
    echo "" >> \${RESULTS_FILE}
    echo "=== ERROR: Model Training Failed ===" >> \${RESULTS_FILE}
fi

# End logging
echo "" >> \${RESULTS_FILE}
echo "Finished at: \$(date)" >> \${RESULTS_FILE}
echo "=== Experiment Completed ===" >> \${RESULTS_FILE}
