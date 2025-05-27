#!/bin/bash

# Experimental Group Partition 1 - Control Experiment Script
# This script runs 5 different configurations of ensemble models for stock return prediction

# Set up error handling
set -e
trap 'echo "Error occurred at line \$LINENO. Command: \$BASH_COMMAND"' ERR

# Define paths
WORKSPACE_DIR="/workspace/starter_code_c68dbfd0-0457-4dd3-8b2c-53698e8de0dc"
RESULTS_DIR="\${WORKSPACE_DIR}/results"
OUTPUT_FILE="\${WORKSPACE_DIR}/results_c68dbfd0-0457-4dd3-8b2c-53698e8de0dc_experimental_group_partition_1.txt"
PYTHON_PATH="/opt/micromamba/envs/curie/bin/python"

# Configuration files
CONFIG_1="\${WORKSPACE_DIR}/config_1_boosting_raw_default.json"
CONFIG_2="\${WORKSPACE_DIR}/config_2_boosting_raw_optimized.json"
CONFIG_3="\${WORKSPACE_DIR}/config_3_boosting_momentum_default.json"
CONFIG_4="\${WORKSPACE_DIR}/config_4_boosting_momentum_optimized.json"
CONFIG_5="\${WORKSPACE_DIR}/config_5_stacking_momentum_optimized.json"

# Record start time
START_TIME=\$(date +%s)

# Create results directory if it doesn't exist
mkdir -p \${RESULTS_DIR}

# Initialize output file
echo "========================================================" > \${OUTPUT_FILE}
echo "EXPERIMENTAL GROUP PARTITION 1 - ENSEMBLE MODEL EXPERIMENTS" >> \${OUTPUT_FILE}
echo "Started at: \$(date)" >> \${OUTPUT_FILE}
echo "========================================================" >> \${OUTPUT_FILE}
echo "" >> \${OUTPUT_FILE}

# Set up OpenCL environment for GPU
echo "Setting up OpenCL environment for GPU..." | tee -a \${OUTPUT_FILE}
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install required packages
echo "Installing required Python packages..." | tee -a \${OUTPUT_FILE}
\${PYTHON_PATH} -m pip install pandas numpy scikit-learn lightgbm pyarrow | tee -a \${OUTPUT_FILE}
\${PYTHON_PATH} -m pip install xgboost catboost | tee -a \${OUTPUT_FILE}

# Create an array of configuration files and their descriptions
CONFIG_FILES=(
    "\${CONFIG_1}"
    "\${CONFIG_2}"
    "\${CONFIG_3}"
    "\${CONFIG_4}"
    "\${CONFIG_5}"
)

CONFIG_DESCRIPTIONS=(
    "Boosting of weak learners (LightGBM+XGBoost+CatBoost) with raw factors and default hyperparameters"
    "Boosting of weak learners (LightGBM+XGBoost+CatBoost) with raw factors and optimized hyperparameters"
    "Boosting of weak learners (LightGBM+XGBoost+CatBoost) with factor momentum + mean reversion and default hyperparameters"
    "Boosting of weak learners (LightGBM+XGBoost+CatBoost) with factor momentum + mean reversion and optimized hyperparameters"
    "Stacking with LightGBM meta-learner (LightGBM+XGBoost+CatBoost) with factor momentum + mean reversion and optimized hyperparameters"
)

# Run each configuration
for i in {0..4}; do
    echo "" | tee -a \${OUTPUT_FILE}
    echo "========================================================" | tee -a \${OUTPUT_FILE}
    echo "Running configuration \$((i+1)): \${CONFIG_DESCRIPTIONS[\$i]}" | tee -a \${OUTPUT_FILE}
    echo "Started at: \$(date)" | tee -a \${OUTPUT_FILE}
    echo "========================================================" | tee -a \${OUTPUT_FILE}
    
    # Run the model training
    CONFIG_START_TIME=\$(date +%s)
    
    \${PYTHON_PATH} \${WORKSPACE_DIR}/ensemble_model_training.py --config \${CONFIG_FILES[\$i]} | tee -a \${OUTPUT_FILE}
    
    CONFIG_END_TIME=\$(date +%s)
    CONFIG_DURATION=\$((CONFIG_END_TIME - CONFIG_START_TIME))
    
    echo "" | tee -a \${OUTPUT_FILE}
    echo "Configuration \$((i+1)) completed at: \$(date)" | tee -a \${OUTPUT_FILE}
    echo "Duration: \$((CONFIG_DURATION / 60)) minutes and \$((CONFIG_DURATION % 60)) seconds" | tee -a \${OUTPUT_FILE}
    echo "" | tee -a \${OUTPUT_FILE}
done

# Collect results
echo "" >> \${OUTPUT_FILE}
echo "========================================================" >> \${OUTPUT_FILE}
echo "SUMMARY OF RESULTS" >> \${OUTPUT_FILE}
echo "========================================================" >> \${OUTPUT_FILE}
echo "" >> \${OUTPUT_FILE}

# Extract metrics from result files
echo "Rank Correlation Metrics:" >> \${OUTPUT_FILE}
for i in {0..4}; do
    LATEST_RESULT=\$(find \${RESULTS_DIR} -name "metrics_*.json" -type f | sort | tail -n 1)
    if [[ -f "\${LATEST_RESULT}" ]]; then
        RANK_CORR=\$(cat \${LATEST_RESULT} | grep -o '"rank_correlation": [0-9.-]*' | cut -d' ' -f2)
        MSE=\$(cat \${LATEST_RESULT} | grep -o '"mean_squared_error": [0-9.-]*' | cut -d' ' -f2)
        DIR_ACC=\$(cat \${LATEST_RESULT} | grep -o '"directional_accuracy": [0-9.-]*' | cut -d' ' -f2)
        echo "Configuration \$((i+1)) (\${CONFIG_DESCRIPTIONS[\$i]}):" >> \${OUTPUT_FILE}
        echo "  - Rank Correlation: \${RANK_CORR}" >> \${OUTPUT_FILE}
        echo "  - Mean Squared Error: \${MSE}" >> \${OUTPUT_FILE}
        echo "  - Directional Accuracy: \${DIR_ACC}" >> \${OUTPUT_FILE}
        echo "" >> \${OUTPUT_FILE}
    else
        echo "Configuration \$((i+1)) (\${CONFIG_DESCRIPTIONS[\$i]}): No results file found" >> \${OUTPUT_FILE}
    fi
done

# Calculate and display total execution time
END_TIME=\$(date +%s)
EXECUTION_TIME=\$((END_TIME - START_TIME))
echo "" >> \${OUTPUT_FILE}
echo "========================================================" >> \${OUTPUT_FILE}
echo "Total experiment completed at: \$(date)" >> \${OUTPUT_FILE}
echo "Total execution time: \$((EXECUTION_TIME / 60)) minutes and \$((EXECUTION_TIME % 60)) seconds" >> \${OUTPUT_FILE}
echo "========================================================" >> \${OUTPUT_FILE}

echo "Experiment completed. Results saved to \${OUTPUT_FILE}"
