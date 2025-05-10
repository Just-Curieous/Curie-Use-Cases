#!/bin/bash

# Set up environment variables
export PATH="/openhands/micromamba/bin:\$PATH"
export VIRTUAL_ENV="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv"
export VENV_PATH="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv"

# Activate the micromamba environment
eval "\$(micromamba shell hook --shell bash)"
micromamba activate \$VENV_PATH/

# Define paths
BASE_DIR="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889"
RESULTS_FILE="\${BASE_DIR}/results_ac20158a-ee6d-48ad-a2a6-9f6cb203d889_experimental_group_partition_1.txt"

# Run the test script
echo "=== Running test script ===" | tee -a \$RESULTS_FILE
python \${BASE_DIR}/test_script.py | tee -a \$RESULTS_FILE

# Check if the summary file exists
SUMMARY_FILE="\${BASE_DIR}/loss_function_comparison_summary.txt"
if [ -f "\$SUMMARY_FILE" ]; then
    echo "=== Test successful: Summary file created ===" | tee -a \$RESULTS_FILE
    cat \$SUMMARY_FILE | tee -a \$RESULTS_FILE
    
    # Extract best performing loss function
    BEST_LOSS=\$(grep "Best performing loss function:" \$SUMMARY_FILE | cut -d':' -f2 | cut -d'(' -f1 | xargs)
    BEST_CORR=\$(grep "Best performing loss function:" \$SUMMARY_FILE | grep -o "Overall Rank Correlation: [0-9.-]*" | cut -d':' -f2 | xargs)
    
    echo "" | tee -a \$RESULTS_FILE
    echo "=== EXPERIMENT CONCLUSION ===" | tee -a \$RESULTS_FILE
    echo "The best performing loss function is: \$BEST_LOSS" | tee -a \$RESULTS_FILE
    echo "With overall rank correlation: \$BEST_CORR" | tee -a \$RESULTS_FILE
    echo "=== Test completed at \$(date) ===" | tee -a \$RESULTS_FILE
else
    echo "=== Test failed: Summary file not created ===" | tee -a \$RESULTS_FILE
fi
