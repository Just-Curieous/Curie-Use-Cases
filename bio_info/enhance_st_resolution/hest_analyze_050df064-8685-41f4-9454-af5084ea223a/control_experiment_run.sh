#!/bin/bash

# Control Experiment Script for HEST Analysis
# Experiment ID: 050df064-8685-41f4-9454-af5084ea223a
# Group: control_group
# Partition: partition_1
# Method: Original non-imputed ST data (baseline)

# Set up paths
WORKSPACE_DIR="/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a"
DATASET_DIR="/workspace/hest_analyze_dataset"
RESULTS_DIR="\${WORKSPACE_DIR}/results"
OUTPUT_FILE="\${WORKSPACE_DIR}/results_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.txt"

# Create necessary directories
mkdir -p "\${RESULTS_DIR}"
mkdir -p "\${DATASET_DIR}/st"

echo "=========================================================="
echo "HEST ST Data Analysis Experiment - Control Group (Partition 1)"
echo "Experiment ID: 050df064-8685-41f4-9454-af5084ea223a"
echo "Date: \$(date)"
echo "=========================================================="

# Run the analysis and redirect all output to the results file
echo "Running spatial transcriptomics analysis (no imputation)..."

python3 "\${WORKSPACE_DIR}/st_analyzer.py" 2>&1 | tee "\${OUTPUT_FILE}"

# Check if the analysis completed successfully
if [ -f "\${RESULTS_DIR}/analysis_results.txt" ]; then
    echo "Analysis completed successfully."
    echo "Results saved to: \${RESULTS_DIR}/analysis_results.txt"
    
    # Copy the script as the experiment file with the required name format
    cp "\${WORKSPACE_DIR}/control_experiment_run.sh" "\${WORKSPACE_DIR}/control_experiment_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.sh"
    
    echo "Control experiment workflow script saved as: control_experiment_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.sh"
    echo "Results saved to: \${OUTPUT_FILE}"
    exit 0
else
    echo "ERROR: Analysis failed. No results file was generated."
    exit 1
fi
