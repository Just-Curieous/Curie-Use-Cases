#!/bin/bash

# Control Experiment Script for HEST Analysis
# Experiment ID: 050df064-8685-41f4-9454-af5084ea223a
# Group: control_group
# Partition: partition_1
# Method: Original non-imputed ST data (baseline)

# Set up paths
WORKSPACE_DIR="/workspace/hest_analyze_050df064-8685-41f4-9454-af5084ea223a"
DATASET_DIR="/workspace/hest_analyze_dataset"
ST_DATA_DIR="\${DATASET_DIR}/st"
RESULTS_DIR="\${WORKSPACE_DIR}/results"
OUTPUT_FILE="\${WORKSPACE_DIR}/results_050df064-8685-41f4-9454-af5084ea223a_control_group_partition_1.txt"

# Create necessary directories
mkdir -p "\${RESULTS_DIR}"
mkdir -p "\${ST_DATA_DIR}"

# Run the analysis and redirect all output to the results file
python3 "\${WORKSPACE_DIR}/st_analyzer.py" 2>&1 | tee "\${OUTPUT_FILE}"

