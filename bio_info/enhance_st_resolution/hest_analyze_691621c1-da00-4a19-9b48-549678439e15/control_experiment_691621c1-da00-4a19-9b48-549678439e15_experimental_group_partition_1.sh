#!/bin/bash

# Experimental Group Script for HEST Analysis
# Experiment ID: 691621c1-da00-4a19-9b48-549678439e15
# Group: experimental_group_partition_1
# Methods: Bicubic interpolation, SRCNN, Histology-guided, Gene VAE

# Set up paths
WORKSPACE_DIR="/workspace/hest_analyze_691621c1-da00-4a19-9b48-549678439e15"
DATASET_DIR="/workspace/hest_analyze_dataset"
RESULTS_FILE="\${WORKSPACE_DIR}/results_691621c1-da00-4a19-9b48-549678439e15_experimental_group_partition_1.txt"
RESULTS_DIR="\${WORKSPACE_DIR}/results"

# Create results directory if it doesn't exist
mkdir -p "\${RESULTS_DIR}"

# Start logging
{
    echo "=========================================================="
    echo "HEST Analysis Experiment - Experimental Group (Partition 1)"
    echo "Experiment ID: 691621c1-da00-4a19-9b48-549678439e15"
    echo "Date: \$(date)"
    echo "=========================================================="
    echo ""

    # Check Python environment
    echo "Checking Python environment..."
    which python3
    python3 --version
    
    # Install required packages
    echo "Installing missing packages..."
    pip install numpy matplotlib torch==2.1.0 scikit-learn scipy scikit-image pandas psutil
    
    echo "=========================================================="
    echo "Starting enhancement experiment with multiple methods:"
    echo "1. Bicubic interpolation"
    echo "2. Deep learning super-resolution (SRCNN)"
    echo "3. Histology-guided deep learning enhancement"
    echo "4. Gene expression aware variational autoencoder (VAE)"
    echo "=========================================================="
    
    # Run the enhancement experiment
    python3 "\${WORKSPACE_DIR}/st_enhancement_experiment.py"
    
    echo "=========================================================="
    echo "Enhancement experiment completed"
    echo "=========================================================="
    
    # List results
    echo "Results files:"
    ls -la "\${RESULTS_DIR}"
    
    echo "=========================================================="
    echo "Experiment completed successfully"
    echo "Results saved to: \${RESULTS_DIR}"
    echo "=========================================================="

} 2>&1 | tee "\${RESULTS_FILE}"

echo "Experiment log saved to: \${RESULTS_FILE}"
