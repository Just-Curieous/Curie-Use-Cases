#!/bin/bash

# Control Experiment Script for HEST Analysis
# Experiment ID: 691621c1-da00-4a19-9b48-549678439e15
# Group: control_group
# Partition: partition_1
# Method: Original unenhanced ST data

# Set up paths
WORKSPACE_DIR="/workspace/hest_analyze_691621c1-da00-4a19-9b48-549678439e15"
DATASET_DIR="/workspace/hest_analyze_dataset"
VENV_PATH="${WORKSPACE_DIR}/venv"
RESULTS_FILE="${WORKSPACE_DIR}/results_691621c1-da00-4a19-9b48-549678439e15_control_group_partition_1.txt"
RESULTS_DIR="${WORKSPACE_DIR}/results"

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Start logging
{
    echo "=========================================================="
    echo "HEST Analysis Experiment - Control Group (Partition 1)"
    echo "Experiment ID: 691621c1-da00-4a19-9b48-549678439e15"
    echo "Method: Original unenhanced ST data"
    echo "Date: $(date)"
    echo "=========================================================="
    echo ""

    # Activate Python environment
    echo "Activating Python environment..."
    export PATH="/openhands/micromamba/bin:$PATH"
    micromamba shell init --shell bash --root-prefix=~/.local/share/mamba
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "$VENV_PATH/"
    
    # Check Python version
    echo "Python version:"
    python --version
    
    # Check GPU availability
    echo "Checking GPU availability..."
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
    
    # Check dataset
    echo "Checking dataset..."
    ls -la "${DATASET_DIR}/st/"
    ls -la "${DATASET_DIR}/thumbnails/"
    
    echo "=========================================================="
    echo "Starting analysis with method: original (control group)"
    echo "=========================================================="
    
    # Run the analysis
    start_time=$(date +%s)
    
    python "${WORKSPACE_DIR}/st_analysis.py"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "=========================================================="
    echo "Analysis completed in ${duration} seconds"
    echo "=========================================================="
    
    # Summarize results
    echo "Results summary:"
    
    # Find all result files
    result_files=$(find "${RESULTS_DIR}" -name "*_original_results.txt")
    
    for file in ${result_files}; do
        echo "----------------------------------------"
        echo "Results from: ${file}"
        echo "----------------------------------------"
        cat "${file}"
        echo ""
    done
    
    # List all generated visualizations
    echo "Generated visualizations:"
    find "${RESULTS_DIR}" -name "*.png" | sort
    
    echo "=========================================================="
    echo "Experiment completed successfully"
    echo "Results saved to: ${RESULTS_DIR}"
    echo "=========================================================="
    
    # Deactivate environment
    micromamba deactivate

} 2>&1 | tee "${RESULTS_FILE}"

echo "Experiment log saved to: ${RESULTS_FILE}"
