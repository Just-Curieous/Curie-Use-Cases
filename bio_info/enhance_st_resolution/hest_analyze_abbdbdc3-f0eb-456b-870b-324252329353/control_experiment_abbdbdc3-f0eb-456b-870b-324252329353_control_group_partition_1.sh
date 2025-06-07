#!/bin/bash

# Control Experiment Script for HEST Analysis
# Experiment ID: abbdbdc3-f0eb-456b-870b-324252329353
# Group: control_group_partition_1
# Method: ST data analysis without histology integration

# Set up paths
WORKSPACE_DIR="/workspace/hest_analyze_abbdbdc3-f0eb-456b-870b-324252329353"
DATASET_DIR="/workspace/hest_analyze_dataset"
RESULTS_FILE="${WORKSPACE_DIR}/results_abbdbdc3-f0eb-456b-870b-324252329353_control_group_partition_1.txt"
RESULTS_DIR="${WORKSPACE_DIR}/results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Start logging to results file
{
    echo "=========================================================="
    echo "HEST Analysis Experiment - Control Group (Partition 1)"
    echo "Experiment ID: abbdbdc3-f0eb-456b-870b-324252329353"
    echo "Date: $(date)"
    echo "=========================================================="
    echo ""

    # Check Python version and packages
    echo "Python version:"
    python3 --version
    
    echo "Checking if required packages are installed..."
    pip list | grep -E "numpy|pandas|matplotlib|scikit-learn|torch"
    
    # Check dataset
    echo "Checking dataset structure..."
    ls -la "${DATASET_DIR}"
    echo "ST data directory:"
    ls -la "${DATASET_DIR}/st"
    
    echo "Running ST data analysis without histology integration..."
    echo "=========================================================="
    
    # Run the analysis
    start_time=$(date +%s)
    
    python3 "${WORKSPACE_DIR}/st_analyzer.py" \
        --dataset_path "${DATASET_DIR}" \
        --output_dir "${RESULTS_DIR}"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "=========================================================="
    echo "Analysis completed in ${duration} seconds"
    echo "=========================================================="
    
    # Display results summary
    echo "Results summary:"
    
    # Show dataset info
    if [ -f "${RESULTS_DIR}/dataset_info.txt" ]; then
        echo "Dataset Information:"
        cat "${RESULTS_DIR}/dataset_info.txt"
        echo ""
    fi
    
    # Show analysis summary
    if [ -f "${RESULTS_DIR}/analysis_summary.txt" ]; then
        echo "Analysis Summary:"
        cat "${RESULTS_DIR}/analysis_summary.txt"
        echo ""
    fi
    
    # List result files
    echo "Generated result files:"
    find "${RESULTS_DIR}" -name "*_analysis_results.txt" | sort
    
    # List visualizations
    echo "Generated visualizations:"
    find "${RESULTS_DIR}" -name "*.png" | sort
    
    echo "=========================================================="
    echo "Experiment completed successfully"
    echo "Results saved to: ${RESULTS_DIR}"
    echo "=========================================================="

} 2>&1 | tee "${RESULTS_FILE}"

echo "Control experiment log saved to: ${RESULTS_FILE}"
