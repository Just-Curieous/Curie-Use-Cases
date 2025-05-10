#!/bin/bash

# Set up environment variables
export PATH="/openhands/micromamba/bin:$PATH"
export VIRTUAL_ENV="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv"
export VENV_PATH="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889/venv"

# Activate the micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate $VENV_PATH/

# Set up OpenCL for GPU support
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Define paths and variables
BASE_DIR="/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889"
RESULTS_FILE="${BASE_DIR}/results_ac20158a-ee6d-48ad-a2a6-9f6cb203d889_experimental_group_partition_1.txt"
SUMMARY_FILE="${BASE_DIR}/loss_function_comparison_summary.txt"
MODEL_SCRIPT="${BASE_DIR}/model_training.py"

# Define loss functions to test
LOSS_FUNCTIONS=("regression_l1" "huber" "fair" "poisson" "quantile")

# Create required directories for each loss function
for LOSS in "${LOSS_FUNCTIONS[@]}"; do
    mkdir -p "${BASE_DIR}/results/${LOSS}"
done

# Initialize results file
echo "Starting experimental group workflow: Testing different LightGBM loss functions" > $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "=======================================================" >> $RESULTS_FILE

# Initialize arrays to store metrics for comparison
declare -A OVERALL_METRICS
declare -A YEARLY_METRICS

# Run model training for each loss function
for LOSS in "${LOSS_FUNCTIONS[@]}"; do
    echo "" >> $RESULTS_FILE
    echo "=======================================================" | tee -a $RESULTS_FILE
    echo "Testing loss function: ${LOSS}" | tee -a $RESULTS_FILE
    echo "=======================================================" | tee -a $RESULTS_FILE
    
    # Define config path for this loss function
    CONFIG_PATH="${BASE_DIR}/${LOSS}_config.json"
    
    # Display the configuration being used
    echo "Configuration file: ${CONFIG_PATH}" | tee -a $RESULTS_FILE
    echo "Configuration parameters:" | tee -a $RESULTS_FILE
    cat $CONFIG_PATH | tee -a $RESULTS_FILE
    echo "" | tee -a $RESULTS_FILE
    
    # Run the model training with the specific loss function
    echo "Starting model training with ${LOSS} loss function..." | tee -a $RESULTS_FILE
    $VENV_PATH/bin/python $MODEL_SCRIPT --config $CONFIG_PATH 2>&1 | tee -a $RESULTS_FILE
    
    # Find the most recent metrics file for this loss function
    LATEST_METRICS=$(ls -t ${BASE_DIR}/results/${LOSS}/metrics_*.json 2>/dev/null | head -n 1)
    
    if [ -f "$LATEST_METRICS" ]; then
        echo "Results from: $LATEST_METRICS" | tee -a $RESULTS_FILE
        echo "Rank Correlation Metrics:" | tee -a $RESULTS_FILE
        
        # Extract and display overall rank correlation
        OVERALL_CORR=$(grep -o '"overall": [0-9.-]*' $LATEST_METRICS | cut -d' ' -f2)
        echo "Overall Rank Correlation: $OVERALL_CORR" | tee -a $RESULTS_FILE
        
        # Store the overall metric for comparison
        OVERALL_METRICS[$LOSS]=$OVERALL_CORR
        
        # Extract and display yearly rank correlations
        for YEAR in {2020..2023}; do
            YEAR_CORR=$(grep -o "\"$YEAR\": [0-9.-]*" $LATEST_METRICS | cut -d' ' -f2)
            if [ ! -z "$YEAR_CORR" ]; then
                echo "$YEAR Rank Correlation: $YEAR_CORR" | tee -a $RESULTS_FILE
                # Store yearly metrics for comparison
                YEARLY_METRICS["${LOSS}_${YEAR}"]=$YEAR_CORR
            fi
        done
        
        echo "Metrics file verified and extracted successfully." | tee -a $RESULTS_FILE
    else
        echo "No metrics file found for ${LOSS}. Check for errors in the experiment." | tee -a $RESULTS_FILE
    fi
    
    echo "=======================================================" | tee -a $RESULTS_FILE
done

# Compare performance across all loss functions
echo "" >> $RESULTS_FILE
echo "=======================================================" | tee -a $RESULTS_FILE
echo "PERFORMANCE COMPARISON ACROSS LOSS FUNCTIONS" | tee -a $RESULTS_FILE
echo "=======================================================" | tee -a $RESULTS_FILE

# Create a summary of overall performance
echo "Overall Rank Correlation by Loss Function:" | tee -a $RESULTS_FILE
for LOSS in "${LOSS_FUNCTIONS[@]}"; do
    if [ ! -z "${OVERALL_METRICS[$LOSS]}" ]; then
        echo "${LOSS}: ${OVERALL_METRICS[$LOSS]}" | tee -a $RESULTS_FILE
    else
        echo "${LOSS}: No data available" | tee -a $RESULTS_FILE
    fi
done

# Create a summary of yearly performance
echo "" | tee -a $RESULTS_FILE
echo "Yearly Rank Correlation by Loss Function:" | tee -a $RESULTS_FILE
for YEAR in {2020..2023}; do
    echo "Year $YEAR:" | tee -a $RESULTS_FILE
    for LOSS in "${LOSS_FUNCTIONS[@]}"; do
        if [ ! -z "${YEARLY_METRICS["${LOSS}_${YEAR}"]}" ]; then
            echo "  ${LOSS}: ${YEARLY_METRICS["${LOSS}_${YEAR}"]}" | tee -a $RESULTS_FILE
        else
            echo "  ${LOSS}: No data available" | tee -a $RESULTS_FILE
        fi
    done
    echo "" | tee -a $RESULTS_FILE
done

# Determine the best performing loss function based on overall rank correlation
echo "Determining best performing loss function..." | tee -a $RESULTS_FILE
BEST_LOSS=""
BEST_CORR=-999

for LOSS in "${LOSS_FUNCTIONS[@]}"; do
    if [ ! -z "${OVERALL_METRICS[$LOSS]}" ]; then
        CORR="${OVERALL_METRICS[$LOSS]}"
        if (( $(echo "$CORR > $BEST_CORR" | bc -l) )); then
            BEST_CORR=$CORR
            BEST_LOSS=$LOSS
        fi
    fi
done

if [ ! -z "$BEST_LOSS" ]; then
    echo "Best performing loss function: $BEST_LOSS with overall rank correlation: $BEST_CORR" | tee -a $RESULTS_FILE
else
    echo "Could not determine best loss function due to missing data" | tee -a $RESULTS_FILE
fi

# Save a summary of the results to a separate file
echo "=======================================================" > $SUMMARY_FILE
echo "LIGHTGBM LOSS FUNCTION COMPARISON SUMMARY" >> $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "Overall Rank Correlation by Loss Function:" >> $SUMMARY_FILE
for LOSS in "${LOSS_FUNCTIONS[@]}"; do
    if [ ! -z "${OVERALL_METRICS[$LOSS]}" ]; then
        echo "${LOSS}: ${OVERALL_METRICS[$LOSS]}" >> $SUMMARY_FILE
    else
        echo "${LOSS}: No data available" >> $SUMMARY_FILE
    fi
done
echo "" >> $SUMMARY_FILE
echo "Best performing loss function: $BEST_LOSS with overall rank correlation: $BEST_CORR" >> $SUMMARY_FILE
echo "=======================================================" >> $SUMMARY_FILE

echo "" | tee -a $RESULTS_FILE
echo "=======================================================" | tee -a $RESULTS_FILE
echo "Experimental workflow completed. Results saved to:" | tee -a $RESULTS_FILE
echo "1. Detailed results: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "2. Summary: $SUMMARY_FILE" | tee -a $RESULTS_FILE
echo "=======================================================" | tee -a $RESULTS_FILE