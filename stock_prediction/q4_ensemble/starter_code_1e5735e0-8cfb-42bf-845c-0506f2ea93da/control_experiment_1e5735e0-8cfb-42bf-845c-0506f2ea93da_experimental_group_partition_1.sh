#!/bin/bash

# Experimental Group Partition 1 - Control Experiment for Stock Return Prediction
# This script runs the experimental group's partition 1 experiment with 5 different configurations

# Define output file for logs
OUTPUT_FILE="/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/results_1e5735e0-8cfb-42bf-845c-0506f2ea93da_experimental_group_partition_1.txt"

# Create results directory if it doesn't exist
mkdir -p "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/results"

# Start timestamp
echo "==================================================" > "$OUTPUT_FILE"
echo "EXPERIMENTAL GROUP PARTITION 1 - STARTED: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"

# Setup OpenCL environment for GPU acceleration
echo "Setting up OpenCL environment..." >> "$OUTPUT_FILE"
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Activate the environment
echo "Activating micromamba environment..." >> "$OUTPUT_FILE"
export PATH="/openhands/micromamba/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"
export VENV_PATH="/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/venv"
micromamba activate "$VENV_PATH/" >> "$OUTPUT_FILE" 2>&1

# Check GPU availability
echo "Checking GPU availability..." >> "$OUTPUT_FILE"
nvidia-smi >> "$OUTPUT_FILE" 2>&1

# Change to the working directory
cd /workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/

# Define configurations to test
CONFIG_FILES=(
    "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/config_1_factor_momentum_mean_reversion_default_equal_weights.json"
    "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/config_2_raw_factors_only_optimized_equal_weights.json"
    "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/config_3_factor_momentum_mean_reversion_optimized_equal_weights.json"
    "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/config_4_factor_momentum_mean_reversion_optimized_performance_based_weights.json"
    "/workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/config_5_factor_interactions_pca_optimized_performance_based_weights.json"
)

CONFIG_DESCRIPTIONS=(
    "Feature Engineering: factor momentum + mean reversion, Hyperparameters: default, Weighting: equal weights"
    "Feature Engineering: raw factors only, Hyperparameters: optimized, Weighting: equal weights"
    "Feature Engineering: factor momentum + mean reversion, Hyperparameters: optimized, Weighting: equal weights"
    "Feature Engineering: factor momentum + mean reversion, Hyperparameters: optimized, Weighting: performance-based weights"
    "Feature Engineering: factor interactions + PCA, Hyperparameters: optimized, Weighting: performance-based weights"
)

# Initialize arrays to store results
RANK_CORRELATIONS=()
SHARPE_RATIOS=()
MSE_VALUES=()
COMPUTATION_TIMES=()

# Run each configuration
for i in "${!CONFIG_FILES[@]}"; do
    CONFIG_FILE="${CONFIG_FILES[$i]}"
    CONFIG_DESC="${CONFIG_DESCRIPTIONS[$i]}"
    
    echo -e "\n\n==================================================" >> "$OUTPUT_FILE"
    echo "RUNNING CONFIGURATION $((i+1))/5: $CONFIG_DESC" >> "$OUTPUT_FILE"
    echo "==================================================" >> "$OUTPUT_FILE"
    echo "Using configuration file: $CONFIG_FILE" >> "$OUTPUT_FILE"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Execute the model training script with the current configuration
    "$VENV_PATH/bin/python" /workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/model_training_experimental.py --config "$CONFIG_FILE" >> "$OUTPUT_FILE" 2>&1
    
    # Check if the experiment completed successfully
    if [ $? -eq 0 ]; then
        echo "Configuration $((i+1)) completed successfully" >> "$OUTPUT_FILE"
        
        # Record end time and calculate duration
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        COMPUTATION_TIMES+=("$DURATION")
        
        # Extract metrics from the latest results file
        LATEST_METRICS=$(ls -t /workspace/starter_code_1e5735e0-8cfb-42bf-845c-0506f2ea93da/results/metrics_*.json | head -n 1)
        
        if [ -f "$LATEST_METRICS" ]; then
            echo "Extracting metrics from $LATEST_METRICS" >> "$OUTPUT_FILE"
            
            # Extract rank correlation
            RANK_CORR=$(grep -o '"overall": [0-9.-]*' "$LATEST_METRICS" | cut -d' ' -f2)
            RANK_CORRELATIONS+=("$RANK_CORR")
            
            # Extract Sharpe ratio
            SHARPE=$(grep -o '"sharpe": [0-9.-]*' "$LATEST_METRICS" | cut -d' ' -f2)
            SHARPE_RATIOS+=("$SHARPE")
            
            # Extract MSE
            MSE=$(grep -o '"mse": [0-9.e-]*' "$LATEST_METRICS" | cut -d' ' -f2)
            MSE_VALUES+=("$MSE")
            
            echo "Rank Correlation: $RANK_CORR" >> "$OUTPUT_FILE"
            echo "Sharpe Ratio: $SHARPE" >> "$OUTPUT_FILE"
            echo "MSE: $MSE" >> "$OUTPUT_FILE"
            echo "Computation Time: $DURATION seconds" >> "$OUTPUT_FILE"
        else
            echo "WARNING: Could not find metrics file for configuration $((i+1))" >> "$OUTPUT_FILE"
            RANK_CORRELATIONS+=("N/A")
            SHARPE_RATIOS+=("N/A")
            MSE_VALUES+=("N/A")
        fi
    else
        echo "ERROR: Configuration $((i+1)) failed" >> "$OUTPUT_FILE"
        RANK_CORRELATIONS+=("N/A")
        SHARPE_RATIOS+=("N/A")
        MSE_VALUES+=("N/A")
        COMPUTATION_TIMES+=("N/A")
    fi
done

# Generate summary table
echo -e "\n\n==================================================" >> "$OUTPUT_FILE"
echo "EXPERIMENTAL GROUP PARTITION 1 - SUMMARY" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"
echo -e "Configuration\tRank Correlation\tSharpe Ratio\tMSE\tComputation Time (s)" >> "$OUTPUT_FILE"

for i in "${!CONFIG_DESCRIPTIONS[@]}"; do
    echo -e "$((i+1)). ${CONFIG_DESCRIPTIONS[$i]}\t${RANK_CORRELATIONS[$i]}\t${SHARPE_RATIOS[$i]}\t${MSE_VALUES[$i]}\t${COMPUTATION_TIMES[$i]}" >> "$OUTPUT_FILE"
done

# Find the best configuration based on rank correlation
BEST_IDX=0
BEST_CORR="${RANK_CORRELATIONS[0]}"

for i in "${!RANK_CORRELATIONS[@]}"; do
    if [[ "${RANK_CORRELATIONS[$i]}" != "N/A" && $(echo "${RANK_CORRELATIONS[$i]} > $BEST_CORR" | bc -l) -eq 1 ]]; then
        BEST_IDX=$i
        BEST_CORR="${RANK_CORRELATIONS[$i]}"
    fi
done

echo -e "\nBest Configuration: $((BEST_IDX+1)). ${CONFIG_DESCRIPTIONS[$BEST_IDX]}" >> "$OUTPUT_FILE"
echo "Best Rank Correlation: ${RANK_CORRELATIONS[$BEST_IDX]}" >> "$OUTPUT_FILE"
echo "Corresponding Sharpe Ratio: ${SHARPE_RATIOS[$BEST_IDX]}" >> "$OUTPUT_FILE"
echo "Corresponding MSE: ${MSE_VALUES[$BEST_IDX]}" >> "$OUTPUT_FILE"

# Completion timestamp
echo -e "\n==================================================" >> "$OUTPUT_FILE"
echo "EXPERIMENTAL GROUP PARTITION 1 - COMPLETED: $(date)" >> "$OUTPUT_FILE"
echo "==================================================" >> "$OUTPUT_FILE"

exit 0