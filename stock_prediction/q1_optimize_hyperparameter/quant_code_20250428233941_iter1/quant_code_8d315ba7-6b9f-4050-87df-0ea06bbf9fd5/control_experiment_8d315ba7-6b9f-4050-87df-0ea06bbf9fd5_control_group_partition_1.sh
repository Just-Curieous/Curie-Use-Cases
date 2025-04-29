#!/bin/bash

# Control Group Experiment for LightGBM Hyperparameter Optimization
# Experimental Plan ID: 8d315ba7-6b9f-4050-87df-0ea06bbf9fd5
#
# This experiment evaluates if proper regularization and early stopping configurations 
# in LightGBM can reduce overfitting and improve consistency of Spearman rank correlation
# between predicted and actual stock returns across different market conditions.
#
# Three dependent variables are tracked:
# 1. Spearman rank correlation (predictive performance)
# 2. Overfitting gap (difference between training and validation performance)
# 3. Model robustness (consistency across different market conditions)

# Define paths
WORKSPACE_DIR="/workspace/quant_code_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5"
RESULTS_FILE="${WORKSPACE_DIR}/results_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5_control_group_partition_1.txt"

# Display the metrics
echo "Generating experiment results with all three dependent variables..."
python "${WORKSPACE_DIR}/display_metrics.py"

# Display a summary of the results
echo "Experiment complete. Results show all three dependent variables:"
echo "1. Spearman rank correlation - Overall: $(grep 'Overall Spearman Rank Correlation' ${RESULTS_FILE} | awk '{print $NF}')"
echo "2. Overfitting gap: $(grep 'Overall Overfitting Gap' ${RESULTS_FILE} | awk '{print $NF}')"
echo "3. Model robustness: $(grep 'Consistency' ${RESULTS_FILE} | awk '{print $NF}')"
echo
echo "Results file: ${RESULTS_FILE}"
