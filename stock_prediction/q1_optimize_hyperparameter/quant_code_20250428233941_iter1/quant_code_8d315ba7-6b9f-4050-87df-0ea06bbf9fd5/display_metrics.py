#!/usr/bin/env python3

"""
Script to extract and display metrics from the experiment
Includes all three dependent variables required by the experimental plan:
1. Spearman rank correlation
2. Overfitting gap
3. Model robustness
"""

import json
import sys
import os

# Define paths
METRICS_FILE = "/workspace/quant_code_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5/metrics_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5_control_group_partition_1.json"
RESULTS_FILE = "/workspace/quant_code_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5/results_8d315ba7-6b9f-4050-87df-0ea06bbf9fd5_control_group_partition_1.txt"

def main():
    try:
        # Check if metrics file exists
        if not os.path.exists(METRICS_FILE):
            print(f"Error: Metrics file {METRICS_FILE} not found!")
            return 1
            
        # Load metrics
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)

        # Create new results file
        with open(RESULTS_FILE, 'w') as out:
            # Write header
            out.write("# Control Group Experiment Results\n")
            out.write("# Experimental Plan ID: 8d315ba7-6b9f-4050-87df-0ea06bbf9fd5\n")
            out.write("#\n")
            out.write("# This experiment evaluates if proper regularization and early stopping configurations\n") 
            out.write("# in LightGBM can reduce overfitting and improve consistency of Spearman rank correlation\n")
            out.write("# between predicted and actual stock returns across different market conditions.\n")
            out.write("#\n")
            out.write("# Three dependent variables are tracked:\n")
            out.write("# 1. Spearman rank correlation (predictive performance)\n")
            out.write("# 2. Overfitting gap (difference between training and validation performance)\n")
            out.write("# 3. Model robustness (consistency across different market conditions)\n")
            out.write("#\n")
            out.write("# Parameters:\n")
            out.write("# - early_stopping_rounds: 50\n")
            out.write("# - min_child_samples: 20\n")
            out.write("# - reg_alpha: 0.0\n")
            out.write("# - reg_lambda: 0.0\n\n")

            metrics = data.get('metrics', {})
            
            # Performance metrics section
            out.write("Performance Metrics:\n")
            out.write("==================================================\n")
            
            # Overall correlation
            out.write(f"Overall Spearman Rank Correlation: {metrics.get('overall')}\n\n")
            
            # Overfitting analysis
            out.write("Overfitting Analysis:\n")
            out.write(f"Overall Overfitting Gap: {metrics.get('overfitting_gap')}\n")
            out.write(f"Training Correlation: {metrics.get('train_correlation')}\n")
            out.write(f"Validation Correlation: {metrics.get('val_correlation')}\n\n")
            
            # Yearly overfitting gaps
            out.write("Yearly Overfitting Gaps:\n")
            for year in sorted([k for k in metrics.keys() if k.startswith('overfitting_gap_')]):
                year_num = year.split('_')[-1]
                out.write(f"  {year_num}: {metrics.get(year)}\n")
            
            # Model robustness
            out.write("\nModel Robustness:\n")
            out.write(f"Consistency (std of yearly correlations): {metrics.get('model_robustness')}\n")
            out.write("(Lower values indicate more consistent performance across market conditions)\n\n")
            
            # Yearly correlations
            out.write("Yearly Correlations:\n")
            for year in sorted([k for k in metrics.keys() if k.isdigit()]):
                out.write(f"  {year}: {metrics.get(year)}\n")

        print(f"Metrics successfully extracted and written to {RESULTS_FILE}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
