#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified experiment script to test the loss function fix
"""

import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the patched model_training module
from model_training import main as train_model

# Define the loss functions to test
LOSS_FUNCTIONS = [
    "regression_l2",  # MSE (control)
    "regression_l1",  # MAE
]

def create_config_for_loss_function(loss_function):
    """Create a configuration file for the specified loss function."""
    # Base config path
    base_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_group_config.json")
    
    # Load the base configuration
    with open(base_config_file, 'r') as f:
        config = json.load(f)
    
    # Ensure the data path is correctly set
    config["data_path"] = "/workspace/starter_code_dataset"
    
    # Update the loss function
    config["lgbm_params"]["loss_function"] = loss_function
    
    # Create a dedicated results path for this loss function
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", loss_function)
    os.makedirs(results_path, exist_ok=True)
    
    # Update the results path in the config
    config["results_path"] = results_path
    
    # Create a config file
    config_filename = f"config_{loss_function}.json"
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_filename)
    
    # Save the configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created configuration file for {loss_function}: {config_path}")
    
    return config_path, results_path

def run_experiment_for_loss_function(loss_function):
    """Run the experiment for a specific loss function."""
    print(f"\n{'='*50}")
    print(f"Starting experiment for loss function: {loss_function}")
    print(f"{'='*50}")
    
    # Create configuration file
    config_path, results_path = create_config_for_loss_function(loss_function)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Prepare arguments for the main function
        sys.argv = [sys.argv[0], "--config", config_path]
        
        # Run the training
        result = train_model()
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Model training completed in {training_time:.2f} seconds")
        
        # Extract metrics
        metrics = result.get('metrics', {})
        
        print("\nResults Summary:")
        print("="*30)
        
        # Overall correlation
        overall_corr = metrics.get('overall', 'N/A')
        print(f"Overall Rank Correlation: {overall_corr}")
        
        # Yearly correlations
        for year in sorted([k for k in metrics.keys() if k != 'overall']):
            print(f"{year} Rank Correlation: {metrics[year]}")
        
        return {
            "loss_function": loss_function,
            "overall_corr": overall_corr,
            "yearly_corrs": {k: v for k, v in metrics.items() if k != 'overall'},
            "training_time": training_time
        }
            
    except Exception as e:
        print(f"Error during model training for {loss_function}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    """Run the simplified experiment."""
    print("Running simplified experiment to test loss function fix")
    print("="*70)
    
    # Create results directory
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
    
    # Run experiments for each loss function
    results = []
    for loss_function in LOSS_FUNCTIONS:
        result = run_experiment_for_loss_function(loss_function)
        if result:
            results.append(result)
    
    # Generate a comparison summary
    print("\n\n" + "="*70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*70)
    
    if not results:
        print("No valid results to compare.")
        return
    
    # Create a table header
    print(f"{'Loss Function':<15} | {'Overall Corr':<15} | {'2020 Corr':<10} | {'2021 Corr':<10} | {'2022 Corr':<10} | {'2023 Corr':<10}")
    print("-" * 80)
    
    for result in results:
        yearly = result.get('yearly_corrs', {})
        print(
            f"{result['loss_function']:<15} | "
            f"{result.get('overall_corr', 'N/A'):<15.6f} | "
            f"{yearly.get('2020', 'N/A'):<10.6f} | "
            f"{yearly.get('2021', 'N/A'):<10.6f} | "
            f"{yearly.get('2022', 'N/A'):<10.6f} | "
            f"{yearly.get('2023', 'N/A'):<10.6f}"
        )
    
    # Create a consolidated summary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"simplified_summary_{timestamp}.txt")
    
    with open(summary_path, 'w') as f:
        f.write("Simplified Experiment Summary\n")
        f.write("="*50 + "\n\n")
        f.write("Comparison of Loss Functions:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Loss Function':<15} | {'Overall Corr':<15} | {'2020 Corr':<10} | {'2021 Corr':<10} | {'2022 Corr':<10} | {'2023 Corr':<10}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            yearly = result.get('yearly_corrs', {})
            f.write(
                f"{result['loss_function']:<15} | "
                f"{result.get('overall_corr', 'N/A'):<15.6f} | "
                f"{yearly.get('2020', 'N/A'):<10.6f} | "
                f"{yearly.get('2021', 'N/A'):<10.6f} | "
                f"{yearly.get('2022', 'N/A'):<10.6f} | "
                f"{yearly.get('2023', 'N/A'):<10.6f}\n"
            )
    
    print(f"\nSummary saved to: {summary_path}")

if __name__ == "__main__":
    main()