#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss Function Experiment Runner

This script runs experiments with different LightGBM loss functions for stock return prediction:
- regression_l1
- huber
- fair
- poisson
- quantile

Each loss function is tested with appropriate parameters and results are consolidated.
"""

import os
import json
import subprocess
import glob
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Set up paths
BASE_DIR = "/workspace/starter_code_ac20158a-ee6d-48ad-a2a6-9f6cb203d889"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONFIG_TEMPLATE_PATH = os.path.join(BASE_DIR, "control_group_config.json")
PYTHON_PATH = os.path.join(BASE_DIR, "venv/bin/python")
MODEL_SCRIPT_PATH = os.path.join(BASE_DIR, "model_training.py")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define loss functions and their special parameters
LOSS_FUNCTIONS = [
    {"name": "regression_l1", "objective": "regression_l1", "params": {}},
    {"name": "huber", "objective": "huber", "params": {"huber_delta": 1.0}},
    {"name": "fair", "objective": "fair", "params": {"fair_c": 1.0}},
    {"name": "poisson", "objective": "poisson", "params": {}},
    {"name": "quantile", "objective": "quantile", "params": {"alpha": 0.5}}
]

def load_config_template():
    """Load the configuration template from the control group config."""
    with open(CONFIG_TEMPLATE_PATH, 'r') as f:
        return json.load(f)

def create_config_for_loss_function(loss_function):
    """Create a configuration file for a specific loss function."""
    config = load_config_template()
    
    # Update the objective and add any special parameters
    config["lgbm_params"]["objective"] = loss_function["objective"]
    for param_name, param_value in loss_function["params"].items():
        config["lgbm_params"][param_name] = param_value
    
    # Create a unique results path for this loss function
    loss_function_dir = os.path.join(RESULTS_DIR, loss_function["name"])
    os.makedirs(loss_function_dir, exist_ok=True)
    config["results_path"] = loss_function_dir
    
    # Save the configuration to a file
    config_path = os.path.join(BASE_DIR, f"{loss_function['name']}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config_path

def run_experiment(loss_function, config_path):
    """Run the model training with a specific loss function configuration."""
    print(f"\n{'='*50}")
    print(f"Running experiment with {loss_function['name']} loss function")
    print(f"Configuration: {config_path}")
    print(f"{'='*50}")
    
    # Run the model training script
    cmd = [PYTHON_PATH, MODEL_SCRIPT_PATH, "--config", config_path]
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output
    print(process.stdout)
    if process.stderr:
        print(f"Errors: {process.stderr}")
    
    return process.returncode == 0

def get_latest_metrics_file(loss_function_dir):
    """Get the most recent metrics file from a directory."""
    metrics_files = glob.glob(os.path.join(loss_function_dir, "metrics_*.json"))
    if not metrics_files:
        return None
    
    # Sort by modification time (newest first)
    return max(metrics_files, key=os.path.getmtime)

def extract_metrics(metrics_file):
    """Extract metrics from a metrics JSON file."""
    if not metrics_file or not os.path.exists(metrics_file):
        return None
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return data.get("metrics", {})

def consolidate_results():
    """Consolidate results from all experiments."""
    results = {}
    
    for loss_function in LOSS_FUNCTIONS:
        loss_name = loss_function["name"]
        loss_dir = os.path.join(RESULTS_DIR, loss_name)
        
        metrics_file = get_latest_metrics_file(loss_dir)
        metrics = extract_metrics(metrics_file)
        
        if metrics:
            results[loss_name] = metrics
    
    return results

def find_best_loss_function(results):
    """Find the best loss function based on overall rank correlation."""
    if not results:
        return None
    
    best_loss = None
    best_corr = -float('inf')
    
    for loss_name, metrics in results.items():
        overall_corr = metrics.get("overall", -float('inf'))
        if overall_corr > best_corr:
            best_corr = overall_corr
            best_loss = loss_name
    
    return best_loss, best_corr

def create_results_summary(results):
    """Create a summary of results for all loss functions."""
    if not results:
        return "No results available."
    
    # Create a DataFrame for easier comparison
    years = set()
    for metrics in results.values():
        years.update(key for key in metrics.keys() if key != "overall")
    
    years = sorted(years)
    columns = ["overall"] + list(years)
    
    # Create DataFrame with results
    df = pd.DataFrame(index=results.keys(), columns=columns)
    
    for loss_name, metrics in results.items():
        for col in columns:
            if col in metrics:
                df.loc[loss_name, col] = metrics[col]
    
    # Format the summary
    summary = f"\n{'='*80}\n"
    summary += "LOSS FUNCTION COMPARISON - RANK CORRELATION\n"
    summary += f"{'='*80}\n\n"
    summary += df.to_string(float_format="{:.4f}".format)
    summary += "\n\n"
    
    # Add best loss function
    best_loss, best_corr = find_best_loss_function(results)
    if best_loss:
        summary += f"Best performing loss function: {best_loss} (Overall Rank Correlation: {best_corr:.4f})\n"
    
    summary += f"{'='*80}\n"
    
    return summary

def main():
    """Main function to run all experiments."""
    start_time = time.time()
    
    print(f"Starting loss function experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing loss functions: {', '.join(lf['name'] for lf in LOSS_FUNCTIONS)}")
    
    # Run experiments for each loss function
    for loss_function in LOSS_FUNCTIONS:
        config_path = create_config_for_loss_function(loss_function)
        success = run_experiment(loss_function, config_path)
        
        if not success:
            print(f"Warning: Experiment for {loss_function['name']} may have failed.")
    
    # Consolidate and summarize results
    results = consolidate_results()
    summary = create_results_summary(results)
    
    # Save the summary to a file
    summary_path = os.path.join(BASE_DIR, "loss_function_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    # Print the summary
    print(summary)
    
    # Print execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()