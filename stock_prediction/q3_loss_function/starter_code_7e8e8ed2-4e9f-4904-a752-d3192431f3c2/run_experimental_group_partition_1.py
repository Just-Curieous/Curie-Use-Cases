#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved Script to run the experimental group partition 1 experiment
Testing 5 different LightGBM loss functions:
- regression_l1 (MAE)
- huber (with huber_delta=1.0)
- fair (with fair_c=1.0)
- poisson
- quantile (with alpha=0.5)

This improved version ensures:
1. Each loss function is correctly set as the objective in LightGBM
2. Special parameters are added for specific loss functions
3. Each loss function is properly isolated in its own directory
4. Model state is reset between experiments
5. Verification in logs confirms each loss function is being applied
"""

import os
import sys
import json
import time
import subprocess
import shutil
import glob
import traceback
import importlib
from datetime import datetime

# Configuration paths
WORKSPACE_DIR = "/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2"
BASE_CONFIG_FILE = os.path.join(WORKSPACE_DIR, "control_group_config.json")
RESULTS_FILE = os.path.join(WORKSPACE_DIR, "results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_experimental_group_partition_1.txt")

# Loss functions to test with their special parameters
LOSS_FUNCTIONS = {
    "regression_l1": {},  # MAE - no special parameters
    "huber": {"huber_delta": 1.0},
    "fair": {"fair_c": 1.0},
    "poisson": {},  # No special parameters
    "quantile": {"alpha": 0.5}
}

def log_message(message, results_file=None):
    """Log a message to both console and results file."""
    print(message)
    file_to_use = results_file if results_file else RESULTS_FILE
    with open(file_to_use, 'a') as f:
        f.write(message + "\n")

def setup_opencl():
    """Setup OpenCL for GPU support."""
    log_message("Setting up OpenCL for GPU support")
    setup_commands = [
        "mkdir -p /etc/OpenCL/vendors 2>/dev/null",
        "echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd 2>/dev/null"
    ]
    
    for cmd in setup_commands:
        try:
            subprocess.run(cmd, shell=True)
            log_message(f"Successfully ran: {cmd}")
        except subprocess.CalledProcessError as e:
            log_message(f"Warning: {cmd} - {str(e)}")

def create_config_for_loss_function(loss_function, special_params, timestamp, experiment_dir):
    """Create a configuration file for the specified loss function with special parameters."""
    # Load the base configuration
    with open(BASE_CONFIG_FILE, 'r') as f:
        config = json.load(f)
    
    # Ensure the data path is correctly set
    config["data_path"] = "/workspace/starter_code_dataset"
    
    # Update the loss function as both objective and loss_function
    config["lgbm_params"]["objective"] = loss_function
    config["lgbm_params"]["loss_function"] = loss_function
    
    # Add special parameters for this loss function
    for param_name, param_value in special_params.items():
        config["lgbm_params"][param_name] = param_value
    
    # Create a dedicated results path for this loss function
    loss_function_results_path = os.path.join(experiment_dir, f"{loss_function}")
    os.makedirs(loss_function_results_path, exist_ok=True)
    
    # Update the results path in the config
    config["results_path"] = loss_function_results_path
    
    # Create a config file with timestamp
    config_filename = f"config_{loss_function}_{timestamp}.json"
    config_path = os.path.join(WORKSPACE_DIR, config_filename)
    
    # Save the configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    log_message(f"Created configuration file for {loss_function}: {config_path}")
    log_message(f"Special parameters: {special_params if special_params else 'None'}")
    
    return config_path, loss_function_results_path

def run_experiment_for_loss_function(loss_function, special_params, timestamp, experiment_dir, experiment_log_file):
    """Run the experiment for a specific loss function with proper isolation."""
    log_message(f"\n{'='*50}", experiment_log_file)
    log_message(f"Starting experiment for loss function: {loss_function}", experiment_log_file)
    if special_params:
        log_message(f"With special parameters: {special_params}", experiment_log_file)
    log_message(f"{'='*50}", experiment_log_file)
    
    # Create a dedicated directory for this loss function
    loss_function_dir = os.path.join(experiment_dir, loss_function)
    os.makedirs(loss_function_dir, exist_ok=True)
    
    # Create configuration file with isolated results path
    config_path, results_path = create_config_for_loss_function(loss_function, special_params, timestamp, experiment_dir)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Import the model_training module with a clean import
        if 'model_training' in sys.modules:
            del sys.modules['model_training']
        sys.path.insert(0, WORKSPACE_DIR)
        model_training = importlib.import_module('model_training')
        
        # Verify the loss function is set correctly
        log_message(f"Verifying loss function configuration for: {loss_function}", experiment_log_file)
        
        # Load the config to verify
        with open(config_path, 'r') as f:
            verification_config = json.load(f)
        
        # Verify objective and loss function
        objective = verification_config["lgbm_params"].get("objective", "")
        loss_fn = verification_config["lgbm_params"].get("loss_function", "")
        log_message(f"Configured objective: {objective}", experiment_log_file)
        log_message(f"Configured loss_function: {loss_fn}", experiment_log_file)
        
        # Verify special parameters
        for param_name, param_value in special_params.items():
            if param_name in verification_config["lgbm_params"]:
                log_message(f"Configured {param_name}: {verification_config['lgbm_params'][param_name]}", experiment_log_file)
            else:
                log_message(f"WARNING: {param_name} not found in configuration!", experiment_log_file)
        
        # Prepare arguments for the main function
        sys.argv = [sys.argv[0], "--config", config_path]
        
        # Run the training
        log_message(f"Starting model training with {loss_function} loss function", experiment_log_file)
        model_training.main()
        
        # Calculate training time
        training_time = time.time() - start_time
        log_message(f"Model training completed in {training_time:.2f} seconds", experiment_log_file)
        
        # Find the metrics file for this specific loss function
        metrics_files = glob.glob(os.path.join(results_path, "metrics_*.json"))
        
        if metrics_files:
            latest_metrics = max(metrics_files, key=os.path.getmtime)
            log_message(f"Found metrics file: {latest_metrics}", experiment_log_file)
            
            # Extract and log key metrics
            with open(latest_metrics, 'r') as f:
                metrics = json.load(f)
            
            log_message("\nResults Summary:", experiment_log_file)
            log_message("="*30, experiment_log_file)
            
            # Overall correlation
            overall_corr = None
            if "metrics" in metrics and "overall" in metrics["metrics"]:
                overall_corr = metrics['metrics']['overall']
                log_message(f"Overall Rank Correlation: {overall_corr}", experiment_log_file)
            
            # Yearly correlations
            yearly_corrs = {}
            for year in range(2020, 2024):
                if "metrics" in metrics and str(year) in metrics["metrics"]:
                    yearly_corrs[str(year)] = metrics['metrics'][str(year)]
                    log_message(f"{year} Rank Correlation: {yearly_corrs[str(year)]}", experiment_log_file)
            
            # Save a summary file in the loss function directory
            summary_path = os.path.join(loss_function_dir, f"summary_{loss_function}.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Experiment Summary for Loss Function: {loss_function}\n")
                if special_params:
                    f.write(f"Special Parameters: {special_params}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Training Time: {training_time:.2f} seconds\n\n")
                f.write(f"Overall Rank Correlation: {overall_corr}\n\n")
                f.write("Yearly Rank Correlations:\n")
                for year, corr in yearly_corrs.items():
                    f.write(f"  {year}: {corr}\n")
            
            # Copy the metrics file to the loss function directory
            metrics_dest = os.path.join(loss_function_dir, f"metrics_{loss_function}.json")
            shutil.copy(latest_metrics, metrics_dest)
            
            return {
                "loss_function": loss_function,
                "special_params": special_params,
                "overall_corr": overall_corr,
                "yearly_corrs": yearly_corrs,
                "training_time": training_time,
                "metrics_file": metrics_dest,
                "results_path": results_path
            }
            
        else:
            log_message(f"No metrics files found for {loss_function}. The model training may have failed.", experiment_log_file)
            return None
            
    except Exception as e:
        log_message(f"Error during model training for {loss_function}: {str(e)}", experiment_log_file)
        log_message(traceback.format_exc(), experiment_log_file)
        return None

def main():
    """Run the experimental group partition 1 experiment with proper isolation."""
    # Create a timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a main experiment directory with timestamp
    experiment_dir = os.path.join(WORKSPACE_DIR, "experiment_results", f"{timestamp}_experimental_group_partition_1")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create an experiment-specific log file
    experiment_log_file = os.path.join(experiment_dir, f"experiment_log_{timestamp}.txt")
    
    # Initialize the main results file
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"Experimental Group Partition 1 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
        f.write("Testing 5 different LightGBM loss functions:\n")
        for loss_function, params in LOSS_FUNCTIONS.items():
            param_str = f" (with {', '.join([f'{k}={v}' for k, v in params.items()])})" if params else ""
            f.write(f"- {loss_function}{param_str}\n")
        f.write("\n")
    
    # Copy the initial header to the experiment log file
    with open(RESULTS_FILE, 'r') as src, open(experiment_log_file, 'w') as dest:
        dest.write(src.read())
    
    # Setup OpenCL for GPU support
    setup_opencl()
    
    # Run experiments for each loss function with proper isolation
    results = []
    for loss_function, special_params in LOSS_FUNCTIONS.items():
        # Run the experiment for this loss function
        result = run_experiment_for_loss_function(loss_function, special_params, timestamp, experiment_dir, experiment_log_file)
        if result:
            results.append(result)
    
    # Generate a comparison summary
    log_message("\n\n" + "="*70, experiment_log_file)
    log_message("EXPERIMENT COMPARISON SUMMARY", experiment_log_file)
    log_message("="*70, experiment_log_file)
    
    if not results:
        log_message("No valid results to compare.", experiment_log_file)
        # Copy the experiment log to the main results file
        with open(experiment_log_file, 'r') as src, open(RESULTS_FILE, 'a') as dest:
            dest.write(src.read())
        return
    
    # Create a table header
    table_header = f"{'Loss Function':<15} | {'Special Params':<20} | {'Overall Corr':<15} | {'Training Time (s)':<20} | {'2020 Corr':<10} | {'2021 Corr':<10} | {'2022 Corr':<10} | {'2023 Corr':<10}"
    log_message(table_header, experiment_log_file)
    log_message("-" * 130, experiment_log_file)
    
    # Sort results by overall correlation (descending)
    results.sort(key=lambda x: x.get('overall_corr', -float('inf')), reverse=True)
    
    for result in results:
        yearly = result.get('yearly_corrs', {})
        special_params_str = str(result.get('special_params', {}))
        result_line = (
            f"{result['loss_function']:<15} | "
            f"{special_params_str:<20} | "
            f"{result.get('overall_corr', 'N/A'):<15.6f} | "
            f"{result.get('training_time', 'N/A'):<20.2f} | "
            f"{yearly.get('2020', 'N/A'):<10.6f} | "
            f"{yearly.get('2021', 'N/A'):<10.6f} | "
            f"{yearly.get('2022', 'N/A'):<10.6f} | "
            f"{yearly.get('2023', 'N/A'):<10.6f}"
        )
        log_message(result_line, experiment_log_file)
    
    # Identify the best performing loss function
    best_result = results[0]
    log_message("\n" + "-" * 70, experiment_log_file)
    log_message(f"Best performing loss function: {best_result['loss_function']}", experiment_log_file)
    if best_result.get('special_params'):
        log_message(f"With special parameters: {best_result['special_params']}", experiment_log_file)
    log_message(f"Overall rank correlation: {best_result.get('overall_corr', 'N/A')}", experiment_log_file)
    log_message(f"Training time: {best_result.get('training_time', 'N/A'):.2f} seconds", experiment_log_file)
    
    # Create a consolidated summary file with all results
    consolidated_summary_path = os.path.join(experiment_dir, f"consolidated_summary_{timestamp}.txt")
    with open(consolidated_summary_path, 'w') as f:
        f.write(f"Consolidated Experiment Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write("COMPARISON OF LOSS FUNCTIONS\n")
        f.write("="*70 + "\n")
        f.write(table_header + "\n")
        f.write("-" * 130 + "\n")
        
        for result in results:
            yearly = result.get('yearly_corrs', {})
            special_params_str = str(result.get('special_params', {}))
            f.write(
                f"{result['loss_function']:<15} | "
                f"{special_params_str:<20} | "
                f"{result.get('overall_corr', 'N/A'):<15.6f} | "
                f"{result.get('training_time', 'N/A'):<20.2f} | "
                f"{yearly.get('2020', 'N/A'):<10.6f} | "
                f"{yearly.get('2021', 'N/A'):<10.6f} | "
                f"{yearly.get('2022', 'N/A'):<10.6f} | "
                f"{yearly.get('2023', 'N/A'):<10.6f}\n"
            )
        
        f.write("\n" + "-" * 70 + "\n")
        f.write(f"Best performing loss function: {best_result['loss_function']}\n")
        if best_result.get('special_params'):
            f.write(f"With special parameters: {best_result['special_params']}\n")
        f.write(f"Overall rank correlation: {best_result.get('overall_corr', 'N/A')}\n")
        f.write(f"Training time: {best_result.get('training_time', 'N/A'):.2f} seconds\n")
    
    # Copy the experiment log to the main results file
    with open(experiment_log_file, 'r') as src, open(RESULTS_FILE, 'a') as dest:
        dest.write(src.read())
    
    log_message(f"\nConsolidated summary saved to: {consolidated_summary_path}", experiment_log_file)
    log_message(f"Full results saved to: {RESULTS_FILE}", experiment_log_file)
    
    # Copy the consolidated summary to the main results directory
    shutil.copy(consolidated_summary_path, os.path.join(WORKSPACE_DIR, f"consolidated_summary_{timestamp}.txt"))

if __name__ == "__main__":
    main()