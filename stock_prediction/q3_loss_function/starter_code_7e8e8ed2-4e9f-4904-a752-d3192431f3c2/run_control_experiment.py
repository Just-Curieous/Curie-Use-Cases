#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple script to run the control experiment
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# Configuration paths
WORKSPACE_DIR = "/workspace/starter_code_7e8e8ed2-4e9f-4904-a752-d3192431f3c2"
CONFIG_FILE = os.path.join(WORKSPACE_DIR, "control_group_config.json")
RESULTS_FILE = os.path.join(WORKSPACE_DIR, "results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_control_group_partition_1.txt")

def main():
    """Run the control experiment."""
    # Create a log file
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"Control Experiment - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
    
    # Setup OpenCL for GPU support
    setup_commands = [
        "mkdir -p /etc/OpenCL/vendors",
        "echo 'libnvidia-opencl.so.1' > /etc/OpenCL/vendors/nvidia.icd"
    ]
    
    for cmd in setup_commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
            log_message(f"Successfully ran: {cmd}")
        except subprocess.CalledProcessError as e:
            log_message(f"Error executing: {cmd}\nError: {str(e)}")
    
    # Run the model training script
    log_message("Starting model training with control group configuration")
    
    try:
        # Import the model_training module
        sys.path.append(WORKSPACE_DIR)
        from model_training import main as train_model
        
        # Prepare arguments for the main function
        sys.argv = [sys.argv[0], "--config", CONFIG_FILE]
        
        # Run the training
        train_model()
        
        log_message("Model training completed successfully")
        
        # Find the latest metrics file
        import glob
        metrics_files = glob.glob(os.path.join(WORKSPACE_DIR, "results", "metrics_*.json"))
        if metrics_files:
            latest_metrics = max(metrics_files, key=os.path.getmtime)
            log_message(f"Found metrics file: {latest_metrics}")
            
            # Extract and log key metrics
            with open(latest_metrics, 'r') as f:
                metrics = json.load(f)
            
            log_message("\nResults Summary:")
            log_message("="*30)
            
            # Overall correlation
            if "metrics" in metrics and "overall" in metrics["metrics"]:
                log_message(f"Overall Rank Correlation: {metrics['metrics']['overall']}")
            
            # Yearly correlations
            for year in range(2020, 2024):
                if "metrics" in metrics and str(year) in metrics["metrics"]:
                    log_message(f"{year} Rank Correlation: {metrics['metrics'][str(year)]}")
            
        else:
            log_message("No metrics files found!")
            
    except Exception as e:
        log_message(f"Error during model training: {str(e)}")
        import traceback
        log_message(traceback.format_exc())
        sys.exit(1)

def log_message(message):
    """Log a message to both console and results file."""
    print(message)
    with open(RESULTS_FILE, 'a') as f:
        f.write(message + "\n")

if __name__ == "__main__":
    main()