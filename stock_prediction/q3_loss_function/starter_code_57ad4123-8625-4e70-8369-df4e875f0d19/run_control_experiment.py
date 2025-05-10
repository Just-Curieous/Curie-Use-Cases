#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Control Experiment Runner for Stock Return Prediction

This script runs the control experiment for stock return prediction using LightGBM
with MSE loss (regression_l2) as specified in the control group configuration.
"""

import os
import json
import logging
import time
from datetime import datetime
import model_training

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("control_experiment_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_timestamp_directory(base_path):
    """Create a timestamped directory for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_path, f"run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Created results directory: {result_dir}")
    return result_dir, timestamp

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def save_metrics_with_timestamp(metrics, config, base_path, timestamp):
    """Save metrics to a timestamped JSON file."""
    results = {
        "metrics": metrics,
        "config": config,
        "timestamp": timestamp
    }
    
    metrics_file = os.path.join(base_path, f"metrics_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_file}")
    return metrics_file

def print_metrics_summary(metrics):
    """Print a summary of the metrics."""
    logger.info(f"\n{'='*50}\nPERFORMANCE METRICS SUMMARY\n{'='*50}")
    logger.info(f"Overall Rank Correlation: {metrics['overall']:.4f}")
    
    for year in sorted(k for k in metrics.keys() if k != 'overall'):
        logger.info(f"{year} Rank Correlation: {metrics[year]:.4f}")
    
    logger.info(f"{'='*50}")

def main():
    """Main function to run the control experiment."""
    start_time = time.time()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_group_config.json")
    config = load_config(config_path)
    if config is None:
        return
    
    # Create timestamped directory for results
    result_dir, timestamp = create_timestamp_directory(config["results_path"])
    
    # Update config with timestamped results path
    config["results_path"] = result_dir
    
    # Run the experiment
    logger.info("Starting control experiment with regression_l2 loss...")
    result = model_training.main(config)
    
    if result is None:
        logger.error("Experiment failed.")
        return
    
    # Save metrics with timestamp
    metrics_file = save_metrics_with_timestamp(
        result['metrics'], 
        config, 
        config["results_path"], 
        timestamp
    )
    
    # Print summary
    print_metrics_summary(result['metrics'])
    
    # Log execution time
    total_time = time.time() - start_time
    logger.info(f"Total experiment time: {total_time:.2f} seconds")
    logger.info(f"Full report saved to: {metrics_file}")

if __name__ == "__main__":
    main()