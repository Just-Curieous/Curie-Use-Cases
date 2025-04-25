#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined Approach for LightGBM Stock Return Prediction
"""

import os
import json
import numpy as np
import pandas as pd
from model_training import DEFAULT_CONFIG

def create_combined_config(hyperparameter_config_path, window_size):
    """Create configuration that combines all optimizations."""
    # Load hyperparameter optimized config
    with open(hyperparameter_config_path, 'r') as f:
        hyperparameter_config = json.load(f)
    
    # Start with hyperparameter optimized config
    combined_config = hyperparameter_config.copy()
    
    # Add optimal window size
    combined_config["num_years_train"] = window_size
    
    # Add feature engineering parameters
    combined_config["feature_engineering"] = {
        "scale_features": True,
        "handle_outliers": True,
        "outlier_method": "clip",
        "outlier_threshold": 3,
        "create_time_features": True,
        "cross_sectional_normalize": True
    }
    
    # Add feature selection parameters
    combined_config["feature_selection"] = {
        "enabled": True,
        "importance_threshold": 0.01,
        "top_n_features": 50
    }
    
    # Save the config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "configs", "combined_approach_config.json")
    with open(config_path, 'w') as f:
        json.dump(combined_config, f, indent=4)
    
    print(f"Combined approach config saved to: {config_path}")
    return config_path

if __name__ == "__main__":
    create_combined_config("configs/hyperparameter_config.json", 3)  # Default values
