#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Window Size Optimization for LightGBM Stock Return Prediction
"""

import os
import json
import numpy as np
import pandas as pd
from model_training import DEFAULT_CONFIG, main as run_model

def optimize_window_size():
    """Test different window sizes and find the optimal one."""
    window_sizes = [1, 2, 3, 4, 5]  # Years
    results = {}
    
    for window_size in window_sizes:
        print(f"Testing window size: {window_size} years")
        
        # Create config with this window size
        config = DEFAULT_CONFIG.copy()
        config["num_years_train"] = window_size
        config["num_simulations"] = 1  # Use 1 simulation for faster testing
        
        # Run model with this config
        result = run_model(config)
        
        # Store result
        results[window_size] = result["metrics"]["overall"]
        print(f"Window size {window_size} years: correlation = {results[window_size]:.6f}")
    
    # Find best window size
    best_window_size = max(results, key=results.get)
    best_correlation = results[best_window_size]
    
    print(f"Best window size: {best_window_size} years with correlation {best_correlation:.6f}")
    
    # Create config with optimal window size
    optimized_config = DEFAULT_CONFIG.copy()
    optimized_config["num_years_train"] = best_window_size
    optimized_config["num_simulations"] = 3  # Reset to 3 simulations for final run
    
    # Save the config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "configs", "window_optimization_config.json")
    with open(config_path, 'w') as f:
        json.dump(optimized_config, f, indent=4)
    
    print(f"Window optimization config saved to: {config_path}")
    return config_path, best_window_size

if __name__ == "__main__":
    optimize_window_size()
