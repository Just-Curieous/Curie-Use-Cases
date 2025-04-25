#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Selection for LightGBM Stock Return Prediction
"""

import os
import json
import numpy as np
import pandas as pd
from model_training import DEFAULT_CONFIG, main as run_model

def create_feature_selection_config():
    """Create configuration for feature selection."""
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Add feature selection parameters
    config["feature_selection"] = {
        "enabled": True,
        "importance_threshold": 0.01,  # Keep features with importance > 1% of total
        "top_n_features": 50  # Alternative: keep top N features
    }
    
    # Save the config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "configs", "feature_selection_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Feature selection config saved to: {config_path}")
    return config_path

if __name__ == "__main__":
    create_feature_selection_config()
