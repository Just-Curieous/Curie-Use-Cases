#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Feature Engineering for LightGBM Stock Return Prediction
"""

import os
import json
import numpy as np
import pandas as pd
from model_training import DEFAULT_CONFIG, main as run_model

def scale_features(factors):
    """Standardize features using z-score normalization."""
    scaled_factors = []
    
    for factor in factors:
        # Compute mean and std for each factor
        mean = factor.mean(axis=1)
        std = factor.std(axis=1)
        
        # Standardize
        scaled_factor = factor.sub(mean, axis=0).div(std, axis=0)
        scaled_factors.append(scaled_factor)
    
    return scaled_factors

def handle_outliers(factors, method='clip', threshold=3):
    """Handle outliers in the data."""
    processed_factors = []
    
    for factor in factors:
        if method == 'clip':
            # Clip values beyond threshold standard deviations
            mean = factor.mean()
            std = factor.std()
            factor_clipped = factor.clip(mean - threshold * std, mean + threshold * std)
            processed_factors.append(factor_clipped)
        elif method == 'winsorize':
            # Winsorize at specified percentiles
            lower = factor.quantile(0.01)
            upper = factor.quantile(0.99)
            factor_winsorized = factor.clip(lower, upper)
            processed_factors.append(factor_winsorized)
    
    return processed_factors

def create_time_features(factors):
    """Create time-based features like moving averages."""
    enhanced_factors = []
    
    for factor in factors:
        # Moving averages
        ma_5 = factor.rolling(window=5).mean()
        ma_20 = factor.rolling(window=20).mean()
        
        # Rate of change
        roc = factor.pct_change()
        
        # Combine features
        enhanced_factors.extend([factor, ma_5, ma_20, roc])
    
    return enhanced_factors

def cross_sectional_normalize(factors):
    """Normalize factors cross-sectionally (across stocks) for each day."""
    normalized_factors = []
    
    for factor in factors:
        # Rank normalization
        normalized = factor.rank(axis=1, pct=True)
        normalized_factors.append(normalized)
    
    return normalized_factors

def create_enhanced_features_config():
    """Create configuration for enhanced feature engineering."""
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Add feature engineering parameters
    config["feature_engineering"] = {
        "scale_features": True,
        "handle_outliers": True,
        "outlier_method": "clip",
        "outlier_threshold": 3,
        "create_time_features": True,
        "cross_sectional_normalize": True
    }
    
    # Save the config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "configs", "feature_engineering_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Enhanced feature engineering config saved to: {config_path}")
    return config_path

if __name__ == "__main__":
    create_enhanced_features_config()
