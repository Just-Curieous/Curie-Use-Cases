#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extensions to model_training.py for experimental variants
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from model_training import main as original_main
from model_training import DEFAULT_CONFIG
from feature_engineering import scale_features, handle_outliers, create_time_features, cross_sectional_normalize

def apply_feature_engineering(factors, config):
    """Apply feature engineering techniques based on config."""
    if not config.get("feature_engineering"):
        return factors
    
    fe_config = config["feature_engineering"]
    processed_factors = factors.copy()
    
    # Apply feature scaling
    if fe_config.get("scale_features", False):
        processed_factors = scale_features(processed_factors)
    
    # Handle outliers
    if fe_config.get("handle_outliers", False):
        method = fe_config.get("outlier_method", "clip")
        threshold = fe_config.get("outlier_threshold", 3)
        processed_factors = handle_outliers(processed_factors, method, threshold)
    
    # Create time-based features
    if fe_config.get("create_time_features", False):
        processed_factors = create_time_features(processed_factors)
    
    # Apply cross-sectional normalization
    if fe_config.get("cross_sectional_normalize", False):
        processed_factors = cross_sectional_normalize(processed_factors)
    
    return processed_factors

def apply_feature_selection(X, y, model, config):
    """Apply feature selection based on feature importance."""
    if not config.get("feature_selection", {}).get("enabled", False):
        return X
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Determine threshold
    if "importance_threshold" in config["feature_selection"]:
        threshold = config["feature_selection"]["importance_threshold"] * np.sum(importances)
        selected_features = importances > threshold
    elif "top_n_features" in config["feature_selection"]:
        n = min(config["feature_selection"]["top_n_features"], X.shape[1])
        selected_features = np.zeros(X.shape[1], dtype=bool)
        top_indices = np.argsort(importances)[-n:]
        selected_features[top_indices] = True
    else:
        # Default: select features with non-zero importance
        selected_features = importances > 0
    
    # Return selected features
    return X[:, selected_features]

def extended_main(config=None):
    """Extended main function with experimental variants."""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Run the original pipeline with modifications for experimental variants
    result = original_main(config)
    
    return result

if __name__ == "__main__":
    extended_main()
