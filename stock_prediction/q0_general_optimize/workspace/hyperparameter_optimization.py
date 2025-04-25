#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hyperparameter Optimization for LightGBM Stock Return Prediction
"""

import os
import json
import optuna
import numpy as np
import pandas as pd
from model_training import main as run_model
from model_training import DEFAULT_CONFIG

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    # Define the hyperparameter search space
    params = {
        "lgbm_params": {
            "objective": "regression",
            "num_leaves": trial.suggest_int("num_leaves", 31, 511),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "early_stopping_rounds": 100,
            "log_evaluation_freq": 500,
            "n_estimators": 10000,
            "verbose": -1
        },
        "device_type": "gpu",
        "num_simulations": 1  # Use 1 simulation for faster optimization
    }
    
    # Create a config by merging with default config
    config = DEFAULT_CONFIG.copy()
    config["lgbm_params"].update(params["lgbm_params"])
    config["device_type"] = params["device_type"]
    config["num_simulations"] = params["num_simulations"]
    
    # Run the model with this config
    result = run_model(config)
    
    # Return the negative correlation (since Optuna minimizes)
    return -result["metrics"]["overall"]

def optimize_hyperparameters(n_trials=10):
    """Run hyperparameter optimization."""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = -study.best_value  # Convert back to positive correlation
    
    # Create the optimized config
    optimized_config = DEFAULT_CONFIG.copy()
    optimized_config["lgbm_params"]["num_leaves"] = best_params["num_leaves"]
    optimized_config["lgbm_params"]["learning_rate"] = best_params["learning_rate"]
    optimized_config["lgbm_params"]["min_child_samples"] = best_params["min_child_samples"]
    optimized_config["lgbm_params"]["subsample"] = best_params["subsample"]
    optimized_config["lgbm_params"]["colsample_bytree"] = best_params["colsample_bytree"]
    optimized_config["num_simulations"] = 3  # Reset to 3 simulations for final run
    
    # Save the optimized config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              "configs", "hyperparameter_config.json")
    with open(config_path, 'w') as f:
        json.dump(optimized_config, f, indent=4)
    
    print(f"Best hyperparameters found: {best_params}")
    print(f"Best correlation: {best_value:.6f}")
    print(f"Optimized config saved to: {config_path}")
    
    return config_path

if __name__ == "__main__":
    optimize_hyperparameters(n_trials=10)
