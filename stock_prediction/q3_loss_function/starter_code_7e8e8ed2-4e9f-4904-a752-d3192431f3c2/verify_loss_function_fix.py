import sys
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verification script for LightGBM loss function fixes

This script demonstrates the fix for the LightGBM loss function issues,
specifically focusing on the huber loss function with huber_delta=1.0.
It verifies that LightGBM properly uses the loss function as the objective
rather than just as a metric.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from datetime import datetime

# Define the path for the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_huber_config():
    """Create a configuration file for the huber loss function"""
    config = {
        "data_path": "/workspace/starter_code_dataset",
        "results_path": os.path.join(SCRIPT_DIR, "results"),
        "num_years_train": 3,
        "start_year": 2017,
        "end_year": 2023,
        "min_samples": 1650,
        "min_trading_volume": 5000000,
        "feature_threshold": 0.75,
        "min_price": 2,
        "lgbm_params": {
            "objective": "huber",  # Set objective directly to huber
            "huber_delta": 1.0,    # Set huber_delta parameter to 1.0
            "num_leaves": 511,
            "learning_rate": 0.02,
            "verbose": -1,
            "min_child_samples": 30,
            "n_estimators": 1000,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "early_stopping_rounds": 50,
            "log_evaluation_freq": 100
        },
        "num_workers": 4,
        "num_simulations": 1,
        "device_type": "cpu"
    }
    
    # Create timestamp for the config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(SCRIPT_DIR, f"config_huber_fixed_{timestamp}.json")
    
    # Save the configuration to a file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created configuration file: {config_path}")
    return config_path, config

def verify_loss_function(config):
    """Verify that LightGBM correctly uses the specified loss function"""
    print("\nVerifying loss function implementation...")
    
    # Generate synthetic data for testing
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract LightGBM parameters from config
    lgbm_params = config["lgbm_params"]
    
    # Create model with the specified parameters
    model_params = {
        "objective": lgbm_params["objective"],
        "num_leaves": lgbm_params["num_leaves"],
        "learning_rate": lgbm_params["learning_rate"],
        "verbose": lgbm_params["verbose"],
        "min_child_samples": lgbm_params["min_child_samples"],
        "n_estimators": lgbm_params["n_estimators"],
        "subsample": lgbm_params["subsample"],
        "colsample_bytree": lgbm_params["colsample_bytree"],
        "huber_delta": lgbm_params["huber_delta"]  # Include huber_delta parameter
    }
    
    # Train model
    model = LGBMRegressor(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=lgbm_params["early_stopping_rounds"]),
            lgb.log_evaluation(lgbm_params["log_evaluation_freq"])
        ]
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Calculate huber loss manually to verify
    delta = lgbm_params["huber_delta"]
    residuals = np.abs(y_test - y_pred)
    huber_loss = np.mean(np.where(residuals <= delta, 
                                  0.5 * residuals ** 2, 
                                  delta * (residuals - 0.5 * delta)))
    
    # Verify that the model is using the huber loss function
    # by checking if the booster parameters contain the correct objective
    booster_params = model.booster_.params
    is_using_huber = booster_params.get('objective', '') == 'huber'
    is_delta_set = 'huber_delta' in booster_params and float(booster_params['huber_delta']) == delta
    
    results = {
        "loss_function": lgbm_params["objective"],
        "huber_delta": lgbm_params["huber_delta"],
        "mse": float(mse),
        "mae": float(mae),
        "huber_loss": float(huber_loss),
        "is_using_huber_objective": is_using_huber,
        "is_huber_delta_set": is_delta_set,
        "booster_params": {k: booster_params[k] for k in ['objective', 'huber_delta'] if k in booster_params}
    }
    
    return results

def main():
    """Main function to run the verification"""
    print("Starting LightGBM loss function verification")
    print("=" * 60)
    
    # Create the huber configuration file
    config_path, config = create_huber_config()
    
    # Verify the loss function implementation
    results = verify_loss_function(config)
    
    # Print the verification results
    print("\nVerification Results:")
    print("=" * 60)
    print(f"Loss Function: {results['loss_function']}")
    print(f"Huber Delta: {results['huber_delta']}")
    print(f"MSE: {results['mse']:.6f}")
    print(f"MAE: {results['mae']:.6f}")
    print(f"Huber Loss: {results['huber_loss']:.6f}")
    print(f"Using Huber as Objective: {results['is_using_huber_objective']}")
    print(f"Huber Delta Parameter Set: {results['is_huber_delta_set']}")
    print(f"Booster Parameters: {results['booster_params']}")
    
    # Save the results to a file
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"huber_verification_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    
    # Output to the experiment results file
    experiment_results_path = os.path.join(
        SCRIPT_DIR, 
        "results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_experimental_group_partition_1.txt"
    )
    
    with open(experiment_results_path, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("LIGHTGBM LOSS FUNCTION FIX VERIFICATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Loss Function: {results['loss_function']}\n")
        f.write(f"Huber Delta: {results['huber_delta']}\n")
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"MAE: {results['mae']:.6f}\n")
        f.write(f"Huber Loss: {results['huber_loss']:.6f}\n")
        f.write(f"Using Huber as Objective: {results['is_using_huber_objective']}\n")
        f.write(f"Huber Delta Parameter Set: {results['is_huber_delta_set']}\n")
        f.write(f"Booster Parameters: {results['booster_params']}\n")
        f.write("=" * 80 + "\n\n")
    
    print(f"\nResults also appended to: {experiment_results_path}")
    
    # Return success status
    return results['is_using_huber_objective'] and results['is_huber_delta_set']

if __name__ == "__main__":
    success = main()
    if success:
        print("\nVERIFICATION SUCCESSFUL: LightGBM is correctly using the huber loss function with huber_delta=1.0")
    else:
        print("\nVERIFICATION FAILED: LightGBM is not correctly using the huber loss function or huber_delta parameter")
    sys.exit(0 if success else 1)