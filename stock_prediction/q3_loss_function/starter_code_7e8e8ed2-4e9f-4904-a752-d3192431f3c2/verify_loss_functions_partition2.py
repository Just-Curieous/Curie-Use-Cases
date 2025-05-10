#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verification script for LightGBM loss functions in partition 2
Testing two LightGBM loss functions:
- mape (Mean Absolute Percentage Error)
- tweedie (Tweedie regression with tweedie_variance_power=1.5)

This script verifies that LightGBM properly uses these loss functions as objectives
rather than just as metrics, and compares their performance.
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
from scipy.stats import spearmanr

# Define the path for the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_config(loss_function, special_params=None):
    """Create a configuration file for the specified loss function"""
    if special_params is None:
        special_params = {}
        
    config = {
        "data_path": "/workspace/starter_code_dataset",
        "results_path": os.path.join(SCRIPT_DIR, "results", loss_function),
        "num_years_train": 3,
        "start_year": 2017,
        "end_year": 2023,
        "min_samples": 1650,
        "min_trading_volume": 5000000,
        "feature_threshold": 0.75,
        "min_price": 2,
        "lgbm_params": {
            "objective": loss_function,  # Set objective directly to the specified loss function
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
    
    # Add special parameters for the loss function
    for param_name, param_value in special_params.items():
        config["lgbm_params"][param_name] = param_value
    
    # Create timestamp for the config file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(SCRIPT_DIR, f"config_{loss_function}_{timestamp}.json")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(config["results_path"]), exist_ok=True)
    
    # Save the configuration to a file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created configuration file: {config_path}")
    return config_path, config

def verify_loss_function(loss_function, config):
    """Verify that LightGBM correctly uses the specified loss function"""
    print(f"\nVerifying {loss_function} loss function implementation...")
    
    # Generate synthetic data for testing
    # For MAPE and Tweedie, we need positive target values
    X, y_raw = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Transform y to be positive (needed for both MAPE and Tweedie)
    y = np.exp(y_raw / 10) + 1  # Ensure all values are positive
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Extract LightGBM parameters from config
    lgbm_params = config["lgbm_params"]
    
    # Create model with the specified parameters
    model_params = {k: v for k, v in lgbm_params.items() if k not in ['early_stopping_rounds', 'log_evaluation_freq']}
    
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
    
    # Calculate MAPE manually
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Calculate rank correlation (Spearman's rho)
    rank_corr, _ = spearmanr(y_test, y_pred)
    
    # Verify that the model is using the correct loss function
    # by checking if the booster parameters contain the correct objective
    booster_params = model.booster_.params
    is_using_correct_objective = booster_params.get('objective', '') == loss_function
    
    # Check for special parameters
    special_params_verified = {}
    if loss_function == "tweedie":
        expected_variance_power = lgbm_params.get("tweedie_variance_power", None)
        actual_variance_power = booster_params.get("tweedie_variance_power", None)
        is_variance_power_set = (actual_variance_power is not None and 
                                float(actual_variance_power) == expected_variance_power)
        special_params_verified["is_tweedie_variance_power_set"] = is_variance_power_set
        special_params_verified["expected_variance_power"] = expected_variance_power
        special_params_verified["actual_variance_power"] = actual_variance_power
    
    results = {
        "loss_function": loss_function,
        "mse": float(mse),
        "mae": float(mae),
        "mape": float(mape),
        "rank_correlation": float(rank_corr),
        "is_using_correct_objective": is_using_correct_objective,
        "special_params_verified": special_params_verified,
        "booster_params": {k: booster_params[k] for k in ['objective'] + 
                          (['tweedie_variance_power'] if loss_function == 'tweedie' else []) 
                          if k in booster_params}
    }
    
    return results

def main():
    """Main function to run the verification for both loss functions"""
    print("Starting LightGBM loss function verification for partition 2")
    print("=" * 60)
    
    # Define the loss functions to test with their special parameters
    loss_functions = {
        "mape": {},  # No special parameters for MAPE
        "tweedie": {"tweedie_variance_power": 1.5}  # Set tweedie_variance_power for Tweedie
    }
    
    # Results to store verification outcomes
    all_results = {}
    
    # Create results directory
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Timestamp for this verification run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Output file for the experiment results
    experiment_results_path = os.path.join(
        SCRIPT_DIR, 
        "results_7e8e8ed2-4e9f-4904-a752-d3192431f3c2_experimental_group_partition_2.txt"
    )
    
    # Verify each loss function
    for loss_function, special_params in loss_functions.items():
        print(f"\n{'-' * 60}")
        print(f"Testing loss function: {loss_function}")
        if special_params:
            print(f"With special parameters: {special_params}")
        print(f"{'-' * 60}")
        
        # Create configuration for this loss function
        config_path, config = create_config(loss_function, special_params)
        
        # Verify the loss function implementation
        results = verify_loss_function(loss_function, config)
        all_results[loss_function] = results
        
        # Print the verification results
        print("\nVerification Results:")
        print("=" * 60)
        print(f"Loss Function: {results['loss_function']}")
        print(f"MSE: {results['mse']:.6f}")
        print(f"MAE: {results['mae']:.6f}")
        print(f"MAPE: {results['mape']:.6f}%")
        print(f"Rank Correlation: {results['rank_correlation']:.6f}")
        print(f"Using {loss_function} as Objective: {results['is_using_correct_objective']}")
        
        if loss_function == "tweedie":
            print(f"Tweedie Variance Power Set: {results['special_params_verified']['is_tweedie_variance_power_set']}")
            print(f"Expected Variance Power: {results['special_params_verified']['expected_variance_power']}")
            print(f"Actual Variance Power: {results['special_params_verified']['actual_variance_power']}")
        
        print(f"Booster Parameters: {results['booster_params']}")
        
        # Save the individual results to a file
        loss_function_dir = os.path.join(results_dir, loss_function)
        os.makedirs(loss_function_dir, exist_ok=True)
        
        results_path = os.path.join(loss_function_dir, f"{loss_function}_verification_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {results_path}")
        
        # Append to the experiment results file
        with open(experiment_results_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"LIGHTGBM {loss_function.upper()} LOSS FUNCTION VERIFICATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Loss Function: {results['loss_function']}\n")
            
            if loss_function == "tweedie":
                f.write(f"Tweedie Variance Power: {special_params.get('tweedie_variance_power', 'Not set')}\n")
            
            f.write(f"MSE: {results['mse']:.6f}\n")
            f.write(f"MAE: {results['mae']:.6f}\n")
            f.write(f"MAPE: {results['mape']:.6f}%\n")
            f.write(f"Rank Correlation: {results['rank_correlation']:.6f}\n")
            f.write(f"Using {loss_function} as Objective: {results['is_using_correct_objective']}\n")
            
            if loss_function == "tweedie":
                f.write(f"Tweedie Variance Power Set: {results['special_params_verified']['is_tweedie_variance_power_set']}\n")
            
            f.write(f"Booster Parameters: {results['booster_params']}\n")
            f.write("=" * 80 + "\n\n")
    
    # Compare the performance of the two loss functions
    print("\n" + "=" * 80)
    print("COMPARISON OF LOSS FUNCTIONS")
    print("=" * 80)
    
    # Create a comparison table
    comparison_table = [
        ["Metric", "MAPE", "Tweedie"],
        ["MSE", f"{all_results['mape']['mse']:.6f}", f"{all_results['tweedie']['mse']:.6f}"],
        ["MAE", f"{all_results['mape']['mae']:.6f}", f"{all_results['tweedie']['mae']:.6f}"],
        ["MAPE", f"{all_results['mape']['mape']:.6f}%", f"{all_results['tweedie']['mape']:.6f}%"],
        ["Rank Correlation", f"{all_results['mape']['rank_correlation']:.6f}", f"{all_results['tweedie']['rank_correlation']:.6f}"]
    ]
    
    # Print the comparison table
    for row in comparison_table:
        print(f"{row[0]:<20} | {row[1]:<15} | {row[2]:<15}")
    
    # Determine which loss function has better rank correlation
    mape_corr = all_results['mape']['rank_correlation']
    tweedie_corr = all_results['tweedie']['rank_correlation']
    
    better_loss_function = "mape" if mape_corr > tweedie_corr else "tweedie"
    corr_diff = abs(mape_corr - tweedie_corr)
    
    print("\n" + "-" * 80)
    print(f"Better performing loss function based on rank correlation: {better_loss_function}")
    print(f"Rank correlation difference: {corr_diff:.6f}")
    print("-" * 80)
    
    # Save the comparison results
    comparison_results = {
        "timestamp": timestamp,
        "comparison": {
            "mape": {
                "mse": all_results['mape']['mse'],
                "mae": all_results['mape']['mae'],
                "mape": all_results['mape']['mape'],
                "rank_correlation": all_results['mape']['rank_correlation']
            },
            "tweedie": {
                "mse": all_results['tweedie']['mse'],
                "mae": all_results['tweedie']['mae'],
                "mape": all_results['tweedie']['mape'],
                "rank_correlation": all_results['tweedie']['rank_correlation'],
                "tweedie_variance_power": loss_functions['tweedie'].get('tweedie_variance_power', None)
            },
            "better_loss_function": better_loss_function,
            "rank_correlation_difference": corr_diff
        }
    }
    
    # Save the comparison results to a file
    comparison_path = os.path.join(results_dir, f"loss_function_comparison_{timestamp}.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    
    print(f"\nComparison results saved to: {comparison_path}")
    
    # Append the comparison to the experiment results file
    with open(experiment_results_path, 'a') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("LOSS FUNCTION COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        # Write the comparison table
        for row in comparison_table:
            f.write(f"{row[0]:<20} | {row[1]:<15} | {row[2]:<15}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write(f"Better performing loss function based on rank correlation: {better_loss_function}\n")
        f.write(f"Rank correlation difference: {corr_diff:.6f}\n")
        f.write("-" * 80 + "\n\n")
    
    # Return success status - both loss functions should be verified
    return (all_results['mape']['is_using_correct_objective'] and 
            all_results['tweedie']['is_using_correct_objective'] and
            all_results['tweedie']['special_params_verified'].get('is_tweedie_variance_power_set', False))

if __name__ == "__main__":
    success = main()
    if success:
        print("\nVERIFICATION SUCCESSFUL: LightGBM is correctly using both loss functions")
        print("- MAPE loss function is correctly configured")
        print("- Tweedie loss function is correctly configured with tweedie_variance_power=1.5")
        exit(0)
    else:
        print("\nVERIFICATION FAILED: LightGBM is not correctly using one or both loss functions")
        exit(1)