#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the fix for the loss function issue
"""

import os
import sys
import json
import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Define test loss functions
LOSS_FUNCTIONS = [
    "regression_l2",  # MSE (control)
    "regression_l1",  # MAE
    "huber",
    "fair"
]

def test_loss_function(loss_function):
    """Test a specific loss function and return metrics"""
    print(f"\nTesting loss function: {loss_function}")
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model with the specified loss function
    model_params = {
        "objective": loss_function,  # This is the key parameter we're testing
        "metric": loss_function,     # Also set as metric for evaluation
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "verbose": -1
    }
    
    # Add special parameters for specific loss functions
    if loss_function == "quantile":
        model_params["alpha"] = 0.5  # Median quantile
    
    # Train model
    model = LGBMRegressor(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(10)
        ]
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"Results for {loss_function}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    return {
        "loss_function": loss_function,
        "mse": mse,
        "mae": mae
    }

def main():
    """Run tests for all loss functions"""
    print("Testing different LightGBM loss functions")
    print("=" * 50)
    
    results = []
    for loss_function in LOSS_FUNCTIONS:
        result = test_loss_function(loss_function)
        results.append(result)
    
    print("\nSummary of Results:")
    print("=" * 50)
    print(f"{'Loss Function':<15} | {'MSE':<10} | {'MAE':<10}")
    print("-" * 40)
    
    for result in results:
        print(f"{result['loss_function']:<15} | {result['mse']:<10.6f} | {result['mae']:<10.6f}")

if __name__ == "__main__":
    main()