#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mock Experiment for Stock Return Prediction

This script creates a simplified mock version of the experiment for testing purposes.
It uses dummy data to simulate the stock return prediction task and trains a simple
LightGBM model with regression_l2 loss function (MSE).
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
WORKSPACE_DIR = "/workspace/starter_code_57ad4123-8625-4e70-8369-df4e875f0d19"
OUTPUT_FILE = os.path.join(WORKSPACE_DIR, "results_57ad4123-8625-4e70-8369-df4e875f0d19_control_group_partition_1.txt")

def generate_synthetic_data(n_samples=1000, n_features=20, n_stocks=50, n_days=100):
    """
    Generate synthetic financial data for testing.
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features per stock
        n_stocks: Number of stocks
        n_days: Number of trading days
        
    Returns:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        dates: List of dates
        stock_ids: List of stock IDs
    """
    logger.info(f"Generating synthetic data with {n_samples} samples, {n_features} features")
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target variable with some noise
    true_weights = np.random.randn(n_features) / np.sqrt(n_features)
    y = np.dot(X, true_weights) + 0.1 * np.random.randn(n_samples)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generate dates and stock IDs for panel data structure
    base_date = datetime(2022, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    stock_ids = [f"STOCK_{i:04d}" for i in range(n_stocks)]
    
    logger.info(f"Generated data with shape: X_train={X_train.shape}, y_train={y_train.shape}")
    
    return X_train, y_train, X_test, y_test, dates, stock_ids

def train_lightgbm_model(X_train, y_train, X_test, y_test):
    """
    Train a LightGBM model with regression_l2 loss function (MSE).
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        model: Trained LightGBM model
        metrics: Dictionary of performance metrics
    """
    logger.info("Training LightGBM model with regression_l2 loss function")
    
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Define parameters
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Store metrics
    metrics = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'best_iteration': model.best_iteration
    }
    
    logger.info(f"Model training completed. Test MSE: {test_mse:.6f}, Test R²: {test_r2:.6f}")
    
    return model, metrics

def calculate_rank_correlation(predictions, returns):
    """
    Calculate rank correlation between predictions and actual returns.
    
    Args:
        predictions: Predicted returns
        returns: Actual returns
        
    Returns:
        corr: Rank correlation
    """
    # Convert to ranks
    pred_ranks = pd.Series(predictions).rank()
    return_ranks = pd.Series(returns).rank()
    
    # Calculate correlation
    corr = pred_ranks.corr(return_ranks, method='spearman')
    
    return corr

def simulate_portfolio_performance(predictions, returns, n_quantiles=5):
    """
    Simulate portfolio performance based on predictions.
    
    Args:
        predictions: Predicted returns
        returns: Actual returns
        n_quantiles: Number of quantiles for portfolio construction
        
    Returns:
        portfolio_returns: Dictionary of portfolio returns
    """
    # Convert to pandas Series
    preds = pd.Series(predictions)
    rets = pd.Series(returns)
    
    # Create quantiles
    quantiles = pd.qcut(preds, n_quantiles, labels=False)
    
    # Calculate returns by quantile
    portfolio_returns = {}
    for i in range(n_quantiles):
        portfolio_returns[f'quantile_{i+1}'] = float(rets[quantiles == i].mean())
    
    # Calculate long-short portfolio return
    portfolio_returns['long_short'] = float(rets[quantiles == n_quantiles-1].mean() - rets[quantiles == 0].mean())
    
    return portfolio_returns

def run_mock_experiment():
    """
    Run the complete mock experiment workflow.
    """
    logger.info("Starting mock experiment for stock return prediction")
    
    # Generate synthetic data
    X_train, y_train, X_test, y_test, dates, stock_ids = generate_synthetic_data()
    
    # Train LightGBM model
    model, training_metrics = train_lightgbm_model(X_train, y_train, X_test, y_test)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate rank correlation
    rank_corr = calculate_rank_correlation(y_pred, y_test)
    logger.info(f"Rank correlation between predictions and actual returns: {rank_corr:.6f}")
    
    # Simulate portfolio performance
    portfolio_returns = simulate_portfolio_performance(y_pred, y_test)
    logger.info(f"Long-short portfolio return: {portfolio_returns['long_short']:.6f}")
    
    # Compile all results
    results = {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'training_metrics': training_metrics,
        'rank_correlation': float(rank_corr),
        'portfolio_returns': portfolio_returns,
        'feature_importance': {
            f'feature_{i}': float(imp) for i, imp in enumerate(model.feature_importance())
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(WORKSPACE_DIR, f"mock_experiment_results_{results['timestamp']}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")
    
    return results

if __name__ == "__main__":
    # Run the experiment
    results = run_mock_experiment()
    
    # Print summary to stdout
    print("\n" + "="*50)
    print("MOCK EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    print(f"Training MSE: {results['training_metrics']['train_mse']:.6f}")
    print(f"Test MSE: {results['training_metrics']['test_mse']:.6f}")
    print(f"Test R²: {results['training_metrics']['test_r2']:.6f}")
    print(f"Rank Correlation: {results['rank_correlation']:.6f}")
    print(f"Long-Short Portfolio Return: {results['portfolio_returns']['long_short']:.6f}")
    print("="*50)
    
    # Top 5 features by importance
    feature_imp = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Features by Importance:")
    for feature, importance in feature_imp:
        print(f"  {feature}: {importance:.6f}")
    print("="*50)