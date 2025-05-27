#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Ensemble Model Training Script

This script extends the original model_training.py to support multiple model types and ensemble methods
for financial factor-based prediction models.
"""

import os
import warnings
import logging
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np
from functools import partial
import json
from typing import List, Dict, Any, Union, Optional, Tuple
import gc

# For parallel processing
from multiprocessing import Pool, cpu_count

# Machine learning models
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
try:
    import catboost as cb
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "catboost"])
    import catboost as cb
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True

# For meta-learners in stacking
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectFromModel

# get the current working directory
cur_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current working directory: {cur_dir}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(cur_dir, "model_training_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Default hyperparameters - extended from original script
DEFAULT_CONFIG = {
    "data_path": "/workspace/starter_code_dataset",
    "results_path": "/workspace/starter_code_c0338e9a-0531-43c2-9a5d-5a6ba5156e3b/results",
    "num_years_train": 3,
    "start_year": 2017,
    "end_year": 2023,
    
    "min_samples": 1650,
    "min_trading_volume": 5000000,
    "feature_threshold": 0.75,
    "min_price": 2,
    
    "ensemble_architecture": "stacking",  # stacking, boosting, or hybrid
    "meta_learner": "linear",  # linear or lightgbm (for stacking)
    "base_models": ["lightgbm", "xgboost", "catboost"],
    "feature_selection": "all",  # all or importance
    "importance_threshold": 0.01,  # for feature importance based selection
    
    "lgbm_params": {
        "objective": "regression",
        "num_leaves": 511,
        "learning_rate": 0.02,
        "verbose": -1,
        "min_child_samples": 30,
        "n_estimators": 10000,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 100,
        "log_evaluation_freq": 500
    },
    
    "xgboost_params": {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "learning_rate": 0.02,
        "n_estimators": 10000,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 100
    },
    
    "catboost_params": {
        "objective": "RMSE",
        "iterations": 10000,
        "learning_rate": 0.02,
        "depth": 8,
        "early_stopping_rounds": 100,
        "verbose": 0
    },
    
    "num_workers": 40,
    "num_simulations": 3,
    "device_type": "gpu",
    "variant_name": "default"
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ensemble model training with configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    return parser.parse_args()

def load_data(config):
    """Load and preprocess data based on configuration."""
    logger.info(f"Loading data from {config['data_path']}...")
    
    # Implement data loading logic
    # This function should return features and targets for training and testing
    
    # Simulating data loading for now
    X_train = pd.DataFrame(np.random.random((1000, 50)))
    y_train = pd.Series(np.random.random(1000))
    X_test = pd.DataFrame(np.random.random((500, 50)))
    y_test = pd.Series(np.random.random(500))
    
    return X_train, y_train, X_test, y_test

def select_features(X_train, y_train, config):
    """Select features based on importance if specified in config."""
    if config["feature_selection"] == "all":
        logger.info("Using all features.")
        return X_train
    
    elif config["feature_selection"] == "importance":
        logger.info("Selecting features based on feature importance.")
        # Train a simple LGBM model to get feature importances
        feature_selector = LGBMRegressor(
            n_estimators=100, 
            learning_rate=0.1
        )
        feature_selector.fit(X_train, y_train)
        
        # Get feature importances
        importances = feature_selector.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        })
        
        # Select features based on importance threshold
        threshold = config["importance_threshold"]
        important_features = feature_importance[feature_importance['importance'] > threshold]['feature'].tolist()
        
        logger.info(f"Selected {len(important_features)} important features out of {X_train.shape[1]}")
        
        return X_train[important_features]
    else:
        logger.warning(f"Unknown feature selection method: {config['feature_selection']}. Using all features.")
        return X_train

def create_base_model(model_type, params, device_type):
    """Create a base model of specified type with given parameters."""
    if model_type == "lightgbm":
        if device_type == "gpu":
            params["device"] = "gpu"
        return LGBMRegressor(**params)
    
    elif model_type == "xgboost":
        if device_type == "gpu":
            params["tree_method"] = "gpu_hist"
        return XGBRegressor(**params)
    
    elif model_type == "catboost":
        if device_type == "gpu":
            params["task_type"] = "GPU"
        return CatBoostRegressor(**params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_stacking_ensemble(X_train, y_train, X_test, config):
    """Train a stacking ensemble model with specified base models and meta-learner."""
    logger.info(f"Training stacking ensemble with {config['meta_learner']} meta-learner")
    
    base_models = []
    device_type = config.get("device_type", "cpu")
    
    # Train base models
    base_predictions_train = np.zeros((X_train.shape[0], len(config['base_models'])))
    base_predictions_test = np.zeros((X_test.shape[0], len(config['base_models'])))
    
    for i, model_type in enumerate(config['base_models']):
        logger.info(f"Training base model {i+1}/{len(config['base_models'])}: {model_type}")
        
        # Get parameters for this model type
        if model_type == "lightgbm":
            params = config['lgbm_params']
        elif model_type == "xgboost":
            params = config['xgboost_params']
        elif model_type == "catboost":
            params = config['catboost_params']
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Create and train the model
        model = create_base_model(model_type, params, device_type)
        model.fit(X_train, y_train)
        base_models.append(model)
        
        # Generate predictions
        base_predictions_train[:, i] = model.predict(X_train)
        base_predictions_test[:, i] = model.predict(X_test)
        
    # Train meta-learner
    if config['meta_learner'] == 'linear':
        meta_model = Ridge(alpha=1.0)
    elif config['meta_learner'] == 'lightgbm':
        meta_model = LGBMRegressor(**{
            **config['lgbm_params'],
            'n_estimators': 100  # Use fewer estimators for meta-model
        })
    else:
        raise ValueError(f"Unsupported meta-learner type: {config['meta_learner']}")
    
    # Train meta-learner
    meta_model.fit(base_predictions_train, y_train)
    
    # Final predictions
    final_predictions = meta_model.predict(base_predictions_test)
    
    return final_predictions, base_models, meta_model

def train_boosting_ensemble(X_train, y_train, X_test, config):
    """Train boosting ensemble by sequentially training models on residuals."""
    logger.info("Training boosting ensemble of weak learners")
    
    device_type = config.get("device_type", "cpu")
    base_models = []
    
    # Initial prediction is zero
    current_pred_train = np.zeros(X_train.shape[0])
    current_pred_test = np.zeros(X_test.shape[0])
    
    # Residuals start as the original target
    residuals = y_train.copy()
    
    for i, model_type in enumerate(config['base_models']):
        logger.info(f"Training boosting model {i+1}/{len(config['base_models'])}: {model_type}")
        
        # Get parameters for this model type
        if model_type == "lightgbm":
            params = config['lgbm_params']
        elif model_type == "xgboost":
            params = config['xgboost_params'] 
        elif model_type == "catboost":
            params = config['catboost_params']
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Create and train the model on residuals
        model = create_base_model(model_type, params, device_type)
        model.fit(X_train, residuals)
        base_models.append(model)
        
        # Update predictions and residuals
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        # Learning rate for boosting
        lr = 0.1
        
        current_pred_train += lr * pred_train
        current_pred_test += lr * pred_test
        
        # Update residuals
        residuals = y_train - current_pred_train
        
    return current_pred_test, base_models, None

def train_hybrid_ensemble(X_train, y_train, X_test, config):
    """Train hybrid ensemble that blends top N models."""
    logger.info(f"Training hybrid ensemble blending top {config['blend_top']} models")
    
    base_models = []
    scores = []
    predictions = []
    device_type = config.get("device_type", "cpu")
    
    # Train all base models
    for i, model_type in enumerate(config['base_models']):
        logger.info(f"Training base model {i+1}/{len(config['base_models'])}: {model_type}")
        
        # Get parameters for this model type
        if model_type == "lightgbm":
            params = config['lgbm_params']
        elif model_type == "xgboost":
            params = config['xgboost_params']
        elif model_type == "catboost":
            params = config['catboost_params']
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Split train data for validation
        X_train_part, X_val, y_train_part, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        # Create and train the model
        model = create_base_model(model_type, params, device_type)
        model.fit(X_train_part, y_train_part)
        base_models.append(model)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val)
        score = np.corrcoef(val_pred, y_val)[0, 1]  # Rank correlation
        scores.append((i, score))
        
        # Generate test predictions
        predictions.append(model.predict(X_test))
    
    # Sort by score and get top N
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in scores[:config['blend_top']]]
    
    logger.info(f"Top {config['blend_top']} models: {[config['base_models'][idx] for idx in top_indices]}")
    
    # Blend predictions of top models
    final_predictions = np.zeros(X_test.shape[0])
    for idx in top_indices:
        final_predictions += predictions[idx] / config['blend_top']
    
    return final_predictions, [base_models[idx] for idx in top_indices], None

def train_ensemble(X_train, y_train, X_test, config):
    """Train the ensemble model based on specified architecture."""
    if config['ensemble_architecture'] == 'stacking':
        return train_stacking_ensemble(X_train, y_train, X_test, config)
    elif config['ensemble_architecture'] == 'boosting':
        return train_boosting_ensemble(X_train, y_train, X_test, config)
    elif config['ensemble_architecture'] == 'hybrid':
        return train_hybrid_ensemble(X_train, y_train, X_test, config)
    else:
        raise ValueError(f"Unsupported ensemble architecture: {config['ensemble_architecture']}")

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using various metrics."""
    metrics = {}
    
    # Rank correlation (Spearman)
    rank_corr = np.corrcoef(pd.Series(y_true).rank(), pd.Series(y_pred).rank())[0, 1]
    metrics['rank_correlation'] = rank_corr
    
    # Mean squared error
    mse = mean_squared_error(y_true, y_pred)
    metrics['mse'] = mse
    
    # Directional accuracy (proportion of correct direction predictions)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    dir_acc = np.mean(direction_true == direction_pred)
    metrics['directional_accuracy'] = dir_acc
    
    return metrics

def main():
    """Main function to run the ensemble model training pipeline."""
    start_time = time.time()
    
    # Parse arguments and load configuration
    args = parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Merge with default configuration
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    
    logger.info(f"Running ensemble model training with configuration: {config['variant_name']}")
    
    try:
        # Create results directory if it doesn't exist
        os.makedirs(config['results_path'], exist_ok=True)
        
        # Load data
        X_train, y_train, X_test, y_test = load_data(config)
        
        # Feature selection if specified
        X_train_selected = select_features(X_train, y_train, config)
        X_test_selected = X_test[X_train_selected.columns]
        
        logger.info(f"Training with {X_train_selected.shape[1]} features")
        
        # Train ensemble model
        predictions, base_models, meta_model = train_ensemble(X_train_selected, y_train, X_test_selected, config)
        
        # Evaluate model
        metrics = evaluate_model(y_test, predictions)
        
        # Log results
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'true_value': y_test,
            'prediction': predictions
        })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_file = os.path.join(config['results_path'], f"predictions_{config['variant_name']}_{timestamp}.parquet")
        metrics_file = os.path.join(config['results_path'], f"metrics_{config['variant_name']}_{timestamp}.json")
        
        results_df.to_parquet(predictions_file)
        
        # Add configuration to metrics for reference
        output = {
            "metrics": metrics,
            "config": config
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(output, f, indent=4)
        
        logger.info(f"Results saved to {predictions_file} and {metrics_file}")
        
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise
    
    finally:
        # Print total runtime
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("")

if __name__ == "__main__":
    main()
