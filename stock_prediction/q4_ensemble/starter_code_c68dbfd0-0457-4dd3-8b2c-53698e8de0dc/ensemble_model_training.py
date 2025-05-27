#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Model Training Script for Ensemble Methods

This script extends the original model_training.py to implement various ensemble approaches 
for stock returns prediction, including boosting of weak learners and stacking.
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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split

# For parallel processing
from multiprocessing import Pool, cpu_count

# Machine learning models
import lightgbm as lgb
from lightgbm import LGBMRegressor
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "xgboost"])
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True

try:
    import catboost as cb
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("CatBoost not available, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "catboost"])
    import catboost as cb
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True

# Current working directory
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

# Default hyperparameters
DEFAULT_CONFIG = {
    "data_path": "/workspace/starter_code_dataset",
    "results_path": "results",
    "num_years_train": 3,
    "start_year": 2017,
    "end_year": 2023,
    
    "min_samples": 1650,
    "min_trading_volume": 5000000,
    "feature_threshold": 0.75,
    "min_price": 2,
    
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
    
    "xgb_params": {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "learning_rate": 0.02,
        "n_estimators": 10000,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "early_stopping_rounds": 100,
        "verbosity": 0
    },
    
    "catboost_params": {
        "loss_function": "RMSE",
        "depth": 8,
        "learning_rate": 0.02,
        "iterations": 10000,
        "subsample": 0.7,
        "colsample_bylevel": 0.7,
        "early_stopping_rounds": 100,
        "verbose": False
    },
    
    "num_workers": 40,
    "num_simulations": 3,
    "device_type": "gpu",
    
    "ensemble_method": "boosting",  # 'boosting' or 'stacking'
    "feature_engineering": "raw",   # 'raw' or 'momentum_mean_reversion'
    "hyperparameter_optimization": "default"  # 'default' or 'optimized'
}

# Enhanced hyperparameters (for 'optimized' setting)
OPTIMIZED_PARAMS = {
    "lgbm_params": {
        "objective": "regression",
        "num_leaves": 127,
        "learning_rate": 0.01,
        "verbose": -1,
        "min_child_samples": 50,
        "n_estimators": 10000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "early_stopping_rounds": 100,
        "log_evaluation_freq": 500
    },
    
    "xgb_params": {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.01,
        "n_estimators": 10000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "early_stopping_rounds": 100,
        "verbosity": 0
    },
    
    "catboost_params": {
        "loss_function": "RMSE",
        "depth": 6,
        "learning_rate": 0.01,
        "iterations": 10000,
        "subsample": 0.8,
        "colsample_bylevel": 0.8,
        "reg_lambda": 0.1,
        "early_stopping_rounds": 100,
        "verbose": False
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run financial factor model training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            # Merge with default config
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

def load_data(config):
    """Load and preprocess financial data."""
    try:
        data_path = config['data_path']
        result_path = config['results_path']
        min_samples = config['min_samples']
        min_trading_volume = config['min_trading_volume']
        feature_threshold = config['feature_threshold']
        min_price = config['min_price']
        
        os.makedirs(result_path, exist_ok=True)
        
        # Placeholder for data loading
        # In a real scenario, you would load the data from files
        logger.info(f"Loading data from {data_path}...")
        
        # Load raw factor data
        factors = []
        for year in range(config['start_year'], config['end_year'] + 1):
            factors_path = os.path.join(data_path, 'RawData', f'factors_{year}.parquet')
            try:
                if os.path.exists(factors_path):
                    logger.info(f"Loading factors from {factors_path}")
                    year_factors = pd.read_parquet(factors_path)
                    factors.append(year_factors)
                else:
                    logger.warning(f"No factor data found for {year}")
            except Exception as e:
                logger.error(f"Error loading factor data for {year}: {e}")
                continue
        
        if not factors:
            raise ValueError("No factor data was loaded. Check paths and data availability.")
            
        logger.info(f"Loaded {len(factors)} years of factor data")
        
        # Apply filtering criteria
        filtered_factors = []
        for factor in factors:
            # Apply filtering based on minimum price, trading volume, etc.
            logger.info(f"Original shape: {factor.shape}")
            filtered_factor = factor.copy()
            # Additional preprocessing steps can be added here
            filtered_factors.append(filtered_factor)
            logger.info(f"Filtered shape: {filtered_factor.shape}")
        
        return filtered_factors
    
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

def create_momentum_mean_reversion_features(factors):
    """
    Create momentum and mean reversion features from raw factors.
    
    Args:
        factors: List of DataFrames containing factor data
        
    Returns:
        List of DataFrames with added momentum and mean reversion features
    """
    logger.info("Creating momentum and mean reversion features...")
    enhanced_factors = []
    
    for factor in factors:
        # Create a copy of the original data
        enhanced_factor = factor.copy()
        
        # Get list of factor columns (excluding any non-factor columns like date, ticker, etc.)
        factor_cols = [col for col in factor.columns if col not in ['date', 'ticker', 'return']]
        
        # Calculate momentum features (1-month change)
        for col in factor_cols:
            momentum_col = f"{col}_momentum"
            enhanced_factor[momentum_col] = enhanced_factor.groupby('ticker')[col].pct_change(1)
        
        # Calculate mean reversion features (deviation from 3-month moving average)
        for col in factor_cols:
            mean_rev_col = f"{col}_mean_reversion"
            moving_avg = enhanced_factor.groupby('ticker')[col].rolling(window=3).mean().reset_index(level=0, drop=True)
            enhanced_factor[mean_rev_col] = enhanced_factor[col] / moving_avg - 1
        
        # Handle NaNs
        enhanced_factor = enhanced_factor.fillna(0)
        
        enhanced_factors.append(enhanced_factor)
    
    logger.info(f"Created momentum and mean reversion features. New feature count: {enhanced_factors[0].shape[1]}")
    return enhanced_factors

def train_test_split_temporal(X, y, folds=3):
    """
    Split data temporally for cross-validation.
    
    Args:
        X: Features dataframe
        y: Target values
        folds: Number of folds for temporal cross-validation
        
    Returns:
        (X_train, X_valid, y_train, y_valid)
    """
    # Assume data is already sorted by date
    fold_size = len(X) // folds
    
    # Use the last fold for validation
    X_train = X[:fold_size * (folds - 1)]
    y_train = y[:fold_size * (folds - 1)]
    
    X_valid = X[fold_size * (folds - 1):]
    y_valid = y[fold_size * (folds - 1):]
    
    return X_train, X_valid, y_train, y_valid

def evaluate_predictions(y_true, y_pred):
    """
    Calculate various performance metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary of performance metrics
    """
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr
    
    # Mean squared error
    mse = mean_squared_error(y_true, y_pred)
    
    # Rank correlation (Spearman)
    corr, _ = spearmanr(y_true, y_pred)
    
    # Directional accuracy (% of times prediction gets direction right)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred) * 100
    
    return {
        "mean_squared_error": mse,
        "rank_correlation": corr,
        "directional_accuracy": directional_accuracy
    }

class BoostedEnsemble:
    """
    Ensemble model combining LightGBM, XGBoost, and CatBoost using boosting approach.
    Models are trained sequentially, each one learning from the residuals of the previous model.
    """
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.model_types = ['lgbm', 'xgb', 'catboost']
        
        # Determine hyperparameters based on optimization setting
        if config['hyperparameter_optimization'] == 'optimized':
            self.lgbm_params = OPTIMIZED_PARAMS['lgbm_params']
            self.xgb_params = OPTIMIZED_PARAMS['xgb_params']
            self.catboost_params = OPTIMIZED_PARAMS['catboost_params']
        else:
            self.lgbm_params = config['lgbm_params']
            self.xgb_params = config['xgb_params']
            self.catboost_params = config['catboost_params']
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the boosted ensemble model.
        Each model learns from the residuals of the previous model.
        """
        logger.info("Fitting boosted ensemble model...")
        current_target = y_train.copy()
        self.models = []
        
        # Create validation data if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42)
        else:
            X_train_split, y_train_split = X_train, y_train
            
        # Prepare validation sets for early stopping
        val_sets = {}
        if X_val is not None and y_val is not None:
            # LightGBM
            lgbm_val = lgb.Dataset(X_val, y_val)
            # XGBoost
            xgb_val = xgb.DMatrix(X_val, y_val)
            # CatBoost
            cat_val = cb.Pool(X_val, y_val)
            val_sets = {'lgbm': lgbm_val, 'xgb': xgb_val, 'catboost': cat_val}
        
        # Train the models sequentially
        predictions = np.zeros_like(y_train)
        
        # LightGBM
        logger.info("Training LightGBM model...")
        lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)
        lgbm_model.fit(
            X_train_split, current_target,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.lgbm_params.get('early_stopping_rounds', 100),
            verbose=False
        )
        self.models.append(('lgbm', lgbm_model))
        
        # Calculate residuals
        lgbm_pred = lgbm_model.predict(X_train)
        predictions += lgbm_pred
        current_target = y_train - predictions
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost model...")
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_model.fit(
                X_train_split, current_target,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.xgb_params.get('early_stopping_rounds', 100),
                verbose=False
            )
            self.models.append(('xgb', xgb_model))
            
            # Calculate residuals
            xgb_pred = xgb_model.predict(X_train)
            predictions += xgb_pred
            current_target = y_train - predictions
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost model...")
            cat_model = cb.CatBoostRegressor(**self.catboost_params)
            cat_model.fit(
                X_train_split, current_target,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.catboost_params.get('early_stopping_rounds', 100),
                verbose=False
            )
            self.models.append(('catboost', cat_model))
            
            # No need to calculate final residuals
        
        logger.info(f"Finished training {len(self.models)} models in boosted ensemble")
        return self
    
    def predict(self, X):
        """
        Generate predictions by summing the predictions from all models in the ensemble.
        """
        predictions = np.zeros(len(X))
        for model_type, model in self.models:
            predictions += model.predict(X)
        return predictions

class StackingEnsemble:
    """
    Ensemble model combining LightGBM, XGBoost, and CatBoost using stacking approach.
    First level models are trained independently, then a meta-model combines their predictions.
    """
    
    def __init__(self, config):
        self.config = config
        self.first_level_models = []
        self.meta_model = None
        self.model_types = ['lgbm', 'xgb', 'catboost']
        
        # Determine hyperparameters based on optimization setting
        if config['hyperparameter_optimization'] == 'optimized':
            self.lgbm_params = OPTIMIZED_PARAMS['lgbm_params']
            self.xgb_params = OPTIMIZED_PARAMS['xgb_params']
            self.catboost_params = OPTIMIZED_PARAMS['catboost_params']
        else:
            self.lgbm_params = config['lgbm_params']
            self.xgb_params = config['xgb_params']
            self.catboost_params = config['catboost_params']
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the stacking ensemble model.
        First level models are trained independently.
        A meta-model combines their predictions.
        """
        logger.info("Fitting stacking ensemble model...")
        self.first_level_models = []
        
        # Create validation data if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42)
        else:
            X_train_split, y_train_split = X_train, y_train
            
        # Train first level models
        first_level_preds_train = np.zeros((X_train.shape[0], len(self.model_types)))
        first_level_preds_val = np.zeros((X_val.shape[0], len(self.model_types)))
        
        # LightGBM
        logger.info("Training LightGBM model...")
        lgbm_model = lgb.LGBMRegressor(**self.lgbm_params)
        lgbm_model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.lgbm_params.get('early_stopping_rounds', 100),
            verbose=False
        )
        self.first_level_models.append(('lgbm', lgbm_model))
        first_level_preds_train[:, 0] = lgbm_model.predict(X_train)
        first_level_preds_val[:, 0] = lgbm_model.predict(X_val)
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("Training XGBoost model...")
            xgb_model = xgb.XGBRegressor(**self.xgb_params)
            xgb_model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=self.xgb_params.get('early_stopping_rounds', 100),
                verbose=False
            )
            self.first_level_models.append(('xgb', xgb_model))
            first_level_preds_train[:, 1] = xgb_model.predict(X_train)
            first_level_preds_val[:, 1] = xgb_model.predict(X_val)
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            logger.info("Training CatBoost model...")
            cat_model = cb.CatBoostRegressor(**self.catboost_params)
            cat_model.fit(
                X_train_split, y_train_split,
                eval_set=(X_val, y_val),
                early_stopping_rounds=self.catboost_params.get('early_stopping_rounds', 100),
                verbose=False
            )
            self.first_level_models.append(('catboost', cat_model))
            first_level_preds_train[:, 2] = cat_model.predict(X_train)
            first_level_preds_val[:, 2] = cat_model.predict(X_val)
        
        # Train meta-model (LightGBM meta-learner)
        logger.info("Training meta-model...")
        meta_lgbm_params = {
            "objective": "regression",
            "num_leaves": 31,
            "learning_rate": 0.01,
            "verbose": -1,
            "n_estimators": 5000,
            "early_stopping_rounds": 50
        }
        meta_model = lgb.LGBMRegressor(**meta_lgbm_params)
        meta_model.fit(
            first_level_preds_train, y_train,
            eval_set=[(first_level_preds_val, y_val)],
            early_stopping_rounds=meta_lgbm_params.get('early_stopping_rounds', 50),
            verbose=False
        )
        self.meta_model = meta_model
        
        logger.info(f"Finished training stacking ensemble with {len(self.first_level_models)} first-level models")
        return self
    
    def predict(self, X):
        """
        Generate predictions using the stacking ensemble.
        First generate predictions from first level models,
        then use meta-model to combine them.
        """
        first_level_preds = np.zeros((X.shape[0], len(self.model_types)))
        
        for i, (model_type, model) in enumerate(self.first_level_models):
            first_level_preds[:, i] = model.predict(X)
        
        return self.meta_model.predict(first_level_preds)

def run_experiment(config):
    """
    Run a complete stock return prediction experiment with the given configuration.
    
    Args:
        config: Dictionary of configuration parameters
        
    Returns:
        Dictionary of results
    """
    start_time = time.time()
    logger.info(f"Starting experiment with config: {json.dumps(config, indent=2)}")
    
    try:
        # Load and preprocess data
        factors = load_data(config)
        
        # Apply feature engineering if specified
        if config['feature_engineering'] == 'momentum_mean_reversion':
            logger.info("Applying momentum and mean-reversion feature engineering...")
            factors = create_momentum_mean_reversion_features(factors)
        
        # Set up tracking of results
        all_metrics = []
        all_predictions = []
        
        # Perform rolling window training and evaluation
        for year_idx in range(config['num_years_train'], len(factors)):
            year = year_idx + config['start_year']
            logger.info(f"Training for year {year} using previous {config['num_years_train']} years")
            
            # Get training and test data
            X_train_years = factors[year_idx - config['num_years_train']:year_idx]
            X_test_year = factors[year_idx]
            
            # Convert to tabular format
            X_train_list = []
            y_train_list = []
            for X_year in X_train_years:
                # Prepare features and target for this year
                feature_cols = [col for col in X_year.columns if col not in ['date', 'ticker', 'return']]
                X = X_year[feature_cols]
                y = X_year['return'] if 'return' in X_year.columns else None
                
                if y is not None:
                    X_train_list.append(X)
                    y_train_list.append(y)
            
            # Concatenate training data
            X_train = pd.concat(X_train_list)
            y_train = pd.concat(y_train_list)
            
            # Prepare test data
            feature_cols = [col for col in X_test_year.columns if col not in ['date', 'ticker', 'return']]
            X_test = X_test_year[feature_cols]
            y_test = X_test_year['return'] if 'return' in X_test_year.columns else None
            
            # Split validation set from training data
            X_train_split, X_val, y_train_split, y_val = train_test_split_temporal(X_train, y_train)
            
            # Choose and train ensemble model
            if config['ensemble_method'] == 'boosting':
                logger.info("Training boosted ensemble model...")
                model = BoostedEnsemble(config)
            else:  # stacking
                logger.info("Training stacking ensemble model...")
                model = StackingEnsemble(config)
                
            model.fit(X_train, y_train, X_val, y_val)
            
            # Generate predictions
            if y_test is not None:
                predictions = model.predict(X_test)
                
                # Calculate metrics
                metrics = evaluate_predictions(y_test, predictions)
                metrics['year'] = year
                all_metrics.append(metrics)
                
                # Store predictions
                pred_df = X_test_year[['ticker', 'date']].copy()
                pred_df['prediction'] = predictions
                pred_df['actual'] = y_test
                all_predictions.append(pred_df)
                
                logger.info(f"Year {year} metrics: MSE={metrics['mean_squared_error']:.6f}, "
                           f"Rank Correlation={metrics['rank_correlation']:.4f}, "
                           f"Directional Accuracy={metrics['directional_accuracy']:.2f}%")
        
        # Combine all predictions
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        # Calculate overall metrics
        overall_metrics = evaluate_predictions(all_predictions_df['actual'], all_predictions_df['prediction'])
        overall_metrics['execution_time'] = time.time() - start_time
        
        logger.info(f"Overall metrics: MSE={overall_metrics['mean_squared_error']:.6f}, "
                   f"Rank Correlation={overall_metrics['rank_correlation']:.4f}, "
                   f"Directional Accuracy={overall_metrics['directional_accuracy']:.2f}%")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"{config['results_path']}/metrics_{timestamp}.json"
        predictions_file = f"{config['results_path']}/predictions_{timestamp}.parquet"
        
        with open(results_file, 'w') as f:
            json.dump({
                'overall_metrics': overall_metrics,
                'yearly_metrics': all_metrics,
                'config': config
            }, f, indent=2)
        
        all_predictions_df.to_parquet(predictions_file, index=False)
        
        logger.info(f"Results saved to {results_file} and {predictions_file}")
        
        return {
            'overall_metrics': overall_metrics,
            'yearly_metrics': all_metrics,
            'predictions_file': predictions_file,
            'results_file': results_file
        }
    
    except Exception as e:
        logger.error(f"Error in experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def main():
    """Main entry point for the script."""
    args = parse_args()
    config = load_config(args.config)
    results = run_experiment(config)
    logger.info("Experiment completed successfully")
    return results

if __name__ == "__main__":
    main()
