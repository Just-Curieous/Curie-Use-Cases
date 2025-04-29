#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structured Model Training Script

This script contains improved implementation of financial factor-based prediction model.
It includes optimized data processing, model training, and evaluation.
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

# For parallel processing
from multiprocessing import Pool, cpu_count

# Machine learning
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# get the current working directory
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current working directory: {cur_dir}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(cur_dir, "model_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Default hyperparameters
DEFAULT_CONFIG = {
    # Data parameters
    "data_path": "/workspace/quant_data/",
    "results_path": os.path.join(cur_dir, "results"),
    "num_years_train": 3,
    "start_year": 2017,
    "end_year": 2023,
    
    # Filtering parameters
    "min_samples": 1650,
    "min_trading_volume": 5000000,
    "min_price": 2,
    
    # Model parameters
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
    
    # Processing parameters
    "num_workers": min(80, cpu_count()),
    "num_simulations": 3,
    "feature_threshold": 0.75,
    "device_type": "gpu"
}

# Create necessary directories
def create_directories(config):
    """Create necessary directories for storing results."""
    os.makedirs(config["results_path"], exist_ok=True)
    logger.info(f"Created or verified directories: {config['results_path']}")

# Helper Functions
def filter_st(signal, is_locked):
    """Filter out locked stocks."""
    mask = (is_locked != 1).replace(False, np.nan)
    return (mask * signal).dropna(how='all')

def get_common_indices(dataframes):
    """Get common indices and columns across dataframes."""
    common_idx = dataframes[0].index
    common_cols = dataframes[0].columns
    
    for df in dataframes:
        common_idx = common_idx.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)
    
    return [df.loc[common_idx, common_cols] for df in dataframes]

def process_factor(factor, is_locked):
    """Process a single factor - used for parallel processing."""
    try:
        result = filter_st(factor, is_locked)
        return result.astype(np.float64).fillna(0)
    except Exception as e:
        logger.error(f"Error processing factor: {e}")
        return None

def factors_process_parallel(factors, is_locked, config):
    """Process all factors in parallel using process pool."""
    logger.info(f"Processing {len(factors)} factors using {config['num_workers']} workers")
    
    start_time = time.time()
    
    # Using partial to create a function with preset parameters
    process_func = partial(process_factor, is_locked=is_locked)
    
    # Using context manager to ensure proper cleanup
    with Pool(config['num_workers']) as pool:
        processed_factors = pool.map(process_func, factors)
    
    # Filter out None values (failed processing)
    valid_factors = [f for f in processed_factors if f is not None]
    
    duration = time.time() - start_time
    logger.info(f"Processed {len(valid_factors)} factors in {duration:.2f} seconds")
    
    return valid_factors

def filter_factors(factors, min_samples=1650, year_range=('2017', '2023')):
    """Filter factors based on sample size within date range."""
    filtered = [f for f in factors if f.dropna(how='all').loc[year_range[0]:year_range[1]].shape[0] > min_samples]
    logger.info(f"Filtered factors from {len(factors)} to {len(filtered)}")
    return filtered

# Model Training and Prediction Functions
def reshape_data(factors, return_data, mask):
    """Reshape factor and return data for model training."""
    # Get dimensions
    nrows = return_data[mask].iloc[:-6, :].shape[0]
    ncols = return_data[mask].iloc[:-6, :].shape[1]
    
    # Extract and reshape factors
    factor_data = [factor[mask].iloc[:-6, :] for factor in factors]
    factor_array = np.asarray(factor_data)
    X = np.reshape(factor_array, (factor_array.shape[0], nrows * ncols))
    
    # Reshape return data
    y = np.reshape(return_data[mask].iloc[:-6, :].values, (nrows * ncols))
    
    return X.T, y

def remove_nan_sparse(X, y, feature_threshold=0.75):
    """Remove rows with NaN values or too many zero features."""
    # Mask for non-NaN target values
    mask_1 = ~np.isnan(y)
    
    # Mask for rows where less than threshold% of features are zero
    mask_2 = (X == 0).sum(axis=1) < X.shape[1] * feature_threshold
    
    # Combine masks
    combined_mask = mask_1 & mask_2
    
    return X[combined_mask], y[combined_mask]

def train_lgbm_model(X_train, y_train, config):
    """Train LightGBM model with early stopping."""
    lgbm_params = config["lgbm_params"]
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=np.random.randint(1000)
    )
    
    # Create and train model
    model = LGBMRegressor(
        objective=lgbm_params["objective"],
        num_leaves=lgbm_params["num_leaves"],
        learning_rate=lgbm_params["learning_rate"],
        verbose=lgbm_params["verbose"],
        min_child_samples=lgbm_params["min_child_samples"],
        n_estimators=lgbm_params["n_estimators"],
        n_jobs=config["num_workers"],
        subsample=lgbm_params["subsample"],
        colsample_bytree=lgbm_params["colsample_bytree"],
        random_state=np.random.randint(1000),
        device_type="gpu"
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_metric='l2',
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=lgbm_params["early_stopping_rounds"]),
            lgb.log_evaluation(lgbm_params["log_evaluation_freq"])
        ]
    )
    
    return model

def make_predictions(factors, mask, model, config):
    """Make predictions for a specific time period."""
    # Extract factor data for the specified mask
    factor_data = [factor[mask] for factor in factors]
    factor_array = np.array(factor_data)
    
    # Initialize predictions array
    predictions = np.zeros([factor_array.shape[1], factor_array.shape[2]])
    
    # For each day in the period
    for day in range(factor_array.shape[1]):
        # Stack features for all stocks on this day
        X = np.column_stack(factor_array[:, day])
        
        # Identify stocks with sufficient non-zero features
        indicator = (X != 0).sum(axis=1) > config["feature_threshold"] * X.shape[1]
        
        # Make predictions for valid stocks
        if np.any(indicator):
            day_predictions = model.predict(X[indicator], num_iteration=model.best_iteration_)
            predictions[day][indicator] = day_predictions
            predictions[day][~indicator] = np.nan
            
    return predictions

def run_prediction(factors, return_data, config):
    """Run prediction for all years in simulation."""
    # Extract configuration parameters
    start_year = config["start_year"]
    end_year = config["end_year"]
    num_years_train = config["num_years_train"]
    num_sims = config["num_simulations"]
    
    # Initialize prediction DataFrame with zeros
    predictions = pd.DataFrame(
        np.zeros(factors[0].shape),
        index=factors[0].index,
        columns=factors[0].columns
    )
    
    # Run multiple simulations to reduce variance
    for sim in range(num_sims):
        logger.info(f"Running simulation {sim+1}/{num_sims}")
        
        # Initialize this simulation's predictions
        sim_predictions = pd.DataFrame(
            index=factors[0].index,
            columns=factors[0].columns
        )
        
        # For each prediction year
        for pred_year in range(start_year + num_years_train, end_year + 1):
            print(f"[{sim+1}/{num_sims}] Predicting for year {pred_year}")
            # Define training and prediction periods
            train_mask = (factors[0].index.year < pred_year) & (factors[0].index.year >= (pred_year - num_years_train))
            pred_mask = factors[0].index.year == pred_year
            
            # Reshape data for training
            X, y = reshape_data(factors, return_data=return_data, mask=train_mask)
            
            # Remove NaN and sparse rows
            X, y = remove_nan_sparse(X, y, config["feature_threshold"])
            
            # Train model
            model = train_lgbm_model(X, y, config)
            
            # Make predictions
            sim_predictions[pred_mask] = make_predictions(factors, mask=pred_mask, model=model, config=config)
        
        # Add this simulation's predictions to the total
        predictions += sim_predictions
    
    # Average the predictions across simulations
    predictions = predictions / num_sims
    
    return predictions

def calculate_metrics(predictions, returns, config):
    """Calculate and return performance metrics."""
    # Apply filtering criteria
    filtered_predictions = predictions.copy()
    
    # Calculate rank correlations by year
    metrics = {}
    
    # Overall metrics
    filtered_predictions_rank = filtered_predictions.rank(axis=1)
    returns_rank = returns.rank(axis=1).shift(-1)
    overall_corr = filtered_predictions_rank.corrwith(returns_rank, axis=1).mean()
    metrics["overall"] = float(overall_corr)
    
    # Yearly metrics
    for year in range(config["start_year"] + config["num_years_train"], config["end_year"] + 1):
        year_mask = filtered_predictions.index.year == year
        if year_mask.sum() > 0:
            year_corr = filtered_predictions.loc[str(year)].rank(axis=1).corrwith(
                returns.rank(axis=1).shift(-1), axis=1
            ).mean()
            metrics[str(year)] = float(year_corr)
    
    return metrics

def apply_filters(predictions, returns, is_locked, trading_volume, prices, config):
    """Apply filters to predictions and returns data."""
    # Create masks for filtering
    volume_mask = trading_volume > config["min_trading_volume"]
    price_mask = prices > config["min_price"]
    lock_mask = is_locked != 1
    
    # Apply all filters
    combined_mask = volume_mask & price_mask & lock_mask
    
    # Apply masks to dataframes
    filtered_predictions = predictions[combined_mask]
    filtered_returns = returns[combined_mask]
    
    logger.info(f"Applied filters: {filtered_predictions.shape[0]} rows remaining")
    
    return filtered_predictions, filtered_returns

def load_data(config):
    """Load all necessary data files."""
    data_path = config["data_path"]
    
    # Load factors
    factor_dir = os.path.join(data_path, 'RawData/NFactors/')
    factors = []
    
    # Check if directory exists
    if not os.path.exists(factor_dir):
        logger.error(f"Factor directory not found: {factor_dir}")
        return None
    
    # Load each factor file
    for filename in os.listdir(factor_dir):
        try:
            file_path = os.path.join(factor_dir, filename)
            df = pd.read_parquet(file_path, engine='pyarrow')
            factors.append(df)
        except Exception as e:
            logger.info(f"Warning: Skip reading {file_path}: {e}")
    
    logger.info(f"Loaded {len(factors)} factor files")
    
    # Load label data
    label_dir = os.path.join(data_path, 'RawData/Label/')
    
    try:
        ret = pd.read_parquet(os.path.join(label_dir, 'ret.parquet'))
        ret_n = pd.read_parquet(os.path.join(label_dir, 'ret_n.parquet'))
    except Exception as e:
        logger.error(f"Error loading return data: {e}")
        return None
    
    # Load daily base data
    daily_base_dir = os.path.join(data_path, 'RawData/DailyBase/')
    
    try:
        is_locked = pd.read_parquet(os.path.join(daily_base_dir, 'is_locked.parquet'))
        tva_0930_1130 = pd.read_parquet(os.path.join(daily_base_dir, 'tva_0930_1130.parquet'))
        vwap_0930_1130 = pd.read_parquet(os.path.join(daily_base_dir, 'vwap_0930_1130.parquet'))
    except Exception as e:
        logger.error(f"Error loading daily base data: {e}")
        return None
    
    logger.info("Successfully loaded all data files")
    
    return {
        'factors': factors,
        'ret': ret,
        'ret_n': ret_n,
        'is_locked': is_locked,
        'tva_0930_1130': tva_0930_1130,
        'vwap_0930_1130': vwap_0930_1130
    }

def save_results(predictions, metrics, config):
    """Save predictions and metrics to files."""
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions to parquet
    pred_file = os.path.join(config["results_path"], f"predictions_{timestamp}.parquet")
    predictions.to_parquet(pred_file)
    
    # Save metrics and config to JSON
    results = {
        "metrics": metrics,
        "config": config
    }
    
    metrics_file = os.path.join(config["results_path"], f"metrics_{timestamp}.json")
    with open(metrics_file, 'a') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {pred_file} and {metrics_file}")
    
    return metrics_file

def main(config=None):
    """Main function to run the entire pipeline."""
    start_time = time.time()
    
    # Use default config if none provided
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Create directories
    create_directories(config)
    
    # Load data
    logger.info("Loading data...")
    data = load_data(config)
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return None
    
    # Filter factors based on sample size
    logger.info("Filtering factors...")
    filtered_factors = filter_factors(
        data['factors'], 
        min_samples=config["min_samples"], 
        year_range=(str(config["start_year"]), str(config["end_year"]))
    )
    
    # Process factors in parallel
    logger.info("Processing factors...")
    processed_factors = factors_process_parallel(
        filtered_factors,
        data['is_locked'],
        config
    )
    
    # Prepare return data
    ret_train = data['ret_n'][data['is_locked'] != 1].shift(-1).dropna(how='all')
    
    # Combine factors with return data and get common indices
    logger.info("Finding common indices...")
    combined_data = processed_factors + [ret_train]
    common_data = get_common_indices(combined_data)
    
    # Extract factors and returns with common indices
    common_factors = common_data[:-1]
    ret_train_common = common_data[-1]
    
    # Run prediction
    logger.info("Running prediction...")
    predictions = run_prediction(common_factors, ret_train_common, config)
    
    # Apply filters
    logger.info("Applying filters...")
    filtered_predictions, filtered_returns = apply_filters(
        predictions,
        data['ret'],
        data['is_locked'],
        data['tva_0930_1130'],
        data['vwap_0930_1130'],
        config
    )
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(filtered_predictions, filtered_returns, config)
    
    # Save results
    logger.info("Saving results...")
    metrics_file = save_results(filtered_predictions, metrics, config)
    
    # Print summary
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    
    # Print metrics report
    logger.info(f"\n{'='*50}\nPERFORMANCE METRICS\n{'='*50}")
    logger.info(f"Overall Rank Correlation: {metrics['overall']:.4f}")
    
    for year in sorted(k for k in metrics.keys() if k != 'overall'):
        logger.info(f"{year} Rank Correlation: {metrics[year]:.4f}")
    
    logger.info(f"{'='*50}\nFull report saved to: {metrics_file}\n{'='*50}")
    
    return {
        'predictions': filtered_predictions,
        'metrics': metrics,
        'config': config
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Financial factor model training")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            # Merge with default config
            config = {**DEFAULT_CONFIG, **custom_config}
    else:
        import sys
        sys.exit("Config file not found. Specify a valid path using --config.")
    
    # Run main function
    main(config)
