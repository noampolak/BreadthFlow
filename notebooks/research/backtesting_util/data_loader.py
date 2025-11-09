"""
Data loading functions for backtesting.

Handles loading of models, stock data, and features cache.
"""

import pandas as pd
import json
import os
import pickle

from .config import (
    DATA_DIR,
    MODELS_DIR,
    LOG_FILE,
    FEATURES_CACHE_DIR,
    FEATURE_WINDOWS
)


def load_models_from_log(include_randomforest=True, include_xgboost=True, model_suffix=None):
    """
    Load models from training log.
    
    Parameters:
    -----------
    include_randomforest : bool
        Whether to include RandomForest models
    include_xgboost : bool
        Whether to include XGBoost models
    model_suffix : str, optional
        Filter models by suffix (e.g., '_2014_2021'). If None, loads all models.
        
    Returns:
    --------
    dict
        Dictionary mapping model_key -> model object
        Separate keys: 'rf_models' and 'xgb_models'
    """
    print(f"Loading models from {LOG_FILE}...")
    
    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"Training log not found: {LOG_FILE}")
    
    with open(LOG_FILE, 'r') as f:
        training_log = json.load(f)
    
    rf_models = {}
    xgb_models = {}
    
    for model_key, info in training_log.items():
        # Filter by suffix if specified
        if model_suffix and not model_key.endswith(model_suffix):
            continue
        
        model_path = info.get('model_path')
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            if 'RandomForest' in model_key and include_randomforest:
                rf_models[model_key] = model
            elif 'XGBoost' in model_key and include_xgboost:
                xgb_models[model_key] = model
    
    print(f"✅ Loaded {len(rf_models)} RandomForest models")
    print(f"✅ Loaded {len(xgb_models)} XGBoost models")
    
    return {
        'rf_models': rf_models,
        'xgb_models': xgb_models,
        'all_models': {**rf_models, **xgb_models}
    }


def load_models_from_directory(model_suffix, model_dir=None):
    """
    Load models directly from directory by suffix pattern.
    Useful when models aren't in training log.
    
    Parameters:
    -----------
    model_suffix : str
        Model suffix to match (e.g., '_2014_2021')
    model_dir : str, optional
        Directory containing models. If None, uses MODELS_DIR from config.
        
    Returns:
    --------
    dict
        Dictionary mapping model_key -> model object
        Separate keys: 'rf_models' and 'xgb_models'
    """
    if model_dir is None:
        model_dir = MODELS_DIR
    
    print(f"Loading models with suffix '{model_suffix}' from {model_dir}...")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Models directory not found: {model_dir}")
    
    rf_models = {}
    xgb_models = {}
    
    # Scan directory for matching model files
    for filename in os.listdir(model_dir):
        if filename.endswith('.pkl') and model_suffix in filename:
            # Extract model key from filename (remove .pkl extension)
            model_key = filename.replace('.pkl', '')
            
            model_path = os.path.join(model_dir, filename)
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                if 'RandomForest' in model_key and 'RandomForest' in filename:
                    rf_models[model_key] = model
                elif 'XGBoost' in model_key and 'XGBoost' in filename:
                    xgb_models[model_key] = model
            except Exception as e:
                print(f"   ⚠️  Failed to load {model_key}: {e}")
    
    print(f"✅ Loaded {len(rf_models)} RandomForest models")
    print(f"✅ Loaded {len(xgb_models)} XGBoost models")
    
    return {
        'rf_models': rf_models,
        'xgb_models': xgb_models,
        'all_models': {**rf_models, **xgb_models}
    }


def load_stock_data(data_file=None):
    """
    Load stock data from pickle file.
    
    Parameters:
    -----------
    data_file : str, optional
        Path to stock data file. If None, uses default location.
        
    Returns:
    --------
    pd.DataFrame or None
        Stock data DataFrame, or None if file not found
    """
    if data_file is None:
        data_file = os.path.join(DATA_DIR, 'stock_data_latest', 'sp500_stock_data_latest.pkl')
    
    if os.path.exists(data_file):
        stock_data = pd.read_pickle(data_file)
        print(f"✅ Loaded stock data: {stock_data.shape}")
        print(f"   From: {data_file}")
        return stock_data
    else:
        print(f"❌ Stock data file not found: {data_file}")
        return None


def load_features_cache(feature_windows=None):
    """
    Load features cache for specified windows.
    
    Parameters:
    -----------
    feature_windows : list of int, optional
        List of feature windows to load. If None, uses FEATURE_WINDOWS from config.
        
    Returns:
    --------
    dict
        Dictionary mapping window -> features DataFrame
    """
    if feature_windows is None:
        feature_windows = FEATURE_WINDOWS
    
    features_cache = {}
    
    print("Loading features cache...")
    for window in feature_windows:
        cache_file = os.path.join(FEATURES_CACHE_DIR, f'features_{window}d.pkl')
        if os.path.exists(cache_file):
            features_cache[window] = pd.read_pickle(cache_file)
            print(f"   ✅ Loaded {window}d features: {len(features_cache[window]):,} rows")
        else:
            print(f"   ⚠️  Features file not found: {cache_file}")
    
    return features_cache


def get_test_period(stock_data, test_split=0.6, test_start_date=None, test_end_date=None):
    """
    Calculate test period dates based on split ratio or use provided dates.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock data with 'date' column
    test_split : float
        Training split ratio (e.g., 0.6 means 60% training, 40% testing).
        Only used if test_start_date is None.
    test_start_date : datetime or str, optional
        Explicit test start date. If provided, overrides test_split.
    test_end_date : datetime or str, optional
        Explicit test end date. If None, uses last date in stock_data.
        
    Returns:
    --------
    tuple
        (test_start_date, test_end_date)
    """
    all_dates = sorted(stock_data['date'].unique())
    
    if test_start_date is not None:
        # Use explicit dates
        test_start_date = pd.to_datetime(test_start_date)
        if test_end_date is not None:
            test_end_date = pd.to_datetime(test_end_date)
        else:
            test_end_date = pd.to_datetime(all_dates[-1])
    else:
        # Use split ratio
        test_start_idx = int(len(all_dates) * test_split)
        test_start_date = all_dates[test_start_idx]
        test_end_date = all_dates[-1] if test_end_date is None else pd.to_datetime(test_end_date)
    
    test_data = stock_data[
        (stock_data['date'] >= test_start_date) & 
        (stock_data['date'] <= test_end_date)
    ].copy()
    
    print(f"📅 Test Period:")
    print(f"   Start: {test_start_date}")
    print(f"   End: {test_end_date}")
    print(f"   Trading days: {len([d for d in all_dates if test_start_date <= pd.to_datetime(d) <= test_end_date])}")
    print(f"   Test data: {len(test_data):,} rows")
    
    return test_start_date, test_end_date


def parse_model_key(model_key):
    """
    Parse model key to extract parameters.
    
    Handles formats like:
    - 'f5d_t5d_XGBoost'
    - 'f5d_t5d_XGBoost_2014_2021'
    
    Parameters:
    -----------
    model_key : str
        Model key in format like 'f5d_t5d_XGBoost' or 'f5d_t5d_XGBoost_2014_2021'
        
    Returns:
    --------
    tuple
        (feature_window, target_window, model_type)
    """
    parts = model_key.split('_')
    feature_window = int(parts[0][1:-1])  # f5d -> 5
    
    # Find target window (t5d, t10d, etc.) - it's the first part starting with 't'
    target_window = None
    for part in parts:
        if part.startswith('t') and part[1:-1].isdigit():
            target_window = int(part[1:-1])  # t5d -> 5
            break
    
    if target_window is None:
        raise ValueError(f"Could not parse target_window from model_key: {model_key}")
    
    if 'RandomForest' in model_key:
        model_type = 'RandomForest'
    elif 'XGBoost' in model_key:
        model_type = 'XGBoost'
    else:
        model_type = 'Unknown'
    
    return feature_window, target_window, model_type
