"""
Main orchestrator function for training all models.
"""

import os
import pickle
import json
import pandas as pd
from tqdm import tqdm

from .config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES_CACHE_DIR, LOG_FILE,
    DEFAULT_FEATURE_WINDOWS, DEFAULT_TARGET_WINDOWS, DEFAULT_MODEL_TYPES,
    DEFAULT_TRAIN_SPLIT
)
from .cache_manager import load_training_log, save_training_log
from .feature_creator import precalculate_all_features
from .target_calculator import precalculate_all_targets
from .model_trainer import train_single_model_optimized

# Try to import NON_FEATURE_COLS
try:
    from training_feature_utils import NON_FEATURE_COLS
except ImportError:
    NON_FEATURE_COLS = [
        'symbol', 'date', 'quarter_end', 'fiscal_year',
        'fiscal_period', 'frame', 'year'
    ]


def train_all_models_optimized(
    df=None,
    feature_windows=None,
    target_windows=None,
    model_types=None,
    train_split=DEFAULT_TRAIN_SPLIT,
    model_suffix=None,
    force_recalculate_features=False,
    models_dir=None,
    results_dir=None,
    log_file=None,
    data_file=None,
    train_start_date=None,
    train_end_date=None,
    verbose=True
):
    """
    Optimized training: Calculate features once, reuse for all models.
    Features and targets are cached to disk for resuming training.
    
    Parameters:
    -----------
    df : DataFrame, optional
        Stock data. If None, will load from data_file or default location.
    data_file : str, optional
        Path to stock data file (pickle or CSV). If None and df is None,
        uses default: {DATA_DIR}/sp500_stock_data_latest.pkl
    feature_windows : list, optional
        List of feature windows to train. If None, uses DEFAULT_FEATURE_WINDOWS.
        Example: [5, 10, 15] to train only 5d, 10d, 15d feature windows
    target_windows : list, optional
        List of target windows to train. If None, uses DEFAULT_TARGET_WINDOWS.
        Example: [5] to train only 5-day target window
    model_types : list, optional
        List of model types to train. If None, uses DEFAULT_MODEL_TYPES.
        Example: ['XGBoost'] to train only XGBoost models
    train_split : float, default=0.6
        Training/test split ratio (e.g., 0.6 = 60% train, 40% test)
    model_suffix : str, optional
        Suffix to add to model keys (e.g., '_2014_2021').
        Model files will be saved as: f{window}d_t{window}d_{model}{suffix}.pkl
    force_recalculate_features : bool
        If True, ignore cache and recalculate all features and targets
    models_dir : str, optional
        Custom models directory. If None, uses MODELS_DIR from config.
    results_dir : str, optional
        Custom results directory. If None, uses RESULTS_DIR from config.
    log_file : str, optional
        Custom log file path. If None, uses LOG_FILE from config.
    train_start_date : str or datetime, optional
        Explicit training start date. If provided, filters data to this date and ignores train_split.
        Example: '2012-01-01'
    train_end_date : str or datetime, optional
        Explicit training end date. If provided, filters data to this date and ignores train_split.
        Example: '2017-12-31'
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    dict : Training log with all trained models
    """
    # Set directories
    if models_dir is None:
        models_dir = MODELS_DIR
    if results_dir is None:
        results_dir = RESULTS_DIR
    if log_file is None:
        log_file = LOG_FILE
    
    # Load data if not provided
    if df is None:
        if data_file is None:
            data_file = os.path.join(DATA_DIR, 'sp500_stock_data_latest.pkl')
        
        if verbose:
            print(f"\n📂 Loading stock data from: {data_file}")
        
        if os.path.exists(data_file):
            if data_file.endswith('.pkl'):
                df = pd.read_pickle(data_file)
            elif data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
                df['date'] = pd.to_datetime(df['date'])
            else:
                raise ValueError(f"Unsupported file format: {data_file}")
            
            if verbose:
                print(f"   ✅ Loaded {len(df):,} records, {df['symbol'].nunique()} symbols")
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Please provide df parameter or ensure data file exists."
            )
    
    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(FEATURES_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Set defaults
    if feature_windows is None:
        feature_windows = DEFAULT_FEATURE_WINDOWS
    if target_windows is None:
        target_windows = DEFAULT_TARGET_WINDOWS
    if model_types is None:
        model_types = DEFAULT_MODEL_TYPES
    
    if verbose:
        print("\n" + "="*70)
        print("🚀 STARTING OPTIMIZED TRAINING PIPELINE")
        print("="*70)
        print(f"📊 Total models to train: {len(feature_windows) * len(target_windows) * len(model_types)}")
        print(f"   Feature windows: {feature_windows}")
        print(f"   Target windows: {target_windows}")
        print(f"   Model types: {model_types}")
        if model_suffix:
            print(f"   Model suffix: {model_suffix}")
        print(f"   Train split: {train_split*100:.0f}% train, {100-train_split*100:.0f}% test")
        print(f"\n💾 Feature cache directory: {FEATURES_CACHE_DIR}")
    
    # Load training log
    log = load_training_log()
    
    # Step 1: Pre-calculate ALL features (once per window, with caching)
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: PRE-CALCULATING FEATURES (with caching)")
        print("="*70)
    feature_dfs = precalculate_all_features(
        df,
        feature_windows=feature_windows,
        force_recalculate=force_recalculate_features
    )
    
    # Step 2: Pre-calculate ALL targets (once per window, with caching)
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: PRE-CALCULATING TARGETS (with caching)")
        print("="*70)
    target_series = precalculate_all_targets(
        df,
        target_windows=target_windows,
        force_recalculate=force_recalculate_features
    )
    
    # Step 3: Train all model combinations (reusing pre-calculated features)
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: TRAINING ALL MODEL COMBINATIONS")
        print("="*70)
    
    total_models = len(feature_windows) * len(target_windows) * len(model_types)
    trained_count = 0
    skipped_count = 0
    
    for feature_window in feature_windows:
        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 FEATURE WINDOW: {feature_window} days")
            print(f"{'='*70}")
        
        # Get pre-calculated features for this window
        feature_df = feature_dfs[feature_window]
        
        for target_window in target_windows:
            if verbose:
                print(f"\n  🎯 TARGET WINDOW: {target_window} days")
            
            # Get pre-calculated target for this window
            target = target_series[target_window]
            
            # Align features and target (same indices)
            combined = feature_df.copy()
            combined['target'] = target
            combined = combined.dropna(subset=['target'])
            # Filter by training date range if provided
            if train_start_date is not None or train_end_date is not None:
                if 'date' in combined.columns:
                    if train_start_date is not None:
                        combined = combined[combined['date'] >= pd.to_datetime(train_start_date)]
                    if train_end_date is not None:
                        combined = combined[combined['date'] <= pd.to_datetime(train_end_date)]
                    # When using explicit dates, use train_split parameter (or default to 0.9)
                    # This ensures we still have test data for evaluation
                    effective_train_split = train_split if train_split is not None else 0.9
                    if verbose:
                        print(f"     📅 Training date range: {train_start_date} to {train_end_date}")
                        print(f"     📊 Using {effective_train_split*100:.0f}% train / {100-effective_train_split*100:.0f}% validation split")
                else:
                    effective_train_split = train_split
            else:
                effective_train_split = train_split
            
            # CRITICAL FIX: Sort by date to ensure chronological order for time-based split
            # This fixes the issue where data was sorted by symbol, then date, causing
            # train/test split to be by symbol groups instead of chronologically
            if 'date' in combined.columns:
                combined = combined.sort_values('date').reset_index(drop=True)
                if verbose:
                    print(f"     📅 Sorted data chronologically: {combined['date'].min()} to {combined['date'].max()}")
            
            # Filter out non-feature columns
            exclude_cols = NON_FEATURE_COLS + ['target']
            valid_cols = [col for col in combined.columns if col not in exclude_cols]
            
            # DIAGNOSTIC: Log feature counts at each stage
            if verbose:
                print(f"     📊 Features after creation: {len(combined.columns)} columns")
                excluded_found = [col for col in combined.columns if col in NON_FEATURE_COLS]
                if excluded_found:
                    print(f"     🔧 Excluded non-feature columns: {excluded_found}")
                print(f"     📊 Features after excluding non-feature cols: {len(valid_cols)}")
            
            # Select only numeric columns
            X = combined[valid_cols].select_dtypes(include=['number'])
            y = combined['target']
            
            if verbose:
                non_numeric = [col for col in valid_cols if col not in X.columns]
                if non_numeric:
                    print(f"     🔧 Excluded non-numeric columns: {non_numeric}")
                print(f"     📊 Features after selecting numeric: {len(X.columns)}")
                print(f"     📊 Training data: {len(X):,} samples, {len(X.columns)} features")
            
            for model_name in model_types:
                # Create model key with optional suffix
                model_key = f"f{feature_window}d_t{target_window}d_{model_name}"
                if model_suffix:
                    model_key += model_suffix
                
                # Check if already trained
                if model_key in log:
                    acc = log[model_key].get('accuracy', 'N/A')
                    acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else acc
                    if verbose:
                        print(f"     ⏭️  {model_name}: Already trained (accuracy: {acc_str})")
                    skipped_count += 1
                    continue
                
                if verbose:
                    print(f"\n     🔧 Training {model_name}...")
                
                try:
                    # Train the model
                    model, metrics = train_single_model_optimized(
                        X, y, model_name, feature_window, target_window,
                        train_split=effective_train_split
                    )
                    
                    # Save model
                    model_path = os.path.join(models_dir, f"{model_key}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    # Save metrics
                    metrics_path = os.path.join(results_dir, f"{model_key}_metrics.json")
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    # Update log
                    log[model_key] = {
                        'feature_window': feature_window,
                        'target_window': target_window,
                        'model_type': model_name,
                        'accuracy': metrics.get('accuracy', 0),
                        'trained_at': pd.Timestamp.now().isoformat(),
                        'model_path': model_path,
                        'metrics_path': metrics_path,
                        'train_split': train_split,
                        'model_suffix': model_suffix
                    }
                    save_training_log(log)
                    
                    trained_count += 1
                    acc = metrics.get('accuracy', 0)
                    f1 = metrics.get('f1_weighted', metrics.get('f1', 0))
                    if verbose:
                        print(f"     ✅ {model_name}: accuracy={acc:.3f}, f1={f1:.3f}")
                    
                except Exception as e:
                    if verbose:
                        print(f"     ❌ {model_name} failed: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    continue
            
            if verbose:
                progress = (trained_count + skipped_count) / total_models * 100
                print(f"\n  📊 Progress: {trained_count + skipped_count}/{total_models} ({progress:.1f}%)")
                print(f"     ✅ Trained: {trained_count}, ⏭️  Skipped: {skipped_count}")
    
    if verbose:
        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print(f"✅ Trained: {trained_count} models")
        print(f"⏭️  Skipped: {skipped_count} models (already trained)")
        print(f"📊 Total: {trained_count + skipped_count}/{total_models} models")
        print(f"\n💾 Feature cache saved to: {FEATURES_CACHE_DIR}")
        print(f"   (Will be reused on next run if data hasn't changed)")
    
    return log

