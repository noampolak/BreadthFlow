"""
Main orchestrator function for ultra-optimized backtesting.

Provides a single entry point to run the complete backtesting pipeline.
"""

from .data_loader import (
    load_models_from_log,
    load_models_from_directory,
    load_stock_data,
    load_features_cache,
    get_test_period
)
from .batch_processor import run_all_models
from .config import DEFAULT_TEST_SPLIT


def run_ultra_optimized_backtesting(
    data_dir=None,
    test_split=DEFAULT_TEST_SPLIT,
    test_start_date=None,
    test_end_date=None,
    max_positions=20,
    verbose=True,
    resume=True,
    trade_when_positions_zero=True,
    include_randomforest=True,
    include_xgboost=True,
    model_suffix=None,
    results_dir=None
):
    """
    Run the complete ultra-optimized backtesting pipeline.
    
    Parameters:
    -----------
    data_dir : str, optional
        Override default data directory
    test_split : float
        Training/test split ratio (used only if test_start_date is None)
    test_start_date : datetime or str, optional
        Explicit test start date. If provided, overrides test_split.
    test_end_date : datetime or str, optional
        Explicit test end date. If None, uses last date in stock_data.
    max_positions : int
        Maximum number of positions
    verbose : bool
        Whether to print progress
    resume : bool
        Whether to skip already-completed models
    trade_when_positions_zero : bool
        If True, only trade when positions == 0
    include_randomforest : bool
        Whether to include RandomForest models
    include_xgboost : bool
        Whether to include XGBoost models
    model_suffix : str, optional
        Filter models by suffix (e.g., '_2014_2021'). If provided, also tries to load
        directly from models directory if not in training log.
    results_dir : str, optional
        Custom results directory. If None, uses default from config.
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with results
    """
    print("🚀 ULTRA-OPTIMIZED BACKTESTING: ALL RANDOMFOREST & XGBOOST MODELS")
    print("="*70)
    print("⚠️  TIMING FIX: Using previous TRADING DAY (not calendar day)")
    print("="*70)
    
    # Step 1: Load models
    print(f"\nStep 1: Loading models...")
    if model_suffix:
        print(f"   Filtering models with suffix: {model_suffix}")
        # Try loading from log first
        models_dict = load_models_from_log(
            include_randomforest=include_randomforest,
            include_xgboost=include_xgboost,
            model_suffix=model_suffix
        )
        all_models = models_dict['all_models']
        
        # If no models found in log, try loading directly from directory
        if not all_models:
            print(f"   No models found in log, trying to load directly from directory...")
            models_dict = load_models_from_directory(
                model_suffix=model_suffix
            )
            all_models = models_dict['all_models']
    else:
        models_dict = load_models_from_log(
            include_randomforest=include_randomforest,
            include_xgboost=include_xgboost
        )
        all_models = models_dict['all_models']
    
    if not all_models:
        print("❌ No models loaded!")
        return None
    
    # Step 2: Load stock data
    print(f"\nStep 2: Loading stock data...")
    stock_data = load_stock_data()
    
    if stock_data is None:
        print("❌ Cannot run backtesting - missing stock data")
        return None
    
    # Step 3: Load features cache
    print(f"\nStep 3: Loading features cache...")
    features_cache = load_features_cache()
    
    if not features_cache:
        print("❌ Cannot run backtesting - missing features")
        return None
    
    # Step 4: Define test period
    print(f"\nStep 4: Defining test period...")
    test_start, test_end = get_test_period(
        stock_data, 
        test_split=test_split,
        test_start_date=test_start_date,
        test_end_date=test_end_date
    )
    
    # Step 5: Run all models
    print(f"\nStep 5: Running all models...")
    results_df = run_all_models(
        all_models=all_models,
        features_cache=features_cache,
        stock_data=stock_data,
        test_start_date=test_start,
        test_end_date=test_end,
        results_dir=results_dir,
        max_positions=max_positions,
        verbose=verbose,
        resume=resume,
        trade_when_positions_zero=trade_when_positions_zero
    )
    
    print(f"\n✅ ALL MODELS PROCESSING COMPLETE!")
    print("⚠️  All timing fixes applied - using previous TRADING DAY from actual data")
    
    return results_df


if __name__ == "__main__":
    # Run when executed directly
    results = run_ultra_optimized_backtesting()
    if results is not None:
        print(f"\n✅ Backtesting complete! Results: {len(results)} models")
