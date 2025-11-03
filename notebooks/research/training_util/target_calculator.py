"""
Target calculation functions for multi-class training.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from .cache_manager import load_target_from_cache, save_target_to_cache
from .config import DEFAULT_TARGET_WINDOWS


def calculate_multiclass_target(df, target_window, top_percent=0.2):
    """
    Calculate multi-class target: Momentum (0), Reversal (1), or Do Nothing (2)
    
    Parameters:
    -----------
    df : DataFrame
        Stock data with price information
    target_window : int
        Target prediction window in days
    top_percent : float
        Percentage of top stocks to consider (default 0.2 for top 20%)
    
    Returns:
    --------
    target : Series
        Multi-class target (0=Momentum, 1=Reversal, 2=Do Nothing)
    """
    
    # Transaction costs: 0.03% commission + 0.1% tax + buffer ≈ 0.3%
    TRANSACTION_COST = 0.003  # 0.3% per round-trip trade
    
    print(f"    🎯 Calculating multi-class target for {target_window}-day window...")
    
    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date']).copy()
    
    # Calculate momentum scores (past performance)
    momentum_scores = df.groupby('symbol')['close'].pct_change(target_window)
    
    # Calculate reversal scores (contrarian - opposite of momentum)
    reversal_scores = -momentum_scores
    
    # Calculate future returns (what actually happens)
    future_returns = df.groupby('symbol')['close'].pct_change(target_window).shift(-target_window)
    
    # Initialize target array
    target = pd.Series(2, index=df.index)  # Default: Do Nothing
    
    # For each date, calculate yields and assign target
    dates = df['date'].unique()
    print(f"    📅 Processing {len(dates):,} dates...")
    
    momentum_count = 0
    reversal_count = 0
    nothing_count = 0
    
    for date in tqdm(dates, desc="    Calculating targets"):
        date_mask = df['date'] == date
        date_indices = df[date_mask].index
        
        # Get stocks available on this date
        date_momentum = momentum_scores[date_mask].dropna()
        date_reversal = reversal_scores[date_mask].dropna()
        date_future_returns = future_returns[date_mask].dropna()
        
        if len(date_momentum) == 0 or len(date_future_returns) == 0:
            continue
        
        # Get top momentum stocks (highest past returns)
        momentum_threshold = date_momentum.quantile(1 - top_percent)
        top_momentum_stocks = date_momentum[date_momentum >= momentum_threshold].index
        
        # Calculate momentum yield (average future return of top momentum stocks)
        momentum_future_returns = date_future_returns[date_future_returns.index.isin(top_momentum_stocks)]
        momentum_yield = momentum_future_returns.mean() if len(momentum_future_returns) > 0 else 0
        
        # Get top reversal stocks (lowest past returns, expecting reversal)
        reversal_threshold = date_reversal.quantile(1 - top_percent)
        top_reversal_stocks = date_reversal[date_reversal >= reversal_threshold].index
        
        # Calculate reversal yield (average future return of top reversal stocks)
        reversal_future_returns = date_future_returns[date_future_returns.index.isin(top_reversal_stocks)]
        reversal_yield = reversal_future_returns.mean() if len(reversal_future_returns) > 0 else 0
        
        # Assign target class based on best strategy
        if momentum_yield > reversal_yield and momentum_yield > TRANSACTION_COST:
            target[date_indices] = 0  # Momentum
            momentum_count += len(date_indices)
        elif reversal_yield > momentum_yield and reversal_yield > TRANSACTION_COST:
            target[date_indices] = 1  # Reversal
            reversal_count += len(date_indices)
        else:
            target[date_indices] = 2  # Do Nothing
            nothing_count += len(date_indices)
    
    # Print distribution
    total = momentum_count + reversal_count + nothing_count
    if total > 0:
        print(f"\n    ✅ Target distribution for {target_window}-day window:")
        print(f"       🚀 Momentum: {momentum_count:,} ({momentum_count/total*100:.1f}%)")
        print(f"       🔄 Reversal: {reversal_count:,} ({reversal_count/total*100:.1f}%)")
        print(f"       ⏸️  Do Nothing: {nothing_count:,} ({nothing_count/total*100:.1f}%)")
    
    return target


def precalculate_all_targets(df, target_windows=None, force_recalculate=False):
    """
    Pre-calculate targets for all windows (done once!).
    If cached targets exist, load them instead of recalculating.
    
    Parameters:
    -----------
    df : DataFrame
        Stock data
    target_windows : list, optional
        List of target windows to calculate. If None, uses DEFAULT_TARGET_WINDOWS.
    force_recalculate : bool
        If True, ignore cache and recalculate all targets
    
    Returns:
    --------
    dict : Dictionary mapping window -> target Series
    """
    if target_windows is None:
        target_windows = DEFAULT_TARGET_WINDOWS
    
    print("\n" + "="*70)
    print("🎯 PRE-CALCULATING TARGETS FOR ALL WINDOWS")
    print("="*70)
    
    if force_recalculate:
        print("⚠️  Force recalculate enabled - ignoring cache")
    
    target_series = {}
    
    for window in target_windows:
        print(f"\n🎯 Window {window} days:")
        
        # Try to load from cache first
        if not force_recalculate:
            cached_target = load_target_from_cache(window)
            if cached_target is not None:
                target_series[window] = cached_target
                
                # Show distribution
                counts = cached_target.value_counts().sort_index()
                print(f"   📊 Distribution:")
                print(f"      Class 0 (Momentum): {counts.get(0, 0):,} samples")
                print(f"      Class 1 (Reversal): {counts.get(1, 0):,} samples")
                print(f"      Class 2 (Do Nothing): {counts.get(2, 0):,} samples")
                continue
        
        # Calculate target if not cached
        print(f"   🔧 Calculating target for {window}-day window...")
        target = calculate_multiclass_target(df, window)
        target_series[window] = target
        
        # Show distribution
        counts = target.value_counts().sort_index()
        print(f"   📊 Distribution:")
        print(f"      Class 0 (Momentum): {counts.get(0, 0):,} samples")
        print(f"      Class 1 (Reversal): {counts.get(1, 0):,} samples")
        print(f"      Class 2 (Do Nothing): {counts.get(2, 0):,} samples")
        
        # Save to cache for future runs
        save_target_to_cache(target, window)
    
    print("\n" + "="*70)
    print(f"✅ TARGETS READY FOR {len(target_windows)} WINDOWS")
    print("="*70)
    
    return target_series
