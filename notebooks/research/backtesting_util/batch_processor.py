"""
Batch processing functions for running backtests on multiple models.
"""

import pandas as pd
import os
from tqdm import tqdm

from .backtest_engine import backtest_model_ultra_optimized
from .data_loader import parse_model_key
from .config import RESULTS_DIR, DEFAULT_INITIAL_CAPITAL


def run_all_models(
    all_models,
    features_cache,
    stock_data,
    test_start_date,
    test_end_date,
    results_dir=None,
    initial_capital=DEFAULT_INITIAL_CAPITAL,
    max_positions=20,
    verbose=True,
    resume=True,
    trade_when_positions_zero=True
):
    """
    Run backtesting on all models.
    
    Parameters:
    -----------
    all_models : dict
        Dictionary mapping model_key -> model object
    features_cache : dict
        Dictionary mapping window -> features DataFrame
    stock_data : pd.DataFrame
        Stock data
    test_start_date : datetime
        Test period start date
    test_end_date : datetime
        Test period end date
    results_dir : str, optional
        Directory to save results. If None, uses default from config.
    initial_capital : float
        Initial capital
    max_positions : int
        Maximum number of positions
    verbose : bool
        Whether to print progress
    resume : bool
        Whether to skip already-completed models
    trade_when_positions_zero : bool
        If True, only trade when positions == 0
        
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with results for all models
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    print(f"✅ Results directory: {results_dir}")
    
    # Track results
    results_summary = []
    successful_models = 0
    failed_models = 0
    
    # Resume support - check existing files
    completed_models = set()
    if resume:
        summary_file = os.path.join(results_dir, 'ultra_optimized_results_summary.csv')
        if os.path.exists(summary_file):
            try:
                prev_summary = pd.read_csv(summary_file)
                if 'model_key' in prev_summary.columns:
                    completed_models.update(prev_summary['model_key'].astype(str).tolist())
            except Exception as e:
                print(f"⚠️  Could not read existing summary: {e}")
        
        # Also check for existing portfolio files (support both naming conventions)
        try:
            for fn in os.listdir(results_dir):
                if fn.endswith('_portfolio_ultra_optimized.csv'):
                    completed_models.add(fn.replace('_portfolio_ultra_optimized.csv', ''))
                elif fn.endswith('_portfolio.csv'):
                    # Also check for simpler naming convention
                    completed_models.add(fn.replace('_portfolio.csv', ''))
        except FileNotFoundError:
            pass
        
        if completed_models:
            print(f"🔁 Resume mode: Found {len(completed_models)} completed models to skip")
    
    print(f"\n🚀 Running {len(all_models)} models:")
    print(f"   📊 Total: {len(all_models)} models")
    
    # Run all models with progress bar
    for model_key, model in tqdm(all_models.items(), desc="Processing models"):
        
        # Skip if already completed
        if resume and model_key in completed_models:
            print(f"\n⏭️  Skipping {model_key} (already completed)")
            continue
        
        if verbose:
            print(f"\n🔄 Processing {model_key}...")
        
        # Determine model type and extract parameters
        feature_window, target_window, model_type = parse_model_key(model_key)
        
        # Get features for this window
        if feature_window in features_cache:
            features_df = features_cache[feature_window]
            
            # Run ultra-optimized backtesting
            try:
                portfolio_df, trades_df, decision_stats = backtest_model_ultra_optimized(
                    model=model,
                    features_df=features_df,
                    stock_data=stock_data,
                    target_window=target_window,
                    start_date=test_start_date,
                    end_date=test_end_date,
                    initial_capital=initial_capital,
                    max_positions=max_positions,
                    verbose=verbose,
                    model_name=model_key,
                    trade_when_positions_zero=trade_when_positions_zero
                )
                
                # Calculate final metrics
                final_value = portfolio_df['total_value'].iloc[-1]
                total_return = (final_value - initial_capital) / initial_capital
                
                # Save results (preserve model suffix in filename)
                # Use simpler naming if results_dir indicates it's a custom directory
                if '2014_2021_models_backtest' in results_dir or 'portfolios' in results_dir:
                    # Match original format: {model_key}_portfolio.csv and {model_key}_trades.csv
                    portfolio_file = os.path.join(results_dir, f"{model_key}_portfolio.csv")
                    trades_file = os.path.join(results_dir, f"{model_key}_trades.csv")
                else:
                    # Use ultra_optimized suffix for default results directory
                    portfolio_file = os.path.join(results_dir, f"{model_key}_portfolio_ultra_optimized.csv")
                    trades_file = os.path.join(results_dir, f"{model_key}_trades_ultra_optimized.csv")
                
                portfolio_df.to_csv(portfolio_file, index=False)
                if not trades_df.empty:
                    trades_df.to_csv(trades_file, index=False)
                
                # Store summary
                results_summary.append({
                    'model_key': model_key,
                    'model_type': model_type,
                    'feature_window': feature_window,
                    'target_window': target_window,
                    'final_value': final_value,
                    'total_return': total_return,
                    'total_trades': len(trades_df),
                    'momentum_trades': decision_stats['total_momentum_trades'],
                    'reversal_trades': decision_stats['total_reversal_trades'],
                    'do_nothing_days': decision_stats['total_do_nothing_days'],
                    'momentum_reversal_ratio': decision_stats['momentum_reversal_ratio'],
                    'efficiency_gain': decision_stats['efficiency_gain']
                })
                
                successful_models += 1
                if verbose:
                    print(f"   ✅ {model_key}: ${final_value:,.2f} ({total_return:.1%}) | {len(trades_df)} trades | {decision_stats['efficiency_gain']:.1f}% efficiency")
                
            except Exception as e:
                print(f"   ❌ {model_key} failed: {str(e)}")
                failed_models += 1
        else:
            print(f"   ❌ Features for window {feature_window} not found")
            failed_models += 1
    
    # Save summary results
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        # Use simpler naming for custom directories
        if '2014_2021_models_backtest' in results_dir or 'portfolios' in results_dir:
            # If results_dir is portfolios/, save summary to parent directory
            if results_dir.endswith('portfolios') or results_dir.endswith('portfolios/'):
                summary_file = os.path.join(os.path.dirname(results_dir), 'backtesting_results_2022_2024.csv')
            else:
                summary_file = os.path.join(results_dir, 'backtesting_results_2022_2024.csv')
        else:
            summary_file = os.path.join(results_dir, 'ultra_optimized_results_summary.csv')
        
        # Merge with prior summary if present (for resume mode)
        if resume and os.path.exists(summary_file):
            try:
                prev_df = pd.read_csv(summary_file)
                results_df = pd.concat([prev_df, results_df], ignore_index=True)
                # Drop duplicates by model_key, keep the latest row
                if 'model_key' in results_df.columns:
                    results_df = results_df.drop_duplicates(subset=['model_key'], keep='last')
            except Exception as e:
                print(f"⚠️  Could not merge with existing summary, overwriting: {e}")
        
        results_df.to_csv(summary_file, index=False)
        
        print(f"\n✅ ULTRA-OPTIMIZED backtesting complete!")
        print(f"   📊 Successful: {successful_models}/{len(all_models)} models")
        print(f"   ❌ Failed: {failed_models}/{len(all_models)} models")
        print(f"   📁 Results saved to: {results_dir}")
        print(f"   📋 Summary saved to: {summary_file}")
        
        # Show top performers
        if len(results_df) > 0:
            print(f"\n🏆 TOP 10 PERFORMERS:")
            top_performers = results_df.nlargest(10, 'total_return')
            for idx, row in top_performers.iterrows():
                print(f"   {row['model_key']}: {row['total_return']:.1%} | {row['total_trades']} trades | {row['momentum_reversal_ratio']:.2f} ratio")
        
        return results_df
    else:
        print(f"\n⚠️  No results to save")
        return pd.DataFrame()
