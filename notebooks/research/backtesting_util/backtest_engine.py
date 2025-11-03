"""
Core ultra-optimized backtesting engine.

Implements the main backtesting function with all timing fixes:
1. Momentum calculation uses previous trading day
2. Entry prices use current day's OPEN
3. Exit dates use trading days (not calendar days)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from .helpers import (
    get_previous_trading_day,
    get_next_trading_day,
    get_price_fast,
    get_price_open,
    get_top_momentum_stocks_fast,
    get_bottom_momentum_stocks_fast,
    create_momentum_matrix
)
from .config import (
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_TOP_PERCENT,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_CASH_BUFFER
)


def backtest_model_ultra_optimized(
    model,
    features_df,
    stock_data,
    target_window,
    start_date,
    end_date,
    initial_capital=DEFAULT_INITIAL_CAPITAL,
    transaction_cost=DEFAULT_TRANSACTION_COST,
    top_percent=DEFAULT_TOP_PERCENT,
    max_positions=DEFAULT_MAX_POSITIONS,
    cash_buffer=DEFAULT_CASH_BUFFER,
    verbose=False,
    model_name="Model",
    trade_when_positions_zero=True
):
    """
    ULTRA-OPTIMIZED backtesting: Trade daily with configurable stock limit
    
    ⚠️  TIMING FIX: Features for date T use data up to T-1
    - Features are already calculated with shifted prices
    - Trading decisions use features for current date
    - Momentum calculation uses previous trading day (from actual data)
    - Entry prices use current day's OPEN price
    - Exit dates use trading days (k trading days ahead)
    
    Parameters:
    -----------
    model : sklearn/XGBoost model
        Trained model for predictions
    features_df : pd.DataFrame
        Features DataFrame with columns ['symbol', 'date', ...features]
    stock_data : pd.DataFrame
        Stock data with columns ['symbol', 'date', 'open', 'close', 'high', 'low', 'volume']
    target_window : int
        Target window size (e.g., 5 for 5-day momentum)
    start_date : datetime or str
        Start date for backtesting
    end_date : datetime or str
        End date for backtesting
    initial_capital : float
        Initial capital
    transaction_cost : float
        Transaction cost as fraction (e.g., 0.003 = 0.3%)
    top_percent : float
        Top/bottom percent of stocks to select (e.g., 0.2 = 20%)
    max_positions : int
        Maximum number of positions to hold
    cash_buffer : float
        Cash buffer fraction (e.g., 0.05 = 5%)
    verbose : bool
        Whether to print progress
    model_name : str
        Model name for progress display
    trade_when_positions_zero : bool
        If True, only trade when positions == 0 (else trade when positions < max_positions)
        
    Returns:
    --------
    portfolio_df : pd.DataFrame
        Daily portfolio values with columns ['date', 'cash', 'positions_value', 
        'total_value', 'num_positions', 'momentum_trades', 'reversal_trades', 
        'do_nothing_days', 'daily_return', 'cumulative_return']
    trades_df : pd.DataFrame
        All executed trades with columns ['symbol', 'entry_date', 'exit_date', 
        'entry_price', 'exit_price', 'shares', 'gross_pnl', 'entry_cost', 
        'exit_cost', 'total_costs', 'net_pnl', 'return_pct', 'strategy_used']
    decision_stats : dict
        Statistics about trading decisions
    """
    
    # Convert dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Get all trading days from stock_data (for previous day lookup)
    all_trading_dates = sorted(stock_data['date'].unique())
    
    # Initialize tracking
    capital = initial_capital
    positions = {}
    daily_values = []
    trades_log = []
    
    # Track decision counts
    momentum_trades = 0
    reversal_trades = 0
    do_nothing_days = 0
    
    # Track optimization stats
    skipped_calculations = 0
    total_calculations = 0
    
    # Get trading days
    trading_days = sorted(features_df['date'].unique())
    trading_days = [pd.to_datetime(d) for d in trading_days if start_date <= pd.to_datetime(d) <= end_date]
    
    # Progress bar for trading days
    if verbose:
        print(f"📊 {model_name}: Backtesting {len(trading_days)} trading days...")
        iterator = tqdm(trading_days, desc=f"   {model_name}")
    else:
        iterator = trading_days
    
    # OPTIMIZATION 1: Pre-calculate momentum matrices
    momentum_matrices = {}
    momentum_matrix = create_momentum_matrix(stock_data, target_window)
    momentum_matrices[target_window] = momentum_matrix
    
    # OPTIMIZATION 2: Pre-filter features for test period
    test_features = features_df[features_df['date'] >= start_date].copy()
    feature_cols = [col for col in test_features.columns 
                   if col not in ['symbol', 'date', 'target']]
    
    # Walk through each day
    for current_date in iterator:
        
        # STEP 1: CHECK FOR POSITION EXITS
        positions_to_close = []
        for symbol, pos in positions.items():
            if current_date >= pos['exit_date']:
                exit_price = get_price_fast(stock_data, symbol, current_date)
                
                if exit_price is not None:
                    shares = pos['shares']
                    entry_value = pos['entry_price'] * shares
                    exit_value = exit_price * shares
                    pnl = exit_value - entry_value
                    exit_cost = exit_value * transaction_cost
                    net_pnl = pnl - exit_cost
                    
                    capital += exit_value - exit_cost
                    
                    trades_log.append({
                        'symbol': symbol,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': shares,
                        'entry_value': entry_value,
                        'exit_value': exit_value,
                        'gross_pnl': pnl,
                        'entry_cost': pos['entry_cost'],
                        'exit_cost': exit_cost,
                        'total_costs': pos['entry_cost'] + exit_cost,
                        'net_pnl': net_pnl,
                        'return_pct': (exit_price - pos['entry_price']) / pos['entry_price'],
                        'strategy_used': pos.get('strategy_used', 'unknown')
                    })
                    
                    positions_to_close.append(symbol)
        
        for symbol in positions_to_close:
            del positions[symbol]
        
        # STEP 2: ULTRA-OPTIMIZED DAILY TRADING LOGIC
        total_calculations += 1
        
        # Check if we can trade
        if trade_when_positions_zero:
            can_trade = (len(positions) == 0 and capital > initial_capital * 0.1)
        else:
            can_trade = (len(positions) < max_positions and capital > initial_capital * 0.1)
        
        if not can_trade:
            skipped_calculations += 1
            # Skip all trade calculations - just update portfolio value
            positions_value = 0
            for symbol, pos in positions.items():
                current_price = get_price_fast(stock_data, symbol, current_date)
                if current_price is not None:
                    positions_value += current_price * pos['shares']
            
            total_portfolio_value = capital + positions_value
            
            daily_values.append({
                'date': current_date,
                'cash': capital,
                'positions_value': positions_value,
                'total_value': total_portfolio_value,
                'num_positions': len(positions),
                'momentum_trades': momentum_trades,
                'reversal_trades': reversal_trades,
                'do_nothing_days': do_nothing_days
            })
            continue
        
        # Only do trade calculations if we can actually trade
        current_features = test_features[test_features['date'] == current_date]
        
        if len(current_features) > 0:
            X = current_features[feature_cols]
            symbols = current_features['symbol'].values
            
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            try:
                # STEP 3: AGGREGATE DECISION
                predictions = model.predict(X)
                prediction_counts = np.bincount(predictions, minlength=3)
                aggregate_signal = np.argmax(prediction_counts)
                
                # Track decisions
                if aggregate_signal == 0:
                    momentum_trades += 1
                elif aggregate_signal == 1:
                    reversal_trades += 1
                else:
                    do_nothing_days += 1
                
                # STEP 4: EXECUTE TRADING DECISION
                if aggregate_signal == 0:  # MOMENTUM
                    # ⚠️  TIMING FIX: Use previous trading day's momentum (from actual data)
                    prev_trading_day = get_previous_trading_day(current_date, all_trading_dates)
                    
                    if prev_trading_day is not None:
                        stocks_to_buy = get_top_momentum_stocks_fast(
                            prev_trading_day, target_window, top_percent, momentum_matrices, symbols
                        )
                        
                        # Filter out stocks we already own
                        stocks_to_buy = [s for s in stocks_to_buy if s not in positions]
                        stocks_to_buy = stocks_to_buy[:max_positions - len(positions)]
                        
                        if len(stocks_to_buy) > 0:
                            available_capital = capital * (1 - cash_buffer)
                            capital_per_stock = available_capital / len(stocks_to_buy)
                            
                            for symbol in stocks_to_buy:
                                # ⚠️  TIMING FIX: Use current day's OPEN price for entry
                                entry_price = get_price_open(stock_data, symbol, current_date)
                                
                                if entry_price is not None and entry_price > 0:
                                    shares = int(capital_per_stock / entry_price)
                                    
                                    if shares > 0:
                                        entry_value = entry_price * shares
                                        entry_cost = entry_value * transaction_cost
                                        total_cost = entry_value + entry_cost
                                        
                                        if total_cost <= capital:
                                            capital -= total_cost
                                            
                                            # ⚠️  TIMING FIX: Exit date uses trading days (k ahead)
                                            exit_date = get_next_trading_day(current_date, all_trading_dates, k=target_window)
                                            
                                            positions[symbol] = {
                                                'entry_date': current_date,
                                                'entry_price': entry_price,
                                                'shares': shares,
                                                'exit_date': exit_date,
                                                'entry_cost': entry_cost,
                                                'strategy_used': 'momentum'
                                            }
                
                elif aggregate_signal == 1:  # REVERSAL
                    # ⚠️  TIMING FIX: Use previous trading day's momentum (from actual data)
                    prev_trading_day = get_previous_trading_day(current_date, all_trading_dates)
                    
                    if prev_trading_day is not None:
                        stocks_to_buy = get_bottom_momentum_stocks_fast(
                            prev_trading_day, target_window, top_percent, momentum_matrices, symbols
                        )
                        
                        # Filter out stocks we already own
                        stocks_to_buy = [s for s in stocks_to_buy if s not in positions]
                        stocks_to_buy = stocks_to_buy[:max_positions - len(positions)]
                        
                        if len(stocks_to_buy) > 0:
                            available_capital = capital * (1 - cash_buffer)
                            capital_per_stock = available_capital / len(stocks_to_buy)
                            
                            for symbol in stocks_to_buy:
                                # ⚠️  TIMING FIX: Use current day's OPEN price for entry
                                entry_price = get_price_open(stock_data, symbol, current_date)
                                
                                if entry_price is not None and entry_price > 0:
                                    shares = int(capital_per_stock / entry_price)
                                    
                                    if shares > 0:
                                        entry_value = entry_price * shares
                                        entry_cost = entry_value * transaction_cost
                                        total_cost = entry_value + entry_cost
                                        
                                        if total_cost <= capital:
                                            capital -= total_cost
                                            
                                            # ⚠️  TIMING FIX: Exit date uses trading days (k ahead)
                                            exit_date = get_next_trading_day(current_date, all_trading_dates, k=target_window)
                                            
                                            positions[symbol] = {
                                                'entry_date': current_date,
                                                'entry_price': entry_price,
                                                'shares': shares,
                                                'exit_date': exit_date,
                                                'entry_cost': entry_cost,
                                                'strategy_used': 'reversal'
                                            }
                
                # Class 2: DO NOTHING - stay in cash
                
            except Exception as e:
                if verbose:
                    print(f"Prediction error on {current_date}: {e}")
        
        # STEP 5: CALCULATE DAILY PORTFOLIO VALUE
        positions_value = 0
        for symbol, pos in positions.items():
            current_price = get_price_fast(stock_data, symbol, current_date)
            if current_price is not None:
                positions_value += current_price * pos['shares']
        
        total_portfolio_value = capital + positions_value
        
        daily_values.append({
            'date': current_date,
            'cash': capital,
            'positions_value': positions_value,
            'total_value': total_portfolio_value,
            'num_positions': len(positions),
            'momentum_trades': momentum_trades,
            'reversal_trades': reversal_trades,
            'do_nothing_days': do_nothing_days
        })
    
    # STEP 6: CALCULATE METRICS
    portfolio_df = pd.DataFrame(daily_values)
    trades_df = pd.DataFrame(trades_log) if len(trades_log) > 0 else pd.DataFrame()
    
    portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
    portfolio_df['cumulative_return'] = (portfolio_df['total_value'] / initial_capital - 1)
    
    return portfolio_df, trades_df, {
        'total_momentum_trades': momentum_trades,
        'total_reversal_trades': reversal_trades,
        'total_do_nothing_days': do_nothing_days,
        'momentum_reversal_ratio': momentum_trades / max(1, reversal_trades),
        'skipped_calculations': skipped_calculations,
        'total_calculations': total_calculations,
        'efficiency_gain': skipped_calculations/total_calculations*100 if total_calculations > 0 else 0
    }
