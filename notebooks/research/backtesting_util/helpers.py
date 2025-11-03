"""
Helper functions for ultra-optimized backtesting.

Includes timing fixes and utility functions for trading day calculations,
price lookups, and momentum stock selection.
"""

import pandas as pd
import numpy as np


def get_previous_trading_day(current_date, trading_dates_list):
    """
    ⚠️  TIMING FIX: Get the previous trading day from actual data
    This ensures we use a real trading day (not calendar day)
    
    Parameters:
    -----------
    current_date : datetime
        Current trading day
    trading_dates_list : list
        Sorted list of all trading days
    
    Returns:
    --------
    datetime or None
        Previous trading day, or None if not found
    """
    current_date = pd.to_datetime(current_date)
    trading_dates = sorted([pd.to_datetime(d) for d in trading_dates_list])
    
    # Find index of current_date
    try:
        current_idx = trading_dates.index(current_date)
        if current_idx > 0:
            return trading_dates[current_idx - 1]
        else:
            return None  # No previous trading day
    except ValueError:
        # current_date not in list, find closest previous one
        prev_date = None
        for date in reversed(trading_dates):
            if date < current_date:
                prev_date = date
                break
        return prev_date


def get_next_trading_day(current_date, trading_dates, k=1):
    """
    Get k trading days ahead.
    
    Parameters:
    -----------
    current_date : datetime
        Current trading day
    trading_dates : list
        Sorted list of all trading days
    k : int
        Number of trading days ahead
        
    Returns:
    --------
    datetime
        The k-th trading day ahead, or the last available date if beyond range
    """
    dates = [pd.to_datetime(x) for x in sorted(trading_dates)]
    try:
        i = dates.index(pd.to_datetime(current_date))
        j = i + k
        return dates[j] if j < len(dates) else dates[-1]
    except (ValueError, IndexError):
        return None


def get_price_fast(stock_data, symbol, date):
    """
    Fast price lookup using pre-indexed data.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock data with columns ['symbol', 'date', 'close', 'open']
    symbol : str
        Stock symbol
    date : datetime
        Trading date
        
    Returns:
    --------
    float or None
        Price (close) at the given date, or None if not found
    """
    try:
        price = stock_data[(stock_data['symbol'] == symbol) & 
                          (stock_data['date'] == date)]['close'].values[0]
        return price
    except (KeyError, IndexError):
        return None


def get_price_open(stock_data, symbol, date):
    """
    Get open price for a symbol on a given date.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock data with columns ['symbol', 'date', 'open']
    symbol : str
        Stock symbol
    date : datetime
        Trading date
        
    Returns:
    --------
    float or None
        Open price at the given date, or None if not found
    """
    try:
        price = stock_data[(stock_data['symbol'] == symbol) & 
                          (stock_data['date'] == date)]['open'].values[0]
        return price
    except (KeyError, IndexError):
        return None


def get_top_momentum_stocks_fast(date, window, top_percent, momentum_matrices, symbols=None):
    """
    Ultra-fast top momentum stocks selection.
    
    Parameters:
    -----------
    date : datetime
        Trading date for momentum calculation
    window : int
        Momentum window size
    top_percent : float
        Top percentage of stocks to select (e.g., 0.2 for top 20%)
    momentum_matrices : dict
        Dictionary mapping window -> momentum DataFrame (index=date, columns=symbols)
    symbols : list, optional
        List of symbols to filter by (if None, uses all symbols in momentum matrix)
        
    Returns:
    --------
    list
        List of top momentum stock symbols
    """
    try:
        momentum_scores = momentum_matrices[window].loc[date].dropna()
        if len(momentum_scores) == 0:
            return []
        
        n_top = max(1, int(len(momentum_scores) * top_percent))
        top_stocks = momentum_scores.nlargest(n_top).index.tolist()
        return top_stocks
    except (KeyError, IndexError):
        return []


def get_bottom_momentum_stocks_fast(date, window, top_percent, momentum_matrices, symbols=None):
    """
    Ultra-fast bottom momentum stocks selection (for reversal strategy).
    
    Parameters:
    -----------
    date : datetime
        Trading date for momentum calculation
    window : int
        Momentum window size
    top_percent : float
        Bottom percentage of stocks to select (e.g., 0.2 for bottom 20%)
    momentum_matrices : dict
        Dictionary mapping window -> momentum DataFrame (index=date, columns=symbols)
    symbols : list, optional
        List of symbols to filter by (if None, uses all symbols in momentum matrix)
        
    Returns:
    --------
    list
        List of bottom momentum stock symbols
    """
    try:
        momentum_scores = momentum_matrices[window].loc[date].dropna()
        if len(momentum_scores) == 0:
            return []
        
        n_bottom = max(1, int(len(momentum_scores) * top_percent))
        bottom_stocks = momentum_scores.nsmallest(n_bottom).index.tolist()
        return bottom_stocks
    except (KeyError, IndexError):
        return []


def create_momentum_matrix(stock_data, target_window):
    """
    Pre-calculate momentum matrix for fast lookups.
    
    Parameters:
    -----------
    stock_data : pd.DataFrame
        Stock data with columns ['symbol', 'date', 'close']
    target_window : int
        Momentum window size
        
    Returns:
    --------
    pd.DataFrame
        Momentum matrix with index=date, columns=symbols, values=momentum_scores
    """
    # Create pivot table for fast momentum calculation
    price_matrix = stock_data.pivot_table(
        index='date', 
        columns='symbol', 
        values='close'
    )
    
    # Pre-calculate momentum for target_window
    momentum_matrix = price_matrix.pct_change(target_window)
    
    return momentum_matrix
