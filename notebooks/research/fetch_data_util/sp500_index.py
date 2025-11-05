"""
S&P 500 index data fetching from Yahoo Finance.

Downloads S&P 500 index OHLCV data with automatic versioning.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import os

from .config import (
    SP500_INDEX_DIR,
    LATEST_DIR,
    SP500_INDEX_PREFIX,
    SP500_INDEX_TICKER
)
from .versioning import save_with_versioning, copy_to_latest


def fetch_sp500_index(
    start_date='2020-12-30',
    end_date=None,
    data_dir=SP500_INDEX_DIR,
    latest_dir=LATEST_DIR,
    verbose=True,
    use_cache=True
):
    """
    Fetch S&P 500 index data from Yahoo Finance.
    
    Automatically handles file versioning and creates latest copy.
    If data for the same date range was fetched today, returns cached data.
    
    Parameters:
    -----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format. If None, uses today
    data_dir : str
        Directory to save versioned files
    latest_dir : str
        Directory to save latest copy
    verbose : bool
        Whether to print progress
    use_cache : bool
        If True, check for existing data fetched today with same date range
        
    Returns:
    --------
    tuple (pd.DataFrame, dict)
        (sp500_index_data, metadata) where metadata contains:
        - filepath: path to saved file
        - latest_filepath: path to latest copy
        - cached: True if data was loaded from cache
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Extract years for filename
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    
    # Get download date
    download_date = datetime.now()
    download_date_str = download_date.strftime('%d_%m_%Y')
    
    # Check for existing data if use_cache is enabled
    if use_cache:
        # Generate expected filename
        filename = f"{SP500_INDEX_PREFIX}_{start_year}-{end_year}_download_date_{download_date_str}.pkl"
        filepath = os.path.join(data_dir, filename)
        
        # Also check latest directory
        latest_path = os.path.join(latest_dir, 'sp500_index_data_latest.pkl')
        
        # Check if file exists
        if os.path.exists(filepath):
            if verbose:
                print(f"📂 Found existing S&P 500 index data for today ({download_date_str}) with date range {start_date} to {end_date}")
                print(f"   Loading from: {os.path.basename(filepath)}")
            try:
                df = pd.read_pickle(filepath)
                
                # Verify date range matches
                df_start = pd.to_datetime(df['date'].min()).strftime('%Y-%m-%d')
                df_end = pd.to_datetime(df['date'].max()).strftime('%Y-%m-%d')
                
                # Check if date range is compatible (cached data should cover requested range)
                requested_start = pd.to_datetime(start_date)
                requested_end = pd.to_datetime(end_date)
                actual_start = pd.to_datetime(df_start)
                actual_end = pd.to_datetime(df_end)
                
                if actual_start <= requested_start and actual_end >= requested_end:
                    if verbose:
                        print(f"   ✅ Cached data covers requested range ({start_date} to {end_date})")
                        print(f"   📊 Records: {len(df):,}")
                        print(f"   📅 Actual range: {df_start} to {df_end}")
                    
                    # Calculate total return
                    total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                    
                    # Create metadata
                    metadata = {
                        'timestamp': download_date.strftime('%Y%m%d_%H%M%S'),
                        'filepath': filepath,
                        'latest_filepath': latest_path if os.path.exists(latest_path) else filepath,
                        'backup_filepath': None,
                        'date_range': {
                            'start': start_date,
                            'end': end_date
                        },
                        'data_info': {
                            'total_records': len(df),
                            'date_range_actual': {
                                'start': str(df['date'].min()),
                                'end': str(df['date'].max())
                            },
                            'price_range': {
                                'min': float(df['close'].min()),
                                'max': float(df['close'].max())
                            },
                            'total_return_pct': float(total_return)
                        },
                        'cached': True
                    }
                    
                    if verbose:
                        print(f"   ✅ Using cached data (no re-fetch needed)")
                    
                    return df, metadata
                else:
                    if verbose:
                        print(f"   ⚠️  Cached data range ({df_start} to {df_end}) doesn't fully cover requested range")
                        print(f"   📥 Will fetch new data...")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Error loading cached data: {str(e)}")
                    print(f"   📥 Will fetch new data...")
        
        # Also check latest file (if it was updated today)
        elif os.path.exists(latest_path):
            # Check if latest file was modified today
            file_mtime = datetime.fromtimestamp(os.path.getmtime(latest_path))
            if file_mtime.date() == download_date.date():
                if verbose:
                    print(f"📂 Found latest data file updated today, checking if it matches...")
                try:
                    df = pd.read_pickle(latest_path)
                    
                    # Verify date range matches
                    df_start = pd.to_datetime(df['date'].min()).strftime('%Y-%m-%d')
                    df_end = pd.to_datetime(df['date'].max()).strftime('%Y-%m-%d')
                    
                    # Check if date range is compatible
                    requested_start = pd.to_datetime(start_date)
                    requested_end = pd.to_datetime(end_date)
                    actual_start = pd.to_datetime(df_start)
                    actual_end = pd.to_datetime(df_end)
                    
                    if actual_start <= requested_start and actual_end >= requested_end:
                        if verbose:
                            print(f"   ✅ Latest data covers requested range ({start_date} to {end_date})")
                            print(f"   📊 Records: {len(df):,}")
                            print(f"   📅 Actual range: {df_start} to {df_end}")
                        
                        # Calculate total return
                        total_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
                        
                        # Create metadata
                        metadata = {
                            'timestamp': download_date.strftime('%Y%m%d_%H%M%S'),
                            'filepath': filepath,  # Would be the new filepath if we were to save
                            'latest_filepath': latest_path,
                            'backup_filepath': None,
                            'date_range': {
                                'start': start_date,
                                'end': end_date
                            },
                            'data_info': {
                                'total_records': len(df),
                                'date_range_actual': {
                                    'start': str(df['date'].min()),
                                    'end': str(df['date'].max())
                                },
                                'price_range': {
                                    'min': float(df['close'].min()),
                                    'max': float(df['close'].max())
                                },
                                'total_return_pct': float(total_return)
                            },
                            'cached': True
                        }
                        
                        if verbose:
                            print(f"   ✅ Using cached data (no re-fetch needed)")
                        
                        return df, metadata
                    else:
                        if verbose:
                            print(f"   ⚠️  Latest data range ({df_start} to {df_end}) doesn't fully cover requested range")
                            print(f"   📥 Will fetch new data...")
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Error loading latest data: {str(e)}")
                        print(f"   📥 Will fetch new data...")
    
    if verbose:
        print(f"📊 Fetching S&P 500 index data from Yahoo Finance...")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"   📈 Ticker: {SP500_INDEX_TICKER}")
    
    try:
        # Create ticker object
        index = yf.Ticker(SP500_INDEX_TICKER)
        
        # Fetch historical data
        hist = index.history(start=start_date, end=end_date)
        
        if hist.empty:
            print(f"    ⚠️  No data available for {SP500_INDEX_TICKER}")
            return None, None
        
        # Reset index to get date as column
        hist = hist.reset_index()
        
        # Add symbol column
        hist['symbol'] = 'SP500'
        
        # Rename columns to lowercase
        hist.columns = [col.lower() for col in hist.columns]
        
        # Select required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        if all(col in hist.columns for col in required_cols):
            sp500_data = hist[required_cols].copy()
            
            # Ensure date is datetime and timezone-naive
            sp500_data['date'] = pd.to_datetime(sp500_data['date'])
            if sp500_data['date'].dt.tz is not None:
                sp500_data['date'] = sp500_data['date'].dt.tz_localize(None)
            
            # Sort by date
            sp500_data = sp500_data.sort_values('date').reset_index(drop=True)
            
            # Calculate daily returns for baseline analysis
            sp500_data['daily_return'] = sp500_data['close'].pct_change()
            
            # Calculate cumulative returns (normalized to start at 100)
            sp500_data['cumulative_return'] = (1 + sp500_data['daily_return'].fillna(0)).cumprod() * 100
            
            if verbose:
                print(f"    ✅ Successfully fetched {len(sp500_data)} records")
                print(f"    📅 Date range: {sp500_data['date'].min()} to {sp500_data['date'].max()}")
                print(f"    📈 Price range: ${sp500_data['close'].min():.2f} - ${sp500_data['close'].max():.2f}")
            
        else:
            missing_cols = [col for col in required_cols if col not in hist.columns]
            print(f"    ❌ Missing required columns: {missing_cols}")
            return None, None
            
    except Exception as e:
        print(f"    ❌ Error fetching S&P 500 index data: {str(e)}")
        import traceback
        if verbose:
            print(traceback.format_exc())
        return None, None
    
    # Generate filename
    filename = f"{SP500_INDEX_PREFIX}_{start_year}-{end_year}_download_date_{download_date_str}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Save with versioning
    if verbose:
        print(f"\n💾 Saving data with versioning...")
    saved_path, backup_path = save_with_versioning(
        sp500_data, filepath, SP500_INDEX_PREFIX, start_year, end_year, download_date_str, save_csv=True
    )
    
    # Copy to latest
    latest_path = copy_to_latest(saved_path, latest_dir, 'sp500_index_data_latest')
    
    # Create metadata
    total_return = ((sp500_data['close'].iloc[-1] / sp500_data['close'].iloc[0]) - 1) * 100
    
    metadata = {
        'timestamp': download_date.strftime('%Y%m%d_%H%M%S'),
        'filepath': saved_path,
        'latest_filepath': latest_path,
        'backup_filepath': backup_path,
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'data_info': {
            'total_records': len(sp500_data),
            'date_range_actual': {
                'start': str(sp500_data['date'].min()),
                'end': str(sp500_data['date'].max())
            },
            'price_range': {
                'min': float(sp500_data['close'].min()),
                'max': float(sp500_data['close'].max())
            },
            'total_return_pct': float(total_return)
        },
        'cached': False
    }
    
    if verbose:
        print(f"\n✅ S&P 500 index data saved successfully!")
        print(f"   📄 Main file: {os.path.basename(saved_path)}")
        print(f"   📋 Latest copy: {os.path.basename(latest_path)}")
        print(f"   📈 Total return: {total_return:.2f}%")
    
    return sp500_data, metadata

