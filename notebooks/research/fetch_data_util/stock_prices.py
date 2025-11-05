"""
Stock price data fetching from Yahoo Finance.

Downloads OHLCV data for S&P 500 stocks with automatic versioning.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime
from tqdm import tqdm
import os

from .config import (
    STOCK_PRICES_DIR,
    LATEST_DIR,
    STOCK_PRICES_PREFIX,
    YAHOO_RATE_LIMIT
)
from .versioning import save_with_versioning, copy_to_latest


def get_sp500_tickers():
    """
    Get curated list of 250+ major S&P 500 tickers.
    
    Returns:
    --------
    list
        List of S&P 500 ticker symbols
    """
    # Curated list of major S&P 500 tickers (250+ companies)
    sp500_tickers = [
        # Technology (50+)
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
        'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'IBM', 'QCOM', 'TXN', 'AVGO', 'AMAT',
        'MU', 'ADI', 'LRCX', 'KLAC', 'MCHP', 'SNPS', 'CDNS', 'ANSS', 'FTNT', 'PANW',
        'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'SNOW', 'PLTR', 'RBLX', 'U', 'TWLO',
        'ZM', 'DOCU', 'SPLK', 'WDAY', 'NOW', 'TEAM', 'VEEV', 'MDB', 'ESTC',
        'CFLT', 'FROG', 'PATH', 'BILL', 'AI', 'SMCI', 'ARM', 'ARMK', 'ARW', 'ASML',
        
        # Healthcare (40+)
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
        'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'SYK', 'ISRG',
        'MDT', 'BSX', 'EW', 'DXCM', 'TECH', 'IQV', 'A', 'WAT', 'PKI', 'WST',
        
        # Financial Services (35+)
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI', 'MCO',
        'V', 'MA', 'PYPL', 'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'NTRS',
        'SCHW', 'ICE', 'CME', 'NDAQ', 'MKTX', 'FIS', 'FISV', 'GPN', 'JKHY', 'FLT',
        'WU', 'TRV', 'ALL', 'PGR', 'AON', 'MMC', 'AFL', 'PRU', 'MET', 'AIG',
        
        # Consumer Discretionary (30+)
        'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'ROST', 'TGT', 'WMT', 'COST',
        'F', 'GM', 'TM', 'HMC', 'NIO', 'XPEV', 'LI',
        'BABA', 'JD', 'PDD', 'BIDU', 'NTES', 'TME', 'VIPS', 'YMM', 'DIDI', 'GRAB',
        
        # Consumer Staples (25+)
        'PG', 'KO', 'PEP', 'CL', 'KMB', 'CHD', 'CLX', 'GIS',
        'K', 'CPB', 'HSY', 'MKC', 'SJM', 'CAG', 'HRL', 'TSN', 'KHC', 'MDLZ',
        'PM', 'MO', 'BTI', 'IMB', 'UL', 'NVS', 'SAP',
        
        # Industrial (30+)
        'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC',
        'GD', 'TDG', 'LHX', 'TDY',
        'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'SWK', 'TXT', 'DOV', 'FTV', 'IEX',
        
        # Energy (20+)
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PXD', 'MPC', 'VLO', 'PSX',
        'KMI', 'WMB', 'EPD', 'OKE', 'ET', 'ENB', 'TRP', 'PPL', 'DUK', 'SO',
        
        # Materials (15+)
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'FCX', 'NEM', 'GOLD', 'AA',
        'X', 'CLF', 'NUE', 'STLD', 'CMC', 'RS', 'VMC', 'MLM', 'EXP', 'SUM',
        
        # Utilities (15+)
        'NEE', 'D', 'EXC', 'AEP', 'XEL', 'PEG', 'ES', 'EIX',
        'SRE', 'WEC', 'AWK', 'LNT', 'CNP', 'ED', 'PCG', 'ETR', 'FE', 'AEE',
        
        # Real Estate (10+)
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'EXR', 'AVB', 'EQR', 'MAA', 'UDR',
        'ESS', 'CPT', 'AIV', 'BXP', 'VTR', 'WELL', 'PEAK', 'HCP',
        
        # Communication Services (15+)
        'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS',
        'TWTR', 'SNAP', 'PINS', 'SPOT', 'MTCH', 'LYFT', 'UBER', 'DASH', 'ABNB', 'BKNG'
    ]
    
    # Remove duplicates and sort
    sp500_tickers = sorted(list(set(sp500_tickers)))
    
    return sp500_tickers


def fetch_stock_prices(
    tickers=None,
    start_date='2014-01-01',
    end_date=None,
    data_dir=STOCK_PRICES_DIR,
    latest_dir=LATEST_DIR,
    verbose=True,
    use_cache=True
):
    """
    Fetch real stock data for S&P 500 tickers from Yahoo Finance.
    
    Automatically handles file versioning and creates latest copy.
    If data for the same date range was fetched today, returns cached data.
    
    Parameters:
    -----------
    tickers : list, optional
        List of ticker symbols. If None, uses get_sp500_tickers()
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
        (stock_data, metadata) where metadata contains:
        - successful_tickers: list of successfully fetched tickers
        - failed_tickers: list of failed tickers
        - filepath: path to saved file
        - latest_filepath: path to latest copy
    """
    if tickers is None:
        tickers = get_sp500_tickers()
    
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
        # Also check latest directory
        latest_path = os.path.join(latest_dir, 'sp500_stock_data_latest.pkl')
        
        # Function to check and load cached file
        def check_and_load_cached_file(filepath_to_check, file_description):
            """Helper to check if a file exists and covers the requested date range"""
            if not os.path.exists(filepath_to_check):
                if verbose:
                    print(f"   ⚠️  {file_description} file not found: {os.path.basename(filepath_to_check)}")
                return None, None
            
            try:
                df = pd.read_pickle(filepath_to_check)
                
                if df.empty:
                    if verbose:
                        print(f"   ⚠️  {file_description} is empty")
                    return None, None
                
                # Verify date range matches
                df_start = pd.to_datetime(df['date'].min()).strftime('%Y-%m-%d')
                df_end = pd.to_datetime(df['date'].max()).strftime('%Y-%m-%d')
                
                # Check if date range is compatible (cached data should cover requested range)
                requested_start = pd.to_datetime(start_date)
                requested_end = pd.to_datetime(end_date)
                actual_start = pd.to_datetime(df_start)
                actual_end = pd.to_datetime(df_end)
                
                if verbose:
                    print(f"   📊 {file_description} has range: {df_start} to {df_end} ({len(df):,} records)")
                    print(f"   📅 Requested range: {start_date} to {end_date}")
                
                if actual_start <= requested_start and actual_end >= requested_end:
                    if verbose:
                        print(f"   ✅ {file_description} covers requested range!")
                        print(f"   📈 Symbols: {df['symbol'].nunique()}")
                    
                    # Create metadata
                    metadata = {
                        'timestamp': download_date.strftime('%Y%m%d_%H%M%S'),
                        'filepath': filepath_to_check,
                        'latest_filepath': latest_path if os.path.exists(latest_path) else filepath_to_check,
                        'backup_filepath': None,
                        'successful_tickers': df['symbol'].unique().tolist(),
                        'failed_tickers': [],
                        'total_tickers': len(tickers),
                        'date_range': {
                            'start': start_date,
                            'end': end_date
                        },
                        'data_info': {
                            'total_records': len(df),
                            'unique_symbols': df['symbol'].nunique(),
                            'date_range_actual': {
                                'start': str(df['date'].min()),
                                'end': str(df['date'].max())
                            }
                        },
                        'cached': True
                    }
                    
                    return df, metadata
                else:
                    if verbose:
                        if actual_start > requested_start:
                            print(f"   ⚠️  Start date too late: {df_start} > {start_date}")
                        if actual_end < requested_end:
                            print(f"   ⚠️  End date too early: {df_end} < {end_date}")
                        print(f"   📥 Will search for other files or fetch new data...")
            except Exception as e:
                if verbose:
                    print(f"   ❌ Error loading {file_description}: {str(e)}")
                    import traceback
                    print(f"   {traceback.format_exc()}")
            
            return None, None
        
        # 1. First check for file created today
        filename_today = f"{STOCK_PRICES_PREFIX}_{start_year}-{end_year}_download_date_{download_date_str}.pkl"
        filepath_today = os.path.join(data_dir, filename_today)
        
        if os.path.exists(filepath_today):
            if verbose:
                print(f"📂 Found existing data for today ({download_date_str}) with date range {start_date} to {end_date}")
                print(f"   Loading from: {os.path.basename(filepath_today)}")
            
            df, metadata = check_and_load_cached_file(filepath_today, "Today's cached data")
            if df is not None:
                if verbose:
                    print(f"   ✅ Using cached data (no re-fetch needed)")
                return df, metadata
        
        # 2. Check latest file (check date range regardless of modification date)
        if os.path.exists(latest_path):
            if verbose:
                print(f"📂 Found latest data file, checking if it matches...")
            
            df, metadata = check_and_load_cached_file(latest_path, "Latest file")
            if df is not None:
                if verbose:
                    print(f"   ✅ Using cached data (no re-fetch needed)")
                return df, metadata
        
        # 3. Search for ANY file with the same date range (regardless of download date)
        if verbose:
            print(f"📂 Searching for existing files with date range {start_year}-{end_year}...")
        
        import glob
        pattern = os.path.join(data_dir, f"{STOCK_PRICES_PREFIX}_{start_year}-{end_year}_download_date_*.pkl")
        matching_files = glob.glob(pattern)
        
        # Exclude versioned files (v1, v2, etc.)
        matching_files = [f for f in matching_files if '_v' not in os.path.basename(f)]
        
        if matching_files:
            # Sort by modification time (newest first)
            matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            for filepath in matching_files:
                if verbose:
                    print(f"   Checking: {os.path.basename(filepath)}")
                
                df, metadata = check_and_load_cached_file(filepath, f"Found file {os.path.basename(filepath)}")
                if df is not None:
                    if verbose:
                        print(f"   ✅ Using cached data (no re-fetch needed)")
                    return df, metadata
        
        if verbose:
            print(f"   📥 No suitable cached data found, will fetch new data...")
    
    if verbose:
        print(f"📥 Fetching real stock data for {len(tickers)} S&P 500 stocks...")
        print(f"📅 Date range: {start_date} to {end_date}")
    
    all_data = []
    successful_tickers = []
    failed_tickers = []
    
    for i, ticker in enumerate(tqdm(tickers, desc="Fetching data", disable=not verbose)):
        if verbose and i % 10 == 0:
            print(f"📊 Fetching {ticker} ({i+1}/{len(tickers)})...")
        
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                if verbose:
                    print(f"    ⚠️  {ticker}: No data available")
                failed_tickers.append(ticker)
                continue
            
            # Reset index to get date as column
            hist = hist.reset_index()
            
            # Add symbol column
            hist['symbol'] = ticker
            
            # Rename columns to lowercase
            hist.columns = [col.lower() for col in hist.columns]
            
            # Ensure we have required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            if all(col in hist.columns for col in required_cols):
                all_data.append(hist[required_cols])
                successful_tickers.append(ticker)
                if verbose and i % 10 == 0:
                    print(f"    ✅ {ticker}: {len(hist)} records")
            else:
                if verbose:
                    print(f"    ⚠️  {ticker}: Missing required columns")
                failed_tickers.append(ticker)
            
            # Rate limiting
            time.sleep(YAHOO_RATE_LIMIT)
            
        except Exception as e:
            if verbose:
                print(f"    ❌ {ticker}: {str(e)}")
            failed_tickers.append(ticker)
            continue
    
    if not all_data:
        print("❌ No data retrieved from any ticker")
        return None, None
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by symbol and date
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # Ensure date is timezone-naive
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    
    if verbose:
        print(f"\n✅ Data collection complete!")
        print(f"   📊 Total records: {len(df):,}")
        print(f"   📈 Successful tickers: {len(successful_tickers)}")
        print(f"   ❌ Failed tickers: {len(failed_tickers)}")
        print(f"   📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   💰 Symbols: {df['symbol'].nunique()}")
    
    # Generate filename
    filename = f"{STOCK_PRICES_PREFIX}_{start_year}-{end_year}_download_date_{download_date_str}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Save with versioning
    if verbose:
        print(f"\n💾 Saving data with versioning...")
    saved_path, backup_path = save_with_versioning(
        df, filepath, STOCK_PRICES_PREFIX, start_year, end_year, download_date_str, save_csv=True
    )
    
    # Copy to latest
    latest_path = copy_to_latest(saved_path, latest_dir, 'sp500_stock_data_latest')
    
    # Create metadata
    metadata = {
        'timestamp': download_date.strftime('%Y%m%d_%H%M%S'),
        'filepath': saved_path,
        'latest_filepath': latest_path,
        'backup_filepath': backup_path,
        'successful_tickers': successful_tickers,
        'failed_tickers': failed_tickers,
        'total_tickers': len(tickers),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'data_info': {
            'total_records': len(df),
            'unique_symbols': df['symbol'].nunique(),
            'date_range_actual': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            }
        },
        'cached': False
    }
    
    if verbose:
        print(f"\n✅ Stock prices saved successfully!")
        print(f"   📄 Main file: {os.path.basename(saved_path)}")
        print(f"   📋 Latest copy: {os.path.basename(latest_path)}")
    
    return df, metadata

