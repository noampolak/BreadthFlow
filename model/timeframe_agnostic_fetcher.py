#!/usr/bin/env python3
"""
Timeframe-Agnostic Data Fetcher

This module provides multi-timeframe data fetching capabilities while maintaining
backward compatibility with existing daily data fetching.

Key Features:
- Support for multiple timeframes: 1min, 5min, 15min, 1hour, 1day
- Multiple data sources: yfinance, alpha_vantage, polygon
- Backward compatibility with existing daily fetching
- Consistent data format across all timeframes
"""

import yfinance as yf
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeAgnosticDataSource(ABC):
    """Abstract base class for timeframe-agnostic data sources."""
    
    @abstractmethod
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for specified symbol and timeframe.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Data timeframe ('1min', '5min', '15min', '1hour', '1day')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        pass
    
    @abstractmethod
    def get_supported_timeframes(self) -> List[str]:
        """Return list of supported timeframes for this data source."""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, int]:
        """Return rate limits for this data source."""
        pass

class YFinanceIntradaySource(TimeframeAgnosticDataSource):
    """yfinance-based intraday data source."""
    
    def __init__(self):
        self.timeframe_mapping = {
            '1min': '1m',
            '5min': '5m', 
            '15min': '15m',
            '1hour': '1h',
            '1day': '1d'
        }
        
        # Rate limits for yfinance (conservative estimates)
        self.rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 2000,
            'requests_per_day': 48000
        }
        
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def get_supported_timeframes(self) -> List[str]:
        """Return supported timeframes."""
        return list(self.timeframe_mapping.keys())
    
    def get_rate_limits(self) -> Dict[str, int]:
        """Return rate limits."""
        return self.rate_limits
    
    def _respect_rate_limits(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data from yfinance with timeframe support.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe ('1min', '5min', '15min', '1hour', '1day')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with standardized columns
        """
        if timeframe not in self.timeframe_mapping:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(self.timeframe_mapping.keys())}")
        
        # Respect rate limits
        self._respect_rate_limits()
        
        try:
            yf_timeframe = self.timeframe_mapping[timeframe]
            
            logger.info(f"Fetching {symbol} data: {timeframe} from {start_date} to {end_date}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # For intraday data, yfinance has limitations
            if timeframe in ['1min', '5min', '15min']:
                # Intraday data is limited to last 60 days for most intervals
                logger.warning(f"Intraday data ({timeframe}) may be limited to recent periods")
                
                # Calculate the maximum allowed start date (60 days ago)
                max_start = datetime.now() - timedelta(days=60)
                actual_start = max(datetime.strptime(start_date, '%Y-%m-%d'), max_start)
                
                if actual_start > datetime.strptime(start_date, '%Y-%m-%d'):
                    logger.warning(f"Adjusted start date to {actual_start.strftime('%Y-%m-%d')} due to yfinance limitations")
                    start_date = actual_start.strftime('%Y-%m-%d')
            
            # Fetch the data
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_timeframe,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol} ({timeframe})")
                return pd.DataFrame()
            
            # Standardize column names and format
            data = data.reset_index()
            
            # Rename columns to match our standard format
            column_mapping = {
                'Date': 'Date',
                'Datetime': 'Date',  # For intraday data
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }
            
            # Handle datetime vs date column
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            
            # Ensure we have all required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            # Select and reorder columns
            data = data[required_columns]
            
            # Convert Date to consistent format
            if timeframe == '1day':
                # For daily data, keep as date
                data['Date'] = pd.to_datetime(data['Date']).dt.date
            else:
                # For intraday data, keep as datetime
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Ensure numeric types
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol} ({timeframe})")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} ({timeframe}): {str(e)}")
            return pd.DataFrame()

class TimeframeAgnosticFetcher:
    """
    Main fetcher class that coordinates multiple data sources and timeframes.
    
    This class maintains backward compatibility while adding multi-timeframe support.
    """
    
    def __init__(self):
        # Initialize data sources
        self.data_sources = {
            'yfinance': YFinanceIntradaySource(),
            # Add more sources as needed
            # 'alpha_vantage': AlphaVantageSource(),
            # 'polygon': PolygonSource(),
        }
        
        # Default source for each timeframe
        self.default_sources = {
            '1min': 'yfinance',
            '5min': 'yfinance', 
            '15min': 'yfinance',
            '1hour': 'yfinance',
            '1day': 'yfinance'  # Maintain backward compatibility
        }
        
        # Timeframe validation
        self.supported_timeframes = ['1min', '5min', '15min', '1hour', '1day']
    
    def get_supported_timeframes(self) -> List[str]:
        """Get all supported timeframes across all data sources."""
        return self.supported_timeframes.copy()
    
    def get_data_sources(self) -> List[str]:
        """Get available data sources."""
        return list(self.data_sources.keys())
    
    def fetch_data(self, symbol: str, timeframe: str = '1day', 
                   start_date: str = None, end_date: str = None,
                   data_source: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch data with timeframe support.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe (default: '1day' for backward compatibility)
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            data_source: Specific data source to use (optional)
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        # Validate timeframe
        if timeframe not in self.supported_timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}")
        
        # Determine data source
        if data_source is None:
            data_source = self.default_sources[timeframe]
        
        if data_source not in self.data_sources:
            raise ValueError(f"Unknown data source: {data_source}. Available: {list(self.data_sources.keys())}")
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # Default to 1 year ago for daily, shorter for intraday
            if timeframe == '1day':
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            elif timeframe in ['1hour']:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            else:  # intraday
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Fetch data
        source = self.data_sources[data_source]
        data = source.fetch_data(symbol, timeframe, start_date, end_date)
        
        # Create metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'data_source': data_source,
            'records_count': len(data),
            'fetch_timestamp': datetime.now().isoformat(),
            'rate_limits': source.get_rate_limits()
        }
        
        return data, metadata
    
    def fetch_multiple_symbols(self, symbols: List[str], timeframe: str = '1day',
                             start_date: str = None, end_date: str = None,
                             data_source: str = None) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            data_source: Data source to use
            
        Returns:
            Dictionary mapping symbol to (DataFrame, metadata)
        """
        results = {}
        
        for symbol in symbols:
            try:
                data, metadata = self.fetch_data(symbol, timeframe, start_date, end_date, data_source)
                results[symbol] = (data, metadata)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {str(e)}")
                results[symbol] = (pd.DataFrame(), {'error': str(e)})
        
        return results

# Factory function for backward compatibility
def create_timeframe_fetcher() -> TimeframeAgnosticFetcher:
    """Create a timeframe-agnostic fetcher instance."""
    return TimeframeAgnosticFetcher()

# Example usage and testing
if __name__ == "__main__":
    # Test the fetcher
    fetcher = create_timeframe_fetcher()
    
    print("Supported timeframes:", fetcher.get_supported_timeframes())
    print("Available data sources:", fetcher.get_data_sources())
    
    # Test daily data (backward compatibility)
    print("\n=== Testing Daily Data (Backward Compatibility) ===")
    data, metadata = fetcher.fetch_data('AAPL', '1day', '2024-08-01', '2024-08-31')
    print(f"Fetched {len(data)} daily records for AAPL")
    if not data.empty:
        print(data.head())
    
    # Test hourly data
    print("\n=== Testing Hourly Data ===")
    data, metadata = fetcher.fetch_data('AAPL', '1hour', '2024-08-20', '2024-08-21')
    print(f"Fetched {len(data)} hourly records for AAPL")
    if not data.empty:
        print(data.head())
