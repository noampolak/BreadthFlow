"""
YFinance Data Source Implementation

Enhanced YFinance data source with multi-resource support for
stock prices, fundamental data, and market cap information.
"""

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from data_source_interface import DataSourceInterface
from resources.data_resources import (
    STOCK_PRICE, REVENUE, MARKET_CAP, validate_resource_data
)

logger = logging.getLogger(__name__)

class YFinanceDataSource(DataSourceInterface):
    """Enhanced YFinance data source with multi-resource support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is not available. Please install it with: pip install yfinance")
        
        self.config = config or {}
        self.name = "yfinance"
        self.supported_resources = ["stock_price", "revenue", "market_cap"]
        
        # Rate limiting
        self.rate_limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 2000,
            'requests_per_day': 10000
        }
        
        # Cache for ticker objects
        self._ticker_cache = {}
    
    def get_name(self) -> str:
        """Get the name of this data source"""
        return self.name
    
    def get_supported_resources(self) -> List[str]:
        """Get list of supported resource names"""
        return self.supported_resources
    
    def get_config(self) -> Dict[str, Any]:
        """Get data source configuration"""
        return self.config
    
    def fetch_data(self, resource_name: str, symbols: List[str], 
                   start_date: datetime, end_date: datetime, 
                   **kwargs) -> pd.DataFrame:
        """Fetch data for specified resource and symbols"""
        
        if resource_name not in self.supported_resources:
            raise ValueError(f"Unsupported resource: {resource_name}")
        
        logger.info(f"Fetching {resource_name} data for {len(symbols)} symbols")
        
        if resource_name == "stock_price":
            return self._fetch_stock_price(symbols, start_date, end_date, **kwargs)
        elif resource_name == "revenue":
            return self._fetch_revenue(symbols, start_date, end_date, **kwargs)
        elif resource_name == "market_cap":
            return self._fetch_market_cap(symbols, start_date, end_date, **kwargs)
        else:
            raise ValueError(f"Unknown resource: {resource_name}")
    
    def _fetch_stock_price(self, symbols: List[str], start_date: datetime, 
                          end_date: datetime, **kwargs) -> pd.DataFrame:
        """Fetch stock price data"""
        
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                
                # Get historical data
                hist = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=kwargs.get('interval', '1d')
                )
                
                if hist.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Reset index to get date as column
                hist = hist.reset_index()
                
                # Add symbol column
                hist['symbol'] = symbol
                
                # Rename columns to match our schema
                hist = hist.rename(columns={
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Adj Close': 'adj_close'
                })
                
                # Select only the columns we need
                columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
                hist = hist[columns]
                
                all_data.append(hist)
                
            except Exception as e:
                logger.error(f"Error fetching stock price data for {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        
        # Validate data
        for _, row in result.iterrows():
            if not validate_resource_data(row.to_dict(), STOCK_PRICE):
                logger.warning(f"Invalid stock price data for {row['symbol']}")
        
        return result
    
    def _fetch_revenue(self, symbols: List[str], start_date: datetime, 
                      end_date: datetime, **kwargs) -> pd.DataFrame:
        """Fetch revenue data"""
        
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                
                # Get financial data
                financials = ticker.financials
                
                if financials.empty:
                    logger.warning(f"No financial data found for {symbol}")
                    continue
                
                # Look for revenue data
                revenue_data = None
                for index in financials.index:
                    if 'revenue' in index.lower() or 'total revenue' in index.lower():
                        revenue_data = financials.loc[index]
                        break
                
                if revenue_data is None:
                    logger.warning(f"No revenue data found for {symbol}")
                    continue
                
                # Convert to DataFrame
                revenue_df = revenue_data.reset_index()
                revenue_df.columns = ['date', 'revenue']
                revenue_df['symbol'] = symbol
                revenue_df['currency'] = 'USD'
                revenue_df['period'] = 'quarterly'
                revenue_df['source'] = 'yfinance'
                
                # Filter by date range
                revenue_df['date'] = pd.to_datetime(revenue_df['date'])
                revenue_df = revenue_df[
                    (revenue_df['date'] >= start_date) & 
                    (revenue_df['date'] <= end_date)
                ]
                
                if not revenue_df.empty:
                    all_data.append(revenue_df)
                
            except Exception as e:
                logger.error(f"Error fetching revenue data for {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        
        # Validate data
        for _, row in result.iterrows():
            if not validate_resource_data(row.to_dict(), REVENUE):
                logger.warning(f"Invalid revenue data for {row['symbol']}")
        
        return result
    
    def _fetch_market_cap(self, symbols: List[str], start_date: datetime, 
                         end_date: datetime, **kwargs) -> pd.DataFrame:
        """Fetch market cap data"""
        
        all_data = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                
                # Get info
                info = ticker.info
                
                if not info:
                    logger.warning(f"No info data found for {symbol}")
                    continue
                
                # Extract market cap
                market_cap = info.get('marketCap')
                shares_outstanding = info.get('sharesOutstanding')
                
                if market_cap is None:
                    logger.warning(f"No market cap data found for {symbol}")
                    continue
                
                # Create data row
                data = {
                    'symbol': symbol,
                    'date': datetime.now(),
                    'market_cap': market_cap,
                    'shares_outstanding': shares_outstanding,
                    'currency': 'USD',
                    'source': 'yfinance'
                }
                
                # Validate data
                if validate_resource_data(data, MARKET_CAP):
                    all_data.append(data)
                else:
                    logger.warning(f"Invalid market cap data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching market cap data for {symbol}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.DataFrame(all_data)
    
    def _get_ticker(self, symbol: str):
        """Get or create ticker object with caching"""
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting information for this data source"""
        return self.rate_limits
    
    def get_data_quality_info(self) -> Dict[str, Any]:
        """Get data quality information for this data source"""
        return {
            'completeness': 0.90,
            'accuracy': 0.95,
            'latency': 'medium',
            'update_frequency': 'real_time',
            'coverage': 'global'
        }
