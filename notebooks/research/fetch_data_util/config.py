"""
Configuration constants for data fetching utilities.
"""

import os

# Default data directory
DATA_DIR = 'data/research'

# Subdirectories for different data types
STOCK_PRICES_DIR = os.path.join(DATA_DIR, 'stock_prices')
SEC_DATA_DIR = os.path.join(DATA_DIR, 'sec_data')
SP500_INDEX_DIR = os.path.join(DATA_DIR, 'sp500_index')
LATEST_DIR = os.path.join(DATA_DIR, 'stock_data_latest')

# File naming patterns
STOCK_PRICES_PREFIX = 'stocks_prices'
SEC_DATA_PREFIX = 'sec_fundamentals'
SP500_INDEX_PREFIX = 'sp500_index'

# Rate limiting (seconds between requests)
YAHOO_RATE_LIMIT = 0.1  # 100ms between Yahoo Finance requests
SEC_RATE_LIMIT = 0.08   # 80ms between SEC API requests

# SEC API configuration
SEC_USER_AGENT = "Noam Polak noampolak@gmail.com"  # Format: "Name email@domain.com" (no parentheses)
SEC_BASE_URL = "https://data.sec.gov"
SEC_CACHE_DIR = "./sec_cache"

# S&P 500 index ticker
SP500_INDEX_TICKER = '^GSPC'

