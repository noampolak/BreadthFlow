"""
Data fetching utilities for stock prices, SEC fundamentals, and S&P 500 index data.

This package provides functions to download and manage financial data with
automatic versioning and file management.
"""

from .stock_prices import fetch_stock_prices
from .sec_data import fetch_sec_fundamentals, merge_fundamentals_with_prices
from .sp500_index import fetch_sp500_index
from .versioning import (
    get_versioned_filename,
    backup_existing_file,
    save_with_versioning,
    copy_to_latest
)

__all__ = [
    'fetch_stock_prices',
    'fetch_sec_fundamentals',
    'merge_fundamentals_with_prices',
    'fetch_sp500_index',
    'get_versioned_filename',
    'backup_existing_file',
    'save_with_versioning',
    'copy_to_latest'
]

