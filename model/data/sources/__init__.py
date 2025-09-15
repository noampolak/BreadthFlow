"""
Data Sources Module

Provides data source implementations and interfaces for fetching
various types of financial data.
"""

from .data_source_interface import DataSourceInterface

try:
    from .yfinance_source import YFinanceDataSource

    __all__ = ["DataSourceInterface", "YFinanceDataSource"]
except ImportError:
    __all__ = ["DataSourceInterface"]
