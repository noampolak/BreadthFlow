"""
Data Management System

This module provides the data fetching, resource management, and validation
components for the BreadthFlow abstraction system.
"""

from .resources.data_resources import (
    ResourceType,
    DataFrequency,
    ResourceField,
    DataResource,
    STOCK_PRICE,
    REVENUE,
    MARKET_CAP,
    NEWS_SENTIMENT,
)
from .sources.data_source_interface import DataSourceInterface
from .validation.data_validator import DataValidator, ValidationRule, ValidationResult
from .universal_data_fetcher import UniversalDataFetcher

__all__ = [
    "ResourceType",
    "DataFrequency",
    "ResourceField",
    "DataResource",
    "STOCK_PRICE",
    "REVENUE",
    "MARKET_CAP",
    "NEWS_SENTIMENT",
    "DataSourceInterface",
    "DataValidator",
    "ValidationRule",
    "ValidationResult",
    "UniversalDataFetcher",
]
