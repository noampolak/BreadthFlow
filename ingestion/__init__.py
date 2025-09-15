"""
Data ingestion module for Breadth/Thrust Signals POC.

Handles data fetching from external sources and replay functionality.
"""

from .data_fetcher import DataFetcher
from .replay import ReplayManager

__all__ = ["DataFetcher", "ReplayManager"]
