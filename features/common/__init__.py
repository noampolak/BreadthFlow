"""
Common utilities for feature computation.

Provides shared functionality for data I/O and processing.
"""

from .config import get_config
from .io import read_delta, write_delta

__all__ = ["write_delta", "read_delta", "get_config"]
