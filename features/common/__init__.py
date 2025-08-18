"""
Common utilities for feature computation.

Provides shared functionality for data I/O and processing.
"""

from .io import write_delta, read_delta
from .config import get_config

__all__ = ['write_delta', 'read_delta', 'get_config']
