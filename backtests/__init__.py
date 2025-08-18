"""
Backtesting module for Breadth/Thrust Signals POC.

Handles performance analysis, portfolio simulation, and risk metrics.
"""

from .engine import BacktestEngine
from .metrics import calculate_metrics

__all__ = ['BacktestEngine', 'calculate_metrics']
