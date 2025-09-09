"""
Signal generation and ML module for Breadth/Thrust Signals POC.

Handles composite scoring, signal generation, and ML-ready interfaces.
"""

# Import from new modular system
from .signals.composite_signal_generator import CompositeSignalGenerator
from .backtesting.analytics.performance_analyzer import PerformanceAnalyzer

__all__ = ['CompositeSignalGenerator', 'PerformanceAnalyzer']
