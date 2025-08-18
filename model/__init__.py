"""
Signal generation and ML module for Breadth/Thrust Signals POC.

Handles composite scoring, signal generation, and ML-ready interfaces.
"""

from .scoring import SignalScoring
from .signal_generator import SignalGenerator

__all__ = ['SignalScoring', 'SignalGenerator']
