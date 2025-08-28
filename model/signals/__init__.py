"""
Signal Generation System

This module provides the signal generation, strategy management, and component
orchestration for the BreadthFlow abstraction system.
"""

from .signal_config import SignalConfig
from .signal_generator_interface import SignalGeneratorInterface
from .components.technical_indicators import TechnicalIndicators
from .components.fundamental_indicators import FundamentalIndicators
from .components.sentiment_indicators import SentimentIndicators
from .strategies.base_signal_strategy import BaseSignalStrategy
from .strategies.technical_analysis_strategy import TechnicalAnalysisStrategy
from .strategies.fundamental_analysis_strategy import FundamentalAnalysisStrategy
from .composite_signal_generator import CompositeSignalGenerator

__all__ = [
    'SignalConfig', 'SignalGeneratorInterface',
    'TechnicalIndicators', 'FundamentalIndicators', 'SentimentIndicators',
    'BaseSignalStrategy', 'TechnicalAnalysisStrategy', 'FundamentalAnalysisStrategy',
    'CompositeSignalGenerator'
]
