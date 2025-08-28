"""
Backtesting System

This module provides the backtesting, execution, risk management, and performance
analytics components for the BreadthFlow abstraction system.
"""

from .backtest_config import BacktestConfig
from .backtest_engine_interface import BacktestEngineInterface
from .engines.base_backtest_engine import BaseBacktestEngine
from .engines.standard_backtest_engine import StandardBacktestEngine
from .engines.high_frequency_backtest_engine import HighFrequencyBacktestEngine
from .execution.execution_engine import ExecutionEngine
from .execution.standard_execution_engine import StandardExecutionEngine
from .execution.high_frequency_execution_engine import HighFrequencyExecutionEngine
from .risk.risk_manager import RiskManager
from .risk.standard_risk_manager import StandardRiskManager
from .risk.var_risk_manager import VaRRiskManager
from .analytics.performance_analyzer import PerformanceAnalyzer

__all__ = [
    'BacktestConfig', 'BacktestEngineInterface',
    'BaseBacktestEngine', 'StandardBacktestEngine', 'HighFrequencyBacktestEngine',
    'ExecutionEngine', 'StandardExecutionEngine', 'HighFrequencyExecutionEngine',
    'RiskManager', 'StandardRiskManager', 'VaRRiskManager',
    'PerformanceAnalyzer'
]
