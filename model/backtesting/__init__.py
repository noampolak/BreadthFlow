"""
Backtesting System

This module provides the backtesting, execution, risk management, and performance
analytics components for the BreadthFlow abstraction system.
"""

from .analytics.performance_analyzer import PerformanceAnalyzer
from .backtest_config import BacktestConfig
from .backtest_engine_interface import BacktestEngineInterface
from .engines.base_backtest_engine import BaseBacktestEngine
from .engines.high_frequency_backtest_engine import HighFrequencyBacktestEngine
from .engines.standard_backtest_engine import StandardBacktestEngine
from .execution.execution_engine import ExecutionEngine
from .execution.high_frequency_execution_engine import HighFrequencyExecutionEngine
from .execution.standard_execution_engine import StandardExecutionEngine
from .risk.risk_manager import RiskManager
from .risk.standard_risk_manager import StandardRiskManager
from .risk.var_risk_manager import VaRRiskManager

__all__ = [
    "BacktestConfig",
    "BacktestEngineInterface",
    "BaseBacktestEngine",
    "StandardBacktestEngine",
    "HighFrequencyBacktestEngine",
    "ExecutionEngine",
    "StandardExecutionEngine",
    "HighFrequencyExecutionEngine",
    "RiskManager",
    "StandardRiskManager",
    "VaRRiskManager",
    "PerformanceAnalyzer",
]
