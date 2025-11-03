"""
Ultra-Optimized Backtesting Utilities

A modular package for running ultra-optimized backtests with timing fixes.

Main components:
- config: Configuration constants and paths
- helpers: Helper functions for trading day calculations and momentum
- backtest_engine: Core backtesting function
- data_loader: Functions to load models, stock data, and features
- batch_processor: Batch processing for multiple models
"""

from .config import (
    DATA_DIR,
    MODELS_DIR,
    LOG_FILE,
    FEATURES_CACHE_DIR,
    RESULTS_DIR,
    FEATURE_WINDOWS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_TRANSACTION_COST,
    DEFAULT_TOP_PERCENT,
    DEFAULT_MAX_POSITIONS,
    DEFAULT_CASH_BUFFER,
    DEFAULT_TEST_SPLIT
)

from .helpers import (
    get_previous_trading_day,
    get_next_trading_day,
    get_price_fast,
    get_price_open,
    get_top_momentum_stocks_fast,
    get_bottom_momentum_stocks_fast,
    create_momentum_matrix
)

from .backtest_engine import backtest_model_ultra_optimized

from .data_loader import (
    load_models_from_log,
    load_models_from_directory,
    load_stock_data,
    load_features_cache,
    get_test_period,
    parse_model_key
)

from .batch_processor import run_all_models
from .main import run_ultra_optimized_backtesting

__all__ = [
    # Config
    'DATA_DIR',
    'MODELS_DIR',
    'LOG_FILE',
    'FEATURES_CACHE_DIR',
    'RESULTS_DIR',
    'FEATURE_WINDOWS',
    'DEFAULT_INITIAL_CAPITAL',
    'DEFAULT_TRANSACTION_COST',
    'DEFAULT_TOP_PERCENT',
    'DEFAULT_MAX_POSITIONS',
    'DEFAULT_CASH_BUFFER',
    'DEFAULT_TEST_SPLIT',
    # Helpers
    'get_previous_trading_day',
    'get_next_trading_day',
    'get_price_fast',
    'get_price_open',
    'get_top_momentum_stocks_fast',
    'get_bottom_momentum_stocks_fast',
    'create_momentum_matrix',
    # Core
    'backtest_model_ultra_optimized',
    # Data loading
    'load_models_from_log',
    'load_models_from_directory',
    'load_stock_data',
    'load_features_cache',
    'get_test_period',
    'parse_model_key',
    # Batch processing
    'run_all_models',
    # Main orchestrator
    'run_ultra_optimized_backtesting',
]
