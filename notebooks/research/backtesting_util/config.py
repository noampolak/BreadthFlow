"""
Configuration constants and paths for ultra-optimized backtesting.
"""

import os

# Default paths
DATA_DIR = 'data/research'
MODELS_DIR = os.path.join(DATA_DIR, 'multiclass_models')
LOG_FILE = os.path.join(DATA_DIR, 'training_log.json')
FEATURES_CACHE_DIR = os.path.join(DATA_DIR, 'features_cache')
RESULTS_DIR = os.path.join(DATA_DIR, 'backtesting', 'ultra_optimized_results')

# Feature windows
FEATURE_WINDOWS = [5, 10, 15, 20, 25, 30]

# Default backtesting parameters
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_TRANSACTION_COST = 0.003
DEFAULT_TOP_PERCENT = 0.2
DEFAULT_MAX_POSITIONS = 20
DEFAULT_CASH_BUFFER = 0.05
DEFAULT_TEST_SPLIT = 0.6  # 60% for training, 40% for testing
DEFAULT_STOP_LOSS = 0.15  # 15% stop-loss (based on analysis showing optimal balance)
