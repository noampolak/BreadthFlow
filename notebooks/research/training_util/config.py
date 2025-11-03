"""
Configuration constants for the training utility package.
"""

import os

# Directory configuration
DATA_DIR = os.path.join('data', 'research')
MODELS_DIR = os.path.join(DATA_DIR, 'multiclass_models')
RESULTS_DIR = os.path.join(DATA_DIR, 'multiclass_results')
FEATURES_CACHE_DIR = os.path.join(DATA_DIR, 'features_cache')
LOG_FILE = os.path.join(DATA_DIR, 'training_log.json')

# Feature and target windows
DEFAULT_FEATURE_WINDOWS = [5, 10, 15, 20, 25, 30]
DEFAULT_TARGET_WINDOWS = [5, 10, 15, 20, 25, 30]

# Model types
DEFAULT_MODEL_TYPES = ['RandomForest', 'XGBoost']

# Training configuration
DEFAULT_TRAIN_SPLIT = 0.6  # 60% train, 40% test

# Class labels
CLASS_LABELS = {
    0: 'Momentum',
    1: 'Reversal',
    2: 'Do Nothing'
}

