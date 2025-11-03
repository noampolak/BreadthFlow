"""
Training Utilities Package

A modular package for training multi-class models with feature and target caching.

Main components:
- config: Configuration constants and paths
- cache_manager: Functions for caching features and targets
- feature_creator: Functions for creating features
- target_calculator: Functions for calculating targets
- model_trainer: Functions for training individual models
- main: Main orchestrator function
"""

from .config import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FEATURES_CACHE_DIR,
    LOG_FILE,
    DEFAULT_FEATURE_WINDOWS,
    DEFAULT_TARGET_WINDOWS,
    DEFAULT_MODEL_TYPES,
    DEFAULT_TRAIN_SPLIT,
    CLASS_LABELS
)

from .cache_manager import (
    load_features_from_cache,
    save_features_to_cache,
    load_target_from_cache,
    save_target_to_cache,
    load_training_log,
    save_training_log
)

from .feature_creator import (
    create_features_for_window,
    precalculate_all_features
)

from .target_calculator import (
    precalculate_all_targets
)

from .model_trainer import (
    train_single_model_optimized
)

from .main import (
    train_all_models_optimized
)

__all__ = [
    # Config
    'DATA_DIR',
    'MODELS_DIR',
    'RESULTS_DIR',
    'FEATURES_CACHE_DIR',
    'LOG_FILE',
    'DEFAULT_FEATURE_WINDOWS',
    'DEFAULT_TARGET_WINDOWS',
    'DEFAULT_MODEL_TYPES',
    'DEFAULT_TRAIN_SPLIT',
    'CLASS_LABELS',
    # Cache management
    'load_features_from_cache',
    'save_features_to_cache',
    'load_target_from_cache',
    'save_target_to_cache',
    'load_training_log',
    'save_training_log',
    # Feature creation
    'create_features_for_window',
    'precalculate_all_features',
    # Target calculation
    'precalculate_all_targets',
    # Model training
    'train_single_model_optimized',
    # Main orchestrator
    'train_all_models_optimized',
]

