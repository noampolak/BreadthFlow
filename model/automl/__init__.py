"""
AutoML Integration Module for BreadthFlow ML Pipeline

This module provides automated machine learning capabilities
using various AutoML frameworks including auto-sklearn, TPOT, and H2O.
"""

from .auto_sklearn_integration import AutoSklearnIntegration
from .automl_manager import AutoMLManager
from .h2o_integration import H2OIntegration
from .tpot_integration import TPOTIntegration

__all__ = ["AutoMLManager", "AutoSklearnIntegration", "TPOTIntegration", "H2OIntegration"]
