"""
Simple Hyperparameter Optimizer
A basic hyperparameter optimizer that doesn't require Optuna.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ML dependencies - imported conditionally to avoid import errors
try:
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

    # Create dummy classes for type hints when sklearn is not available
    class DummySklearn:
        def __getattr__(self, name):
            raise ImportError("scikit-learn is not available. Install with: poetry install --with ml")

    joblib = DummySklearn()
    accuracy_score = DummySklearn()
    f1_score = DummySklearn()
    precision_score = DummySklearn()
    recall_score = DummySklearn()
    TimeSeriesSplit = DummySklearn()
    cross_val_score = DummySklearn()

logger = logging.getLogger(__name__)


class SimpleHyperparameterOptimizer:
    """
    Simple hyperparameter optimization without external dependencies.
    """

    def __init__(self, n_trials: int = 50, cv_folds: int = 5, scoring: str = "accuracy"):
        """
        Initialize the hyperparameter optimizer.

        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is not available. Install with: poetry install --with ml")

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.best_params = None
        self.best_score = None

        logger.info(f"SimpleHyperparameterOptimizer initialized with {n_trials} trials")

    def optimize(self, model_class, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, List]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.

        Args:
            model_class: Model class to optimize
            X: Training features
            y: Training target
            param_grid: Parameter grid for optimization

        Returns:
            Best parameters and score
        """
        logger.info(f"Starting hyperparameter optimization for {model_class.__name__}")

        best_score = -np.inf if self.scoring in ["accuracy", "f1", "precision", "recall"] else np.inf
        best_params = None

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)

        # Limit to n_trials
        if len(param_combinations) > self.n_trials:
            np.random.seed(42)
            param_combinations = np.random.choice(param_combinations, size=self.n_trials, replace=False).tolist()

        for i, params in enumerate(param_combinations):
            try:
                # Create model with current parameters
                model = model_class(**params)

                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring)

                mean_score = np.mean(cv_scores)

                # Update best if better
                is_better = (
                    mean_score > best_score
                    if self.scoring in ["accuracy", "f1", "precision", "recall"]
                    else mean_score < best_score
                )

                if is_better:
                    best_score = mean_score
                    best_params = params

                logger.info(f"Trial {i+1}/{len(param_combinations)}: {params} -> {mean_score:.4f}")

            except Exception as e:
                logger.warning(f"Trial {i+1} failed with params {params}: {e}")
                continue

        self.best_params = best_params
        self.best_score = best_score

        logger.info(f"Optimization completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return {"best_params": best_params, "best_score": best_score, "n_trials": len(param_combinations)}

    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found during optimization."""
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """Get the best score found during optimization."""
        return self.best_score
