"""
Auto-sklearn Integration

Integration with auto-sklearn for automated machine learning
including automated model selection and hyperparameter optimization.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import auto-sklearn
try:
    import autosklearn.classification
    import autosklearn.regression

    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False
    autosklearn = None

logger = logging.getLogger(__name__)


class AutoSklearnIntegration:
    """
    Integration with auto-sklearn for automated machine learning.

    Features:
    - Automated model selection
    - Hyperparameter optimization
    - Ensemble learning
    - Time series support
    """

    def __init__(self, time_limit: int = 300, memory_limit: int = 3072):
        """
        Initialize auto-sklearn integration.

        Args:
            time_limit: Time limit in seconds for optimization
            memory_limit: Memory limit in MB for optimization
        """
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.logger = logging.getLogger(__name__)

        if not AUTOSKLEARN_AVAILABLE:
            self.logger.warning("auto-sklearn not available. Install with: pip install auto-sklearn")

        self.logger.info(f"AutoSklearnIntegration initialized (available: {AUTOSKLEARN_AVAILABLE})")

    def train_classification_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_name: str = "breadthflow_classification",
        include_preprocessors: List[str] = None,
        include_estimators: List[str] = None,
        exclude_estimators: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a classification model using auto-sklearn.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            task_name: Name for the task
            include_preprocessors: List of preprocessors to include
            include_estimators: List of estimators to include
            exclude_estimators: List of estimators to exclude

        Returns:
            Dictionary with training results
        """
        try:
            if not AUTOSKLEARN_AVAILABLE:
                return {"error": "auto-sklearn not available"}

            self.logger.info("Starting auto-sklearn classification training")
            start_time = datetime.now()

            # Initialize auto-sklearn classifier
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=self.time_limit,
                memory_limit=self.memory_limit,
                include_preprocessors=include_preprocessors,
                include_estimators=include_estimators,
                exclude_estimators=exclude_estimators,
                n_jobs=1,  # Single job for stability
                seed=42,
            )

            # Train the model
            automl.fit(X_train, y_train)

            # Make predictions
            y_train_pred = automl.predict(X_train)
            y_test_pred = automl.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            train_metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "train_precision": precision_score(y_train, y_train_pred, average="weighted", zero_division=0),
                "train_recall": recall_score(y_train, y_train_pred, average="weighted", zero_division=0),
                "train_f1": f1_score(y_train, y_train_pred, average="weighted", zero_division=0),
            }

            test_metrics = {
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "test_recall": recall_score(y_test, y_test_pred, average="weighted", zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
            }

            # Get model information
            model_info = {
                "model": automl,
                "ensemble_size": len(automl.get_models_with_weights()),
                "models_used": [str(model) for model in automl.get_models_with_weights()],
                "best_individual_score": automl.score(X_test, y_test),
                "training_time": (datetime.now() - start_time).total_seconds(),
            }

            self.logger.info(f"Auto-sklearn training completed in {model_info['training_time']:.2f}s")

            return {
                "success": True,
                "model": automl,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "model_info": model_info,
            }

        except Exception as e:
            self.logger.error(f"Error in auto-sklearn classification: {str(e)}")
            return {"success": False, "error": str(e)}

    def train_regression_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_name: str = "breadthflow_regression",
        include_preprocessors: List[str] = None,
        include_estimators: List[str] = None,
        exclude_estimators: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a regression model using auto-sklearn.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            task_name: Name for the task
            include_preprocessors: List of preprocessors to include
            include_estimators: List of estimators to include
            exclude_estimators: List of estimators to exclude

        Returns:
            Dictionary with training results
        """
        try:
            if not AUTOSKLEARN_AVAILABLE:
                return {"error": "auto-sklearn not available"}

            self.logger.info("Starting auto-sklearn regression training")
            start_time = datetime.now()

            # Initialize auto-sklearn regressor
            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=self.time_limit,
                memory_limit=self.memory_limit,
                include_preprocessors=include_preprocessors,
                include_estimators=include_estimators,
                exclude_estimators=exclude_estimators,
                n_jobs=1,  # Single job for stability
                seed=42,
            )

            # Train the model
            automl.fit(X_train, y_train)

            # Make predictions
            y_train_pred = automl.predict(X_train)
            y_test_pred = automl.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            train_metrics = {
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_r2": r2_score(y_train, y_train_pred),
            }

            test_metrics = {
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_r2": r2_score(y_test, y_test_pred),
            }

            # Get model information
            model_info = {
                "model": automl,
                "ensemble_size": len(automl.get_models_with_weights()),
                "models_used": [str(model) for model in automl.get_models_with_weights()],
                "best_individual_score": automl.score(X_test, y_test),
                "training_time": (datetime.now() - start_time).total_seconds(),
            }

            self.logger.info(f"Auto-sklearn regression training completed in {model_info['training_time']:.2f}s")

            return {
                "success": True,
                "model": automl,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "model_info": model_info,
            }

        except Exception as e:
            self.logger.error(f"Error in auto-sklearn regression: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_available_estimators(self) -> Dict[str, List[str]]:
        """
        Get available estimators and preprocessors.

        Returns:
            Dictionary with available components
        """
        try:
            if not AUTOSKLEARN_AVAILABLE:
                return {"error": "auto-sklearn not available"}

            # Get available preprocessors
            preprocessors = [
                "no_preprocessing",
                "normalizer",
                "minmax_scaler",
                "standard_scaler",
                "robust_scaler",
                "quantile_transformer",
                "power_transformer",
                "polynomial_features",
                "feature_selection",
                "pca",
                "truncated_svd",
                "fast_ica",
                "kernel_pca",
                "random_trees_embedding",
            ]

            # Get available estimators for classification
            classification_estimators = [
                "adaboost",
                "bernoulli_nb",
                "decision_tree",
                "extra_trees",
                "gaussian_nb",
                "gradient_boosting",
                "k_nearest_neighbors",
                "lda",
                "liblinear_svc",
                "libsvm_svc",
                "multinomial_nb",
                "passive_aggressive",
                "qda",
                "random_forest",
                "sgd",
                "xgradient_boosting",
            ]

            # Get available estimators for regression
            regression_estimators = [
                "adaboost",
                "ard_regression",
                "decision_tree",
                "extra_trees",
                "gaussian_process",
                "gradient_boosting",
                "k_nearest_neighbors",
                "liblinear_svr",
                "libsvm_svr",
                "passive_aggressive",
                "random_forest",
                "sgd",
                "xgradient_boosting",
            ]

            return {
                "preprocessors": preprocessors,
                "classification_estimators": classification_estimators,
                "regression_estimators": regression_estimators,
            }

        except Exception as e:
            self.logger.error(f"Error getting available estimators: {str(e)}")
            return {"error": str(e)}

    def get_model_explanations(self, model, X_test: pd.DataFrame, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Get model explanations and feature importance.

        Args:
            model: Trained auto-sklearn model
            X_test: Test features
            feature_names: Names of features

        Returns:
            Dictionary with model explanations
        """
        try:
            if not AUTOSKLEARN_AVAILABLE:
                return {"error": "auto-sklearn not available"}

            explanations = {}

            # Get ensemble weights
            models_with_weights = model.get_models_with_weights()
            explanations["ensemble_weights"] = [
                {"model": str(model), "weight": weight} for model, weight in models_with_weights
            ]

            # Get performance statistics
            explanations["performance_stats"] = model.get_models_with_weights()

            # Get configuration space
            explanations["configuration_space"] = str(model.get_configuration_space())

            return explanations

        except Exception as e:
            self.logger.error(f"Error getting model explanations: {str(e)}")
            return {"error": str(e)}

    def is_available(self) -> bool:
        """
        Check if auto-sklearn is available.

        Returns:
            True if available, False otherwise
        """
        return AUTOSKLEARN_AVAILABLE
