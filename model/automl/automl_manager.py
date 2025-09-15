"""
AutoML Manager

Orchestrates multiple AutoML frameworks including auto-sklearn,
TPOT, and H2O AutoML for comprehensive automated machine learning.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .auto_sklearn_integration import AutoSklearnIntegration
from .h2o_integration import H2OIntegration
from .tpot_integration import TPOTIntegration

logger = logging.getLogger(__name__)


class AutoMLManager:
    """
    Manager for multiple AutoML frameworks.

    Features:
    - Integration with auto-sklearn, TPOT, and H2O AutoML
    - Automated framework selection
    - Ensemble of AutoML results
    - Performance comparison and ranking
    """

    def __init__(self, time_limit: int = 300, memory_limit: int = 3072, include_frameworks: List[str] = None):
        """
        Initialize AutoML manager.

        Args:
            time_limit: Time limit in seconds for optimization
            memory_limit: Memory limit in MB for optimization
            include_frameworks: List of frameworks to include
        """
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.logger = logging.getLogger(__name__)

        # Initialize framework integrations
        self.frameworks = {}

        if include_frameworks is None:
            include_frameworks = ["auto_sklearn", "tpot", "h2o"]

        # Initialize available frameworks
        if "auto_sklearn" in include_frameworks:
            self.frameworks["auto_sklearn"] = AutoSklearnIntegration(time_limit=time_limit, memory_limit=memory_limit)

        if "tpot" in include_frameworks:
            self.frameworks["tpot"] = TPOTIntegration(generations=5, population_size=20)

        if "h2o" in include_frameworks:
            self.frameworks["h2o"] = H2OIntegration(max_models=20, max_runtime_secs=time_limit)

        self.logger.info(f"AutoMLManager initialized with frameworks: {list(self.frameworks.keys())}")

    def train_all_frameworks(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str = "classification",
        target_column: str = "target",
    ) -> Dict[str, Any]:
        """
        Train models using all available AutoML frameworks.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            problem_type: Type of problem (classification or regression)
            target_column: Name of the target column

        Returns:
            Dictionary with results from all frameworks
        """
        try:
            self.logger.info(f"Starting AutoML training with {len(self.frameworks)} frameworks")
            start_time = datetime.now()

            results = {}

            # Train with each framework
            for framework_name, framework in self.frameworks.items():
                self.logger.info(f"Training with {framework_name}")

                try:
                    if framework_name == "auto_sklearn":
                        if problem_type == "classification":
                            result = framework.train_classification_model(X_train, y_train, X_test, y_test)
                        else:
                            result = framework.train_regression_model(X_train, y_train, X_test, y_test)

                    elif framework_name == "tpot":
                        if problem_type == "classification":
                            result = framework.train_classification_pipeline(X_train, y_train, X_test, y_test)
                        else:
                            result = framework.train_regression_pipeline(X_train, y_train, X_test, y_test)

                    elif framework_name == "h2o":
                        result = framework.train_automl(
                            X_train, y_train, X_test, y_test, target_column=target_column, problem_type=problem_type
                        )

                    results[framework_name] = result

                except Exception as e:
                    self.logger.error(f"Error training with {framework_name}: {str(e)}")
                    results[framework_name] = {"success": False, "error": str(e)}

            # Compare results
            comparison = self._compare_framework_results(results, problem_type)

            total_time = (datetime.now() - start_time).total_seconds()

            return {
                "success": True,
                "results": results,
                "comparison": comparison,
                "total_time": total_time,
                "frameworks_used": list(self.frameworks.keys()),
            }

        except Exception as e:
            self.logger.error(f"Error in AutoML training: {str(e)}")
            return {"success": False, "error": str(e)}

    def train_best_framework(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str = "classification",
        target_column: str = "target",
        metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        Train with the best performing framework based on previous results.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            problem_type: Type of problem (classification or regression)
            target_column: Name of the target column
            metric: Metric to use for comparison

        Returns:
            Dictionary with results from the best framework
        """
        try:
            # Quick evaluation of all frameworks
            self.logger.info("Quick evaluation of all frameworks")

            # Use a subset of data for quick evaluation
            subset_size = min(1000, len(X_train))
            X_train_subset = X_train.head(subset_size)
            y_train_subset = y_train.head(subset_size)
            X_test_subset = X_test.head(min(200, len(X_test)))
            y_test_subset = y_test.head(min(200, len(y_test)))

            # Quick training with reduced parameters
            quick_results = {}

            for framework_name, framework in self.frameworks.items():
                try:
                    if framework_name == "auto_sklearn":
                        # Use shorter time limit for quick evaluation
                        framework.time_limit = 60
                        if problem_type == "classification":
                            result = framework.train_classification_model(
                                X_train_subset, y_train_subset, X_test_subset, y_test_subset
                            )
                        else:
                            result = framework.train_regression_model(
                                X_train_subset, y_train_subset, X_test_subset, y_test_subset
                            )

                    elif framework_name == "tpot":
                        # Use fewer generations for quick evaluation
                        framework.generations = 2
                        if problem_type == "classification":
                            result = framework.train_classification_pipeline(
                                X_train_subset, y_train_subset, X_test_subset, y_test_subset
                            )
                        else:
                            result = framework.train_regression_pipeline(
                                X_train_subset, y_train_subset, X_test_subset, y_test_subset
                            )

                    elif framework_name == "h2o":
                        # Use fewer models for quick evaluation
                        framework.max_models = 5
                        result = framework.train_automl(
                            X_train_subset,
                            y_train_subset,
                            X_test_subset,
                            y_test_subset,
                            target_column=target_column,
                            problem_type=problem_type,
                        )

                    quick_results[framework_name] = result

                except Exception as e:
                    self.logger.error(f"Error in quick evaluation with {framework_name}: {str(e)}")
                    quick_results[framework_name] = {"success": False, "error": str(e)}

            # Find best framework
            best_framework = self._find_best_framework(quick_results, metric)

            if best_framework is None:
                return {"success": False, "error": "No framework performed successfully"}

            # Train with best framework on full data
            self.logger.info(f"Training with best framework: {best_framework}")

            framework = self.frameworks[best_framework]

            if best_framework == "auto_sklearn":
                if problem_type == "classification":
                    result = framework.train_classification_model(X_train, y_train, X_test, y_test)
                else:
                    result = framework.train_regression_model(X_train, y_train, X_test, y_test)

            elif best_framework == "tpot":
                if problem_type == "classification":
                    result = framework.train_classification_pipeline(X_train, y_train, X_test, y_test)
                else:
                    result = framework.train_regression_pipeline(X_train, y_train, X_test, y_test)

            elif best_framework == "h2o":
                result = framework.train_automl(
                    X_train, y_train, X_test, y_test, target_column=target_column, problem_type=problem_type
                )

            result["best_framework"] = best_framework
            return result

        except Exception as e:
            self.logger.error(f"Error in best framework training: {str(e)}")
            return {"success": False, "error": str(e)}

    def _compare_framework_results(self, results: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
        """
        Compare results from different frameworks.

        Args:
            results: Results from all frameworks
            problem_type: Type of problem (classification or regression)

        Returns:
            Dictionary with comparison results
        """
        try:
            comparison_data = []

            for framework_name, result in results.items():
                if result.get("success", False) and "test_metrics" in result:
                    test_metrics = result["test_metrics"]

                    if problem_type == "classification":
                        score = test_metrics.get("test_accuracy", 0)
                    else:
                        score = test_metrics.get("test_r2", 0)

                    comparison_data.append(
                        {
                            "framework": framework_name,
                            "score": score,
                            "metrics": test_metrics,
                            "training_time": result.get("model_info", {}).get("training_time", 0),
                        }
                    )

            if not comparison_data:
                return {"error": "No successful results to compare"}

            # Sort by score
            comparison_data.sort(key=lambda x: x["score"], reverse=True)

            best_framework = comparison_data[0]["framework"]
            best_score = comparison_data[0]["score"]

            return {
                "best_framework": best_framework,
                "best_score": best_score,
                "comparison_table": comparison_data,
                "summary": {
                    "total_frameworks": len(comparison_data),
                    "successful_frameworks": len([r for r in results.values() if r.get("success", False)]),
                    "failed_frameworks": len([r for r in results.values() if not r.get("success", False)]),
                },
            }

        except Exception as e:
            self.logger.error(f"Error comparing framework results: {str(e)}")
            return {"error": str(e)}

    def _find_best_framework(self, results: Dict[str, Any], metric: str) -> Optional[str]:
        """
        Find the best performing framework.

        Args:
            results: Results from frameworks
            metric: Metric to use for comparison

        Returns:
            Name of the best framework or None
        """
        try:
            best_framework = None
            best_score = -float("inf")

            for framework_name, result in results.items():
                if result.get("success", False) and "test_metrics" in result:
                    test_metrics = result["test_metrics"]
                    score = test_metrics.get(f"test_{metric}", 0)

                    if score > best_score:
                        best_score = score
                        best_framework = framework_name

            return best_framework

        except Exception as e:
            self.logger.error(f"Error finding best framework: {str(e)}")
            return None

    def get_framework_status(self) -> Dict[str, Any]:
        """
        Get status of all available frameworks.

        Returns:
            Dictionary with framework status
        """
        status = {}

        for framework_name, framework in self.frameworks.items():
            status[framework_name] = {"available": framework.is_available(), "type": type(framework).__name__}

        return status

    def shutdown(self):
        """
        Shutdown all frameworks.
        """
        try:
            for framework_name, framework in self.frameworks.items():
                if hasattr(framework, "shutdown"):
                    framework.shutdown()

            self.logger.info("All frameworks shutdown")

        except Exception as e:
            self.logger.error(f"Error shutting down frameworks: {str(e)}")

    def get_available_frameworks(self) -> List[str]:
        """
        Get list of available frameworks.

        Returns:
            List of available framework names
        """
        return [name for name, framework in self.frameworks.items() if framework.is_available()]
