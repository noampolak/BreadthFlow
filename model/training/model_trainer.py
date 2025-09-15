"""
Model Trainer for ML Pipeline

Comprehensive model training with support for multiple algorithms,
cross-validation, and performance evaluation.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .experiment_manager import ExperimentManager
from .simple_hyperparameter_optimizer import SimpleHyperparameterOptimizer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model trainer for machine learning models.

    Features:
    - Support for multiple ML algorithms
    - Time series aware cross-validation
    - Comprehensive performance evaluation
    - Model persistence and loading
    - Integration with MLflow
    """

    def __init__(
        self, experiment_manager: ExperimentManager = None, hyperparameter_optimizer: SimpleHyperparameterOptimizer = None
    ):
        """
        Initialize the model trainer.

        Args:
            experiment_manager: MLflow experiment manager
            hyperparameter_optimizer: Hyperparameter optimization tool
        """
        self.experiment_manager = experiment_manager or ExperimentManager()
        self.hyperparameter_optimizer = hyperparameter_optimizer or SimpleHyperparameterOptimizer()
        self.logger = logging.getLogger(__name__)

        # Model registry
        self.models = {}
        self.scalers = {}

        self.logger.info("ModelTrainer initialized")

    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare data for training.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            scale_features: Whether to scale features

        Returns:
            Dictionary with prepared data
        """
        try:
            self.logger.info("Preparing data for training")

            # Handle missing values
            X_clean = X.fillna(X.mean())
            y_clean = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)

            # Encode categorical variables
            categorical_columns = X_clean.select_dtypes(include=["object", "category"]).columns
            label_encoders = {}

            for col in categorical_columns:
                le = LabelEncoder()
                X_clean[col] = le.fit_transform(X_clean[col].astype(str))
                label_encoders[col] = le

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean,
                y_clean,
                test_size=test_size,
                random_state=random_state,
                stratify=y_clean if y_clean.nunique() > 1 else None,
            )

            # Scale features if requested
            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Store scaler
                self.scalers["default"] = scaler
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Convert back to DataFrames
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_clean.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_clean.columns, index=X_test.index)

            self.logger.info(f"Data prepared: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")

            return {
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train,
                "y_test": y_test,
                "scaler": scaler,
                "label_encoders": label_encoders,
                "feature_names": X_clean.columns.tolist(),
            }

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_model(
        self,
        model_class,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict[str, Any] = None,
        experiment_name: str = "breadthflow_trading",
        run_name: str = None,
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.

        Args:
            model_class: Model class to train
            model_name: Name for the model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            params: Model parameters
            experiment_name: MLflow experiment name
            run_name: MLflow run name

        Returns:
            Dictionary with training results
        """
        try:
            self.logger.info(f"Training {model_name} model")
            start_time = datetime.now()

            # Initialize model
            model_params = params or {}
            model = model_class(**model_params)

            # Start MLflow run
            with self.experiment_manager.start_run(experiment_name, run_name) as run:
                # Log parameters
                self.experiment_manager.log_parameters(
                    {"model_name": model_name, "model_class": model_class.__name__, **model_params}
                )

                # Train model
                model.fit(X_train, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_train_pred, "train")
                test_metrics = self._calculate_metrics(y_test, y_test_pred, "test")

                # Log metrics
                self.experiment_manager.log_metrics(train_metrics)
                self.experiment_manager.log_metrics(test_metrics)

                # Log model
                model_uri = self.experiment_manager.log_model(model=model, model_name=model_name, model_type="sklearn")

                # Store model
                self.models[model_name] = {
                    "model": model,
                    "metrics": {**train_metrics, **test_metrics},
                    "model_uri": model_uri,
                    "run_id": run.info.run_id,
                }

                training_time = (datetime.now() - start_time).total_seconds()

                self.logger.info(f"Model {model_name} trained successfully in {training_time:.2f}s")

                return {
                    "model": model,
                    "model_name": model_name,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "model_uri": model_uri,
                    "run_id": run.info.run_id,
                    "training_time": training_time,
                }

        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {str(e)}")
            raise

    def train_multiple_models(
        self,
        model_configs: List[Dict[str, Any]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        experiment_name: str = "breadthflow_trading",
    ) -> Dict[str, Any]:
        """
        Train multiple models and compare their performance.

        Args:
            model_configs: List of model configurations
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            experiment_name: MLflow experiment name

        Returns:
            Dictionary with training results for all models
        """
        try:
            self.logger.info(f"Training {len(model_configs)} models")
            results = {}

            for config in model_configs:
                model_name = config["name"]
                model_class = config["class"]
                params = config.get("params", {})

                try:
                    result = self.train_model(
                        model_class=model_class,
                        model_name=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        params=params,
                        experiment_name=experiment_name,
                        run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    )
                    results[model_name] = result

                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {str(e)}")
                    results[model_name] = {"error": str(e)}

            # Compare models
            comparison = self._compare_models(results)

            return {
                "results": results,
                "comparison": comparison,
                "best_model": comparison["best_model"] if comparison else None,
            }

        except Exception as e:
            self.logger.error(f"Error training multiple models: {str(e)}")
            raise

    def optimize_and_train(
        self,
        model_class,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_space: Dict[str, Any],
        experiment_name: str = "breadthflow_trading",
        optimization_method: str = "sklearn",
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters and train the best model.

        Args:
            model_class: Model class to optimize
            model_name: Name for the model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            param_space: Parameter space for optimization
            experiment_name: MLflow experiment name
            optimization_method: Optimization method to use

        Returns:
            Dictionary with optimization and training results
        """
        try:
            self.logger.info(f"Optimizing and training {model_name}")

            # Optimize hyperparameters
            if optimization_method == "sklearn":
                optimization_result = self.hyperparameter_optimizer.optimize_sklearn_model(
                    model_class=model_class, X=X_train, y=y_train, param_space=param_space
                )
            elif optimization_method == "xgboost":
                optimization_result = self.hyperparameter_optimizer.optimize_xgboost(X=X_train, y=y_train)
            elif optimization_method == "lightgbm":
                optimization_result = self.hyperparameter_optimizer.optimize_lightgbm(X=X_train, y=y_train)
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")

            # Train model with best parameters
            best_params = optimization_result["best_params"]
            training_result = self.train_model(
                model_class=model_class,
                model_name=f"{model_name}_optimized",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=best_params,
                experiment_name=experiment_name,
                run_name=f"{model_name}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            return {"optimization_result": optimization_result, "training_result": training_result, "best_params": best_params}

        except Exception as e:
            self.logger.error(f"Error optimizing and training {model_name}: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            prefix: Prefix for metric names

        Returns:
            Dictionary with metrics
        """
        try:
            metrics = {}

            # Basic metrics
            metrics[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred)
            metrics[f"{prefix}_precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics[f"{prefix}_recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics[f"{prefix}_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            # ROC AUC (for binary classification)
            if len(np.unique(y_true)) == 2:
                try:
                    metrics[f"{prefix}_roc_auc"] = roc_auc_score(y_true, y_pred)
                except:
                    metrics[f"{prefix}_roc_auc"] = 0.0

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple models.

        Args:
            results: Dictionary with model results

        Returns:
            Dictionary with comparison results
        """
        try:
            comparison_data = []

            for model_name, result in results.items():
                if "error" not in result:
                    comparison_data.append(
                        {
                            "model_name": model_name,
                            "test_accuracy": result["test_metrics"].get("test_accuracy", 0),
                            "test_f1": result["test_metrics"].get("test_f1", 0),
                            "training_time": result.get("training_time", 0),
                        }
                    )

            if not comparison_data:
                return {"error": "No successful model training results"}

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("test_accuracy", ascending=False)

            best_model = comparison_df.iloc[0]["model_name"]

            return {
                "comparison_table": comparison_df,
                "best_model": best_model,
                "best_accuracy": comparison_df.iloc[0]["test_accuracy"],
            }

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {"error": str(e)}

    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to file.

        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            model_data = self.models[model_name]
            joblib.dump(model_data, filepath)

            self.logger.info(f"Model {model_name} saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {str(e)}")
            raise

    def load_model(self, model_name: str, filepath: str) -> None:
        """
        Load a trained model from file.

        Args:
            model_name: Name for the loaded model
            filepath: Path to the model file
        """
        try:
            model_data = joblib.load(filepath)
            self.models[model_name] = model_data

            self.logger.info(f"Model {model_name} loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the model to use
            X: Feature matrix

        Returns:
            Predictions
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            model = self.models[model_name]["model"]

            # Scale features if scaler is available
            if "default" in self.scalers:
                X_scaled = self.scalers["default"].transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            else:
                X_scaled = X

            predictions = model.predict(X_scaled)

            return predictions

        except Exception as e:
            self.logger.error(f"Error making predictions with {model_name}: {str(e)}")
            raise

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a trained model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model information
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            model_data = self.models[model_name]

            return {
                "model_name": model_name,
                "metrics": model_data["metrics"],
                "model_uri": model_data.get("model_uri"),
                "run_id": model_data.get("run_id"),
            }

        except Exception as e:
            self.logger.error(f"Error getting model info for {model_name}: {str(e)}")
            raise
