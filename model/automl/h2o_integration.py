"""
H2O AutoML Integration

Integration with H2O AutoML for comprehensive automated
machine learning including deep learning and ensemble methods.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import H2O
try:
    import h2o
    from h2o.automl import H2OAutoML

    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    h2o = None
    H2OAutoML = None

logger = logging.getLogger(__name__)


class H2OIntegration:
    """
    Integration with H2O AutoML for comprehensive automated machine learning.

    Features:
    - Comprehensive AutoML with deep learning
    - Ensemble methods and stacking
    - Model interpretability
    - Distributed computing support
    """

    def __init__(self, max_models: int = 20, max_runtime_secs: int = 3600, seed: int = 42, nfolds: int = 5):
        """
        Initialize H2O integration.

        Args:
            max_models: Maximum number of models to train
            max_runtime_secs: Maximum runtime in seconds
            seed: Random seed
            nfolds: Number of cross-validation folds
        """
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.nfolds = nfolds
        self.logger = logging.getLogger(__name__)

        if not H2O_AVAILABLE:
            self.logger.warning("H2O not available. Install with: pip install h2o")
        else:
            # Initialize H2O
            try:
                h2o.init(nthreads=1, max_mem_size="2G")
                self.logger.info("H2O initialized successfully")
            except Exception as e:
                self.logger.error(f"Error initializing H2O: {str(e)}")
                # Don't modify the global variable, just log the error

        self.logger.info(f"H2OIntegration initialized (available: {H2O_AVAILABLE})")

    def train_automl(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_column: str = "target",
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """
        Train an H2O AutoML model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            target_column: Name of the target column
            problem_type: Type of problem (classification or regression)

        Returns:
            Dictionary with training results
        """
        try:
            if not H2O_AVAILABLE:
                return {"error": "H2O not available"}

            self.logger.info("Starting H2O AutoML training")
            start_time = datetime.now()

            # Prepare data
            train_data = X_train.copy()
            train_data[target_column] = y_train

            test_data = X_test.copy()
            test_data[target_column] = y_test

            # Convert to H2O frames
            train_h2o = h2o.H2OFrame(train_data)
            test_h2o = h2o.H2OFrame(test_data)

            # Identify feature columns
            feature_columns = [col for col in train_h2o.columns if col != target_column]

            # Initialize H2O AutoML
            aml = H2OAutoML(
                max_models=self.max_models,
                max_runtime_secs=self.max_runtime_secs,
                seed=self.seed,
                nfolds=self.nfolds,
                sort_metric="AUTO",
            )

            # Train the model
            aml.train(x=feature_columns, y=target_column, training_frame=train_h2o, validation_frame=test_h2o)

            # Get the best model
            best_model = aml.leader

            # Make predictions
            train_pred = best_model.predict(train_h2o)
            test_pred = best_model.predict(test_h2o)

            # Convert predictions to numpy arrays
            if problem_type == "classification":
                train_pred_array = train_pred.as_data_frame()["predict"].values
                test_pred_array = test_pred.as_data_frame()["predict"].values
            else:
                train_pred_array = train_pred.as_data_frame()["predict"].values
                test_pred_array = test_pred.as_data_frame()["predict"].values

            # Calculate metrics
            if problem_type == "classification":
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

                train_metrics = {
                    "train_accuracy": accuracy_score(y_train, train_pred_array),
                    "train_precision": precision_score(y_train, train_pred_array, average="weighted", zero_division=0),
                    "train_recall": recall_score(y_train, train_pred_array, average="weighted", zero_division=0),
                    "train_f1": f1_score(y_train, train_pred_array, average="weighted", zero_division=0),
                }

                test_metrics = {
                    "test_accuracy": accuracy_score(y_test, test_pred_array),
                    "test_precision": precision_score(y_test, test_pred_array, average="weighted", zero_division=0),
                    "test_recall": recall_score(y_test, test_pred_array, average="weighted", zero_division=0),
                    "test_f1": f1_score(y_test, test_pred_array, average="weighted", zero_division=0),
                }
            else:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                train_metrics = {
                    "train_mse": mean_squared_error(y_train, train_pred_array),
                    "train_mae": mean_absolute_error(y_train, train_pred_array),
                    "train_r2": r2_score(y_train, train_pred_array),
                }

                test_metrics = {
                    "test_mse": mean_squared_error(y_test, test_pred_array),
                    "test_mae": mean_absolute_error(y_test, test_pred_array),
                    "test_r2": r2_score(y_test, test_pred_array),
                }

            # Get leaderboard
            leaderboard = aml.leaderboard.as_data_frame()

            # Get model information
            model_info = {
                "model": aml,
                "best_model": best_model,
                "leaderboard": leaderboard,
                "training_time": (datetime.now() - start_time).total_seconds(),
                "models_trained": len(leaderboard),
                "best_model_id": best_model.model_id,
            }

            self.logger.info(f"H2O AutoML training completed in {model_info['training_time']:.2f}s")

            return {
                "success": True,
                "model": aml,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "model_info": model_info,
            }

        except Exception as e:
            self.logger.error(f"Error in H2O AutoML training: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_model_interpretability(self, model, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Get model interpretability information.

        Args:
            model: Trained H2O AutoML model
            X_test: Test features

        Returns:
            Dictionary with interpretability information
        """
        try:
            if not H2O_AVAILABLE:
                return {"error": "H2O not available"}

            best_model = model.leader

            # Get variable importance
            var_imp = best_model.varimp(use_pandas=True)

            # Get partial dependence plots (if available)
            try:
                pdp_data = best_model.partial_plot(
                    data=h2o.H2OFrame(X_test), cols=best_model.varimp(use_pandas=True).head(5)["variable"].tolist(), plot=False
                )
            except:
                pdp_data = None

            interpretability = {
                "variable_importance": var_imp,
                "partial_dependence": pdp_data,
                "model_summary": best_model.summary(),
            }

            return interpretability

        except Exception as e:
            self.logger.error(f"Error getting model interpretability: {str(e)}")
            return {"error": str(e)}

    def get_ensemble_info(self, model) -> Dict[str, Any]:
        """
        Get information about the ensemble models.

        Args:
            model: Trained H2O AutoML model

        Returns:
            Dictionary with ensemble information
        """
        try:
            if not H2O_AVAILABLE:
                return {"error": "H2O not available"}

            leaderboard = model.leaderboard.as_data_frame()

            ensemble_info = {
                "leaderboard": leaderboard,
                "ensemble_models": leaderboard[leaderboard["model_id"].str.contains("StackedEnsemble")],
                "base_models": leaderboard[~leaderboard["model_id"].str.contains("StackedEnsemble")],
                "best_model_id": model.leader.model_id,
                "best_model_type": type(model.leader).__name__,
            }

            return ensemble_info

        except Exception as e:
            self.logger.error(f"Error getting ensemble info: {str(e)}")
            return {"error": str(e)}

    def save_model(self, model, path: str) -> bool:
        """
        Save the H2O model to disk.

        Args:
            model: Trained H2O model
            path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            if not H2O_AVAILABLE:
                return False

            model_path = h2o.save_model(model.leader, path=path, force=True)
            self.logger.info(f"Model saved to {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, path: str) -> Any:
        """
        Load an H2O model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded H2O model
        """
        try:
            if not H2O_AVAILABLE:
                return None

            model = h2o.load_model(path)
            self.logger.info(f"Model loaded from {path}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def shutdown(self):
        """
        Shutdown H2O cluster.
        """
        try:
            if H2O_AVAILABLE:
                h2o.cluster().shutdown()
                self.logger.info("H2O cluster shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down H2O: {str(e)}")

    def is_available(self) -> bool:
        """
        Check if H2O is available.

        Returns:
            True if available, False otherwise
        """
        return H2O_AVAILABLE
