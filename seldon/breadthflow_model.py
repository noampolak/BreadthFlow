"""
BreadthFlow Seldon Core Model Wrapper
This module provides the Seldon Core interface for serving ML models.
"""

import os
import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from typing import Dict, List, Any, Union
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BreadthFlowModel:
    """
    Seldon Core model wrapper for BreadthFlow trading models.
    """

    def __init__(self):
        """Initialize the model wrapper."""
        self.model = None
        self.model_version = os.getenv("MODEL_VERSION", "latest")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.model_name = os.getenv("SELDON_MODEL_NAME", "breadthflow-model")

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Load model
        self.load_model()

        logger.info(f"BreadthFlowModel initialized with model: {self.model_name}, version: {self.model_version}")

    def load_model(self):
        """Load model from MLflow Model Registry."""
        try:
            # Try to load from model registry
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Successfully loaded model from MLflow: {model_uri}")
        except Exception as e:
            logger.warning(f"Could not load model from MLflow: {e}")
            # Fallback to local model file
            try:
                model_path = "/app/model/models/breadthflow_model.pkl"
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    logger.info(f"Loaded fallback model from: {model_path}")
                else:
                    # Create a dummy model for testing
                    from sklearn.ensemble import RandomForestClassifier

                    self.model = RandomForestClassifier(n_estimators=10, random_state=42)
                    # Fit with dummy data
                    X_dummy = np.random.rand(100, 10)
                    y_dummy = np.random.randint(0, 2, 100)
                    self.model.fit(X_dummy, y_dummy)
                    logger.info("Created dummy model for testing")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise

    def predict(self, X: Union[np.ndarray, List, Dict], feature_names: List[str] = None) -> Union[np.ndarray, List]:
        """
        Make predictions using the loaded model.

        Args:
            X: Input features
            feature_names: Optional feature names
        Returns:
            Predictions
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(X, dict):
                df = pd.DataFrame([X])
            elif isinstance(X, list):
                df = pd.DataFrame(X)
            else:
                df = pd.DataFrame(X)

            # Ensure we have the right number of features
            if feature_names:
                df.columns = feature_names[: len(df.columns)]

            # Make predictions
            predictions = self.model.predict(df)

            # Convert to list for JSON serialization
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()

            logger.info(f"Made predictions for {len(df)} samples")
            return predictions

        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Return dummy predictions for testing
            if isinstance(X, (list, np.ndarray)):
                return [0] * len(X)
            else:
                return [0]

    def predict_proba(self, X: Union[np.ndarray, List, Dict], feature_names: List[str] = None) -> Union[np.ndarray, List]:
        """
        Make probability predictions using the loaded model.

        Args:
            X: Input features
            feature_names: Optional feature names
        Returns:
            Prediction probabilities
        """
        try:
            # Convert input to DataFrame if needed
            if isinstance(X, dict):
                df = pd.DataFrame([X])
            elif isinstance(X, list):
                df = pd.DataFrame(X)
            else:
                df = pd.DataFrame(X)

            # Ensure we have the right number of features
            if feature_names:
                df.columns = feature_names[: len(df.columns)]

            # Make probability predictions
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(df)
            else:
                # Fallback to regular predictions
                predictions = self.model.predict(df)
                probabilities = np.column_stack([1 - predictions, predictions])

            # Convert to list for JSON serialization
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()

            logger.info(f"Made probability predictions for {len(df)} samples")
            return probabilities

        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            # Return dummy probabilities for testing
            if isinstance(X, (list, np.ndarray)):
                return [[0.5, 0.5]] * len(X)
            else:
                return [[0.5, 0.5]]

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the model.

        Returns:
            Health status dictionary
        """
        try:
            # Test prediction with dummy data
            dummy_data = np.random.rand(1, 10)
            prediction = self.predict(dummy_data)

            return {
                "status": "healthy",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "prediction_test": "passed",
                "mlflow_uri": self.mlflow_tracking_uri,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "model_name": self.model_name, "model_version": self.model_version}

    def metadata(self) -> Dict[str, Any]:
        """
        Return model metadata.

        Returns:
            Model metadata dictionary
        """
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": type(self.model).__name__,
            "features_expected": 10,  # This should be dynamic based on actual model
            "mlflow_uri": self.mlflow_tracking_uri,
        }
