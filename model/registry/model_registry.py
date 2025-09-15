"""
Model Registry Service
Manages model versioning, deployment, and A/B testing.
"""

import os
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Optional imports with fallbacks
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Some features will be limited.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model Registry for managing model versions and deployments.
    """

    def __init__(self, mlflow_tracking_uri: str = "http://mlflow:5000"):
        """Initialize the model registry."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = "breadthflow_trading"

        # Ensure experiment exists
        try:
            mlflow.create_experiment(self.experiment_name)
        except:
            pass  # Experiment already exists

        mlflow.set_experiment(self.experiment_name)
        logger.info(f"ModelRegistry initialized with MLflow URI: {mlflow_tracking_uri}")

    def register_model(
        self, model, model_name: str, version: str = None, metrics: Dict[str, float] = None, tags: Dict[str, str] = None
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version (auto-generated if None)
            metrics: Model performance metrics
            tags: Additional tags

        Returns:
            Model version
        """
        try:
            with mlflow.start_run(run_name=f"{model_name}_{version or 'latest'}"):
                # Log model
                mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

                # Log metrics
                if metrics:
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)

                # Log tags
                if tags:
                    for key, value in tags.items():
                        mlflow.log_param(key, value)

                # Log metadata
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("registered_at", datetime.now().isoformat())

                # Get the registered model version
                client = mlflow.tracking.MlflowClient()
                model_version = client.get_latest_versions(model_name)[0].version

                logger.info(f"Successfully registered model {model_name} version {model_version}")
                return model_version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a model.

        Args:
            model_name: Name of the model

        Returns:
            List of model versions with metadata
        """
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name)

            result = []
            for version in versions:
                result.append(
                    {
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "description": version.description,
                        "user_id": version.user_id,
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []

    def promote_model(self, model_name: str, version: str, stage: str) -> bool:
        """
        Promote a model to a specific stage.

        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (Staging, Production, Archived)

        Returns:
            Success status
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(name=model_name, version=version, stage=stage)

            logger.info(f"Successfully promoted {model_name} version {version} to {stage}")
            return True

        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            return False

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current production model.

        Args:
            model_name: Name of the model

        Returns:
            Production model metadata or None
        """
        try:
            client = mlflow.tracking.MlflowClient()
            production_models = client.get_latest_versions(model_name, stages=["Production"])

            if production_models:
                model = production_models[0]
                return {
                    "version": model.version,
                    "stage": model.current_stage,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "user_id": model.user_id,
                }

            return None

        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None

    def create_ab_test(
        self, model_name: str, model_a_version: str, model_b_version: str, traffic_split: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create an A/B test configuration.

        Args:
            model_name: Name of the model
            model_a_version: Version A
            model_b_version: Version B
            traffic_split: Traffic split ratio (0.0 to 1.0)

        Returns:
            A/B test configuration
        """
        ab_test_config = {
            "test_id": f"ab_test_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": model_name,
            "model_a_version": model_a_version,
            "model_b_version": model_b_version,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "active",
        }

        # Save A/B test configuration
        ab_test_file = f"/app/seldon/ab_tests/{ab_test_config['test_id']}.json"
        os.makedirs(os.path.dirname(ab_test_file), exist_ok=True)

        with open(ab_test_file, "w") as f:
            json.dump(ab_test_config, f, indent=2)

        logger.info(f"Created A/B test: {ab_test_config['test_id']}")
        return ab_test_config

    def get_ab_tests(self) -> List[Dict[str, Any]]:
        """
        Get all A/B tests.

        Returns:
            List of A/B test configurations
        """
        ab_tests = []
        ab_tests_dir = "/app/seldon/ab_tests"

        if os.path.exists(ab_tests_dir):
            for filename in os.listdir(ab_tests_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(ab_tests_dir, filename), "r") as f:
                            ab_test = json.load(f)
                            ab_tests.append(ab_test)
                    except Exception as e:
                        logger.error(f"Error loading A/B test {filename}: {e}")

        return ab_tests

    def stop_ab_test(self, test_id: str) -> bool:
        """
        Stop an A/B test.

        Args:
            test_id: A/B test ID

        Returns:
            Success status
        """
        try:
            ab_test_file = f"/app/seldon/ab_tests/{test_id}.json"

            if os.path.exists(ab_test_file):
                with open(ab_test_file, "r") as f:
                    ab_test = json.load(f)

                ab_test["status"] = "stopped"
                ab_test["stopped_at"] = datetime.now().isoformat()

                with open(ab_test_file, "w") as f:
                    json.dump(ab_test, f, indent=2)

                logger.info(f"Stopped A/B test: {test_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error stopping A/B test: {e}")
            return False
