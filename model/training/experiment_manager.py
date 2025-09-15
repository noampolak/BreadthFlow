"""
Experiment Manager for MLflow Integration

Manages MLflow experiments, runs, and model registry operations
for the BreadthFlow ML training pipeline.
"""

import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    Manages MLflow experiments and model registry operations.

    Features:
    - Experiment creation and management
    - Run tracking and logging
    - Model registry operations
    - Artifact management
    - Model versioning and deployment
    """

    def __init__(self, tracking_uri: str = "http://mlflow:5000"):
        """
        Initialize the experiment manager.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri
        self.logger = logging.getLogger(__name__)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Initialize experiment names
        self.experiments = {
            "breadthflow_trading": "BreadthFlow Trading Models",
            "feature_engineering": "Feature Engineering Experiments",
            "hyperparameter_tuning": "Hyperparameter Optimization",
            "model_comparison": "Model Comparison Studies",
        }

        self.logger.info(f"ExperimentManager initialized with tracking URI: {tracking_uri}")

    def create_experiment(self, experiment_name: str, description: str = None) -> str:
        """
        Create a new MLflow experiment.

        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment

        Returns:
            Experiment ID
        """
        try:
            # Check if experiment already exists
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment:
                self.logger.info(f"Experiment '{experiment_name}' already exists")
                return existing_experiment.experiment_id

            # Create new experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name, tags={"description": description or f"Experiment: {experiment_name}"}
            )

            self.logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
            return experiment_id

        except Exception as e:
            self.logger.error(f"Error creating experiment '{experiment_name}': {str(e)}")
            raise

    def start_run(self, experiment_name: str, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of the run
            tags: Additional tags for the run

        Returns:
            Active MLflow run
        """
        try:
            # Get or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                experiment_id = self.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id

            # Set experiment
            mlflow.set_experiment(experiment_name)

            # Start run
            run = mlflow.start_run(run_name=run_name, tags=tags)

            self.logger.info(f"Started run '{run_name}' in experiment '{experiment_name}'")
            return run

        except Exception as e:
            self.logger.error(f"Error starting run: {str(e)}")
            raise

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameters to log
        """
        try:
            mlflow.log_params(params)
            self.logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            self.logger.error(f"Error logging parameters: {str(e)}")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        try:
            if step is not None:
                for name, value in metrics.items():
                    mlflow.log_metric(name, value, step=step)
            else:
                mlflow.log_metrics(metrics)

            self.logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")
            raise

    def log_model(
        self,
        model: Any,
        model_name: str,
        model_type: str = "sklearn",
        signature: mlflow.models.ModelSignature = None,
        input_example: Any = None,
    ) -> str:
        """
        Log a model to MLflow.

        Args:
            model: The trained model
            model_name: Name of the model
            model_type: Type of model (sklearn, xgboost, lightgbm)
            signature: Model signature
            input_example: Example input for the model

        Returns:
            Model URI
        """
        try:
            if model_type == "sklearn":
                model_uri = mlflow.sklearn.log_model(
                    sk_model=model, artifact_path=model_name, signature=signature, input_example=input_example
                )
            elif model_type == "xgboost":
                model_uri = mlflow.xgboost.log_model(
                    xgb_model=model, artifact_path=model_name, signature=signature, input_example=input_example
                )
            elif model_type == "lightgbm":
                model_uri = mlflow.lightgbm.log_model(
                    lgb_model=model, artifact_path=model_name, signature=signature, input_example=input_example
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.logger.info(f"Logged {model_type} model '{model_name}'")
            return model_uri

        except Exception as e:
            self.logger.error(f"Error logging model: {str(e)}")
            raise

    def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Log artifacts to the current run.

        Args:
            artifacts: Dictionary of artifacts to log
        """
        try:
            for name, artifact in artifacts.items():
                if isinstance(artifact, (pd.DataFrame, np.ndarray)):
                    # Save as CSV or NPY
                    if isinstance(artifact, pd.DataFrame):
                        artifact.to_csv(f"/tmp/{name}.csv", index=False)
                        mlflow.log_artifact(f"/tmp/{name}.csv", artifact_path="data")
                    else:
                        np.save(f"/tmp/{name}.npy", artifact)
                        mlflow.log_artifact(f"/tmp/{name}.npy", artifact_path="data")
                elif isinstance(artifact, str) and os.path.exists(artifact):
                    # Log file artifact
                    mlflow.log_artifact(artifact, artifact_path="files")
                else:
                    # Log as text
                    with open(f"/tmp/{name}.txt", "w") as f:
                        f.write(str(artifact))
                    mlflow.log_artifact(f"/tmp/{name}.txt", artifact_path="text")

            self.logger.info(f"Logged {len(artifacts)} artifacts")
        except Exception as e:
            self.logger.error(f"Error logging artifacts: {str(e)}")
            raise

    def register_model(self, model_uri: str, model_name: str, tags: Dict[str, str] = None) -> str:
        """
        Register a model in the MLflow Model Registry.

        Args:
            model_uri: URI of the logged model
            model_name: Name for the registered model
            tags: Tags for the registered model

        Returns:
            Registered model version
        """
        try:
            result = mlflow.register_model(model_uri=model_uri, name=model_name, tags=tags)

            self.logger.info(f"Registered model '{model_name}' version {result.version}")
            return result.version

        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """
        Transition a model to a specific stage.

        Args:
            model_name: Name of the registered model
            version: Version of the model
            stage: Target stage (Staging, Production, Archived)
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(name=model_name, version=version, stage=stage)

            self.logger.info(f"Transitioned model '{model_name}' v{version} to {stage}")
        except Exception as e:
            self.logger.error(f"Error transitioning model stage: {str(e)}")
            raise

    def get_best_model(self, experiment_name: str, metric: str = "accuracy") -> Dict[str, Any]:
        """
        Get the best model from an experiment based on a metric.

        Args:
            experiment_name: Name of the experiment
            metric: Metric to use for comparison

        Returns:
            Dictionary with best model information
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            # Get all runs from the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            if runs.empty:
                raise ValueError(f"No runs found in experiment '{experiment_name}'")

            # Find best run based on metric
            if metric not in runs.columns:
                raise ValueError(f"Metric '{metric}' not found in runs")

            best_run = runs.loc[runs[metric].idxmax()]

            return {
                "run_id": best_run["run_id"],
                "metric_value": best_run[metric],
                "metric_name": metric,
                "model_uri": f"runs:/{best_run['run_id']}/model",
            }

        except Exception as e:
            self.logger.error(f"Error getting best model: {str(e)}")
            raise

    def compare_models(self, experiment_name: str, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare models from an experiment.

        Args:
            experiment_name: Name of the experiment
            metrics: List of metrics to compare

        Returns:
            DataFrame with model comparison results
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            # Get all runs from the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            if runs.empty:
                raise ValueError(f"No runs found in experiment '{experiment_name}'")

            # Select metrics columns
            if metrics:
                metric_cols = [col for col in metrics if col in runs.columns]
            else:
                metric_cols = [col for col in runs.columns if col.startswith("metrics.")]

            # Create comparison DataFrame
            comparison = runs[["run_id", "run_name"] + metric_cols].copy()
            comparison = comparison.sort_values(by=metric_cols[0], ascending=False)

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise

    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get summary of an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment summary
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment '{experiment_name}' not found")

            # Get all runs from the experiment
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            summary = {
                "experiment_name": experiment_name,
                "experiment_id": experiment.experiment_id,
                "total_runs": len(runs),
                "creation_time": experiment.creation_time,
                "tags": experiment.tags,
            }

            if not runs.empty:
                # Get metric statistics
                metric_cols = [col for col in runs.columns if col.startswith("metrics.")]
                if metric_cols:
                    summary["metrics"] = runs[metric_cols].describe().to_dict()

                # Get parameter statistics
                param_cols = [col for col in runs.columns if col.startswith("params.")]
                if param_cols:
                    summary["parameters"] = runs[param_cols].nunique().to_dict()

            return summary

        except Exception as e:
            self.logger.error(f"Error getting experiment summary: {str(e)}")
            raise
