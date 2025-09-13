"""
Seldon Core Management Script
Provides utilities for managing Seldon deployments and A/B tests.
"""

import os
import yaml
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeldonManager:
    """
    Manager for Seldon Core deployments and A/B tests.
    """
    
    def __init__(self, seldon_api_url: str = "http://seldon-core-operator:8080"):
        """Initialize the Seldon manager."""
        self.seldon_api_url = seldon_api_url
        self.namespace = "seldon-system"
        
    def deploy_model(self, model_name: str, model_version: str = "latest", 
                    replicas: int = 2) -> Dict[str, Any]:
        """
        Deploy a model using Seldon Core.
        
        Args:
            model_name: Name of the model
            model_version: Model version
            replicas: Number of replicas
            
        Returns:
            Deployment status
        """
        try:
            # Load deployment template
            deployment_template = self._load_deployment_template()
            
            # Update template with model details
            deployment_template["metadata"]["name"] = model_name
            deployment_template["spec"]["name"] = model_name
            
            # Update model version in environment variables
            for predictor in deployment_template["spec"]["predictors"]:
                for container_spec in predictor["componentSpecs"]:
                    for container in container_spec["spec"]["containers"]:
                        if container["name"] == "model":
                            container["env"] = [
                                {"name": "SELDON_MODEL_NAME", "value": model_name},
                                {"name": "MODEL_VERSION", "value": model_version},
                                {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                            ]
                            container["replicas"] = replicas
            
            # Deploy using Seldon Core API
            response = requests.post(
                f"{self.seldon_api_url}/api/v1/namespaces/{self.namespace}/seldondeployments",
                json=deployment_template,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully deployed model {model_name} version {model_version}")
                return {
                    "status": "success",
                    "model_name": model_name,
                    "model_version": model_version,
                    "replicas": replicas,
                    "deployment_url": f"{self.seldon_api_url}/seldondeployments/{model_name}"
                }
            else:
                logger.error(f"Failed to deploy model: {response.text}")
                return {
                    "status": "failed",
                    "error": response.text
                }
                
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def create_ab_test(self, model_name: str, model_a_version: str, 
                      model_b_version: str, traffic_split: float = 0.5) -> Dict[str, Any]:
        """
        Create an A/B test between two model versions.
        
        Args:
            model_name: Name of the model
            model_a_version: Version A
            model_b_version: Version B
            traffic_split: Traffic split ratio (0.0 to 1.0)
            
        Returns:
            A/B test status
        """
        try:
            # Load A/B test template
            ab_test_template = self._load_ab_test_template()
            
            # Update template with model details
            ab_test_name = f"{model_name}-ab-test"
            ab_test_template["metadata"]["name"] = ab_test_name
            ab_test_template["spec"]["name"] = ab_test_name
            
            # Update model versions and traffic split
            predictors = ab_test_template["spec"]["predictors"]
            
            # Model A
            predictors[0]["traffic"] = int(traffic_split * 100)
            for container_spec in predictors[0]["componentSpecs"]:
                for container in container_spec["spec"]["containers"]:
                    if container["name"] == "model":
                        container["env"] = [
                            {"name": "SELDON_MODEL_NAME", "value": model_name},
                            {"name": "MODEL_VERSION", "value": model_a_version},
                            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                        ]
            
            # Model B
            predictors[1]["traffic"] = int((1 - traffic_split) * 100)
            for container_spec in predictors[1]["componentSpecs"]:
                for container in container_spec["spec"]["containers"]:
                    if container["name"] == "model":
                        container["env"] = [
                            {"name": "SELDON_MODEL_NAME", "value": model_name},
                            {"name": "MODEL_VERSION", "value": model_b_version},
                            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                        ]
            
            # Deploy A/B test
            response = requests.post(
                f"{self.seldon_api_url}/api/v1/namespaces/{self.namespace}/seldondeployments",
                json=ab_test_template,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201]:
                logger.info(f"Successfully created A/B test for {model_name}")
                return {
                    "status": "success",
                    "ab_test_name": ab_test_name,
                    "model_name": model_name,
                    "model_a_version": model_a_version,
                    "model_b_version": model_b_version,
                    "traffic_split": traffic_split,
                    "deployment_url": f"{self.seldon_api_url}/seldondeployments/{ab_test_name}"
                }
            else:
                logger.error(f"Failed to create A/B test: {response.text}")
                return {
                    "status": "failed",
                    "error": response.text
                }
                
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """
        Get the status of a Seldon deployment.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Deployment status
        """
        try:
            response = requests.get(
                f"{self.seldon_api_url}/api/v1/namespaces/{self.namespace}/seldondeployments/{deployment_name}"
            )
            
            if response.status_code == 200:
                deployment = response.json()
                return {
                    "status": "success",
                    "deployment": deployment
                }
            else:
                return {
                    "status": "failed",
                    "error": f"Deployment {deployment_name} not found"
                }
                
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def list_deployments(self) -> List[Dict[str, Any]]:
        """
        List all Seldon deployments.
        
        Returns:
            List of deployments
        """
        try:
            response = requests.get(
                f"{self.seldon_api_url}/api/v1/namespaces/{self.namespace}/seldondeployments"
            )
            
            if response.status_code == 200:
                deployments = response.json()
                return deployments.get("items", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return []
    
    def delete_deployment(self, deployment_name: str) -> Dict[str, Any]:
        """
        Delete a Seldon deployment.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Deletion status
        """
        try:
            response = requests.delete(
                f"{self.seldon_api_url}/api/v1/namespaces/{self.namespace}/seldondeployments/{deployment_name}"
            )
            
            if response.status_code in [200, 404]:
                logger.info(f"Successfully deleted deployment {deployment_name}")
                return {
                    "status": "success",
                    "message": f"Deployment {deployment_name} deleted"
                }
            else:
                logger.error(f"Failed to delete deployment: {response.text}")
                return {
                    "status": "failed",
                    "error": response.text
                }
                
        except Exception as e:
            logger.error(f"Error deleting deployment: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _load_deployment_template(self) -> Dict[str, Any]:
        """Load the deployment template."""
        template_path = "/app/seldon/seldon_deployment.yaml"
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default template
            return {
                "apiVersion": "machinelearning.seldon.io/v1",
                "kind": "SeldonDeployment",
                "metadata": {
                    "name": "breadthflow-trading",
                    "namespace": "seldon-system"
                },
                "spec": {
                    "name": "breadthflow-trading",
                    "predictors": [{
                        "name": "default",
                        "replicas": 2,
                        "componentSpecs": [{
                            "spec": {
                                "containers": [{
                                    "name": "model",
                                    "image": "breadthflow-seldon:latest",
                                    "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                                    "env": [
                                        {"name": "SELDON_MODEL_NAME", "value": "breadthflow-model"},
                                        {"name": "MODEL_VERSION", "value": "latest"},
                                        {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                                    ]
                                }]
                            }
                        }],
                        "traffic": 100
                    }]
                }
            }
    
    def _load_ab_test_template(self) -> Dict[str, Any]:
        """Load the A/B test template."""
        template_path = "/app/seldon/ab_test_config.yaml"
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default A/B test template
            return {
                "apiVersion": "machinelearning.seldon.io/v1",
                "kind": "SeldonDeployment",
                "metadata": {
                    "name": "breadthflow-ab-test",
                    "namespace": "seldon-system"
                },
                "spec": {
                    "name": "breadthflow-ab-test",
                    "predictors": [
                        {
                            "name": "model-a",
                            "replicas": 2,
                            "componentSpecs": [{
                                "spec": {
                                    "containers": [{
                                        "name": "model",
                                        "image": "breadthflow-seldon:latest",
                                        "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                                        "env": [
                                            {"name": "SELDON_MODEL_NAME", "value": "breadthflow-model"},
                                            {"name": "MODEL_VERSION", "value": "v1"},
                                            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                                        ]
                                    }]
                                }
                            }],
                            "traffic": 50
                        },
                        {
                            "name": "model-b",
                            "replicas": 2,
                            "componentSpecs": [{
                                "spec": {
                                    "containers": [{
                                        "name": "model",
                                        "image": "breadthflow-seldon:latest",
                                        "ports": [{"containerPort": 8000, "protocol": "TCP"}],
                                        "env": [
                                            {"name": "SELDON_MODEL_NAME", "value": "breadthflow-model"},
                                            {"name": "MODEL_VERSION", "value": "v2"},
                                            {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"}
                                        ]
                                    }]
                                }
                            }],
                            "traffic": 50
                        }
                    ]
                }
            }
