"""
Model Training API Service

FastAPI service for model training operations including
experiment tracking, hyperparameter optimization, and model management.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pyspark.sql import SparkSession

from .model_trainer import ModelTrainer
from .experiment_manager import ExperimentManager
from .hyperparameter_optimizer import HyperparameterOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class TrainingRequest(BaseModel):
    data: Dict[str, Any]  # DataFrame as dictionary
    target_column: str
    model_configs: List[Dict[str, Any]]
    experiment_name: str = "breadthflow_training"
    test_size: float = 0.2

class OptimizationRequest(BaseModel):
    data: Dict[str, Any]  # DataFrame as dictionary
    target_column: str
    model_class: str
    param_space: Dict[str, Any]
    experiment_name: str = "breadthflow_training"
    optimization_method: str = "sklearn"

class PredictionRequest(BaseModel):
    model_name: str
    data: Dict[str, Any]  # DataFrame as dictionary

# Initialize FastAPI app
app = FastAPI(
    title="Model Training API",
    description="API for model training and experiment tracking in ML pipeline",
    version="1.0.0"
)

# Global services
spark_session = None
model_trainer = None
experiment_manager = None
hyperparameter_optimizer = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global spark_session, model_trainer, experiment_manager, hyperparameter_optimizer
    
    try:
        # Initialize Spark session
        spark_session = SparkSession.builder \
            .appName("BreadthFlowModelTraining") \
            .master("spark://spark-master:7077") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        logger.info("Spark session initialized")
        
        # Initialize services
        experiment_manager = ExperimentManager()
        hyperparameter_optimizer = HyperparameterOptimizer()
        model_trainer = ModelTrainer(experiment_manager, hyperparameter_optimizer)
        
        logger.info("Model training services initialized")
        
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global spark_session
    
    if spark_session:
        spark_session.stop()
        logger.info("Spark session stopped")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Model Training API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Spark session
        spark_status = "healthy" if spark_session and not spark_session.sparkContext._jsc.sc().isStopped() else "unhealthy"
        
        # Check MLflow connection
        mlflow_status = "healthy"
        try:
            experiment_manager.get_experiment_summary("breadthflow_training")
        except:
            mlflow_status = "unhealthy"
        
        return {
            "status": "healthy" if spark_status == "healthy" and mlflow_status == "healthy" else "unhealthy",
            "spark": spark_status,
            "mlflow": mlflow_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/train-models")
async def train_models(request: TrainingRequest):
    """Train multiple models and compare their performance."""
    try:
        logger.info("Starting model training request")
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)
        
        # Prepare data
        data_prep = model_trainer.prepare_data(
            X=df.drop(columns=[request.target_column]),
            y=df[request.target_column],
            test_size=request.test_size
        )
        
        # Convert model class strings to actual classes
        model_configs = []
        for config in request.model_configs:
            model_config = config.copy()
            model_class_name = model_config["class"]
            
            # Map string names to actual classes
            if model_class_name == "RandomForestClassifier":
                from sklearn.ensemble import RandomForestClassifier
                model_config["class"] = RandomForestClassifier
            elif model_class_name == "XGBClassifier":
                import xgboost as xgb
                model_config["class"] = xgb.XGBClassifier
            elif model_class_name == "LGBMClassifier":
                import lightgbm as lgb
                model_config["class"] = lgb.LGBMClassifier
            elif model_class_name == "SVC":
                from sklearn.svm import SVC
                model_config["class"] = SVC
            elif model_class_name == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression
                model_config["class"] = LogisticRegression
            else:
                raise ValueError(f"Unknown model class: {model_class_name}")
            
            model_configs.append(model_config)
        
        # Train models
        results = model_trainer.train_multiple_models(
            model_configs=model_configs,
            X_train=data_prep["X_train"],
            y_train=data_prep["y_train"],
            X_test=data_prep["X_test"],
            y_test=data_prep["y_test"],
            experiment_name=request.experiment_name
        )
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-and-train")
async def optimize_and_train(request: OptimizationRequest):
    """Optimize hyperparameters and train the best model."""
    try:
        logger.info("Starting hyperparameter optimization and training")
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)
        
        # Prepare data
        data_prep = model_trainer.prepare_data(
            X=df.drop(columns=[request.target_column]),
            y=df[request.target_column]
        )
        
        # Convert model class string to actual class
        if request.model_class == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
        elif request.model_class == "XGBClassifier":
            import xgboost as xgb
            model_class = xgb.XGBClassifier
        elif request.model_class == "LGBMClassifier":
            import lightgbm as lgb
            model_class = lgb.LGBMClassifier
        else:
            raise ValueError(f"Unknown model class: {request.model_class}")
        
        # Optimize and train
        results = model_trainer.optimize_and_train(
            model_class=model_class,
            model_name=f"{request.model_class}_optimized",
            X_train=data_prep["X_train"],
            y_train=data_prep["y_train"],
            X_test=data_prep["X_test"],
            y_test=data_prep["y_test"],
            param_space=request.param_space,
            experiment_name=request.experiment_name,
            optimization_method=request.optimization_method
        )
        
        return {
            "success": True,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in optimization and training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using a trained model."""
    try:
        logger.info(f"Making predictions with model: {request.model_name}")
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model_trainer.predict(request.model_name, df)
        
        return {
            "success": True,
            "predictions": predictions.tolist(),
            "model_name": request.model_name,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments")
async def list_experiments():
    """List all MLflow experiments."""
    try:
        experiments = experiment_manager.experiments
        return {
            "experiments": experiments,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/{experiment_name}/summary")
async def get_experiment_summary(experiment_name: str):
    """Get summary of an experiment."""
    try:
        summary = experiment_manager.get_experiment_summary(experiment_name)
        return {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/{experiment_name}/best-model")
async def get_best_model(experiment_name: str, metric: str = "test_accuracy"):
    """Get the best model from an experiment."""
    try:
        best_model = experiment_manager.get_best_model(experiment_name, metric)
        return {
            "best_model": best_model,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all trained models."""
    try:
        models = list(model_trainer.models.keys())
        return {
            "models": models,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    try:
        model_info = model_trainer.get_model_info(model_name)
        return {
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
