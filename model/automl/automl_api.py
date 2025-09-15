"""
AutoML API Service

FastAPI service for automated machine learning operations including
auto-sklearn, TPOT, and H2O AutoML integration.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pyspark.sql import SparkSession

from .automl_manager import AutoMLManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class AutoMLRequest(BaseModel):
    data: Dict[str, Any]  # DataFrame as dictionary
    target_column: str
    problem_type: str = "classification"
    frameworks: List[str] = ["auto_sklearn", "tpot", "h2o"]
    time_limit: int = 300
    metric: str = "accuracy"


class QuickAutoMLRequest(BaseModel):
    data: Dict[str, Any]  # DataFrame as dictionary
    target_column: str
    problem_type: str = "classification"
    metric: str = "accuracy"


# Initialize FastAPI app
app = FastAPI(title="AutoML API", description="API for automated machine learning operations", version="1.0.0")

# Global services
spark_session = None
automl_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global spark_session, automl_manager

    try:
        # Initialize Spark session
        spark_session = (
            SparkSession.builder.appName("BreadthFlowAutoML")
            .master("spark://spark-master:7077")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )

        logger.info("Spark session initialized")

        # Initialize AutoML manager
        automl_manager = AutoMLManager()

        logger.info("AutoML services initialized")

    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global spark_session, automl_manager

    if automl_manager:
        automl_manager.shutdown()

    if spark_session:
        spark_session.stop()
        logger.info("Spark session stopped")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "AutoML API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Spark session
        spark_status = "healthy" if spark_session and not spark_session.sparkContext._jsc.sc().isStopped() else "unhealthy"

        # Check AutoML frameworks
        framework_status = automl_manager.get_framework_status() if automl_manager else {}

        return {
            "status": "healthy" if spark_status == "healthy" else "unhealthy",
            "spark": spark_status,
            "frameworks": framework_status,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}


@app.post("/train-automl")
async def train_automl(request: AutoMLRequest):
    """Train models using all available AutoML frameworks."""
    try:
        logger.info("Starting AutoML training request")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)

        # Prepare data
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if request.problem_type == "classification" else None
        )

        # Train with all frameworks
        results = automl_manager.train_all_frameworks(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            problem_type=request.problem_type,
            target_column=request.target_column,
        )

        return {"success": True, "results": results, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error in AutoML training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train-best-automl")
async def train_best_automl(request: QuickAutoMLRequest):
    """Train with the best performing AutoML framework."""
    try:
        logger.info("Starting best AutoML training request")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)

        # Prepare data
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if request.problem_type == "classification" else None
        )

        # Train with best framework
        result = automl_manager.train_best_framework(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            problem_type=request.problem_type,
            target_column=request.target_column,
            metric=request.metric,
        )

        return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error in best AutoML training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frameworks")
async def get_frameworks():
    """Get available AutoML frameworks."""
    try:
        status = automl_manager.get_framework_status()
        available = automl_manager.get_available_frameworks()

        return {"frameworks": status, "available": available, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/frameworks/{framework_name}/info")
async def get_framework_info(framework_name: str):
    """Get detailed information about a specific framework."""
    try:
        if framework_name not in automl_manager.frameworks:
            raise HTTPException(status_code=404, detail="Framework not found")

        framework = automl_manager.frameworks[framework_name]

        info = {"name": framework_name, "available": framework.is_available(), "type": type(framework).__name__}

        # Add framework-specific information
        if framework_name == "auto_sklearn" and framework.is_available():
            info["estimators"] = framework.get_available_estimators()
        elif framework_name == "tpot" and framework.is_available():
            info["description"] = "Tree-based Pipeline Optimization Tool"
        elif framework_name == "h2o" and framework.is_available():
            info["description"] = "H2O AutoML with Deep Learning"

        return {"framework_info": info, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare-frameworks")
async def compare_frameworks(request: AutoMLRequest):
    """Compare performance of different AutoML frameworks."""
    try:
        logger.info("Starting framework comparison")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)

        # Prepare data
        X = df.drop(columns=[request.target_column])
        y = df[request.target_column]

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if request.problem_type == "classification" else None
        )

        # Train with all frameworks
        results = automl_manager.train_all_frameworks(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            problem_type=request.problem_type,
            target_column=request.target_column,
        )

        # Extract comparison data
        comparison_data = []
        for framework_name, result in results["results"].items():
            if result.get("success", False):
                if "test_metrics" in result:
                    test_metrics = result["test_metrics"]
                    training_time = result.get("model_info", {}).get("training_time", 0)

                    comparison_data.append(
                        {"framework": framework_name, "metrics": test_metrics, "training_time": training_time, "success": True}
                    )
                else:
                    comparison_data.append(
                        {"framework": framework_name, "success": False, "error": result.get("error", "Unknown error")}
                    )
            else:
                comparison_data.append(
                    {"framework": framework_name, "success": False, "error": result.get("error", "Unknown error")}
                )

        return {
            "success": True,
            "comparison": comparison_data,
            "summary": results.get("comparison", {}),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in framework comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
