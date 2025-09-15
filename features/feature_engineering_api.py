"""
Feature Engineering API Service

FastAPI service for feature engineering operations including
technical indicators, time features, and microstructure analysis.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pyspark.sql import SparkSession

from .feature_engineering_service import FeatureEngineeringService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class FeatureEngineeringRequest(BaseModel):
    data: Dict[str, Any]  # DataFrame as dictionary
    feature_types: List[str] = ["technical", "time", "microstructure"]
    target_column: Optional[str] = None


class BatchFeatureEngineeringRequest(BaseModel):
    symbols: List[str]
    feature_types: List[str] = ["technical", "time", "microstructure"]
    target_column: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Feature Engineering API", description="API for feature engineering operations in ML pipeline", version="1.0.0"
)

# Global services
spark_session = None
feature_engineering_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global spark_session, feature_engineering_service

    try:
        # Initialize Spark session
        spark_session = (
            SparkSession.builder.appName("BreadthFlowFeatureEngineering")
            .master("spark://spark-master:7077")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate()
        )

        logger.info("Spark session initialized")

        # Initialize feature engineering service
        feature_engineering_service = FeatureEngineeringService(spark_session)

        logger.info("Feature engineering services initialized")

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
    return {"service": "Feature Engineering API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Spark session
        spark_status = "healthy" if spark_session and not spark_session.sparkContext._jsc.sc().isStopped() else "unhealthy"

        # Check feature engineering service
        service_status = feature_engineering_service.get_service_status()

        return {
            "status": "healthy" if spark_status == "healthy" and service_status["status"] == "healthy" else "unhealthy",
            "spark": spark_status,
            "feature_engineering": service_status,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}


@app.post("/engineer-features")
async def engineer_features(request: FeatureEngineeringRequest):
    """Engineer features for a single dataset."""
    try:
        logger.info("Starting feature engineering request")

        # Convert dictionary to DataFrame
        df = pd.DataFrame(request.data)

        # Engineer features
        result = feature_engineering_service.engineer_features(
            df=df, feature_types=request.feature_types, target_column=request.target_column
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        # Convert result DataFrame to dictionary for JSON response
        result_data = result["data"].to_dict(orient="records")

        return {"success": True, "data": result_data, "metadata": result["metadata"], "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/engineer-features-batch")
async def engineer_features_batch(request: BatchFeatureEngineeringRequest, background_tasks: BackgroundTasks):
    """Engineer features for multiple symbols."""
    try:
        logger.info(f"Starting batch feature engineering for {len(request.symbols)} symbols")

        # This would typically load data from MinIO for each symbol
        # For now, we'll create mock data
        data_dict = {}
        for symbol in request.symbols:
            # Create mock OHLCV data
            dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
            mock_data = pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": np.random.uniform(100, 200, len(dates)),
                    "high": np.random.uniform(100, 200, len(dates)),
                    "low": np.random.uniform(100, 200, len(dates)),
                    "close": np.random.uniform(100, 200, len(dates)),
                    "volume": np.random.uniform(1000000, 10000000, len(dates)),
                }
            )
            data_dict[symbol] = mock_data

        # Engineer features for all symbols
        result = feature_engineering_service.engineer_features_for_symbols(
            data_dict=data_dict, feature_types=request.feature_types
        )

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "success": True,
            "results": result["results"],
            "total_symbols": result["total_symbols"],
            "total_duration_seconds": result["total_duration_seconds"],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in batch feature engineering: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-summary")
async def get_feature_summary():
    """Get summary of available feature types."""
    try:
        summary = {
            "technical_indicators": {
                "description": "Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)",
                "categories": ["moving_averages", "momentum", "volatility", "trend", "volume"],
            },
            "time_features": {
                "description": "Time-based and cyclical features",
                "categories": ["cyclical", "seasonal", "business_calendar", "market_sessions"],
            },
            "microstructure_features": {
                "description": "Market microstructure and order flow features",
                "categories": ["volume_patterns", "price_volume", "order_flow", "liquidity"],
            },
        }

        return {"feature_types": summary, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/service-status")
async def get_service_status():
    """Get detailed service status."""
    try:
        status = feature_engineering_service.get_service_status()
        return {"service_status": status, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
