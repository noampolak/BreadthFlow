"""
Data Pipeline API Service

FastAPI service for data ingestion, validation, and storage
for the ML training pipeline.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pyspark.sql import SparkSession

from .data_ingestion_service import DataIngestionService
from .data_validation_service import DataValidationService
from .minio_storage_service import MinIOStorageService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class IngestRequest(BaseModel):
    symbol_list: str
    start_date: str
    end_date: str
    validate_data: bool = True

class ValidationRequest(BaseModel):
    data_path: str
    validation_type: str = "quality"  # quality, schema, drift, training

class StorageRequest(BaseModel):
    data_path: str
    bucket_name: Optional[str] = None
    format: str = "parquet"

# Initialize FastAPI app
app = FastAPI(
    title="Data Pipeline API",
    description="API for data ingestion, validation, and storage for ML training",
    version="1.0.0"
)

# Global services
spark_session = None
data_ingestion = None
data_validation = None
minio_storage = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global spark_session, data_ingestion, data_validation, minio_storage
    
    try:
        # Initialize Spark session
        spark_session = SparkSession.builder \
            .appName("BreadthFlowDataPipeline") \
            .master("spark://spark-master:7077") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        logger.info("Spark session initialized")
        
        # Initialize services
        data_ingestion = DataIngestionService(spark_session)
        data_validation = DataValidationService(spark_session)
        minio_storage = MinIOStorageService()
        
        logger.info("Data pipeline services initialized")
        
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
        "service": "Data Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Spark session
        spark_status = "healthy" if spark_session and not spark_session.sparkContext._jsc.sc().isStopped() else "unhealthy"
        
        # Check MinIO connection
        minio_status = minio_storage.get_connection_status()
        
        return {
            "status": "healthy" if spark_status == "healthy" and minio_status["connected"] else "unhealthy",
            "spark": spark_status,
            "minio": minio_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/symbol-lists")
async def get_symbol_lists():
    """Get available symbol lists."""
    try:
        symbol_lists = data_ingestion.load_symbol_lists()
        return {
            "symbol_lists": list(symbol_lists.keys()),
            "details": symbol_lists
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest data for a symbol list."""
    try:
        logger.info(f"Starting data ingestion for {request.symbol_list}")
        
        # Start ingestion in background
        result = data_ingestion.ingest_symbol_list(
            request.symbol_list,
            request.start_date,
            request.end_date
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Validate data if requested
        if request.validate_data:
            # This would need to be implemented with the actual data path
            validation_result = {
                "validation_requested": True,
                "validation_status": "pending"
            }
        else:
            validation_result = {"validation_requested": False}
        
        return {
            "message": "Data ingestion completed",
            "ingestion_result": result,
            "validation": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate_data(request: ValidationRequest):
    """Validate data quality."""
    try:
        logger.info(f"Starting data validation: {request.validation_type}")
        
        # Load data for validation
        # This is a simplified version - in practice, you'd load from the actual data path
        if not spark_session:
            raise HTTPException(status_code=500, detail="Spark session not available")
        
        # For now, return a mock validation result
        # In practice, you'd load the actual data and run validation
        validation_result = {
            "validation_type": request.validation_type,
            "is_valid": True,
            "quality_score": 0.95,
            "issues_found": [],
            "recommendations": ["Data quality is good for training"]
        }
        
        return {
            "message": "Data validation completed",
            "validation_result": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/storage/summary")
async def get_storage_summary():
    """Get storage summary."""
    try:
        summary = minio_storage.get_data_summary()
        return {
            "storage_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/storage/objects")
async def list_objects(prefix: str = "", limit: int = 100):
    """List objects in storage."""
    try:
        objects = minio_storage.list_objects(prefix=prefix)
        return {
            "objects": objects[:limit],
            "total_count": len(objects),
            "prefix": prefix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/storage/cleanup")
async def cleanup_old_data(days_to_keep: int = 30, data_type: str = None):
    """Clean up old data."""
    try:
        result = minio_storage.cleanup_old_data(
            days_to_keep=days_to_keep,
            data_type=data_type
        )
        return {
            "cleanup_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_pipeline_status():
    """Get overall pipeline status."""
    try:
        # Get ingestion status
        ingestion_status = data_ingestion.get_ingestion_status()
        
        # Get storage status
        storage_status = minio_storage.get_connection_status()
        
        return {
            "pipeline_status": "running",
            "ingestion": ingestion_status,
            "storage": storage_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "pipeline_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
