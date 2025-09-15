"""
Model Registry API
FastAPI application for model registry and A/B testing management.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

from model.registry.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Model Registry API",
    description="API for model versioning, deployment, and A/B testing management.",
    version="1.0.0",
)

# Initialize the model registry
model_registry = ModelRegistry()


class ModelRegistrationRequest(BaseModel):
    model_name: str
    version: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None


class ModelPromotionRequest(BaseModel):
    model_name: str
    version: str
    stage: str


class ABTestRequest(BaseModel):
    model_name: str
    model_a_version: str
    model_b_version: str
    traffic_split: float = 0.5


@app.get("/health", summary="Health Check")
async def health_check():
    """Perform a health check on the model registry service."""
    try:
        return {"status": "healthy", "message": "Model Registry service is operational."}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.get("/models", summary="List All Models")
async def list_models():
    """List all registered models."""
    try:
        # This would require additional implementation to list all models
        return {"models": ["breadthflow-model"], "message": "Model listing not fully implemented"}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {e}")


@app.get("/models/{model_name}/versions", summary="Get Model Versions")
async def get_model_versions(model_name: str):
    """Get all versions of a specific model."""
    try:
        versions = model_registry.get_model_versions(model_name)
        return {"model_name": model_name, "versions": versions}
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model versions: {e}")


@app.get("/models/{model_name}/production", summary="Get Production Model")
async def get_production_model(model_name: str):
    """Get the current production model."""
    try:
        production_model = model_registry.get_production_model(model_name)
        if production_model:
            return {"model_name": model_name, "production_model": production_model}
        else:
            return {"model_name": model_name, "production_model": None, "message": "No production model found"}
    except Exception as e:
        logger.error(f"Error getting production model: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting production model: {e}")


@app.post("/models/{model_name}/promote", summary="Promote Model")
async def promote_model(model_name: str, request: ModelPromotionRequest):
    """Promote a model to a specific stage."""
    try:
        success = model_registry.promote_model(model_name=request.model_name, version=request.version, stage=request.stage)

        if success:
            return {"message": f"Successfully promoted {model_name} version {request.version} to {request.stage}"}
        else:
            raise HTTPException(status_code=400, detail="Failed to promote model")
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=500, detail=f"Error promoting model: {e}")


@app.post("/ab-tests", summary="Create A/B Test")
async def create_ab_test(request: ABTestRequest):
    """Create a new A/B test."""
    try:
        ab_test = model_registry.create_ab_test(
            model_name=request.model_name,
            model_a_version=request.model_a_version,
            model_b_version=request.model_b_version,
            traffic_split=request.traffic_split,
        )

        return {"message": "A/B test created successfully", "ab_test": ab_test}
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating A/B test: {e}")


@app.get("/ab-tests", summary="List A/B Tests")
async def list_ab_tests():
    """List all A/B tests."""
    try:
        ab_tests = model_registry.get_ab_tests()
        return {"ab_tests": ab_tests}
    except Exception as e:
        logger.error(f"Error listing A/B tests: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing A/B tests: {e}")


@app.post("/ab-tests/{test_id}/stop", summary="Stop A/B Test")
async def stop_ab_test(test_id: str):
    """Stop an A/B test."""
    try:
        success = model_registry.stop_ab_test(test_id)

        if success:
            return {"message": f"A/B test {test_id} stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="A/B test not found")
    except Exception as e:
        logger.error(f"Error stopping A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping A/B test: {e}")


@app.get("/ab-tests/{test_id}", summary="Get A/B Test")
async def get_ab_test(test_id: str):
    """Get details of a specific A/B test."""
    try:
        ab_tests = model_registry.get_ab_tests()
        ab_test = next((test for test in ab_tests if test["test_id"] == test_id), None)

        if ab_test:
            return {"ab_test": ab_test}
        else:
            raise HTTPException(status_code=404, detail="A/B test not found")
    except Exception as e:
        logger.error(f"Error getting A/B test: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting A/B test: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
