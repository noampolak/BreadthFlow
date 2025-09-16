from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from fastapi_app.apps.infrastructure.schemas import ServiceStatus, SystemHealth
from fastapi_app.apps.infrastructure.utils import InfrastructureService
from fastapi_app.core.dependencies import get_db

router = APIRouter(prefix="/infrastructure", tags=["infrastructure"])


@router.get("/health", response_model=SystemHealth)
async def get_infrastructure_health_route(db: Session = Depends(get_db)):
    """Get overall system health and service status"""
    try:
        service = InfrastructureService(db)
        return service.get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch infrastructure health: {str(e)}")


@router.get("/metrics")
async def get_infrastructure_metrics(db: Session = Depends(get_db)):
    """Get system metrics"""
    try:
        service = InfrastructureService(db)
        return service.get_system_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch infrastructure metrics: {str(e)}")


@router.get("/logs")
async def get_infrastructure_logs(level: str = None, limit: int = 100, db: Session = Depends(get_db)):
    """Get system logs"""
    try:
        service = InfrastructureService(db)
        return service.get_system_logs(level=level, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch infrastructure logs: {str(e)}")
