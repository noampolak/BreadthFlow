from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from core.dependencies import get_db_session
from apps.infrastructure.schemas import SystemHealth, ServiceStatus
from apps.infrastructure.utils import InfrastructureService

router = APIRouter(prefix="/infrastructure", tags=["infrastructure"])

@router.get("/health", response_model=SystemHealth)
async def get_infrastructure_health_route(
    db: Session = Depends(get_db_session)
):
    """Get overall system health and service status"""
    try:
        service = InfrastructureService(db)
        return service.get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch infrastructure health: {str(e)}")
