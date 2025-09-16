from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from fastapi_app.core.dependencies import get_db

from .schemas import DashboardStatsResponse, DashboardSummaryResponse
from .utils import DashboardService

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummaryResponse)
async def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get complete dashboard summary with stats and recent runs"""
    service = DashboardService(db)
    return await service.get_dashboard_summary()


@router.get("/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics only"""
    service = DashboardService(db)
    return await service.get_dashboard_stats()
