import json
import os
from datetime import datetime, timedelta
from typing import List

from fastapi_app.apps.signals.schemas import SignalExportResponse, SignalStats, TradingSignal
from fastapi_app.apps.signals.utils import SignalService
from fastapi_app.core.dependencies import get_db_session
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("/latest", response_model=dict)
async def get_latest_signals_route(db: Session = Depends(get_db_session)):
    """Get latest trading signals with statistics"""
    try:
        service = SignalService(db)
        signals, stats = service.get_latest_signals()
        return {"signals": signals, "stats": stats, "last_updated": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch signals: {str(e)}")


@router.get("/export", response_model=SignalExportResponse)
async def export_signals_route(format: str = "json", db: Session = Depends(get_db_session)):
    """Export trading signals in CSV or JSON format"""
    try:
        service = SignalService(db)
        return service.export_signals(format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export signals: {str(e)}")
