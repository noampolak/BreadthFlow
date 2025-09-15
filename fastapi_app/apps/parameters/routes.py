from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime

from core.dependencies import get_db_session
from apps.parameters.schemas import ParameterGroup, ParameterValue, ParameterUpdate
from apps.parameters.utils import ParametersService

router = APIRouter(prefix="/parameters", tags=["parameters"])


@router.get("/groups", response_model=List[ParameterGroup])
async def get_parameter_groups(db: Session = Depends(get_db_session)):
    """Get all parameter groups and their values"""
    try:
        service = ParametersService(db)
        return service.get_parameter_groups()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parameters: {str(e)}")


@router.put("/update", response_model=Dict[str, str])
async def update_parameters(updates: List[ParameterUpdate], db: Session = Depends(get_db_session)):
    """Update parameter values"""
    try:
        service = ParametersService(db)
        service.update_parameters(updates)
        return {"message": "Parameters updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update parameters: {str(e)}")


@router.post("/reset/{group_name}")
async def reset_parameter_group(group_name: str, db: Session = Depends(get_db_session)):
    """Reset parameter group to default values"""
    try:
        service = ParametersService(db)
        service.reset_parameter_group(group_name)
        return {"message": f"Parameter group '{group_name}' reset to defaults"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset parameters: {str(e)}")


@router.get("/export")
async def export_parameters(format: str = "json", db: Session = Depends(get_db_session)):
    """Export parameters in JSON or YAML format"""
    try:
        service = ParametersService(db)
        return service.export_parameters(format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export parameters: {str(e)}")


@router.post("/import")
async def import_parameters(parameters_data: Dict[str, Any], db: Session = Depends(get_db_session)):
    """Import parameters from JSON or YAML"""
    try:
        service = ParametersService(db)
        service.import_parameters(parameters_data)
        return {"message": "Parameters imported successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import parameters: {str(e)}")


@router.get("/history")
async def get_parameter_history(limit: int = 50, db: Session = Depends(get_db_session)):
    """Get parameter change history"""
    try:
        service = ParametersService(db)
        return service.get_parameter_history(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch parameter history: {str(e)}")
