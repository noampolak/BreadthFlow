import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from fastapi_app.apps.commands.schemas import CommandHistory, CommandRequest, CommandResponse
from fastapi_app.apps.commands.utils import CommandService
from fastapi_app.core.dependencies import get_db

router = APIRouter(prefix="/commands", tags=["commands"])


@router.post("/execute", response_model=CommandResponse)
async def execute_command(request: CommandRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Execute a command and return the result"""
    try:
        service = CommandService(db)
        result = await service.execute_command(request, background_tasks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute command: {str(e)}")


@router.get("/history", response_model=List[CommandHistory])
async def get_command_history(limit: int = 50, db: Session = Depends(get_db)):
    """Get command execution history"""
    try:
        service = CommandService(db)
        return service.get_command_history(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch command history: {str(e)}")


@router.get("/quick-flows", response_model=List[Dict[str, Any]])
async def get_quick_flows():
    """Get available quick flow templates"""
    return [
        {
            "id": "data_fetch_flow",
            "name": "Data Fetch Flow",
            "description": "Fetch data for multiple symbols",
            "commands": [
                "data fetch --symbols AAPL,MSFT,GOOGL --timeframe 1day --start-date 2024-01-01 --end-date 2024-12-31"
            ],
        },
        {
            "id": "signals_flow",
            "name": "Signals Generation Flow",
            "description": "Generate trading signals",
            "commands": ["signals generate --symbols AAPL,MSFT --timeframe 1day --strategy momentum"],
        },
        {
            "id": "backtest_flow",
            "name": "Backtesting Flow",
            "description": "Run backtest analysis",
            "commands": [
                "backtest run --symbols AAPL,MSFT --timeframe 1day --start-date 2024-01-01 --end-date 2024-12-31 --capital 100000"
            ],
        },
        {
            "id": "pipeline_flow",
            "name": "Pipeline Flow",
            "description": "Start automated pipeline",
            "commands": ["pipeline start --mode demo"],
        },
    ]


@router.get("/templates")
async def get_command_templates():
    """Get command templates for different operations"""
    return {
        "data_commands": [
            {
                "name": "Fetch Data",
                "template": "data fetch --symbols {symbols} --timeframe {timeframe} --start-date {start_date} --end-date {end_date}",
                "parameters": ["symbols", "timeframe", "start_date", "end_date"],
            }
        ],
        "signal_commands": [
            {
                "name": "Generate Signals",
                "template": "signals generate --symbols {symbols} --timeframe {timeframe} --strategy {strategy}",
                "parameters": ["symbols", "timeframe", "strategy"],
            }
        ],
        "backtest_commands": [
            {
                "name": "Run Backtest",
                "template": "backtest run --symbols {symbols} --timeframe {timeframe} --start-date {start_date} --end-date {end_date} --capital {capital}",
                "parameters": ["symbols", "timeframe", "start_date", "end_date", "capital"],
            }
        ],
        "pipeline_commands": [
            {"name": "Start Pipeline", "template": "pipeline start --mode {mode}", "parameters": ["mode"]},
            {"name": "Stop Pipeline", "template": "pipeline stop", "parameters": []},
            {"name": "Pipeline Status", "template": "pipeline status", "parameters": []},
        ],
    }
