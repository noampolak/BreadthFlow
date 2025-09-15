from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class PipelineConfig(BaseModel):
    mode: str = Field(..., description="Pipeline mode (demo, small, medium, full)")
    interval: str = Field(..., description="Execution interval")
    timeframe: str = Field(..., description="Data timeframe")
    symbols: Optional[str] = Field(None, description="Custom symbols list")
    data_source: str = Field("yfinance", description="Data source")


class PipelineRunBase(BaseModel):
    command: str = Field(..., description="Pipeline command to execute")
    run_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PipelineRunCreate(PipelineRunBase):
    pass


class PipelineRunResponse(PipelineRunBase):
    run_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class PipelineStatusResponse(BaseModel):
    state: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    stopped_runs: int
    success_rate: float
    average_duration: float
    uptime_seconds: Optional[int] = None
    last_run_time: Optional[str] = None
