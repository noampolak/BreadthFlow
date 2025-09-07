from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List

class DashboardStatsResponse(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    recent_runs_24h: int
    average_duration: float
    success_rate: float
    last_updated: datetime

class RecentRunResponse(BaseModel):
    run_id: str
    command: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class DashboardSummaryResponse(BaseModel):
    stats: DashboardStatsResponse
    recent_runs: List[RecentRunResponse]
    last_updated: datetime
