from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CommandStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CommandRequest(BaseModel):
    command: str = Field(..., description="Command to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Command parameters")
    background: bool = Field(False, description="Run command in background")


class CommandResponse(BaseModel):
    command_id: str
    command: str
    status: CommandStatus
    output: Optional[str] = None
    error: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None


class CommandHistory(BaseModel):
    command_id: str
    command: str
    status: CommandStatus
    output: Optional[str] = None
    error: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    class Config:
        from_attributes = True
