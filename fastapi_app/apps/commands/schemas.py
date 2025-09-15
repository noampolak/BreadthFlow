from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


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
