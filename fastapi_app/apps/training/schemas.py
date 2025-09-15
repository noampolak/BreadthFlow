from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to train on")
    timeframe: str = Field(..., description="Data timeframe")
    start_date: str = Field(..., description="Training start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Training end date (YYYY-MM-DD)")
    strategy: str = Field(..., description="Trading strategy")
    model_type: str = Field(..., description="Machine learning model type")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")
    test_split: float = Field(0.2, description="Test data split ratio")


class TrainingResponse(BaseModel):
    training_id: str
    status: TrainingStatus
    message: str
    start_time: datetime
    estimated_duration: Optional[int] = None  # in minutes


class TrainingHistory(BaseModel):
    training_id: str
    symbols: List[str]
    timeframe: str
    strategy: str
    model_type: str
    status: TrainingStatus
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class ModelInfo(BaseModel):
    model_id: str
    name: str
    strategy: str
    model_type: str
    symbols: List[str]
    timeframe: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    created_at: datetime
    last_used: Optional[datetime] = None
    is_deployed: bool = False

    class Config:
        from_attributes = True
