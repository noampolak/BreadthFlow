from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTISELECT = "multiselect"

class ParameterValue(BaseModel):
    name: str = Field(..., description="Parameter name")
    value: Any = Field(..., description="Current parameter value")
    default_value: Any = Field(..., description="Default parameter value")
    description: str = Field(..., description="Parameter description")
    parameter_type: ParameterType = Field(..., description="Parameter data type")
    options: Optional[List[str]] = Field(None, description="Available options for select/multiselect")
    min_value: Optional[float] = Field(None, description="Minimum value for numeric parameters")
    max_value: Optional[float] = Field(None, description="Maximum value for numeric parameters")
    required: bool = Field(True, description="Whether parameter is required")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")

class ParameterGroup(BaseModel):
    group_name: str = Field(..., description="Parameter group name")
    display_name: str = Field(..., description="Display name for the group")
    description: str = Field(..., description="Group description")
    parameters: List[ParameterValue] = Field(..., description="Parameters in this group")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")

class ParameterUpdate(BaseModel):
    group_name: str = Field(..., description="Parameter group name")
    parameter_name: str = Field(..., description="Parameter name")
    value: Any = Field(..., description="New parameter value")

class ParameterHistory(BaseModel):
    history_id: str
    group_name: str
    parameter_name: str
    old_value: Any
    new_value: Any
    changed_by: str
    change_time: datetime
    change_reason: Optional[str] = None
    
    class Config:
        from_attributes = True

