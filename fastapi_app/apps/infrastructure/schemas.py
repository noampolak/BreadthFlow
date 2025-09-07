from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ServiceStatus(BaseModel):
    name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status (healthy, warning, error, unknown)")
    url: Optional[str] = Field(None, description="Service URL")
    response_time: Optional[float] = Field(None, description="Response time in milliseconds")
    last_check: str = Field(..., description="Last health check timestamp")
    details: Optional[str] = Field(None, description="Additional service details")

class SystemResources(BaseModel):
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    network_status: str = Field(..., description="Network status (connected, disconnected)")

class DatabaseStatus(BaseModel):
    connected: bool = Field(..., description="Database connection status")
    response_time: Optional[float] = Field(None, description="Database response time in milliseconds")
    active_connections: Optional[int] = Field(None, description="Number of active connections")

class SystemHealth(BaseModel):
    overall_status: str = Field(..., description="Overall system status (healthy, warning, error)")
    services: List[ServiceStatus] = Field(..., description="List of service statuses")
    system_resources: SystemResources = Field(..., description="System resource usage")
    database_status: DatabaseStatus = Field(..., description="Database status")
    last_updated: str = Field(..., description="Last update timestamp")
