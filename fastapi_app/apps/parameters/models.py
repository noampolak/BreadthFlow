import uuid
from datetime import datetime

from fastapi_app.shared.models import Base
from sqlalchemy import Boolean, Column, DateTime, Enum, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID

from .schemas import ParameterType


class ParameterConfig(Base):
    __tablename__ = "parameter_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_name = Column(String(50), nullable=False)
    parameter_name = Column(String(100), nullable=False)
    value = Column(Text, nullable=False)  # JSON string
    default_value = Column(Text, nullable=False)  # JSON string
    description = Column(Text, nullable=False)
    parameter_type = Column(Enum(ParameterType), nullable=False)
    options = Column(Text, nullable=True)  # JSON string for select options
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    required = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_modified = Column(DateTime, nullable=True)


class ParameterHistory(Base):
    __tablename__ = "parameter_history"

    history_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_name = Column(String(50), nullable=False)
    parameter_name = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=False)  # JSON string
    new_value = Column(Text, nullable=False)  # JSON string
    changed_by = Column(String(100), nullable=False)
    change_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    change_reason = Column(String(200), nullable=True)
