import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Enum, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID

from fastapi_app.shared.models import Base

from .schemas import CommandStatus


class CommandExecution(Base):
    __tablename__ = "command_executions"

    command_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    command = Column(Text, nullable=False)
    status = Column(Enum(CommandStatus), nullable=False, default=CommandStatus.PENDING)
    output = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
