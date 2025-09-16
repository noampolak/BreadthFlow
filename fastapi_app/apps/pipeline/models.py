from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from fastapi_app.shared.models import Base


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id = Column(String(255), primary_key=True, index=True)
    command = Column(String(500))
    status = Column(String(50), default="pending")
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration = Column(Float, nullable=True)
    error_message = Column(String(1000), nullable=True)
    run_metadata = Column("metadata", JSON, nullable=True)

    def __repr__(self):
        return f"<PipelineRun(run_id='{self.run_id}', status='{self.status}')>"
