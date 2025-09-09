from sqlalchemy import Column, String, DateTime, Float, Text, Boolean, Enum
from sqlalchemy.dialects.postgresql import UUID, JSON
from shared.models import Base
import uuid
from datetime import datetime
from .schemas import TrainingStatus

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    training_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbols = Column(Text, nullable=False)  # JSON string
    timeframe = Column(String(20), nullable=False)
    strategy = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    start_date = Column(String(10), nullable=False)
    end_date = Column(String(10), nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string
    test_split = Column(Float, nullable=False, default=0.2)
    status = Column(Enum(TrainingStatus), nullable=False, default=TrainingStatus.PENDING)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)  # in seconds
    error_message = Column(Text, nullable=True)

class TrainedModel(Base):
    __tablename__ = "trained_models"
    
    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(100), nullable=False)
    strategy = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    symbols = Column(Text, nullable=False)  # JSON string
    timeframe = Column(String(20), nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    is_deployed = Column(Boolean, nullable=False, default=False)
    model_path = Column(String(200), nullable=False)

