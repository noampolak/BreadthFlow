from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
import uuid
import asyncio
from datetime import datetime, timedelta
import json
import os

from .models import TrainingSession, TrainedModel
from .schemas import TrainingRequest, TrainingResponse, TrainingHistory, ModelInfo, TrainingStatus

class TrainingService:
    def __init__(self, db: Session):
        self.db = db
    
    async def start_training(self, request: TrainingRequest, background_tasks) -> TrainingResponse:
        """Start model training"""
        training_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create training session record
        training_session = TrainingSession(
            training_id=training_id,
            symbols=json.dumps(request.symbols),
            timeframe=request.timeframe,
            strategy=request.strategy,
            model_type=request.model_type,
            start_date=request.start_date,
            end_date=request.end_date,
            parameters=json.dumps(request.parameters or {}),
            test_split=request.test_split,
            status=TrainingStatus.PENDING,
            start_time=start_time
        )
        
        self.db.add(training_session)
        self.db.commit()
        
        # Start training in background
        background_tasks.add_task(self._train_model, training_id, request)
        
        # Estimate duration based on model type and data size
        estimated_duration = self._estimate_training_duration(request)
        
        return TrainingResponse(
            training_id=training_id,
            status=TrainingStatus.PENDING,
            message="Training started successfully",
            start_time=start_time,
            estimated_duration=estimated_duration
        )
    
    async def _train_model(self, training_id: str, request: TrainingRequest):
        """Train model in background"""
        try:
            # Update status to running
            training_session = self.db.query(TrainingSession).filter(
                TrainingSession.training_id == training_id
            ).first()
            
            if training_session:
                training_session.status = TrainingStatus.RUNNING
                self.db.commit()
            
            # Simulate training process
            # In a real implementation, this would:
            # 1. Fetch historical data
            # 2. Prepare features
            # 3. Train the model
            # 4. Evaluate performance
            # 5. Save the model
            
            await asyncio.sleep(10)  # Simulate training time
            
            # Simulate training results
            accuracy = 0.75 + (hash(training_id) % 20) / 100  # Random accuracy between 0.75-0.95
            precision = accuracy - 0.05
            recall = accuracy + 0.02
            f1_score = 2 * (precision * recall) / (precision + recall)
            
            end_time = datetime.now()
            duration = (end_time - training_session.start_time).total_seconds()
            
            # Update training session
            if training_session:
                training_session.status = TrainingStatus.COMPLETED
                training_session.accuracy = accuracy
                training_session.precision = precision
                training_session.recall = recall
                training_session.f1_score = f1_score
                training_session.end_time = end_time
                training_session.duration = duration
                self.db.commit()
            
            # Create trained model record
            model_id = str(uuid.uuid4())
            model_name = f"{request.strategy}_{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trained_model = TrainedModel(
                model_id=model_id,
                training_id=training_id,
                name=model_name,
                strategy=request.strategy,
                model_type=request.model_type,
                symbols=json.dumps(request.symbols),
                timeframe=request.timeframe,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                created_at=end_time,
                model_path=f"/models/{model_id}.pkl"
            )
            
            self.db.add(trained_model)
            self.db.commit()
            
        except Exception as e:
            # Update training session with error
            training_session = self.db.query(TrainingSession).filter(
                TrainingSession.training_id == training_id
            ).first()
            
            if training_session:
                training_session.status = TrainingStatus.FAILED
                training_session.error_message = str(e)
                training_session.end_time = datetime.now()
                training_session.duration = (training_session.end_time - training_session.start_time).total_seconds()
                self.db.commit()
    
    def _estimate_training_duration(self, request: TrainingRequest) -> int:
        """Estimate training duration in minutes"""
        base_duration = {
            "random_forest": 5,
            "xgboost": 8,
            "neural_network": 15,
            "svm": 10,
            "logistic_regression": 3
        }
        
        duration = base_duration.get(request.model_type, 10)
        
        # Adjust based on number of symbols
        duration += len(request.symbols) * 2
        
        # Adjust based on timeframe (more data = longer training)
        timeframe_multiplier = {
            "1min": 3,
            "5min": 2,
            "15min": 1.5,
            "1hour": 1.2,
            "1day": 1
        }
        
        duration *= timeframe_multiplier.get(request.timeframe, 1)
        
        return int(duration)
    
    def get_training_history(self, limit: int = 20) -> List[TrainingHistory]:
        """Get training history"""
        training_sessions = self.db.query(TrainingSession).order_by(
            desc(TrainingSession.start_time)
        ).limit(limit).all()
        
        return [
            TrainingHistory(
                training_id=session.training_id,
                symbols=json.loads(session.symbols),
                timeframe=session.timeframe,
                strategy=session.strategy,
                model_type=session.model_type,
                status=session.status,
                accuracy=session.accuracy,
                precision=session.precision,
                recall=session.recall,
                f1_score=session.f1_score,
                start_time=session.start_time,
                end_time=session.end_time,
                duration=session.duration,
                error_message=session.error_message
            )
            for session in training_sessions
        ]
    
    def get_models(self) -> List[ModelInfo]:
        """Get available trained models"""
        models = self.db.query(TrainedModel).order_by(
            desc(TrainedModel.created_at)
        ).all()
        
        return [
            ModelInfo(
                model_id=model.model_id,
                name=model.name,
                strategy=model.strategy,
                model_type=model.model_type,
                symbols=json.loads(model.symbols),
                timeframe=model.timeframe,
                accuracy=model.accuracy,
                precision=model.precision,
                recall=model.recall,
                f1_score=model.f1_score,
                created_at=model.created_at,
                last_used=model.last_used,
                is_deployed=model.is_deployed
            )
            for model in models
        ]
    
    async def delete_model(self, model_id: str):
        """Delete a trained model"""
        model = self.db.query(TrainedModel).filter(
            TrainedModel.model_id == model_id
        ).first()
        
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Delete model file if it exists
        if os.path.exists(model.model_path):
            os.remove(model.model_path)
        
        # Delete from database
        self.db.delete(model)
        self.db.commit()

