from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime

from core.dependencies import get_db_session
from apps.training.schemas import TrainingRequest, TrainingResponse, TrainingHistory, ModelInfo
from apps.training.utils import TrainingService

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_session)
):
    """Start model training"""
    try:
        service = TrainingService(db)
        result = await service.start_training(request, background_tasks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/history", response_model=List[TrainingHistory])
async def get_training_history(
    limit: int = 20,
    db: Session = Depends(get_db_session)
):
    """Get training history"""
    try:
        service = TrainingService(db)
        return service.get_training_history(limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch training history: {str(e)}")

@router.get("/models", response_model=List[ModelInfo])
async def get_models(
    db: Session = Depends(get_db_session)
):
    """Get available trained models"""
    try:
        service = TrainingService(db)
        return service.get_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    db: Session = Depends(get_db_session)
):
    """Delete a trained model"""
    try:
        service = TrainingService(db)
        await service.delete_model(model_id)
        return {"message": f"Model {model_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.get("/configurations", response_model=Dict[str, Any])
async def get_training_configurations():
    """Get available training configurations and templates"""
    return {
        "strategies": [
            {"id": "momentum", "name": "Momentum Strategy", "description": "Price momentum-based signals"},
            {"id": "mean_reversion", "name": "Mean Reversion", "description": "Mean reversion strategy"},
            {"id": "breakout", "name": "Breakout Strategy", "description": "Support/resistance breakout signals"},
            {"id": "rsi", "name": "RSI Strategy", "description": "Relative Strength Index signals"},
            {"id": "macd", "name": "MACD Strategy", "description": "Moving Average Convergence Divergence"}
        ],
        "model_types": [
            {"id": "random_forest", "name": "Random Forest", "description": "Ensemble learning with decision trees"},
            {"id": "xgboost", "name": "XGBoost", "description": "Gradient boosting framework"},
            {"id": "neural_network", "name": "Neural Network", "description": "Deep learning model"},
            {"id": "svm", "name": "Support Vector Machine", "description": "SVM classifier"},
            {"id": "logistic_regression", "name": "Logistic Regression", "description": "Linear classification model"}
        ],
        "timeframes": [
            {"id": "1min", "name": "1 Minute", "description": "High-frequency trading"},
            {"id": "5min", "name": "5 Minutes", "description": "Short-term trading"},
            {"id": "15min", "name": "15 Minutes", "description": "Intraday trading"},
            {"id": "1hour", "name": "1 Hour", "description": "Swing trading"},
            {"id": "1day", "name": "1 Day", "description": "Position trading"}
        ],
        "symbols": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC",
            "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "BND", "TLT", "GLD", "SLV"
        ]
    }

