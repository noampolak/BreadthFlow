import asyncio
import httpx
from sqlalchemy.orm import Session
from core.database import get_db
from .models import PipelineRun
from .schemas import PipelineConfig
from shared.websocket import WebSocketManager
from datetime import datetime

# Global WebSocket manager (will be injected)
websocket_manager = None

def set_websocket_manager(manager: WebSocketManager):
    global websocket_manager
    websocket_manager = manager

async def start_pipeline_task(pipeline_id: int, config: dict):
    """Background task to start pipeline execution"""
    db = next(get_db())
    try:
        # Update status to running
        pipeline_run = db.query(PipelineRun).filter(PipelineRun.id == pipeline_id).first()
        if pipeline_run:
            pipeline_run.status = "running"
            pipeline_run.is_active = True
            db.commit()
            
            # Broadcast update via WebSocket
            if websocket_manager:
                await websocket_manager.broadcast_pipeline_update(
                    pipeline_run.run_id, 
                    "running"
                )
        
        # Execute pipeline via Spark command server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://spark-master:8081/execute",
                json={
                    "command": "spark_streaming_start_demo",
                    "parameters": config
                },
                timeout=300.0
            )
            
            if response.status_code == 200:
                # Update status to completed
                pipeline_run.status = "completed"
                pipeline_run.is_active = False
                pipeline_run.end_time = datetime.now()
                pipeline_run.duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
                
                # Broadcast update via WebSocket
                if websocket_manager:
                    await websocket_manager.broadcast_pipeline_update(
                        pipeline_run.run_id, 
                        "completed"
                    )
            else:
                # Update status to failed
                pipeline_run.status = "failed"
                pipeline_run.is_active = False
                pipeline_run.end_time = datetime.now()
                pipeline_run.error_message = f"HTTP {response.status_code}: {response.text}"
                
                # Broadcast update via WebSocket
                if websocket_manager:
                    await websocket_manager.broadcast_pipeline_update(
                        pipeline_run.run_id, 
                        "failed"
                    )
            
            db.commit()
            
    except Exception as e:
        # Update status to failed
        pipeline_run = db.query(PipelineRun).filter(PipelineRun.id == pipeline_id).first()
        if pipeline_run:
            pipeline_run.status = "failed"
            pipeline_run.is_active = False
            pipeline_run.end_time = datetime.now()
            pipeline_run.error_message = str(e)
            db.commit()
            
            # Broadcast update via WebSocket
            if websocket_manager:
                await websocket_manager.broadcast_pipeline_update(
                    pipeline_run.run_id, 
                    "failed"
                )
    finally:
        db.close()

async def stop_pipeline_task(run_id: str):
    """Background task to stop pipeline execution"""
    db = next(get_db())
    try:
        # Stop pipeline via Spark command server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://spark-master:8081/execute",
                json={
                    "command": "spark_streaming_stop",
                    "parameters": {"run_id": run_id}
                },
                timeout=30.0
            )
            
            # Update status to stopped
            pipeline_run = db.query(PipelineRun).filter(PipelineRun.run_id == run_id).first()
            if pipeline_run:
                pipeline_run.status = "stopped"
                pipeline_run.is_active = False
                pipeline_run.end_time = datetime.now()
                pipeline_run.duration = (pipeline_run.end_time - pipeline_run.start_time).total_seconds()
                db.commit()
                
                # Broadcast update via WebSocket
                if websocket_manager:
                    await websocket_manager.broadcast_pipeline_update(
                        pipeline_run.run_id, 
                        "stopped"
                    )
                
    except Exception as e:
        # Log error but don't fail the task
        print(f"Error stopping pipeline {run_id}: {e}")
    finally:
        db.close()
