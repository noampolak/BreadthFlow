from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from core.dependencies import get_db
from .schemas import PipelineRunCreate, PipelineRunResponse, PipelineConfig, PipelineStatusResponse
from .utils import PipelineService
from .tasks import start_pipeline_task, stop_pipeline_task

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

@router.get("/runs", response_model=List[PipelineRunResponse])
async def get_pipeline_runs(
    skip: int = 0,
    limit: int = 10,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get paginated pipeline runs with optional status filter"""
    service = PipelineService(db)
    return await service.get_runs(skip=skip, limit=limit, status=status)

@router.post("/start", response_model=PipelineRunResponse)
async def start_pipeline(
    config: PipelineConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a new pipeline with background task execution"""
    service = PipelineService(db)
    
    # Check if pipeline is already running
    running_pipeline = await service.get_running_pipeline()
    if running_pipeline:
        raise HTTPException(
            status_code=400, 
            detail=f"Pipeline {running_pipeline.run_id} is already running"
        )
    
    # Create pipeline run record
    pipeline_run = await service.create_pipeline_run(config)
    
    # Start pipeline in background
    background_tasks.add_task(
        start_pipeline_task, 
        pipeline_run.id, 
        config.dict()
    )
    
    return pipeline_run

@router.post("/stop/{run_id}")
async def stop_pipeline(
    run_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Stop a running pipeline"""
    service = PipelineService(db)
    pipeline_run = await service.get_run_by_id(run_id)
    
    if not pipeline_run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    if pipeline_run.status != "running":
        raise HTTPException(status_code=400, detail="Pipeline is not running")
    
    # Stop pipeline in background
    background_tasks.add_task(stop_pipeline_task, run_id)
    
    return {"message": f"Pipeline {run_id} stop requested"}

@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(db: Session = Depends(get_db)):
    """Get current pipeline status and statistics"""
    service = PipelineService(db)
    return await service.get_status()

@router.get("/runs/{run_id}", response_model=PipelineRunResponse)
async def get_pipeline_run(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Get specific pipeline run details"""
    service = PipelineService(db)
    pipeline_run = await service.get_run_by_id(run_id)
    
    if not pipeline_run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    
    return pipeline_run
