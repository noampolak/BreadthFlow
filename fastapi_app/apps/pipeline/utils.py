from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from .models import PipelineRun
from .schemas import PipelineRunCreate, PipelineConfig, PipelineStatus, PipelineStatusResponse
import uuid
from datetime import datetime

class PipelineService:
    def __init__(self, db: Session):
        self.db = db
    
    async def create_pipeline_run(self, config: PipelineConfig) -> PipelineRun:
        """Create a new pipeline run record"""
        run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        pipeline_run = PipelineRun(
            run_id=run_id,
            command=f"spark_streaming_start_{config.mode}",
            status=PipelineStatus.PENDING,
            run_metadata=config.dict()
        )
        
        self.db.add(pipeline_run)
        self.db.commit()
        self.db.refresh(pipeline_run)
        
        return pipeline_run
    
    async def get_runs(
        self, 
        skip: int = 0, 
        limit: int = 10, 
        status: Optional[str] = None
    ) -> List[PipelineRun]:
        """Get paginated pipeline runs"""
        query = self.db.query(PipelineRun)
        
        if status:
            query = query.filter(PipelineRun.status == status)
        
        return query.order_by(desc(PipelineRun.start_time)).offset(skip).limit(limit).all()
    
    async def get_run_by_id(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run by ID"""
        return self.db.query(PipelineRun).filter(PipelineRun.run_id == run_id).first()
    
    async def get_running_pipeline(self) -> Optional[PipelineRun]:
        """Get currently running pipeline"""
        return self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.RUNNING
        ).first()
    
    async def get_status(self) -> PipelineStatusResponse:
        """Get pipeline status and statistics"""
        total_runs = self.db.query(PipelineRun).count()
        successful_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.COMPLETED
        ).count()
        failed_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.FAILED
        ).count()
        running_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.RUNNING
        ).count()
        stopped_runs = self.db.query(PipelineRun).filter(
            PipelineRun.status == PipelineStatus.STOPPED
        ).count()
        
        avg_duration = self.db.query(func.avg(PipelineRun.duration)).filter(
            PipelineRun.duration.isnot(None)
        ).scalar() or 0
        
        # Get the latest run for uptime calculation
        latest_run = self.db.query(PipelineRun).order_by(desc(PipelineRun.start_time)).first()
        uptime_seconds = None
        last_run_time = None
        
        if latest_run:
            last_run_time = latest_run.start_time.isoformat()
            if latest_run.status == PipelineStatus.RUNNING:
                uptime_seconds = int((datetime.now() - latest_run.start_time).total_seconds())
        
        # Determine overall state
        if running_runs > 0:
            state = "running"
        elif total_runs == 0:
            state = "idle"
        else:
            state = "stopped"
        
        return PipelineStatusResponse(
            state=state,
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            stopped_runs=stopped_runs,
            success_rate=round((successful_runs / total_runs * 100) if total_runs > 0 else 0, 2),
            average_duration=round(avg_duration, 2),
            uptime_seconds=uptime_seconds,
            last_run_time=last_run_time
        )
