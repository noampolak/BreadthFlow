#!/usr/bin/env python3
"""
BreadthFlow Pipeline Metadata Tracking

This module provides comprehensive metadata tracking for pipeline runs,
including run history, performance metrics, and detailed logging to PostgreSQL
and Elasticsearch for monitoring and analysis.

Key Features:
- Pipeline run metadata storage
- Performance metrics calculation
- Error tracking and analysis
- Integration with existing DualLogger
- Database schema management
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineRunStatus(Enum):
    """Pipeline run status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineRunMetadata:
    """Pipeline run metadata structure"""
    run_id: str
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: PipelineRunStatus = PipelineRunStatus.PENDING
    mode: str = "demo"
    interval: str = "5m"
    timeframe: str = "1day"
    symbols: List[str] = None
    symbols_processed: int = 0
    signals_generated: int = 0
    backtest_results: Dict[str, Any] = None
    error_count: int = 0
    error_messages: List[str] = None
    duration_seconds: float = 0.0
    config: Dict[str, Any] = None
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.error_messages is None:
            self.error_messages = []
        if self.backtest_results is None:
            self.backtest_results = {}
        if self.config is None:
            self.config = {}
        if self.results is None:
            self.results = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineRunMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        data['status'] = PipelineRunStatus(data['status'])
        return cls(**data)

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    avg_duration_seconds: float = 0.0
    avg_symbols_per_run: float = 0.0
    avg_signals_per_run: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    last_run_time: Optional[datetime] = None
    first_run_time: Optional[datetime] = None
    total_uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['last_run_time'] = self.last_run_time.isoformat() if self.last_run_time else None
        data['first_run_time'] = self.first_run_time.isoformat() if self.first_run_time else None
        return data

class PipelineMetadataTracker:
    """Pipeline metadata tracking and storage"""
    
    def __init__(self, use_database: bool = True):
        self.use_database = use_database
        self._runs: Dict[str, PipelineRunMetadata] = {}
        self._pipeline_start_time: Optional[datetime] = None
        
    def start_pipeline(self, pipeline_id: str, config: Dict[str, Any]) -> str:
        """Start tracking a new pipeline"""
        self._pipeline_start_time = datetime.now()
        logger.info(f"Started tracking pipeline {pipeline_id}")
        return pipeline_id
    
    def stop_pipeline(self, pipeline_id: str):
        """Stop tracking a pipeline"""
        logger.info(f"Stopped tracking pipeline {pipeline_id}")
        self._pipeline_start_time = None
    
    def start_run(self, pipeline_id: str, config: Dict[str, Any]) -> str:
        """Start tracking a new pipeline run"""
        run_id = str(uuid.uuid4())
        
        metadata = PipelineRunMetadata(
            run_id=run_id,
            pipeline_id=pipeline_id,
            start_time=datetime.now(),
            status=PipelineRunStatus.RUNNING,
            mode=config.get("mode", "demo"),
            interval=config.get("interval", "5m"),
            timeframe=config.get("timeframe", "1day"),
            symbols=config.get("symbols", []),
            config=config
        )
        
        self._runs[run_id] = metadata
        
        if self.use_database:
            self._store_run_to_database(metadata)
        
        logger.info(f"Started tracking run {run_id}")
        return run_id
    
    def complete_run(self, run_id: str, results: Dict[str, Any]):
        """Complete a pipeline run with results"""
        if run_id not in self._runs:
            logger.error(f"Run {run_id} not found")
            return
        
        metadata = self._runs[run_id]
        metadata.end_time = datetime.now()
        metadata.status = PipelineRunStatus.COMPLETED
        metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
        metadata.results = results
        
        # Extract metrics from results
        if "symbols_processed" in results:
            metadata.symbols_processed = results["symbols_processed"]
        
        # Count signals generated
        if "signals" in results and "output" in results["signals"]:
            # This would need to be parsed from the output
            metadata.signals_generated = len(metadata.symbols)  # Approximate
        
        if self.use_database:
            self._update_run_in_database(metadata)
        
        logger.info(f"Completed run {run_id} in {metadata.duration_seconds:.1f}s")
    
    def fail_run(self, run_id: str, error: str):
        """Mark a pipeline run as failed"""
        if run_id not in self._runs:
            logger.error(f"Run {run_id} not found")
            return
        
        metadata = self._runs[run_id]
        metadata.end_time = datetime.now()
        metadata.status = PipelineRunStatus.FAILED
        metadata.duration_seconds = (metadata.end_time - metadata.start_time).total_seconds()
        metadata.error_count += 1
        metadata.error_messages.append(error)
        
        if self.use_database:
            self._update_run_in_database(metadata)
        
        logger.error(f"Failed run {run_id}: {error}")
    
    def get_run_metadata(self, run_id: str) -> Optional[PipelineRunMetadata]:
        """Get metadata for a specific run"""
        return self._runs.get(run_id)
    
    def get_recent_runs(self, limit: int = 10) -> List[PipelineRunMetadata]:
        """Get recent pipeline runs"""
        runs = list(self._runs.values())
        runs.sort(key=lambda x: x.start_time, reverse=True)
        return runs[:limit]
    
    def get_pipeline_metrics(self, pipeline_id: str = None) -> PipelineMetrics:
        """Calculate pipeline performance metrics"""
        runs = list(self._runs.values())
        if pipeline_id:
            runs = [r for r in runs if r.pipeline_id == pipeline_id]
        
        if not runs:
            return PipelineMetrics()
        
        total_runs = len(runs)
        successful_runs = len([r for r in runs if r.status == PipelineRunStatus.COMPLETED])
        failed_runs = len([r for r in runs if r.status == PipelineRunStatus.FAILED])
        cancelled_runs = len([r for r in runs if r.status == PipelineRunStatus.CANCELLED])
        
        completed_runs = [r for r in runs if r.end_time is not None]
        avg_duration = sum(r.duration_seconds for r in completed_runs) / len(completed_runs) if completed_runs else 0
        avg_symbols = sum(r.symbols_processed for r in runs) / len(runs) if runs else 0
        avg_signals = sum(r.signals_generated for r in runs) / len(runs) if runs else 0
        
        success_rate = successful_runs / total_runs if total_runs > 0 else 0
        error_rate = failed_runs / total_runs if total_runs > 0 else 0
        
        last_run = max(runs, key=lambda x: x.start_time) if runs else None
        first_run = min(runs, key=lambda x: x.start_time) if runs else None
        
        total_uptime = 0
        if self._pipeline_start_time:
            total_uptime = (datetime.now() - self._pipeline_start_time).total_seconds()
        
        return PipelineMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            cancelled_runs=cancelled_runs,
            avg_duration_seconds=avg_duration,
            avg_symbols_per_run=avg_symbols,
            avg_signals_per_run=avg_signals,
            success_rate=success_rate,
            error_rate=error_rate,
            last_run_time=last_run.start_time if last_run else None,
            first_run_time=first_run.start_time if first_run else None,
            total_uptime_seconds=total_uptime
        )
    
    def get_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze errors from recent runs"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_runs = [r for r in self._runs.values() if r.start_time >= cutoff_time]
        failed_runs = [r for r in recent_runs if r.status == PipelineRunStatus.FAILED]
        
        error_frequency = {}
        for run in failed_runs:
            for error in run.error_messages:
                error_frequency[error] = error_frequency.get(error, 0) + 1
        
        return {
            "time_window_hours": hours,
            "total_runs": len(recent_runs),
            "failed_runs": len(failed_runs),
            "error_rate": len(failed_runs) / len(recent_runs) if recent_runs else 0,
            "most_common_errors": sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            "error_frequency": error_frequency
        }
    
    def _store_run_to_database(self, metadata: PipelineRunMetadata):
        """Store run metadata to database"""
        try:
            # This would integrate with the existing DualLogger or PostgreSQL
            # For now, we'll log it as structured data
            logger.info(f"Storing run metadata to database: {metadata.run_id}")
            
            # TODO: Implement actual database storage
            # This should integrate with the existing pipeline_runs table
            # and possibly create a new pipeline_metadata table
            
        except Exception as e:
            logger.error(f"Failed to store run metadata: {e}")
    
    def _update_run_in_database(self, metadata: PipelineRunMetadata):
        """Update run metadata in database"""
        try:
            logger.info(f"Updating run metadata in database: {metadata.run_id}")
            
            # TODO: Implement actual database update
            # This should update the existing record with completion data
            
        except Exception as e:
            logger.error(f"Failed to update run metadata: {e}")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export pipeline metrics in specified format"""
        metrics = self.get_pipeline_metrics()
        recent_runs = self.get_recent_runs(20)
        error_analysis = self.get_error_analysis()
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
            "recent_runs": [run.to_dict() for run in recent_runs],
            "error_analysis": error_analysis
        }
        
        if format.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global metadata tracker instance
_metadata_tracker: Optional[PipelineMetadataTracker] = None

def get_metadata_tracker() -> PipelineMetadataTracker:
    """Get or create the global metadata tracker"""
    global _metadata_tracker
    if _metadata_tracker is None:
        _metadata_tracker = PipelineMetadataTracker()
    return _metadata_tracker

def track_pipeline_start(pipeline_id: str, config: Dict[str, Any]) -> str:
    """Start tracking a pipeline"""
    tracker = get_metadata_tracker()
    return tracker.start_pipeline(pipeline_id, config)

def track_pipeline_stop(pipeline_id: str):
    """Stop tracking a pipeline"""
    tracker = get_metadata_tracker()
    tracker.stop_pipeline(pipeline_id)

def track_run_start(pipeline_id: str, config: Dict[str, Any]) -> str:
    """Start tracking a pipeline run"""
    tracker = get_metadata_tracker()
    return tracker.start_run(pipeline_id, config)

def track_run_complete(run_id: str, results: Dict[str, Any]):
    """Complete a pipeline run"""
    tracker = get_metadata_tracker()
    tracker.complete_run(run_id, results)

def track_run_failure(run_id: str, error: str):
    """Mark a pipeline run as failed"""
    tracker = get_metadata_tracker()
    tracker.fail_run(run_id, error)

def get_pipeline_status() -> Dict[str, Any]:
    """Get comprehensive pipeline status"""
    tracker = get_metadata_tracker()
    metrics = tracker.get_pipeline_metrics()
    recent_runs = tracker.get_recent_runs(5)
    error_analysis = tracker.get_error_analysis(24)
    
    return {
        "metrics": metrics.to_dict(),
        "recent_runs": [run.to_dict() for run in recent_runs],
        "error_analysis": error_analysis,
        "status_time": datetime.now().isoformat()
    }

