from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List
from datetime import datetime, timedelta
from .models import DashboardStats
from .schemas import DashboardStatsResponse, RecentRunResponse, DashboardSummaryResponse
from apps.pipeline.models import PipelineRun


class DashboardService:
    def __init__(self, db: Session):
        self.db = db

    async def get_dashboard_summary(self) -> DashboardSummaryResponse:
        """Get complete dashboard summary with stats and recent runs"""

        # Get pipeline statistics
        total_runs = self.db.query(PipelineRun).count()
        successful_runs = self.db.query(PipelineRun).filter(PipelineRun.status == "completed").count()
        failed_runs = self.db.query(PipelineRun).filter(PipelineRun.status == "failed").count()

        # Get recent runs (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_runs_24h = self.db.query(PipelineRun).filter(PipelineRun.start_time >= yesterday).count()

        # Get average duration
        avg_duration = self.db.query(func.avg(PipelineRun.duration)).filter(PipelineRun.duration.isnot(None)).scalar() or 0.0

        # Calculate success rate
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0.0

        # Create stats response
        stats = DashboardStatsResponse(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            recent_runs_24h=recent_runs_24h,
            average_duration=round(avg_duration, 2),
            success_rate=round(success_rate, 2),
            last_updated=datetime.now(),
        )

        # Get recent runs
        recent_runs = self.db.query(PipelineRun).order_by(desc(PipelineRun.start_time)).limit(10).all()

        recent_runs_response = [
            RecentRunResponse(
                run_id=run.run_id,
                command=run.command,
                status=run.status,
                start_time=run.start_time,
                end_time=run.end_time,
                duration=run.duration,
                error_message=run.error_message,
            )
            for run in recent_runs
        ]

        return DashboardSummaryResponse(stats=stats, recent_runs=recent_runs_response, last_updated=datetime.now())

    async def get_dashboard_stats(self) -> DashboardStatsResponse:
        """Get dashboard statistics only"""
        summary = await self.get_dashboard_summary()
        return summary.stats
