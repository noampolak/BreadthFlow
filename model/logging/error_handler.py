"""
Error Handler Implementation

Centralized error handling and logging for BreadthFlow system
with error tracking, severity classification, and rollback mechanisms.
"""

from typing import Dict, List, Any, Optional, Callable
import logging
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Record of an error occurrence"""

    timestamp: datetime
    component: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class ErrorHandler:
    """Centralized error handling and logging"""

    def __init__(self, log_level: str = "INFO", max_errors: int = 1000):
        self.log_level = log_level
        self.max_errors = max_errors
        self.error_counts = defaultdict(int)
        self.error_records = deque(maxlen=max_errors)
        self.error_thresholds = {"LOW": 100, "MEDIUM": 50, "HIGH": 20, "CRITICAL": 5}
        self.rollback_threshold = 0.05  # 5% error rate triggers rollback
        self._setup_logging()

    def handle_error(self, error: Exception, context: Dict[str, Any], component: str = "unknown", operation: str = "unknown"):
        """Handle and log errors"""

        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            severity=self._determine_severity(error, context),
        )

        # Add to records
        self.error_records.append(error_record)

        # Update error counts
        error_key = f"{component}:{operation}"
        self.error_counts[error_key] += 1

        # Log error
        self._log_error(error_record)

        # Check if rollback is needed
        if self.should_rollback(component, operation):
            self._trigger_rollback(component, operation)

        return error_record

    def record_error(self, error: Exception, context: Dict[str, Any], component: str = "unknown", operation: str = "unknown"):
        """Record an error (alias for handle_error)"""
        return self.handle_error(error, context, component, operation)

    def get_error_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get error summary for monitoring"""

        if time_window is None:
            time_window = timedelta(hours=1)

        cutoff_time = datetime.now() - time_window
        recent_errors = [error for error in self.error_records if error.timestamp >= cutoff_time]

        # Group by component
        component_errors = defaultdict(list)
        for error in recent_errors:
            component_errors[error.component].append(error)

        # Calculate error rates
        error_rates = {}
        for component, errors in component_errors.items():
            total_operations = self._get_operation_count(component, time_window)
            error_rate = len(errors) / max(total_operations, 1)
            error_rates[component] = error_rate

        return {
            "total_errors": len(recent_errors),
            "error_rates": dict(error_rates),
            "component_breakdown": {component: len(errors) for component, errors in component_errors.items()},
            "severity_breakdown": {
                severity: len([e for e in recent_errors if e.severity == severity])
                for severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            },
            "time_window": str(time_window),
        }

    def should_rollback(self, component: str, operation: str) -> bool:
        """Determine if rollback is needed"""

        error_key = f"{component}:{operation}"
        error_count = self.error_counts[error_key]

        # Get recent error rate
        recent_errors = [
            error
            for error in self.error_records
            if error.component == component
            and error.operation == operation
            and error.timestamp >= datetime.now() - timedelta(minutes=5)
        ]

        if len(recent_errors) == 0:
            return False

        # Calculate error rate
        total_operations = self._get_operation_count(component, timedelta(minutes=5))
        error_rate = len(recent_errors) / max(total_operations, 1)

        return error_rate > self.rollback_threshold

    def resolve_error(self, error_record: ErrorRecord, resolution_notes: str = ""):
        """Mark an error as resolved"""
        error_record.resolved = True
        error_record.resolution_time = datetime.now()

        # Update context with resolution notes
        error_record.context["resolution_notes"] = resolution_notes

        logger.info(f"✅ Error resolved: {error_record.component}:{error_record.operation}")

    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component"""

        recent_errors = [
            error
            for error in self.error_records
            if error.component == component and error.timestamp >= datetime.now() - timedelta(hours=1)
        ]

        total_operations = self._get_operation_count(component, timedelta(hours=1))
        error_rate = len(recent_errors) / max(total_operations, 1)

        # Determine health status
        if error_rate == 0:
            health_status = "HEALTHY"
        elif error_rate < 0.01:
            health_status = "WARNING"
        elif error_rate < 0.05:
            health_status = "DEGRADED"
        else:
            health_status = "CRITICAL"

        return {
            "component": component,
            "health_status": health_status,
            "error_rate": error_rate,
            "total_errors": len(recent_errors),
            "total_operations": total_operations,
            "last_error": recent_errors[-1] if recent_errors else None,
        }

    def _determine_severity(self, error: Exception, context: Dict[str, Any]) -> str:
        """Determine error severity"""

        # Check error type
        critical_errors = ["ConnectionError", "TimeoutError", "AuthenticationError"]
        high_errors = ["ValidationError", "DataError", "ConfigurationError"]
        medium_errors = ["RateLimitError", "TemporaryError"]

        error_type = type(error).__name__

        if error_type in critical_errors:
            return "CRITICAL"
        elif error_type in high_errors:
            return "HIGH"
        elif error_type in medium_errors:
            return "MEDIUM"
        else:
            return "LOW"

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""

        log_message = f"Error in {error_record.component}:{error_record.operation} - {error_record.error_message}"

        if error_record.severity == "CRITICAL":
            logger.critical(log_message, extra={"error_record": asdict(error_record)})
        elif error_record.severity == "HIGH":
            logger.error(log_message, extra={"error_record": asdict(error_record)})
        elif error_record.severity == "MEDIUM":
            logger.warning(log_message, extra={"error_record": asdict(error_record)})
        else:
            logger.info(log_message, extra={"error_record": asdict(error_record)})

    def _trigger_rollback(self, component: str, operation: str):
        """Trigger rollback for component"""

        logger.warning(f"🚨 Triggering rollback for {component}:{operation}")

        # Notify monitoring system
        self._notify_monitoring(component, operation, "ROLLBACK_TRIGGERED")

        # Could integrate with migration system here
        # migrator.rollback_component(component)

    def _notify_monitoring(self, component: str, operation: str, event: str):
        """Notify monitoring system of events"""

        # This could send alerts, update dashboards, etc.
        notification = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "event": event,
            "error_rate": self.error_counts[f"{component}:{operation}"],
        }

        # Send to monitoring system (implementation depends on monitoring setup)
        logger.info(f"📊 Monitoring notification: {notification}")

    def _get_operation_count(self, component: str, time_window: timedelta) -> int:
        """Get total operation count for component in time window"""

        # This would typically come from metrics/monitoring system
        # For now, return a reasonable estimate
        return 100  # Placeholder

    def _setup_logging(self):
        """Setup logging configuration"""

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("logs/error_handler.log"), logging.StreamHandler()],
        )
