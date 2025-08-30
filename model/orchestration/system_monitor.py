"""
System Monitor
=============

Provides real-time monitoring, health checks, and performance metrics
for the entire BreadthFlow system.
"""

import asyncio
import time

# Optional psutil import
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

from ..logging.enhanced_logger import EnhancedLogger
from ..logging.error_handler import ErrorHandler


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """System metric definition"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemStatus:
    """Complete system status"""
    timestamp: datetime
    overall_health: HealthStatus
    health_checks: List[HealthCheck]
    metrics: List[Metric]
    alerts: List[str] = field(default_factory=list)


class SystemMonitor:
    """
    System monitoring and health checking
    
    Features:
    - Real-time system metrics collection
    - Health checks for all components
    - Performance monitoring
    - Alert generation
    - Historical data tracking
    - Resource usage monitoring
    """
    
    def __init__(self, update_interval: int = 30):
        """
        Initialize the system monitor
        
        Args:
            update_interval: Update interval in seconds
        """
        self.logger = EnhancedLogger("SystemMonitor")
        self.error_handler = ErrorHandler()
        
        # Configuration
        self.update_interval = update_interval
        self.monitoring_enabled = True
        
        # Storage
        self.metrics_history: List[Metric] = []
        self.health_history: List[HealthCheck] = []
        self.alerts: List[str] = []
        
        # Current status
        self.current_status: Optional[SystemStatus] = None
        
        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Health check functions
        self.health_checks: Dict[str, Callable] = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'error_rate': 0.1,
            'response_time': 5.0
        }
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks"""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("python_process", self._check_python_process)
        self.register_health_check("error_rate", self._check_error_rate)
        self.register_health_check("component_registry", self._check_component_registry)
    
    def register_health_check(self, name: str, check_function: Callable):
        """
        Register a health check function
        
        Args:
            name: Name of the health check
            check_function: Function that returns HealthCheck
        """
        self.health_checks[name] = check_function
        self.logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        if self.monitor_thread:
            self.stop_event.set()
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
            self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                self._collect_metrics()
                
                # Run health checks
                self._run_health_checks()
                
                # Update system status
                self._update_system_status()
                
                # Check for alerts
                self._check_alerts()
                
                # Wait for next update
                self.stop_event.wait(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self.error_handler.record_error(
                    error=e,
                    context={"monitoring_loop": True},
                    component="SystemMonitor",
                    operation="monitoring_loop"
                )
    
    def _collect_metrics(self):
        """Collect system metrics"""
        timestamp = datetime.now()
        
        if not PSUTIL_AVAILABLE:
            # Create basic metrics when psutil is not available
            metrics = [
                Metric("system_monitoring", 1.0, MetricType.GAUGE, timestamp,
                       description="System monitoring active"),
                Metric("psutil_available", 0.0, MetricType.GAUGE, timestamp,
                       description="psutil library available")
            ]
        else:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Create metrics
            metrics = [
                Metric("cpu_usage_percent", cpu_percent, MetricType.GAUGE, timestamp, 
                       description="CPU usage percentage"),
                Metric("memory_usage_percent", memory.percent, MetricType.GAUGE, timestamp,
                       description="Memory usage percentage"),
                Metric("memory_available_gb", memory.available / (1024**3), MetricType.GAUGE, timestamp,
                       description="Available memory in GB"),
                Metric("disk_usage_percent", disk.percent, MetricType.GAUGE, timestamp,
                       description="Disk usage percentage"),
                Metric("disk_free_gb", disk.free / (1024**3), MetricType.GAUGE, timestamp,
                       description="Free disk space in GB"),
                Metric("process_cpu_percent", process_cpu, MetricType.GAUGE, timestamp,
                       description="Process CPU usage percentage"),
                Metric("process_memory_mb", process_memory.rss / (1024**2), MetricType.GAUGE, timestamp,
                       description="Process memory usage in MB"),
                Metric("network_bytes_sent", network.bytes_sent, MetricType.COUNTER, timestamp,
                       description="Network bytes sent"),
                Metric("network_bytes_recv", network.bytes_recv, MetricType.COUNTER, timestamp,
                       description="Network bytes received"),
                Metric("psutil_available", 1.0, MetricType.GAUGE, timestamp,
                       description="psutil library available")
            ]
        
        # Add to history (keep last 1000 metrics)
        self.metrics_history.extend(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _run_health_checks(self):
        """Run all registered health checks"""
        timestamp = datetime.now()
        health_checks = []
        
        for name, check_function in self.health_checks.items():
            try:
                health_check = check_function()
                if health_check:
                    health_check.timestamp = timestamp
                    health_checks.append(health_check)
                else:
                    # Create default health check if function doesn't return one
                    health_checks.append(HealthCheck(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message="Health check function returned None",
                        timestamp=timestamp
                    ))
                    
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                health_checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    timestamp=timestamp
                ))
        
        # Add to history (keep last 100 health checks)
        self.health_history.extend(health_checks)
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
    
    def _check_system_resources(self) -> HealthCheck:
        """Check system resource usage"""
        if not PSUTIL_AVAILABLE:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="System resource monitoring not available (psutil not installed)",
                timestamp=datetime.now(),
                details={'psutil_available': False}
            )
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine overall status
        if (cpu_percent > 90 or memory.percent > 95 or disk.percent > 95):
            status = HealthStatus.CRITICAL
            message = "System resources critically low"
        elif (cpu_percent > 80 or memory.percent > 85 or disk.percent > 90):
            status = HealthStatus.WARNING
            message = "System resources usage high"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources normal"
        
        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=datetime.now(),
            details={
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3)
            }
        )
    
    def _check_python_process(self) -> HealthCheck:
        """Check Python process health"""
        if not PSUTIL_AVAILABLE:
            return HealthCheck(
                name="python_process",
                status=HealthStatus.UNKNOWN,
                message="Process monitoring not available (psutil not installed)",
                timestamp=datetime.now(),
                details={'psutil_available': False}
            )
        
        try:
            process = psutil.Process()
            
            # Check if process is responsive
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            
            # Check for memory leaks (if memory usage is very high)
            memory_mb = memory_info.rss / (1024**2)
            
            if memory_mb > 1000:  # More than 1GB
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_mb:.1f} MB"
            elif memory_mb > 2000:  # More than 2GB
                status = HealthStatus.CRITICAL
                message = f"Very high memory usage: {memory_mb:.1f} MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Process memory usage normal: {memory_mb:.1f} MB"
            
            return HealthCheck(
                name="python_process",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'num_threads': process.num_threads(),
                    'open_files': len(process.open_files()),
                    'connections': len(process.connections())
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="python_process",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check process: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_error_rate(self) -> HealthCheck:
        """Check error rate from error handler"""
        try:
            # Get recent errors (last hour)
            recent_errors = [
                error for error in self.error_handler.get_errors()
                if error.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            # Calculate error rate
            total_operations = 1000  # Estimate - in real system this would be tracked
            error_rate = len(recent_errors) / total_operations if total_operations > 0 else 0
            
            if error_rate > 0.1:  # More than 10% error rate
                status = HealthStatus.CRITICAL
                message = f"High error rate: {error_rate:.2%}"
            elif error_rate > 0.05:  # More than 5% error rate
                status = HealthStatus.WARNING
                message = f"Elevated error rate: {error_rate:.2%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Error rate normal: {error_rate:.2%}"
            
            return HealthCheck(
                name="error_rate",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'error_rate': error_rate,
                    'recent_errors': len(recent_errors),
                    'critical_errors': len([e for e in recent_errors if e.severity == 'CRITICAL']),
                    'total_operations': total_operations
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="error_rate",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check error rate: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_component_registry(self) -> HealthCheck:
        """Check component registry health"""
        try:
            # This would check the actual component registry if available
            # For now, return a basic health check
            return HealthCheck(
                name="component_registry",
                status=HealthStatus.HEALTHY,
                message="Component registry operational",
                timestamp=datetime.now(),
                details={
                    'components_registered': 0,  # Would be actual count
                    'registry_accessible': True
                }
            )
            
        except Exception as e:
            return HealthCheck(
                name="component_registry",
                status=HealthStatus.CRITICAL,
                message=f"Component registry check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _update_system_status(self):
        """Update overall system status"""
        if not self.health_history:
            return
        
        # Get latest health checks
        latest_checks = [check for check in self.health_history 
                        if check.timestamp > datetime.now() - timedelta(minutes=5)]
        
        if not latest_checks:
            return
        
        # Determine overall health
        critical_count = sum(1 for check in latest_checks if check.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for check in latest_checks if check.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            overall_health = HealthStatus.CRITICAL
        elif warning_count > 0:
            overall_health = HealthStatus.WARNING
        else:
            overall_health = HealthStatus.HEALTHY
        
        # Get latest metrics
        latest_metrics = [metric for metric in self.metrics_history 
                         if metric.timestamp > datetime.now() - timedelta(minutes=5)]
        
        self.current_status = SystemStatus(
            timestamp=datetime.now(),
            overall_health=overall_health,
            health_checks=latest_checks,
            metrics=latest_metrics,
            alerts=self.alerts.copy()
        )
    
    def _check_alerts(self):
        """Check for alert conditions"""
        if not self.metrics_history:
            return
        
        # Get latest metrics
        latest_metrics = [metric for metric in self.metrics_history 
                         if metric.timestamp > datetime.now() - timedelta(minutes=5)]
        
        for metric in latest_metrics:
            threshold = self.alert_thresholds.get(metric.name)
            if threshold and metric.value > threshold:
                alert_msg = f"Alert: {metric.name} = {metric.value} (threshold: {threshold})"
                if alert_msg not in self.alerts:
                    self.alerts.append(alert_msg)
                    self.logger.warning(alert_msg)
        
        # Clean old alerts (older than 1 hour)
        self.alerts = [alert for alert in self.alerts 
                      if "Alert:" not in alert or 
                      datetime.now() - timedelta(hours=1) < datetime.now()]
    
    def get_system_status(self) -> Optional[SystemStatus]:
        """Get current system status"""
        if self.current_status is None:
            # Initialize status if not available
            self._update_system_status()
            # If still None, create a basic status
            if self.current_status is None:
                self.current_status = SystemStatus(
                    timestamp=datetime.now(),
                    overall_health=HealthStatus.HEALTHY,
                    health_checks=[],
                    metrics=[],
                    alerts=[]
                )
        return self.current_status
    
    def get_metrics(self, metric_name: Optional[str] = None, 
                   time_range: Optional[timedelta] = None) -> List[Metric]:
        """Get metrics with optional filtering"""
        metrics = self.metrics_history
        
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            metrics = [m for m in metrics if m.timestamp > cutoff_time]
        
        return metrics
    
    def get_health_history(self, check_name: Optional[str] = None,
                          time_range: Optional[timedelta] = None) -> List[HealthCheck]:
        """Get health check history with optional filtering"""
        checks = self.health_history
        
        if check_name:
            checks = [c for c in checks if c.name == check_name]
        
        if time_range:
            cutoff_time = datetime.now() - time_range
            checks = [c for c in checks if c.timestamp > cutoff_time]
        
        return checks
    
    def get_alerts(self) -> List[str]:
        """Get current alerts"""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a metric"""
        self.alert_thresholds[metric_name] = threshold
        self.logger.info(f"Set alert threshold for {metric_name}: {threshold}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        # Get metrics from last hour
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp > datetime.now() - timedelta(hours=1)]
        
        if not recent_metrics:
            return {}
        
        # Group by metric name
        metric_groups = {}
        for metric in recent_metrics:
            if metric.name not in metric_groups:
                metric_groups[metric.name] = []
            metric_groups[metric.name].append(metric.value)
        
        # Calculate statistics
        summary = {}
        for name, values in metric_groups.items():
            summary[name] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'latest': values[-1] if values else None
            }
        
        return summary
