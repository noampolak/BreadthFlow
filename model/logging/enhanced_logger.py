"""
Enhanced Logger Implementation

Enhanced logging with structured data and performance tracking
for the BreadthFlow system.
"""

from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
from contextlib import contextmanager
import time

class EnhancedLogger:
    """Enhanced logging with structured data and performance tracking"""
    
    def __init__(self, name: str, component: str = "unknown"):
        self.logger = logging.getLogger(name)
        self.component = component
        self.performance_metrics = {}
    
    def log_operation(self, operation: str, data: Dict[str, Any] = None, 
                     level: str = "INFO"):
        """Log operation with structured data"""
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'component': self.component,
            'operation': operation,
            'data': data or {}
        }
        
        log_message = f"{operation} - {json.dumps(log_data)}"
        
        if level == "DEBUG":
            self.logger.debug(log_message)
        elif level == "INFO":
            self.logger.info(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "ERROR":
            self.logger.error(log_message)
        elif level == "CRITICAL":
            self.logger.critical(log_message)
    
    @contextmanager
    def log_performance(self, operation: str):
        """Context manager for logging operation performance"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log performance metrics
            self.log_operation(
                f"{operation}_performance",
                {
                    'duration_seconds': duration,
                    'memory_delta_mb': memory_delta,
                    'success': success
                }
            )
            
            # Store for monitoring
            self.performance_metrics[operation] = {
                'duration': duration,
                'memory_delta': memory_delta,
                'success': success,
                'timestamp': datetime.now()
            }
    
    def log_data_quality(self, data_source: str, quality_metrics: Dict[str, Any]):
        """Log data quality metrics"""
        
        self.log_operation(
            "data_quality_check",
            {
                'data_source': data_source,
                'quality_metrics': quality_metrics
            }
        )
    
    def log_component_health(self, health_status: Dict[str, Any]):
        """Log component health status"""
        
        self.log_operation(
            "component_health",
            health_status
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
