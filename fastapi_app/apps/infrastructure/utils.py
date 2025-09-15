from sqlalchemy.orm import Session
from datetime import datetime
import psutil
import time

from apps.infrastructure.schemas import SystemHealth, ServiceStatus, SystemResources, DatabaseStatus


class InfrastructureService:
    def __init__(self, db: Session):
        self.db = db

    def get_system_health(self) -> SystemHealth:
        """Get overall system health and service status"""
        try:
            # Check database connection
            db_status = self._check_database_health()

            # Get system resources
            system_resources = self._get_system_resources()

            # Check services
            services = self._check_services()

            # Determine overall status
            overall_status = self._determine_overall_status(services, db_status, system_resources)

            return SystemHealth(
                overall_status=overall_status,
                services=services,
                system_resources=system_resources,
                database_status=db_status,
                last_updated=datetime.now().isoformat(),
            )
        except Exception as e:
            print(f"Error getting system health: {e}")
            # Return basic health status
            return SystemHealth(
                overall_status="error",
                services=[],
                system_resources=SystemResources(cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0, network_status="unknown"),
                database_status=DatabaseStatus(connected=False, response_time=None, active_connections=None),
                last_updated=datetime.now().isoformat(),
            )

    def _check_database_health(self) -> DatabaseStatus:
        """Check database connection health"""
        try:
            start_time = time.time()
            # Test database connection
            from sqlalchemy import text

            self.db.execute(text("SELECT 1"))
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get active connections (simplified)
            result = self.db.execute(text("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'"))
            active_connections = result.scalar() if result else 0

            return DatabaseStatus(connected=True, response_time=round(response_time, 2), active_connections=active_connections)
        except Exception as e:
            print(f"Database health check failed: {e}")
            return DatabaseStatus(connected=False, response_time=None, active_connections=None)

    def _get_system_resources(self) -> SystemResources:
        """Get system resource usage"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Check network connectivity (simplified)
            network_status = "connected"  # Assume connected if we can get system info

            return SystemResources(
                cpu_usage=round(cpu_usage, 1),
                memory_usage=round(memory.percent, 1),
                disk_usage=round((disk.used / disk.total) * 100, 1),
                network_status=network_status,
            )
        except Exception as e:
            print(f"Error getting system resources: {e}")
            return SystemResources(cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0, network_status="unknown")

    def _check_services(self) -> list[ServiceStatus]:
        """Check status of various services"""
        services = []

        # Check FastAPI service (self)
        services.append(
            ServiceStatus(
                name="FastAPI Backend",
                status="healthy",
                url="http://localhost:8005",
                response_time=0.0,  # Self-check
                last_check=datetime.now().isoformat(),
                details="API server running normally",
            )
        )

        # Check PostgreSQL
        try:
            start_time = time.time()
            from sqlalchemy import text

            self.db.execute(text("SELECT 1"))
            response_time = (time.time() - start_time) * 1000
            services.append(
                ServiceStatus(
                    name="PostgreSQL Database",
                    status="healthy",
                    url="postgresql://localhost:5432/breadthflow",
                    response_time=round(response_time, 2),
                    last_check=datetime.now().isoformat(),
                    details="Database connection active",
                )
            )
        except Exception as e:
            services.append(
                ServiceStatus(
                    name="PostgreSQL Database",
                    status="error",
                    url="postgresql://localhost:5432/breadthflow",
                    response_time=None,
                    last_check=datetime.now().isoformat(),
                    details=f"Database connection failed: {str(e)}",
                )
            )

        # Check Redis (real check)
        redis_status = self._check_redis_health()
        services.append(redis_status)

        # Check Spark Command Server (real check)
        spark_status = self._check_spark_health()
        services.append(spark_status)

        return services

    def _check_redis_health(self) -> ServiceStatus:
        """Check Redis service health"""
        try:
            import redis

            start_time = time.time()
            r = redis.Redis(host="redis", port=6379, decode_responses=True)
            r.ping()
            response_time = (time.time() - start_time) * 1000

            return ServiceStatus(
                name="Redis Cache",
                status="healthy",
                url="redis://redis:6379",
                response_time=round(response_time, 2),
                last_check=datetime.now().isoformat(),
                details="Cache service running",
            )
        except Exception as e:
            return ServiceStatus(
                name="Redis Cache",
                status="error",
                url="redis://redis:6379",
                response_time=None,
                last_check=datetime.now().isoformat(),
                details=f"Cache service failed: {str(e)}",
            )

    def _check_spark_health(self) -> ServiceStatus:
        """Check Spark Command Server health"""
        try:
            import requests

            start_time = time.time()
            response = requests.get("http://spark-master:8081/health", timeout=5)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return ServiceStatus(
                    name="Spark Command Server",
                    status="healthy",
                    url="http://spark-master:8081",
                    response_time=round(response_time, 2),
                    last_check=datetime.now().isoformat(),
                    details="Command execution service active",
                )
            else:
                return ServiceStatus(
                    name="Spark Command Server",
                    status="warning",
                    url="http://spark-master:8081",
                    response_time=round(response_time, 2),
                    last_check=datetime.now().isoformat(),
                    details=f"Service returned status {response.status_code}",
                )
        except Exception as e:
            return ServiceStatus(
                name="Spark Command Server",
                status="error",
                url="http://spark-master:8081",
                response_time=None,
                last_check=datetime.now().isoformat(),
                details=f"Command service failed: {str(e)}",
            )

    def _determine_overall_status(
        self, services: list[ServiceStatus], db_status: DatabaseStatus, system_resources: SystemResources
    ) -> str:
        """Determine overall system status based on individual components"""
        # Check for any errors
        error_services = [s for s in services if s.status == "error"]
        if error_services or not db_status.connected:
            return "error"

        # Check for warnings
        warning_services = [s for s in services if s.status == "warning"]
        if warning_services:
            return "warning"

        # Check system resources
        if system_resources.cpu_usage > 80 or system_resources.memory_usage > 80 or system_resources.disk_usage > 90:
            return "warning"

        return "healthy"
