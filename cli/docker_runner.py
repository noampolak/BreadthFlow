#!/usr/bin/env python3
"""
Docker-based PySpark runner for BreadthFlow

This module provides a way to run PySpark operations inside Docker containers
to avoid local Java requirements while still using the Docker infrastructure.
"""

import subprocess
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class DockerPySparkRunner:
    """Runs PySpark operations inside Docker containers."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        # Use the existing Spark master container
        self.docker_image = "spark-master"
    
    def _ensure_image_exists(self):
        """Check if the Spark worker container is running."""
        try:
            # Check if container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.docker_image}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            
            if not result.stdout.strip():
                raise Exception("Spark worker container is not running. Please start infrastructure first.")
                    
        except Exception as e:
            print(f"Warning: {e}")
            # Fallback to a simpler approach
            self.docker_image = "python:3.9-slim"
        
    def run_pyspark_operation(
        self, 
        operation: str, 
        args: Dict[str, Any],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run a PySpark operation inside a Docker container.
        
        Args:
            operation: The operation to run (e.g., 'fetch', 'summary')
            args: Arguments for the operation
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with operation results
        """
        # Create temporary script
        script_content = self._create_pyspark_script(operation, args)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            f.write(script_content)
        
        try:
            # Run the script in Docker
            if self.docker_image == "spark-master":
                # Copy script to running container
                container_script_path = "/tmp/script.py"
                subprocess.run([
                    "docker", "cp", script_path, f"{self.docker_image}:{container_script_path}"
                ], check=True)
                
                result = self._run_docker_command(container_script_path, timeout)
                
                # Clean up script in container
                subprocess.run([
                    "docker", "exec", self.docker_image, "rm", container_script_path
                ], check=False)
            else:
                result = self._run_docker_command(script_path, timeout)
            
            return result
        finally:
            # Clean up local script
            os.unlink(script_path)
    
    def _create_pyspark_script(self, operation: str, args: Dict[str, Any]) -> str:
        """Create a Python script for the PySpark operation."""
        
        if operation == "fetch":
            return self._create_fetch_script(args)
        elif operation == "summary":
            return self._create_summary_script(args)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _create_fetch_script(self, args: Dict[str, Any]) -> str:
        """Create script for data fetching operation."""
        return f"""
import sys
import os
sys.path.append('/app')

from pyspark.sql import SparkSession
from ingestion.data_fetcher import create_data_fetcher
import json

# Set environment variable to indicate we're in Docker
os.environ['RUNNING_IN_DOCKER'] = 'true'

# Create Spark session
spark = SparkSession.builder \\
    .appName("DataFetcher") \\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
    .getOrCreate()

# Create fetcher
fetcher = create_data_fetcher(spark)

# Fetch data
result = fetcher.fetch_and_store(
    symbols={args.get('symbols', [])},
    start_date="{args.get('start_date', '2023-01-01')}",
    end_date="{args.get('end_date', '2024-12-31')}",
    table_path="{args.get('table_path', 's3a://breadthflow/ohlcv')}",
    max_workers={args.get('max_workers', 10)}
)

# Output result as JSON
print(json.dumps(result))

spark.stop()
"""
    
    def _create_summary_script(self, args: Dict[str, Any]) -> str:
        """Create script for data summary operation."""
        return f"""
import sys
import os
sys.path.append('/app')

from pyspark.sql import SparkSession
from ingestion.data_fetcher import create_data_fetcher
import json

# Set environment variable to indicate we're in Docker
os.environ['RUNNING_IN_DOCKER'] = 'true'

# Create Spark session
spark = SparkSession.builder \\
    .appName("DataSummary") \\
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
    .getOrCreate()

# Create fetcher
fetcher = create_data_fetcher(spark)

# Get summary
summary = fetcher.get_data_summary(
    table_path="{args.get('table_path', 's3a://breadthflow/ohlcv')}"
)

# Output result as JSON
print(json.dumps(summary))

spark.stop()
"""
    
    def _run_docker_command(self, script_path: str, timeout: int) -> Dict[str, Any]:
        """Run the Python script inside a Docker container."""
        
        # Copy script to container
        container_script_path = "/tmp/script.py"
        
        # Run Docker command
        if self.docker_image == "spark-master":
            # Use exec to run in existing container
            cmd = [
                "docker", "exec",
                self.docker_image,
                "python", container_script_path
            ]
        else:
            # Use run for new container
            cmd = [
                "docker", "run", "--rm",
                "--network", "breadthflow-network",
                "-v", f"{self.project_root}:/app",
                "-v", f"{script_path}:{container_script_path}",
                "-e", "AWS_ACCESS_KEY_ID=minioadmin",
                "-e", "AWS_SECRET_ACCESS_KEY=minioadmin",
                "-e", "AWS_DEFAULT_REGION=us-east-1",
                "-e", "AWS_ENDPOINT_URL=http://minio:9000",
                "-w", "/app",
                self.docker_image,
                "python", container_script_path
            ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # Parse JSON output
                output_lines = result.stdout.strip().split('\n')
                json_output = output_lines[-1]  # Last line should be JSON
                return json.loads(json_output)
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Operation timed out after {timeout} seconds"
            }
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"Invalid JSON output: {result.stdout}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def create_docker_runner() -> DockerPySparkRunner:
    """Factory function to create DockerPySparkRunner instance."""
    return DockerPySparkRunner()
