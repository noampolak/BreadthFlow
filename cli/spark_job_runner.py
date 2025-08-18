#!/usr/bin/env python3
"""
Spark Job Runner for BreadthFlow

Runs Spark jobs using spark-submit from within the container.
"""

import subprocess
import tempfile
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

logger = logging.getLogger(__name__)

class SparkJobRunner:
    """Run Spark jobs using spark-submit."""
    
    def __init__(self):
        self.spark_master_url = "spark://spark-master:7077"
        
    def run_data_fetch_job(self, symbols: List[str], start_date: str, end_date: str, max_workers: int = 2) -> Dict[str, Any]:
        """Run data fetch job using spark-submit."""
        
        # Set environment variables for Spark
        env = os.environ.copy()
        env['HOME'] = '/opt/bitnami/spark'
        env['USER'] = '1001'
        env['SPARK_HOME'] = '/opt/bitnami/spark'
        env['SPARK_CONF_DIR'] = '/opt/bitnami/spark/conf'
        env['SPARK_LOCAL_DIRS'] = '/tmp/spark-local'
        env['IVY_HOME'] = '/opt/bitnami/spark/.ivy2'
        
        # Create temporary Python script for the job
        job_script = self._create_fetch_job_script(symbols, start_date, end_date, max_workers)
        
        try:
            # Submit job using spark-submit
            cmd = [
                "spark-submit",
                "--master", self.spark_master_url,
                "--deploy-mode", "client",
                "--conf", "spark.executor.memory=1g",
                "--conf", "spark.executor.cores=1",
                "--conf", "spark.driver.memory=1g",
                # Delta Lake extensions are causing Ivy issues, using JARs directly instead
                "--conf", "spark.hadoop.fs.s3a.endpoint=http://minio:9000",
                "--conf", "spark.hadoop.fs.s3a.access.key=minioadmin",
                "--conf", "spark.hadoop.fs.s3a.secret.key=minioadmin",
                "--conf", "spark.hadoop.fs.s3a.path.style.access=true",
                "--conf", "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem",
                "--conf", "spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
                # Removed spark.jars.packages to avoid Ivy dependency resolution
                "--conf", "spark.driver.extraClassPath=/opt/bitnami/spark/jars/*",
                "--conf", "spark.executor.extraClassPath=/opt/bitnami/spark/jars/*",
                "--py-files", "/opt/bitnami/spark/jobs/ingestion/data_fetcher.py",
                job_script
            ]
            
            logger.info(f"🔄 Running Spark job: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            
            logger.info("✅ Job completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr}")
                
            return {"success": True, "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Job failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            return {"success": False, "error": str(e), "stderr": e.stderr}
        finally:
            # Clean up temporary script
            if os.path.exists(job_script):
                os.unlink(job_script)
    
    def _create_fetch_job_script(self, symbols: List[str], start_date: str, end_date: str, max_workers: int) -> str:
        """Create a temporary Python script for the fetch job."""
        
        script_content = f'''
import sys
import os
sys.path.insert(0, '/opt/bitnami/spark/jobs')

from pyspark.sql import SparkSession
from ingestion.data_fetcher import DataFetcher

def main():
    print("🚀 Starting BreadthFlow data fetch job...")
    
    # Create Spark session
    spark = SparkSession.builder \\
        .appName("BreadthFlow-DataFetch") \\
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \\
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \\
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \\
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \\
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \\
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \\
        .getOrCreate()
    
    # Configure Delta Lake programmatically
    from delta import configure_spark_with_delta_pip
    spark = configure_spark_with_delta_pip(spark)
    
    print("✅ Spark session created successfully")
    
    # Initialize DataFetcher
    fetcher = DataFetcher(spark)
    
    # Fetch data
    print(f"📊 Fetching data for symbols: {{symbols}}")
    result = fetcher.fetch_and_store(
        symbols={symbols},
        start_date="{start_date}",
        end_date="{end_date}",
        max_workers={max_workers}
    )
    
    print(f"✅ Fetch result: {{result}}")
    
    spark.stop()
    print("🏁 Job completed successfully")

if __name__ == "__main__":
    main()
'''
        
        # Create temporary file
        fd, script_path = tempfile.mkstemp(suffix='.py', prefix='breadthflow_fetch_')
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def check_cluster_status(self) -> bool:
        """Check if the Spark cluster is healthy."""
        try:
            import requests
            response = requests.get("http://localhost:8080/json/", timeout=5)
            data = response.json()
            worker_count = len(data.get('workers', []))
            logger.info(f"✅ Spark cluster healthy with {worker_count} workers")
            return True
        except Exception as e:
            logger.error(f"❌ Spark cluster not healthy: {e}")
            return False
