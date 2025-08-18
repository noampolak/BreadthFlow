#!/usr/bin/env python3
"""
Direct Spark Runner for BreadthFlow

Runs Spark jobs directly without using spark-submit to avoid Ivy issues.
"""

import tempfile
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import subprocess

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

logger = logging.getLogger(__name__)

class DirectSparkRunner:
    """Run Spark jobs directly without spark-submit."""
    
    def __init__(self):
        self.spark_master_url = "spark://spark-master:7077"
        
    def run_data_fetch_job(self, symbols: List[str], start_date: str, end_date: str, max_workers: int = 2) -> Dict[str, Any]:
        """Run data fetch job directly."""
        
        # Set environment variables for Spark
        env = os.environ.copy()
        env['HOME'] = '/opt/bitnami/spark'
        env['USER'] = '1001'
        env['SPARK_HOME'] = '/opt/bitnami/spark'
        env['SPARK_CONF_DIR'] = '/opt/bitnami/spark/conf'
        env['SPARK_LOCAL_DIRS'] = '/tmp/spark-local'
        env['IVY_HOME'] = '/opt/bitnami/spark/.ivy2'
        env['IVY_LOCAL_REPO'] = '/opt/bitnami/spark/.ivy2/local'
        env['IVY_SHARED_REPO'] = '/opt/bitnami/spark/.ivy2/shared'
        env['IVY_CACHE_DIR'] = '/opt/bitnami/spark/.ivy2/cache'
        
        # Create temporary Python script for the job
        job_script = self._create_fetch_job_script(symbols, start_date, end_date, max_workers)
        
        try:
            # Run the Python script directly
            cmd = ["python3", job_script]
            
            logger.info(f"🔄 Running direct Spark job: {' '.join(cmd)}")
            
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
    
    # Create Spark session in local mode first to avoid Ivy issues
    spark = SparkSession.builder \\
        .appName("BreadthFlow-DataFetch") \\
        .master("local[*]") \\
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \\
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \\
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \\
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \\
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \\
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \\
        .config("spark.jars.packages", "") \\
        .config("spark.jars.excludes", "*") \\
        .getOrCreate()
    
    # Configure Delta Lake after session creation
    spark.conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    spark.conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    
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
