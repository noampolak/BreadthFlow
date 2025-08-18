"""
Spark Job Submitter for BreadthFlow

Submits Spark jobs to the containerized Spark cluster where Delta Lake is properly available.
"""

import subprocess
import tempfile
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SparkSubmitter:
    """Submit Spark jobs to the containerized cluster."""
    
    def __init__(self, spark_master_url="spark://spark-master:7077"):
        self.spark_master_url = spark_master_url
        self.project_root = Path(__file__).parent.parent
        
    def submit_data_fetch_job(self, symbols, start_date, end_date, max_workers=2):
        """Submit data fetch job to Spark cluster."""
        
        # Create temporary Python script for the job
        job_script = self._create_fetch_job_script(symbols, start_date, end_date, max_workers)
        
        try:
            # Submit job using spark-submit inside the master container
            cmd = [
                "docker", "exec", "spark-master",
                "spark-submit",
                "--master", self.spark_master_url,
                "--deploy-mode", "cluster",
                "--class", "org.apache.spark.deploy.PythonRunner",
                "--conf", "spark.executor.memory=1g",
                "--conf", "spark.executor.cores=1",
                "--conf", "spark.driver.memory=1g",
                "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension",
                "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog",
                "--conf", "spark.hadoop.fs.s3a.endpoint=http://minio:9000",
                "--conf", "spark.hadoop.fs.s3a.access.key=minioadmin",
                "--conf", "spark.hadoop.fs.s3a.secret.key=minioadmin",
                "--conf", "spark.hadoop.fs.s3a.path.style.access=true",
                "--conf", "spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem",
                "--conf", "spark.hadoop.fs.s3a.aws.credentials.provider=org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
                "--py-files", "/opt/bitnami/spark/jobs/breadthflow_jobs.py",
                job_script
            ]
            
            logger.info(f"Submitting job with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info("Job submitted successfully")
            logger.info(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.warning(f"STDERR: {result.stderr}")
                
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Job submission failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
        finally:
            # Clean up temporary script
            if os.path.exists(job_script):
                os.unlink(job_script)
    
    def _create_fetch_job_script(self, symbols, start_date, end_date, max_workers):
        """Create a temporary Python script for the fetch job."""
        
        script_content = f'''
import sys
import os
sys.path.append('/opt/bitnami/spark/jobs')

from pyspark.sql import SparkSession
from ingestion.data_fetcher import DataFetcher

def main():
    # Create Spark session inside container
    spark = SparkSession.builder \\
        .appName("BreadthFlow-DataFetch") \\
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \\
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \\
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \\
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \\
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \\
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \\
        .getOrCreate()
    
    # Initialize DataFetcher
    fetcher = DataFetcher(spark)
    
    # Fetch data
    result = fetcher.fetch_and_store(
        symbols={symbols},
        start_date="{start_date}",
        end_date="{end_date}",
        max_workers={max_workers}
    )
    
    print(f"Fetch result: {{result}}")
    
    spark.stop()

if __name__ == "__main__":
    main()
'''
        
        # Create temporary file
        fd, script_path = tempfile.mkstemp(suffix='.py', prefix='breadthflow_fetch_')
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        
        return script_path

