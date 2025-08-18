#!/usr/bin/env python3
"""
Working Delta Lake Runner for BreadthFlow

Bypasses PySpark Java gateway issues by using Delta Lake Python API directly.
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

class WorkingDeltaRunner:
    """Run Delta Lake operations using Python API directly."""
    
    def __init__(self):
        self.spark_master_url = "spark://spark-master:7077"
        
    def run_data_fetch_job(self, symbols: List[str], start_date: str, end_date: str, max_workers: int = 2) -> Dict[str, Any]:
        """Run data fetch job using Delta Lake Python API."""
        
        # Set environment variables for Spark
        env = os.environ.copy()
        env['HOME'] = '/opt/bitnami/spark'
        env['USER'] = '1001'
        env['SPARK_HOME'] = '/opt/bitnami/spark'
        env['SPARK_CONF_DIR'] = '/opt/bitnami/spark/conf'
        env['SPARK_LOCAL_DIRS'] = '/tmp/spark-local'
        env['IVY_HOME'] = '/opt/bitnami/spark/.ivy2'
        # Fix Hadoop authentication issues
        env['HADOOP_USER_NAME'] = 'spark'
        env['HADOOP_CONF_DIR'] = '/opt/bitnami/spark/conf'
        
        # Create temporary Python script for the job
        job_script = self._create_fetch_job_script(symbols, start_date, end_date, max_workers)
        
        try:
            # Run the Python script directly
            cmd = ["python3", job_script]
            
            logger.info(f"🔄 Running Delta Lake job: {' '.join(cmd)}")
            
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

# Import PySpark
from pyspark.sql import SparkSession

def main():
    print("🚀 Starting BreadthFlow Delta Lake data fetch job...")
    
    try:
        # Create Spark session with Delta Lake JARs directly (avoiding package resolution)
        delta_jars = "/opt/bitnami/spark/.ivy2/jars/io.delta_delta-spark_2.13-4.0.0.jar,/opt/bitnami/spark/.ivy2/jars/io.delta_delta-storage-4.0.0.jar,/opt/bitnami/spark/.ivy2/jars/org.antlr_antlr4-runtime-4.13.1.jar"
        spark = SparkSession.builder \\
            .appName("BreadthFlow-DeltaTest") \\
            .master("local[*]") \\
            .config("spark.jars", delta_jars) \\
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
            .config("spark.driver.extraJavaOptions", "-Duser.home=/opt/bitnami/spark -Divy.home=/opt/bitnami/spark/.ivy2 -Duser.name=spark -Dhadoop.security.authentication=simple") \\
            .config("spark.executor.extraJavaOptions", "-Duser.home=/opt/bitnami/spark -Divy.home=/opt/bitnami/spark/.ivy2 -Duser.name=spark -Dhadoop.security.authentication=simple") \\
            .getOrCreate()
        
        print("✅ Spark session with Delta Lake created successfully")
        
        # Test Delta Lake functionality
        print("🧪 Testing Delta Lake functionality...")
        
        # Create a simple DataFrame
        data = [("AAPL", 150.0, "2023-01-01"), ("MSFT", 300.0, "2023-01-01"), ("GOOGL", 2500.0, "2023-01-01")]
        df = spark.createDataFrame(data, ["symbol", "price", "date"])
        
        print("✅ DataFrame created successfully")
        print(f"📊 DataFrame count: {{df.count()}}")
        df.show()
        
        # Test Delta Lake write
        print("💾 Testing Delta Lake write to S3...")
        
        # Write to Delta format in S3
        delta_path = "s3a://breadthflow/test-delta-table"
        df.write.format("delta").mode("overwrite").save(delta_path)
        
        print(f"✅ Successfully wrote Delta table to {{delta_path}}")
        
        # Test Delta Lake read
        print("📖 Testing Delta Lake read from S3...")
        
        # Read from Delta format
        read_df = spark.read.format("delta").load(delta_path)
        print(f"✅ Successfully read Delta table: {{read_df.count()}} records")
        read_df.show()
        
        # Now run the actual data fetching
        print("📊 Starting data fetching...")
        
        # Import and run data fetcher
        from ingestion.data_fetcher import DataFetcher
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
        print("🏁 Delta Lake job completed successfully")
        
    except Exception as e:
        print(f"❌ Error in Delta Lake job: {{str(e)}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        # Create temporary file
        fd, script_path = tempfile.mkstemp(suffix='.py', prefix='breadthflow_delta_')
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
