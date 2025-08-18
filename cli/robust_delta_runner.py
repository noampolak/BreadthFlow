#!/usr/bin/env python3
"""
Robust Delta Lake Runner for BreadthFlow

A simplified, reliable approach focused on making Spark + Delta Lake work 
without complex subprocess calls or temporary files.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

logger = logging.getLogger(__name__)

class RobustDeltaRunner:
    """Simplified, robust Delta Lake runner that works reliably."""
    
    def __init__(self):
        self.spark = None
        
    def _create_spark_session(self):
        """Create a robust Spark session with Delta Lake support."""
        if self.spark is not None:
            return self.spark
            
        try:
            from pyspark.sql import SparkSession
            
            # Create Spark session with comprehensive configuration
            builder = SparkSession.builder \
                .appName("BreadthFlow-Robust") \
                .master("spark://spark-master:7077") \
                .config("spark.driver.host", "spark-master") \
                .config("spark.driver.bindAddress", "0.0.0.0") \
                .config("spark.driver.extraJavaOptions", 
                       "-Duser.home=/opt/bitnami/spark " +
                       "-Duser.name=spark " +
                       "-Divy.home=/opt/bitnami/spark/.ivy2 " +
                       "-Dhadoop.security.authentication=simple " +
                       "-Djava.security.auth.login.config= " +
                       "-Dhadoop.home.dir=/opt/bitnami/spark") \
                .config("spark.executor.extraJavaOptions", 
                       "-Duser.home=/opt/bitnami/spark " +
                       "-Duser.name=spark " +
                       "-Divy.home=/opt/bitnami/spark/.ivy2 " +
                       "-Dhadoop.security.authentication=simple " +
                       "-Djava.security.auth.login.config= " +
                       "-Dhadoop.home.dir=/opt/bitnami/spark") \
                .config("spark.hadoop.hadoop.security.authentication", "simple") \
                .config("spark.hadoop.hadoop.security.authorization", "false") \
                .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
                .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
                .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
                .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
                .config("spark.hadoop.fs.s3a.path.style.access", "true") \
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
                .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
            
            # Try to add Delta Lake if available
            try:
                # Check if Delta Lake JARs are available
                delta_jars_path = "/opt/bitnami/spark/.ivy2/jars"
                if os.path.exists(delta_jars_path):
                    delta_jars = []
                    for jar_file in os.listdir(delta_jars_path):
                        if jar_file.startswith("io.delta") or jar_file.startswith("org.antlr"):
                            delta_jars.append(os.path.join(delta_jars_path, jar_file))
                    
                    if delta_jars:
                        builder = builder.config("spark.jars", ",".join(delta_jars)) \
                                        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                                        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                        logger.info(f"Found Delta Lake JARs: {len(delta_jars)} files")
                    else:
                        logger.warning("No Delta Lake JARs found, running without Delta Lake")
                else:
                    logger.warning("Delta Lake JAR directory not found, running without Delta Lake")
                    
            except Exception as e:
                logger.warning(f"Could not configure Delta Lake: {e}")
                
            self.spark = builder.getOrCreate()
            logger.info("✅ Spark session created successfully")
            return self.spark
            
        except Exception as e:
            logger.error(f"Failed to create Spark session: {e}")
            raise
    
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic Spark functionality."""
        try:
            spark = self._create_spark_session()
            
            # Test basic DataFrame operations
            logger.info("🧪 Testing basic DataFrame operations...")
            data = [("AAPL", 150.0, "2023-01-01"), ("MSFT", 300.0, "2023-01-01")]
            df = spark.createDataFrame(data, ["symbol", "price", "date"])
            
            count = df.count()
            logger.info(f"✅ DataFrame created successfully with {count} rows")
            
            return {"success": True, "row_count": count, "message": "Basic functionality working"}
            
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_delta_functionality(self) -> Dict[str, Any]:
        """Test Delta Lake functionality if available."""
        try:
            spark = self._create_spark_session()
            
            # Check if Delta Lake is available
            try:
                # Try to use Delta Lake format
                logger.info("🧪 Testing Delta Lake functionality...")
                data = [("TEST", 100.0, "2023-01-01")]
                df = spark.createDataFrame(data, ["symbol", "price", "date"])
                
                # Try to write to Delta format in a temp location
                test_path = "/tmp/test-delta-table"
                df.write.format("delta").mode("overwrite").save(test_path)
                
                # Try to read back
                read_df = spark.read.format("delta").load(test_path)
                read_count = read_df.count()
                
                logger.info(f"✅ Delta Lake functionality working! Read {read_count} rows")
                return {"success": True, "delta_available": True, "row_count": read_count}
                
            except Exception as delta_error:
                logger.warning(f"Delta Lake not available: {delta_error}")
                return {"success": True, "delta_available": False, "message": "Spark works but Delta Lake not available"}
                
        except Exception as e:
            logger.error(f"❌ Delta functionality test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_data_fetch_job(self, symbols: List[str], start_date: str, end_date: str, max_workers: int = 2) -> Dict[str, Any]:
        """Run a data fetch job (simplified for robustness)."""
        try:
            logger.info(f"🚀 Starting data fetch for symbols: {symbols}")
            
            # First test basic functionality
            basic_result = self.test_basic_functionality()
            if not basic_result["success"]:
                return basic_result
                
            # Test Delta Lake
            delta_result = self.test_delta_functionality()
            
            # Try to import and use the actual data fetcher if Spark is working
            spark = self._create_spark_session()
            
            try:
                from ingestion.data_fetcher import DataFetcher
                logger.info("📊 Using actual DataFetcher...")
                
                fetcher = DataFetcher(spark)
                result = fetcher.fetch_and_store(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    max_workers=max_workers
                )
                
                logger.info(f"✅ Data fetch completed: {result}")
                return {"success": True, "fetch_result": result, "delta_available": delta_result.get("delta_available", False)}
                
            except ImportError as e:
                logger.warning(f"DataFetcher not available: {e}")
                return {"success": True, "message": "Spark working but DataFetcher not available", "basic_test": basic_result, "delta_test": delta_result}
            except Exception as e:
                logger.error(f"DataFetcher failed: {e}")
                return {"success": False, "error": f"DataFetcher failed: {e}", "basic_test": basic_result, "delta_test": delta_result}
                
        except Exception as e:
            logger.error(f"❌ Data fetch job failed: {e}")
            return {"success": False, "error": str(e)}
    
    def close(self):
        """Close the Spark session."""
        if self.spark:
            self.spark.stop()
            self.spark = None

if __name__ == "__main__":
    runner = RobustDeltaRunner()
    try:
        result = runner.run_data_fetch_job(['AAPL'], '2023-01-01', '2023-01-05', 1)
        print(f"Result: {result}")
    finally:
        runner.close()
