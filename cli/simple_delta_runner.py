#!/usr/bin/env python3
"""
Simple Delta Lake Runner - No Authentication, No Complexity

Designed for POC environments where we completely disable all authentication.
"""

import sys
import logging

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

def test_basic_spark():
    """Test the most basic Spark functionality."""
    try:
        from pyspark.sql import SparkSession
        
        print("🚀 Testing basic Spark (no auth, no complexity)...")
        
        # Create the simplest possible Spark session
        spark = SparkSession.builder \
            .appName("SimplePOC") \
            .master("local[*]") \
            .getOrCreate()
        
        # Test basic DataFrame
        data = [("AAPL", 150.0), ("MSFT", 300.0)]
        df = spark.createDataFrame(data, ["symbol", "price"])
        count = df.count()
        
        print(f"✅ Basic Spark works! Created DataFrame with {count} rows")
        df.show()
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ Basic Spark failed: {e}")
        return False

def test_cluster_spark():
    """Test Spark cluster connectivity."""
    try:
        from pyspark.sql import SparkSession
        
        print("🚀 Testing Spark cluster connectivity...")
        
        # Connect to the cluster with minimal config
        spark = SparkSession.builder \
            .appName("ClusterPOC") \
            .master("spark://spark-master:7077") \
            .config("spark.driver.host", "spark-master") \
            .config("spark.driver.bindAddress", "0.0.0.0") \
            .getOrCreate()
        
        # Test basic DataFrame on cluster
        data = [("TEST", 100.0), ("CLUSTER", 200.0)]
        df = spark.createDataFrame(data, ["symbol", "price"])
        count = df.count()
        
        print(f"✅ Cluster Spark works! Processed {count} rows on cluster")
        df.show()
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ Cluster Spark failed: {e}")
        return False

def test_simple_s3():
    """Test basic S3 connectivity without Delta Lake."""
    try:
        from pyspark.sql import SparkSession
        
        print("🚀 Testing S3 connectivity...")
        
        spark = SparkSession.builder \
            .appName("S3POC") \
            .master("local[*]") \
            .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
            .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
            .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
            .getOrCreate()
        
        # Create test data
        data = [("S3TEST", 999.0)]
        df = spark.createDataFrame(data, ["symbol", "price"])
        
        # Try to write to S3 in parquet format (simpler than Delta)
        test_path = "s3a://breadthflow/simple-test-parquet"
        df.write.mode("overwrite").parquet(test_path)
        
        # Try to read back
        read_df = spark.read.parquet(test_path)
        count = read_df.count()
        
        print(f"✅ S3 works! Wrote and read {count} rows to/from MinIO")
        read_df.show()
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ S3 test failed: {e}")
        return False

def test_delta_lake():
    """Test Delta Lake functionality with pre-downloaded JARs."""
    try:
        from pyspark.sql import SparkSession
        import os
        
        print("🚀 Testing Delta Lake...")
        
        # Check if Delta JARs exist
        jar_dir = "/opt/bitnami/spark/.ivy2/jars"
        delta_jars = []
        if os.path.exists(jar_dir):
            for jar_file in os.listdir(jar_dir):
                if "delta" in jar_file.lower() or "antlr" in jar_file.lower():
                    delta_jars.append(os.path.join(jar_dir, jar_file))
        
        if not delta_jars:
            print("❌ No Delta Lake JARs found")
            return False
            
        print(f"Found {len(delta_jars)} Delta Lake JARs")
        
        # Create Spark session with Delta Lake
        spark = SparkSession.builder \
            .appName("DeltaPOC") \
            .master("local[*]") \
            .config("spark.jars", ",".join(delta_jars)) \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        # Create test data
        data = [("DELTA", 777.0), ("LAKE", 888.0)]
        df = spark.createDataFrame(data, ["symbol", "price"])
        
        # Test Delta Lake write to local filesystem first
        local_delta_path = "/tmp/simple-delta-test"
        df.write.format("delta").mode("overwrite").save(local_delta_path)
        
        # Test Delta Lake read
        read_df = spark.read.format("delta").load(local_delta_path)
        count = read_df.count()
        
        print(f"✅ Delta Lake works! Wrote and read {count} rows")
        read_df.show()
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"❌ Delta Lake test failed: {e}")
        return False

def main():
    """Run all tests in order of complexity."""
    print("=" * 60)
    print("🔧 Simple POC Tests - No Auth, Maximum Compatibility")
    print("=" * 60)
    
    tests = [
        ("Basic Spark (Local)", test_basic_spark),
        ("Spark Cluster", test_cluster_spark), 
        ("S3 Connectivity", test_simple_s3),
        ("Delta Lake", test_delta_lake)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        results[test_name] = test_func()
        print()
    
    print("=" * 60)
    print("📊 FINAL RESULTS:")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    total_passed = sum(results.values())
    print(f"\nPassed: {total_passed}/{len(tests)} tests")
    
    if total_passed == len(tests):
        print("🎉 ALL TESTS PASSED! Infrastructure is working!")
    elif total_passed >= 2:
        print("⚠️  PARTIAL SUCCESS - Core functionality working")
    else:
        print("💥 MAJOR ISSUES - Infrastructure needs fixing")

if __name__ == "__main__":
    main()
