#!/usr/bin/env python3
"""
Simple test Spark job to verify basic functionality.
"""

import sys
import os
sys.path.insert(0, '/opt/bitnami/spark/jobs')

from pyspark.sql import SparkSession

def main():
    print("🚀 Starting simple Spark test job...")
    
    # Create a simple Spark session without Delta Lake
    spark = SparkSession.builder \
        .appName("SimpleTest") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    print("✅ Spark session created successfully")
    
    # Create a simple DataFrame
    data = [("AAPL", 150.0), ("MSFT", 300.0), ("GOOGL", 2500.0)]
    df = spark.createDataFrame(data, ["symbol", "price"])
    
    print("✅ DataFrame created successfully")
    print(f"📊 DataFrame count: {df.count()}")
    df.show()
    
    spark.stop()
    print("🏁 Test job completed successfully")

if __name__ == "__main__":
    main()
