#!/usr/bin/env python3
"""
Basic Spark job without Delta Lake - test basic functionality.
"""

import sys
import os
sys.path.insert(0, '/opt/bitnami/spark/jobs')

from pyspark.sql import SparkSession

def main():
    print("🚀 Starting basic Spark job...")
    
    # Create a basic Spark session without Delta Lake
    spark = SparkSession.builder \
        .appName("BasicSparkJob") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    print("✅ Spark session created successfully")
    
    # Create a simple DataFrame
    data = [("AAPL", 150.0), ("MSFT", 300.0), ("GOOGL", 2500.0)]
    df = spark.createDataFrame(data, ["symbol", "price"])
    
    print("✅ DataFrame created successfully")
    print(f"📊 DataFrame count: {df.count()}")
    df.show()
    
    # Test basic operations
    result = df.filter(df.price > 200).count()
    print(f"📊 Symbols with price > 200: {result}")
    
    spark.stop()
    print("🏁 Basic Spark job completed successfully")

if __name__ == "__main__":
    main()
