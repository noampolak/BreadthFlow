#!/usr/bin/env python3
"""
Local Spark test to verify basic PySpark functionality.
"""

import sys
import os
sys.path.insert(0, '/opt/bitnami/spark/jobs')

from pyspark.sql import SparkSession

def main():
    print("🚀 Starting local Spark test...")
    
    # Create a local Spark session
    spark = SparkSession.builder \
        .appName("LocalTest") \
        .master("local[*]") \
        .getOrCreate()
    
    print("✅ Local Spark session created successfully")
    
    # Create a simple DataFrame
    data = [("AAPL", 150.0), ("MSFT", 300.0), ("GOOGL", 2500.0)]
    df = spark.createDataFrame(data, ["symbol", "price"])
    
    print("✅ DataFrame created successfully")
    print(f"📊 DataFrame count: {df.count()}")
    df.show()
    
    spark.stop()
    print("🏁 Local test completed successfully")

if __name__ == "__main__":
    main()
