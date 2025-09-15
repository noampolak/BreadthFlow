"""
Delta Lake I/O utilities for Breadth/Thrust Signals POC.

Provides functions for reading and writing Delta tables with proper partitioning.
"""

import os
from typing import List, Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col


def write_delta(df: DataFrame, path: str, partition_cols: Optional[List[str]] = None, mode: str = "append") -> None:
    """
    Write DataFrame to Delta Lake with partitioning.

    Args:
        df: Spark DataFrame to write
        path: Delta table path
        partition_cols: List of columns to partition by
        mode: Write mode (append, overwrite, errorIfExists, ignore)
    """
    writer = df.write.format("delta").mode(mode)

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.save(path)


def read_delta(spark: SparkSession, path: str) -> DataFrame:
    """
    Read DataFrame from Delta Lake.

    Args:
        spark: SparkSession instance
        path: Delta table path

    Returns:
        Spark DataFrame
    """
    return spark.read.format("delta").load(path)


def read_delta_partition(spark: SparkSession, path: str, partition_filter: str) -> DataFrame:
    """
    Read DataFrame from Delta Lake with partition filtering.

    Args:
        spark: SparkSession instance
        path: Delta table path
        partition_filter: Partition filter expression (e.g., "date >= '2024-01-01'")

    Returns:
        Spark DataFrame
    """
    return spark.read.format("delta").load(path).filter(partition_filter)


def upsert_delta(df: DataFrame, path: str, merge_key: str, partition_cols: Optional[List[str]] = None) -> None:
    """
    Upsert DataFrame to Delta Lake using merge operation.

    Args:
        df: Spark DataFrame to upsert
        path: Delta table path
        merge_key: Column to use for merge matching
        partition_cols: List of columns to partition by
    """
    from delta.tables import DeltaTable

    # Check if table exists
    if DeltaTable.isDeltaTable(spark, path):
        table = DeltaTable.forPath(spark, path)

        # Perform merge
        table.alias("target").merge(
            df.alias("source"), f"target.{merge_key} = source.{merge_key}"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    else:
        # Create new table
        write_delta(df, path, partition_cols, mode="overwrite")


def optimize_delta_table(spark: SparkSession, path: str) -> None:
    """
    Optimize Delta table for better query performance.

    Args:
        spark: SparkSession instance
        path: Delta table path
    """
    spark.sql(f"OPTIMIZE '{path}'")


def vacuum_delta_table(spark: SparkSession, path: str, retention_hours: int = 168) -> None:
    """
    Vacuum Delta table to remove old files.

    Args:
        spark: SparkSession instance
        path: Delta table path
        retention_hours: Hours to retain files (default: 7 days)
    """
    spark.sql(f"VACUUM '{path}' RETAIN {retention_hours} HOURS")


def get_delta_table_info(spark: SparkSession, path: str) -> dict:
    """
    Get information about Delta table.

    Args:
        spark: SparkSession instance
        path: Delta table path

    Returns:
        Dictionary with table information
    """
    from delta.tables import DeltaTable

    if not DeltaTable.isDeltaTable(spark, path):
        return {"exists": False}

    table = DeltaTable.forPath(spark, path)
    detail = table.detail().collect()[0]

    return {
        "exists": True,
        "name": detail["name"],
        "location": detail["location"],
        "format": detail["format"],
        "partitionColumns": detail["partitionColumns"],
        "numFiles": detail["numFiles"],
        "sizeInBytes": detail["sizeInBytes"],
    }
