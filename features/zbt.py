"""
Zweig Breadth Thrust (ZBT) for Breadth/Thrust Signals POC

Implements the Zweig Breadth Thrust calculation for detecting powerful market momentum signals.
The ZBT is a rare but powerful signal that occurs when market breadth suddenly improves dramatically.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col,
    when,
    sum as spark_sum,
    avg,
    lag,
    lead,
    window,
    expr,
    row_number,
    rank,
    dense_rank,
    date_format,
    year,
    month,
    dayofmonth,
    count,
)
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType, BooleanType

from features.common.config import get_config
from features.common.io import read_delta, write_delta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZweigBreadthThrust:
    """
    Zweig Breadth Thrust (ZBT) Calculator.

    Calculates the Zweig Breadth Thrust indicator:
    - Measures the percentage of advancing issues over a 10-day period
    - ZBT occurs when this percentage exceeds 61.5% for the first time in 6 months
    - Extremely rare but powerful bullish signal
    - Helps identify major market bottoms and trend reversals
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize ZBT calculator.

        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()

        # ZBT calculation parameters
        self.zbt_period = int(self.config.get("ZBT_PERIOD", 10))
        self.zbt_threshold = float(self.config.get("ZBT_THRESHOLD", 0.615))  # 61.5%
        self.zbt_lookback_months = int(self.config.get("ZBT_LOOKBACK_MONTHS", 6))
        self.min_data_points = int(self.config.get("ZBT_MIN_DATA", 50))

        # Feature schema
        self.zbt_schema = StructType(
            [
                StructField("date", TimestampType(), False),
                StructField("advancing_issues", LongType(), True),
                StructField("declining_issues", LongType(), True),
                StructField("unchanged_issues", LongType(), True),
                StructField("total_issues", LongType(), True),
                StructField("advancing_percentage", DoubleType(), True),
                StructField("zbt_10day_percentage", DoubleType(), True),
                StructField("zbt_signal", BooleanType(), True),
                StructField("zbt_strength", DoubleType(), True),
                StructField("zbt_momentum", DoubleType(), True),
                StructField("zbt_rarity_score", DoubleType(), True),
                StructField("zbt_confidence", DoubleType(), True),
                StructField("computed_at", TimestampType(), False),
            ]
        )

        logger.info(f"ZBT initialized with {self.zbt_period}-day period and {self.zbt_threshold*100}% threshold")

    def calculate_zbt(
        self,
        df: DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> DataFrame:
        """
        Calculate ZBT for given data.

        Args:
            df: DataFrame with OHLCV data
            symbols: List of symbols to include (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            DataFrame with ZBT values
        """
        logger.info("Calculating Zweig Breadth Thrust")

        # Apply filters
        if symbols:
            df = df.filter(col("symbol").isin(symbols))
            logger.info(f"Filtered to {len(symbols)} symbols")

        if start_date:
            df = df.filter(col("date") >= start_date)

        if end_date:
            df = df.filter(col("date") <= end_date)

        # Filter out low-quality data
        df = self._filter_quality_data(df)

        # Calculate daily advancing/declining issues
        daily_issues = self._calculate_daily_issues(df)

        # Calculate 10-day advancing percentage
        zbt_percentage = self._calculate_zbt_percentage(daily_issues)

        # Detect ZBT signals
        zbt_signals = self._detect_zbt_signals(zbt_percentage)

        # Calculate additional ZBT metrics
        zbt_metrics = self._calculate_zbt_metrics(zbt_signals)

        # Add computed timestamp
        result = zbt_metrics.withColumn("computed_at", expr("current_timestamp()"))

        logger.info(f"ZBT calculated for {result.count()} days")
        return result

    def _filter_quality_data(self, df: DataFrame) -> DataFrame:
        """
        Filter out low-quality data points.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        return df.filter((col("close").isNotNull()) & (col("close") > 0) & (col("volume").isNotNull()) & (col("volume") > 0))

    def _calculate_daily_issues(self, df: DataFrame) -> DataFrame:
        """
        Calculate daily advancing/declining issues.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with daily issues counts
        """
        # Calculate price changes
        window_spec = Window.partitionBy("symbol").orderBy("date")

        df_with_changes = df.withColumn(
            "price_change_pct", (col("close") - lag("close", 1).over(window_spec)) / lag("close", 1).over(window_spec)
        )

        # Classify stocks as advancing, declining, or unchanged
        df_classified = df_with_changes.withColumn(
            "issue_status",
            when(col("price_change_pct") > 0.001, "advancing")
            .when(col("price_change_pct") < -0.001, "declining")
            .otherwise("unchanged"),
        )

        # Aggregate by date
        daily_issues = df_classified.groupBy("date").agg(
            spark_sum(when(col("issue_status") == "advancing", 1).otherwise(0)).alias("advancing_issues"),
            spark_sum(when(col("issue_status") == "declining", 1).otherwise(0)).alias("declining_issues"),
            spark_sum(when(col("issue_status") == "unchanged", 1).otherwise(0)).alias("unchanged_issues"),
        )

        # Calculate total issues and advancing percentage
        daily_issues = daily_issues.withColumn(
            "total_issues", col("advancing_issues") + col("declining_issues") + col("unchanged_issues")
        ).withColumn(
            "advancing_percentage", when(col("total_issues") > 0, col("advancing_issues") / col("total_issues")).otherwise(0.0)
        )

        return daily_issues.orderBy("date")

    def _calculate_zbt_percentage(self, daily_issues: DataFrame) -> DataFrame:
        """
        Calculate 10-day advancing percentage for ZBT.

        Args:
            daily_issues: DataFrame with daily issues

        Returns:
            DataFrame with ZBT percentage
        """
        # Calculate 10-day rolling sum of advancing and total issues
        window_spec = Window.orderBy("date").rowsBetween(-self.zbt_period + 1, 0)

        zbt_percentage = (
            daily_issues.withColumn("zbt_advancing_sum", spark_sum("advancing_issues").over(window_spec))
            .withColumn("zbt_total_sum", spark_sum("total_issues").over(window_spec))
            .withColumn(
                "zbt_10day_percentage",
                when(col("zbt_total_sum") > 0, col("zbt_advancing_sum") / col("zbt_total_sum")).otherwise(0.0),
            )
        )

        return zbt_percentage

    def _detect_zbt_signals(self, zbt_percentage: DataFrame) -> DataFrame:
        """
        Detect ZBT signals based on threshold and rarity conditions.

        Args:
            zbt_percentage: DataFrame with ZBT percentage

        Returns:
            DataFrame with ZBT signals
        """
        # Window specification for lookback calculations
        window_spec = Window.orderBy("date")

        # Check if current percentage exceeds threshold
        zbt_signals = zbt_percentage.withColumn("above_threshold", col("zbt_10day_percentage") >= self.zbt_threshold)

        # Check if this is the first time above threshold in 6 months
        # For simplicity, we'll use a 180-day lookback
        lookback_days = self.zbt_lookback_months * 30

        zbt_signals = zbt_signals.withColumn(
            "recent_above_threshold",
            spark_sum(when(col("above_threshold"), 1).otherwise(0)).over(
                Window.orderBy("date").rowsBetween(-lookback_days, -1)
            ),
        ).withColumn("zbt_signal", (col("above_threshold")) & (col("recent_above_threshold") == 0))

        return zbt_signals

    def _calculate_zbt_metrics(self, zbt_signals: DataFrame) -> DataFrame:
        """
        Calculate additional ZBT metrics and confidence scores.

        Args:
            zbt_signals: DataFrame with ZBT signals

        Returns:
            DataFrame with ZBT metrics
        """
        # Window specification for calculations
        window_spec = Window.orderBy("date")

        # Calculate ZBT strength (how far above threshold)
        zbt_metrics = zbt_signals.withColumn(
            "zbt_strength", when(col("zbt_signal"), col("zbt_10day_percentage") - self.zbt_threshold).otherwise(0.0)
        )

        # Calculate ZBT momentum (rate of change)
        zbt_metrics = zbt_metrics.withColumn(
            "zbt_momentum", col("zbt_10day_percentage") - lag("zbt_10day_percentage", 5).over(window_spec)
        )

        # Calculate rarity score (how unusual this percentage is)
        window_60d = Window.orderBy("date").rowsBetween(-60, 0)
        zbt_metrics = (
            zbt_metrics.withColumn("avg_60d_percentage", avg("zbt_10day_percentage").over(window_60d))
            .withColumn(
                "std_60d_percentage",
                expr("stddev(zbt_10day_percentage) over (order by date rows between 60 preceding and current row)"),
            )
            .withColumn(
                "zbt_rarity_score",
                when(
                    col("std_60d_percentage") > 0,
                    (col("zbt_10day_percentage") - col("avg_60d_percentage")) / col("std_60d_percentage"),
                ).otherwise(0.0),
            )
        )

        # Calculate confidence score
        zbt_metrics = zbt_metrics.withColumn(
            "zbt_confidence",
            when(
                col("zbt_signal"), (col("zbt_strength") * 0.4 + col("zbt_momentum") * 0.3 + col("zbt_rarity_score") * 0.3)
            ).otherwise(0.0),
        )

        return zbt_metrics

    def calculate_zbt_from_delta(
        self,
        table_path: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> DataFrame:
        """
        Calculate ZBT from Delta Lake table.

        Args:
            table_path: Path to Delta table with OHLCV data
            symbols: List of symbols to include
            start_date: Start date filter
            end_date: End date filter
            output_path: Path to save results (optional)

        Returns:
            DataFrame with ZBT values
        """
        logger.info(f"Loading data from {table_path}")

        # Read from Delta Lake
        df = read_delta(self.spark, table_path)

        # Calculate ZBT
        zbt_result = self.calculate_zbt(df, symbols, start_date, end_date)

        # Save to Delta Lake if output path provided
        if output_path:
            logger.info(f"Saving ZBT to {output_path}")
            write_delta(df=zbt_result, path=output_path, partition_cols=["year", "month"], mode="append")

        return zbt_result

    def get_zbt_summary(self, zbt_result: DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for ZBT.

        Args:
            zbt_result: DataFrame with ZBT values

        Returns:
            Dictionary with summary statistics
        """
        summary = zbt_result.agg(
            expr("avg(zbt_10day_percentage)").alias("avg_zbt_percentage"),
            expr("min(zbt_10day_percentage)").alias("min_zbt_percentage"),
            expr("max(zbt_10day_percentage)").alias("max_zbt_percentage"),
            expr("stddev(zbt_10day_percentage)").alias("zbt_percentage_volatility"),
            expr("avg(zbt_strength)").alias("avg_zbt_strength"),
            expr("avg(zbt_momentum)").alias("avg_zbt_momentum"),
            expr("avg(zbt_rarity_score)").alias("avg_zbt_rarity"),
            expr("avg(zbt_confidence)").alias("avg_zbt_confidence"),
            spark_sum(when(col("zbt_signal"), 1).otherwise(0)).alias("total_zbt_signals"),
            spark_sum(when(col("above_threshold"), 1).otherwise(0)).alias("days_above_threshold"),
        ).collect()[0]

        return {
            "avg_zbt_percentage": summary.avg_zbt_percentage,
            "min_zbt_percentage": summary.min_zbt_percentage,
            "max_zbt_percentage": summary.max_zbt_percentage,
            "zbt_percentage_volatility": summary.zbt_percentage_volatility,
            "avg_zbt_strength": summary.avg_zbt_strength,
            "avg_zbt_momentum": summary.avg_zbt_momentum,
            "avg_zbt_rarity": summary.avg_zbt_rarity,
            "avg_zbt_confidence": summary.avg_zbt_confidence,
            "total_zbt_signals": summary.total_zbt_signals,
            "days_above_threshold": summary.days_above_threshold,
        }

    def detect_zbt_signals(self, zbt_result: DataFrame) -> DataFrame:
        """
        Detect comprehensive ZBT-based trading signals.

        Args:
            zbt_result: DataFrame with ZBT values

        Returns:
            DataFrame with trading signals
        """
        # Define comprehensive signal conditions
        signals = (
            zbt_result.withColumn(
                "zbt_bullish_signal", (col("zbt_signal")) & (col("zbt_confidence") > 0.7) & (col("zbt_strength") > 0.05)
            )
            .withColumn(
                "zbt_strong_signal",
                (col("zbt_signal"))
                & (col("zbt_confidence") > 0.8)
                & (col("zbt_strength") > 0.1)
                & (col("zbt_rarity_score") > 2.0),
            )
            .withColumn(
                "zbt_momentum_signal",
                (col("zbt_10day_percentage") > 0.7) & (col("zbt_momentum") > 0.05) & (col("zbt_rarity_score") > 1.5),
            )
            .withColumn(
                "zbt_breakout_signal", (col("zbt_signal")) & (col("zbt_10day_percentage") > 0.75) & (col("zbt_momentum") > 0.1)
            )
        )

        return signals

    def calculate_zbt_rolling_metrics(self, zbt_result: DataFrame, window_days: int = 20) -> DataFrame:
        """
        Calculate rolling ZBT metrics.

        Args:
            zbt_result: DataFrame with ZBT values
            window_days: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        window_spec = Window.orderBy("date").rowsBetween(-window_days, 0)

        rolling_metrics = (
            zbt_result.withColumn(
                "rolling_zbt_percentage",
                expr(f"avg(zbt_10day_percentage) over (order by date rows between {window_days} preceding and current row)"),
            )
            .withColumn(
                "rolling_zbt_momentum",
                expr(f"avg(zbt_momentum) over (order by date rows between {window_days} preceding and current row)"),
            )
            .withColumn(
                "rolling_zbt_rarity",
                expr(f"avg(zbt_rarity_score) over (order by date rows between {window_days} preceding and current row)"),
            )
            .withColumn(
                "zbt_trend",
                expr(f"avg(zbt_10day_percentage) over (order by date rows between {window_days} preceding and current row)")
                - expr(
                    f"avg(zbt_10day_percentage) over (order by date rows between {window_days*2} preceding and {window_days} preceding)"
                ),
            )
        )

        return rolling_metrics

    def get_zbt_historical_analysis(self, zbt_result: DataFrame) -> Dict[str, Any]:
        """
        Get historical analysis of ZBT signals.

        Args:
            zbt_result: DataFrame with ZBT values

        Returns:
            Dictionary with historical analysis
        """
        # Get all ZBT signals
        zbt_signals = zbt_result.filter(col("zbt_signal"))

        if zbt_signals.count() == 0:
            return {"message": "No ZBT signals found in the data"}

        # Analyze signal characteristics
        analysis = zbt_signals.agg(
            expr("avg(zbt_10day_percentage)").alias("avg_signal_percentage"),
            expr("avg(zbt_strength)").alias("avg_signal_strength"),
            expr("avg(zbt_confidence)").alias("avg_signal_confidence"),
            expr("avg(zbt_rarity_score)").alias("avg_signal_rarity"),
            expr("min(date)").alias("first_signal_date"),
            expr("max(date)").alias("last_signal_date"),
            count("*").alias("total_signals"),
        ).collect()[0]

        return {
            "total_signals": analysis.total_signals,
            "avg_signal_percentage": analysis.avg_signal_percentage,
            "avg_signal_strength": analysis.avg_signal_strength,
            "avg_signal_confidence": analysis.avg_signal_confidence,
            "avg_signal_rarity": analysis.avg_signal_rarity,
            "first_signal_date": analysis.first_signal_date,
            "last_signal_date": analysis.last_signal_date,
            "signal_frequency": f"Every {zbt_result.count() / analysis.total_signals:.1f} days on average",
        }


def create_zbt(spark: SparkSession) -> ZweigBreadthThrust:
    """
    Factory function to create ZweigBreadthThrust instance.

    Args:
        spark: PySpark session

    Returns:
        Configured ZweigBreadthThrust instance
    """
    return ZweigBreadthThrust(spark)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # Create Spark session
    spark = (
        SparkSession.builder.appName("ZBTTest")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    # Create ZBT calculator
    zbt = create_zbt(spark)

    try:
        # Calculate ZBT from Delta table
        result = zbt.calculate_zbt_from_delta(
            table_path="data/ohlcv",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "CRM", "ADBE"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path="data/zbt",
        )

        # Get summary
        summary = zbt.get_zbt_summary(result)
        print(f"ZBT Summary: {summary}")

        # Get historical analysis
        analysis = zbt.get_zbt_historical_analysis(result)
        print(f"ZBT Historical Analysis: {analysis}")

        # Detect signals
        signals = zbt.detect_zbt_signals(result)
        print(f"Detected {signals.filter(col('zbt_bullish_signal')).count()} bullish signals")
        print(f"Detected {signals.filter(col('zbt_strong_signal')).count()} strong signals")
        print(f"Detected {signals.filter(col('zbt_momentum_signal')).count()} momentum signals")
        print(f"Detected {signals.filter(col('zbt_breakout_signal')).count()} breakout signals")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        spark.stop()
