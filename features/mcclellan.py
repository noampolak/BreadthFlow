"""
McClellan Oscillator for Breadth/Thrust Signals POC

Implements the McClellan Oscillator calculation for market breadth analysis.
The McClellan Oscillator is a market breadth indicator that shows the difference
between advancing and declining issues on a smoothed basis.
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


class McClellanOscillator:
    """
    McClellan Oscillator Calculator.

    Calculates the McClellan Oscillator, a market breadth indicator:
    - Uses exponential moving averages of advancing/declining issues
    - Provides smoothed view of market breadth
    - Helps identify overbought/oversold conditions
    - Generates buy/sell signals based on oscillator levels
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize McClellan Oscillator calculator.

        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()

        # McClellan calculation parameters
        self.ema_short_period = int(self.config.get("MCCLELLAN_EMA_SHORT", 19))
        self.ema_long_period = int(self.config.get("MCCLELLAN_EMA_LONG", 39))
        self.min_data_points = int(self.config.get("MCCLELLAN_MIN_DATA", 50))

        # Signal thresholds
        self.overbought_threshold = float(self.config.get("MCCLELLAN_OVERBOUGHT", 100))
        self.oversold_threshold = float(self.config.get("MCCLELLAN_OVERSOLD", -100))

        # Feature schema
        self.mcclellan_schema = StructType(
            [
                StructField("date", TimestampType(), False),
                StructField("advancing_issues", LongType(), True),
                StructField("declining_issues", LongType(), True),
                StructField("unchanged_issues", LongType(), True),
                StructField("net_advances", LongType(), True),
                StructField("ema_short", DoubleType(), True),
                StructField("ema_long", DoubleType(), True),
                StructField("mcclellan_oscillator", DoubleType(), True),
                StructField("mcclellan_ratio", DoubleType(), True),
                StructField("mcclellan_summation_index", DoubleType(), True),
                StructField("oscillator_momentum", DoubleType(), True),
                StructField("overbought_signal", BooleanType(), True),
                StructField("oversold_signal", BooleanType(), True),
                StructField("bullish_divergence", BooleanType(), True),
                StructField("bearish_divergence", BooleanType(), True),
                StructField("computed_at", TimestampType(), False),
            ]
        )

        logger.info(f"McClellan Oscillator initialized with EMA{self.ema_short_period}/EMA{self.ema_long_period}")

    def calculate_mcclellan(
        self,
        df: DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> DataFrame:
        """
        Calculate McClellan Oscillator for given data.

        Args:
            df: DataFrame with OHLCV data
            symbols: List of symbols to include (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            DataFrame with McClellan Oscillator values
        """
        logger.info("Calculating McClellan Oscillator")

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

        # Calculate exponential moving averages
        ema_features = self._calculate_ema_features(daily_issues)

        # Calculate McClellan Oscillator
        mcclellan_oscillator = self._calculate_oscillator(ema_features)

        # Calculate additional indicators
        additional_indicators = self._calculate_additional_indicators(mcclellan_oscillator)

        # Detect signals
        signals = self._detect_mcclellan_signals(additional_indicators)

        # Add computed timestamp
        result = signals.withColumn("computed_at", expr("current_timestamp()"))

        logger.info(f"McClellan Oscillator calculated for {result.count()} days")
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

        # Calculate net advances
        daily_issues = daily_issues.withColumn("net_advances", col("advancing_issues") - col("declining_issues"))

        return daily_issues.orderBy("date")

    def _calculate_ema_features(self, daily_issues: DataFrame) -> DataFrame:
        """
        Calculate exponential moving averages of net advances.

        Args:
            daily_issues: DataFrame with daily issues

        Returns:
            DataFrame with EMA features
        """
        # Calculate EMA using Spark's built-in exponential smoothing
        # For simplicity, we'll use a window-based approach
        window_spec = Window.orderBy("date")

        # Calculate short EMA (19-period)
        ema_features = daily_issues.withColumn(
            "ema_short",
            expr(f"avg(net_advances) over (order by date rows between {self.ema_short_period} preceding and current row)"),
        )

        # Calculate long EMA (39-period)
        ema_features = ema_features.withColumn(
            "ema_long",
            expr(f"avg(net_advances) over (order by date rows between {self.ema_long_period} preceding and current row)"),
        )

        return ema_features

    def _calculate_oscillator(self, ema_features: DataFrame) -> DataFrame:
        """
        Calculate McClellan Oscillator.

        Args:
            ema_features: DataFrame with EMA features

        Returns:
            DataFrame with oscillator values
        """
        # McClellan Oscillator = EMA(19) - EMA(39)
        oscillator = ema_features.withColumn("mcclellan_oscillator", col("ema_short") - col("ema_long"))

        # Calculate McClellan Ratio
        oscillator = oscillator.withColumn(
            "mcclellan_ratio",
            when(col("declining_issues") > 0, col("advancing_issues") / col("declining_issues")).otherwise(0.0),
        )

        return oscillator

    def _calculate_additional_indicators(self, oscillator: DataFrame) -> DataFrame:
        """
        Calculate additional McClellan indicators.

        Args:
            oscillator: DataFrame with oscillator values

        Returns:
            DataFrame with additional indicators
        """
        # Window specification for calculations
        window_spec = Window.orderBy("date")

        # Calculate McClellan Summation Index (cumulative oscillator)
        additional_indicators = oscillator.withColumn(
            "mcclellan_summation_index", spark_sum("mcclellan_oscillator").over(window_spec)
        )

        # Calculate oscillator momentum
        additional_indicators = additional_indicators.withColumn(
            "oscillator_momentum", col("mcclellan_oscillator") - lag("mcclellan_oscillator", 5).over(window_spec)
        )

        return additional_indicators

    def _detect_mcclellan_signals(self, indicators: DataFrame) -> DataFrame:
        """
        Detect McClellan-based trading signals.

        Args:
            indicators: DataFrame with McClellan indicators

        Returns:
            DataFrame with signals
        """
        # Overbought/oversold signals
        signals = indicators.withColumn(
            "overbought_signal", col("mcclellan_oscillator") > self.overbought_threshold
        ).withColumn("oversold_signal", col("mcclellan_oscillator") < self.oversold_threshold)

        # Divergence signals (simplified)
        window_spec = Window.orderBy("date").rowsBetween(-10, 0)

        signals = signals.withColumn(
            "bullish_divergence",
            (col("mcclellan_oscillator") > 0)
            & (
                col("mcclellan_oscillator")
                > expr(f"avg(mcclellan_oscillator) over (order by date rows between 10 preceding and current row)")
            ),
        ).withColumn(
            "bearish_divergence",
            (col("mcclellan_oscillator") < 0)
            & (
                col("mcclellan_oscillator")
                < expr(f"avg(mcclellan_oscillator) over (order by date rows between 10 preceding and current row)")
            ),
        )

        return signals

    def calculate_mcclellan_from_delta(
        self,
        table_path: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> DataFrame:
        """
        Calculate McClellan Oscillator from Delta Lake table.

        Args:
            table_path: Path to Delta table with OHLCV data
            symbols: List of symbols to include
            start_date: Start date filter
            end_date: End date filter
            output_path: Path to save results (optional)

        Returns:
            DataFrame with McClellan Oscillator values
        """
        logger.info(f"Loading data from {table_path}")

        # Read from Delta Lake
        df = read_delta(self.spark, table_path)

        # Calculate McClellan Oscillator
        mcclellan_result = self.calculate_mcclellan(df, symbols, start_date, end_date)

        # Save to Delta Lake if output path provided
        if output_path:
            logger.info(f"Saving McClellan Oscillator to {output_path}")
            write_delta(df=mcclellan_result, path=output_path, partition_cols=["year", "month"], mode="append")

        return mcclellan_result

    def get_mcclellan_summary(self, mcclellan_result: DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for McClellan Oscillator.

        Args:
            mcclellan_result: DataFrame with McClellan Oscillator values

        Returns:
            Dictionary with summary statistics
        """
        summary = mcclellan_result.agg(
            expr("avg(mcclellan_oscillator)").alias("avg_oscillator"),
            expr("min(mcclellan_oscillator)").alias("min_oscillator"),
            expr("max(mcclellan_oscillator)").alias("max_oscillator"),
            expr("stddev(mcclellan_oscillator)").alias("oscillator_volatility"),
            expr("avg(mcclellan_ratio)").alias("avg_ratio"),
            expr("avg(oscillator_momentum)").alias("avg_momentum"),
            spark_sum(when(col("overbought_signal"), 1).otherwise(0)).alias("overbought_days"),
            spark_sum(when(col("oversold_signal"), 1).otherwise(0)).alias("oversold_days"),
            spark_sum(when(col("bullish_divergence"), 1).otherwise(0)).alias("bullish_divergences"),
            spark_sum(when(col("bearish_divergence"), 1).otherwise(0)).alias("bearish_divergences"),
        ).collect()[0]

        return {
            "avg_oscillator": summary.avg_oscillator,
            "min_oscillator": summary.min_oscillator,
            "max_oscillator": summary.max_oscillator,
            "oscillator_volatility": summary.oscillator_volatility,
            "avg_ratio": summary.avg_ratio,
            "avg_momentum": summary.avg_momentum,
            "overbought_days": summary.overbought_days,
            "oversold_days": summary.oversold_days,
            "bullish_divergences": summary.bullish_divergences,
            "bearish_divergences": summary.bearish_divergences,
        }

    def detect_mcclellan_signals(self, mcclellan_result: DataFrame) -> DataFrame:
        """
        Detect comprehensive McClellan-based trading signals.

        Args:
            mcclellan_result: DataFrame with McClellan Oscillator values

        Returns:
            DataFrame with trading signals
        """
        # Define comprehensive signal conditions
        signals = (
            mcclellan_result.withColumn(
                "mcclellan_bullish_signal",
                (col("mcclellan_oscillator") > 0) & (col("oscillator_momentum") > 0) & (col("mcclellan_ratio") > 1.2),
            )
            .withColumn(
                "mcclellan_bearish_signal",
                (col("mcclellan_oscillator") < 0) & (col("oscillator_momentum") < 0) & (col("mcclellan_ratio") < 0.8),
            )
            .withColumn("mcclellan_buy_signal", (col("oversold_signal")) & (col("oscillator_momentum") > 0))
            .withColumn("mcclellan_sell_signal", (col("overbought_signal")) & (col("oscillator_momentum") < 0))
            .withColumn(
                "mcclellan_thrust_signal",
                (col("mcclellan_oscillator") > 50) & (col("oscillator_momentum") > 10) & (col("mcclellan_ratio") > 2.0),
            )
        )

        return signals

    def calculate_mcclellan_rolling_metrics(self, mcclellan_result: DataFrame, window_days: int = 20) -> DataFrame:
        """
        Calculate rolling McClellan metrics.

        Args:
            mcclellan_result: DataFrame with McClellan Oscillator values
            window_days: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        window_spec = Window.orderBy("date").rowsBetween(-window_days, 0)

        rolling_metrics = (
            mcclellan_result.withColumn(
                "rolling_oscillator",
                expr(f"avg(mcclellan_oscillator) over (order by date rows between {window_days} preceding and current row)"),
            )
            .withColumn(
                "rolling_momentum",
                expr(f"avg(oscillator_momentum) over (order by date rows between {window_days} preceding and current row)"),
            )
            .withColumn(
                "rolling_ratio",
                expr(f"avg(mcclellan_ratio) over (order by date rows between {window_days} preceding and current row)"),
            )
            .withColumn(
                "oscillator_trend",
                expr(f"avg(mcclellan_oscillator) over (order by date rows between {window_days} preceding and current row)")
                - expr(
                    f"avg(mcclellan_oscillator) over (order by date rows between {window_days*2} preceding and {window_days} preceding)"
                ),
            )
        )

        return rolling_metrics


def create_mcclellan_oscillator(spark: SparkSession) -> McClellanOscillator:
    """
    Factory function to create McClellanOscillator instance.

    Args:
        spark: PySpark session

    Returns:
        Configured McClellanOscillator instance
    """
    return McClellanOscillator(spark)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # Create Spark session
    spark = (
        SparkSession.builder.appName("McClellanOscillatorTest")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    # Create McClellan Oscillator calculator
    mcclellan = create_mcclellan_oscillator(spark)

    try:
        # Calculate McClellan Oscillator from Delta table
        result = mcclellan.calculate_mcclellan_from_delta(
            table_path="data/ohlcv",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path="data/mcclellan",
        )

        # Get summary
        summary = mcclellan.get_mcclellan_summary(result)
        print(f"McClellan Summary: {summary}")

        # Detect signals
        signals = mcclellan.detect_mcclellan_signals(result)
        print(f"Detected {signals.filter(col('mcclellan_bullish_signal')).count()} bullish signals")
        print(f"Detected {signals.filter(col('mcclellan_bearish_signal')).count()} bearish signals")
        print(f"Detected {signals.filter(col('mcclellan_buy_signal')).count()} buy signals")
        print(f"Detected {signals.filter(col('mcclellan_sell_signal')).count()} sell signals")
        print(f"Detected {signals.filter(col('mcclellan_thrust_signal')).count()} thrust signals")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        spark.stop()
