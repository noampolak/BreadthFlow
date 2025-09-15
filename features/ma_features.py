"""
Moving Average (MA) Features for Breadth/Thrust Signals POC

Implements MA20/MA50 calculations and crossover signals for market trend analysis.
These indicators help identify trend direction and potential reversal points.
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


class MAFeatures:
    """
    Moving Average (MA) Features Calculator.

    Calculates moving average indicators and crossover signals:
    - MA20: 20-day simple moving average
    - MA50: 50-day simple moving average
    - Golden Cross: MA20 crosses above MA50
    - Death Cross: MA20 crosses below MA50
    - Price vs MA relationships
    - MA momentum and slope
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize MAFeatures calculator.

        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()

        # MA calculation parameters
        self.ma_short_period = int(self.config.get("MA_SHORT_PERIOD", 20))
        self.ma_long_period = int(self.config.get("MA_LONG_PERIOD", 50))
        self.min_data_points = int(self.config.get("MA_MIN_DATA_POINTS", 50))

        # Feature schema
        self.ma_schema = StructType(
            [
                StructField("symbol", StringType(), False),
                StructField("date", TimestampType(), False),
                StructField("close", DoubleType(), True),
                StructField("ma_short", DoubleType(), True),
                StructField("ma_long", DoubleType(), True),
                StructField("ma_short_slope", DoubleType(), True),
                StructField("ma_long_slope", DoubleType(), True),
                StructField("price_vs_ma_short", DoubleType(), True),
                StructField("price_vs_ma_long", DoubleType(), True),
                StructField("ma_cross_signal", StringType(), True),
                StructField("golden_cross", BooleanType(), True),
                StructField("death_cross", BooleanType(), True),
                StructField("ma_momentum", DoubleType(), True),
                StructField("ma_trend_strength", DoubleType(), True),
                StructField("computed_at", TimestampType(), False),
            ]
        )

        logger.info(f"MAFeatures initialized with MA{self.ma_short_period}/MA{self.ma_long_period}")

    def calculate_ma_features(
        self,
        df: DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> DataFrame:
        """
        Calculate MA features for given data.

        Args:
            df: DataFrame with OHLCV data
            symbols: List of symbols to include (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            DataFrame with MA features
        """
        logger.info("Calculating MA features")

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

        # Calculate moving averages
        ma_features = self._calculate_moving_averages(df)

        # Calculate crossover signals
        crossover_signals = self._calculate_crossover_signals(ma_features)

        # Calculate momentum and trend indicators
        momentum_features = self._calculate_momentum_features(crossover_signals)

        # Add computed timestamp
        result = momentum_features.withColumn("computed_at", expr("current_timestamp()"))

        logger.info(f"MA features calculated for {result.count()} records")
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

    def _calculate_moving_averages(self, df: DataFrame) -> DataFrame:
        """
        Calculate moving averages for each symbol.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with moving averages
        """
        # Window specification for each symbol
        window_spec_short = Window.partitionBy("symbol").orderBy("date").rowsBetween(-self.ma_short_period + 1, 0)
        window_spec_long = Window.partitionBy("symbol").orderBy("date").rowsBetween(-self.ma_long_period + 1, 0)

        # Calculate moving averages
        ma_features = df.withColumn("ma_short", avg("close").over(window_spec_short)).withColumn(
            "ma_long", avg("close").over(window_spec_long)
        )

        # Calculate price vs MA relationships
        ma_features = ma_features.withColumn(
            "price_vs_ma_short", (col("close") - col("ma_short")) / col("ma_short")
        ).withColumn("price_vs_ma_long", (col("close") - col("ma_long")) / col("ma_long"))

        return ma_features

    def _calculate_crossover_signals(self, ma_features: DataFrame) -> DataFrame:
        """
        Calculate MA crossover signals.

        Args:
            ma_features: DataFrame with moving averages

        Returns:
            DataFrame with crossover signals
        """
        # Window specification for lag calculations
        window_spec = Window.partitionBy("symbol").orderBy("date")

        # Calculate lagged values for crossover detection
        crossover_signals = ma_features.withColumn("ma_short_lag", lag("ma_short", 1).over(window_spec)).withColumn(
            "ma_long_lag", lag("ma_long", 1).over(window_spec)
        )

        # Detect crossovers
        crossover_signals = (
            crossover_signals.withColumn(
                "golden_cross", (col("ma_short") > col("ma_long")) & (col("ma_short_lag") <= col("ma_long_lag"))
            )
            .withColumn("death_cross", (col("ma_short") < col("ma_long")) & (col("ma_short_lag") >= col("ma_long_lag")))
            .withColumn(
                "ma_cross_signal",
                when(col("golden_cross"), "golden_cross").when(col("death_cross"), "death_cross").otherwise("none"),
            )
        )

        return crossover_signals

    def _calculate_momentum_features(self, crossover_signals: DataFrame) -> DataFrame:
        """
        Calculate MA momentum and trend strength indicators.

        Args:
            crossover_signals: DataFrame with crossover signals

        Returns:
            DataFrame with momentum features
        """
        # Window specification for slope calculations
        window_spec = Window.partitionBy("symbol").orderBy("date")

        # Calculate MA slopes (momentum)
        momentum_features = crossover_signals.withColumn(
            "ma_short_slope", (col("ma_short") - lag("ma_short", 5).over(window_spec)) / 5
        ).withColumn("ma_long_slope", (col("ma_long") - lag("ma_long", 10).over(window_spec)) / 10)

        # Calculate overall MA momentum
        momentum_features = momentum_features.withColumn("ma_momentum", (col("ma_short_slope") + col("ma_long_slope")) / 2)

        # Calculate trend strength
        momentum_features = momentum_features.withColumn(
            "ma_trend_strength", abs(col("ma_short") - col("ma_long")) / col("ma_long")
        )

        return momentum_features

    def calculate_ma_features_from_delta(
        self,
        table_path: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> DataFrame:
        """
        Calculate MA features from Delta Lake table.

        Args:
            table_path: Path to Delta table with OHLCV data
            symbols: List of symbols to include
            start_date: Start date filter
            end_date: End date filter
            output_path: Path to save results (optional)

        Returns:
            DataFrame with MA features
        """
        logger.info(f"Loading data from {table_path}")

        # Read from Delta Lake
        df = read_delta(self.spark, table_path)

        # Calculate features
        ma_features = self.calculate_ma_features(df, symbols, start_date, end_date)

        # Save to Delta Lake if output path provided
        if output_path:
            logger.info(f"Saving MA features to {output_path}")
            write_delta(df=ma_features, path=output_path, partition_cols=["year", "month"], mode="append")

        return ma_features

    def get_ma_summary(self, ma_features: DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for MA features.

        Args:
            ma_features: DataFrame with MA features

        Returns:
            Dictionary with summary statistics
        """
        summary = ma_features.agg(
            count("symbol").alias("total_records"),
            count("symbol").alias("unique_symbols"),
            expr("avg(price_vs_ma_short)").alias("avg_price_vs_ma_short"),
            expr("avg(price_vs_ma_long)").alias("avg_price_vs_ma_long"),
            expr("avg(ma_momentum)").alias("avg_ma_momentum"),
            expr("avg(ma_trend_strength)").alias("avg_ma_trend_strength"),
            spark_sum(when(col("golden_cross"), 1).otherwise(0)).alias("total_golden_crosses"),
            spark_sum(when(col("death_cross"), 1).otherwise(0)).alias("total_death_crosses"),
            expr("stddev(ma_momentum)").alias("ma_momentum_volatility"),
        ).collect()[0]

        return {
            "total_records": summary.total_records,
            "unique_symbols": summary.unique_symbols,
            "avg_price_vs_ma_short": summary.avg_price_vs_ma_short,
            "avg_price_vs_ma_long": summary.avg_price_vs_ma_long,
            "avg_ma_momentum": summary.avg_ma_momentum,
            "avg_ma_trend_strength": summary.avg_ma_trend_strength,
            "total_golden_crosses": summary.total_golden_crosses,
            "total_death_crosses": summary.total_death_crosses,
            "ma_momentum_volatility": summary.ma_momentum_volatility,
        }

    def detect_ma_signals(self, ma_features: DataFrame) -> DataFrame:
        """
        Detect MA-based trading signals.

        Args:
            ma_features: DataFrame with MA features

        Returns:
            DataFrame with signals
        """
        # Define signal conditions
        signals = (
            ma_features.withColumn(
                "ma_bullish_signal", (col("golden_cross")) & (col("ma_momentum") > 0) & (col("price_vs_ma_short") > 0.02)
            )
            .withColumn(
                "ma_bearish_signal", (col("death_cross")) & (col("ma_momentum") < 0) & (col("price_vs_ma_short") < -0.02)
            )
            .withColumn(
                "ma_trend_following_signal",
                (col("ma_short") > col("ma_long")) & (col("ma_momentum") > 0.01) & (col("ma_trend_strength") > 0.05),
            )
            .withColumn("ma_reversal_signal", (col("ma_cross_signal") != "none") & (abs(col("ma_momentum")) > 0.02))
        )

        return signals

    def calculate_ma_rolling_metrics(self, ma_features: DataFrame, window_days: int = 20) -> DataFrame:
        """
        Calculate rolling MA metrics.

        Args:
            ma_features: DataFrame with MA features
            window_days: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        window_spec = Window.partitionBy("symbol").orderBy("date").rowsBetween(-window_days, 0)

        rolling_metrics = (
            ma_features.withColumn(
                "rolling_ma_momentum",
                expr(
                    f"avg(ma_momentum) over (partition by symbol order by date rows between {window_days} preceding and current row)"
                ),
            )
            .withColumn(
                "rolling_ma_trend_strength",
                expr(
                    f"avg(ma_trend_strength) over (partition by symbol order by date rows between {window_days} preceding and current row)"
                ),
            )
            .withColumn(
                "rolling_price_vs_ma_short",
                expr(
                    f"avg(price_vs_ma_short) over (partition by symbol order by date rows between {window_days} preceding and current row)"
                ),
            )
            .withColumn(
                "rolling_price_vs_ma_long",
                expr(
                    f"avg(price_vs_ma_long) over (partition by symbol order by date rows between {window_days} preceding and current row)"
                ),
            )
        )

        return rolling_metrics

    def get_symbol_ma_summary(self, ma_features: DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Get MA summary for a specific symbol.

        Args:
            ma_features: DataFrame with MA features
            symbol: Symbol to analyze

        Returns:
            Dictionary with symbol-specific MA summary
        """
        symbol_data = ma_features.filter(col("symbol") == symbol)

        if symbol_data.count() == 0:
            return {"error": f"No data found for symbol {symbol}"}

        summary = symbol_data.agg(
            expr("avg(ma_short)").alias("avg_ma_short"),
            expr("avg(ma_long)").alias("avg_ma_long"),
            expr("avg(ma_momentum)").alias("avg_ma_momentum"),
            expr("avg(ma_trend_strength)").alias("avg_ma_trend_strength"),
            spark_sum(when(col("golden_cross"), 1).otherwise(0)).alias("golden_crosses"),
            spark_sum(when(col("death_cross"), 1).otherwise(0)).alias("death_crosses"),
            expr("max(close)").alias("max_price"),
            expr("min(close)").alias("min_price"),
            expr("stddev(ma_momentum)").alias("momentum_volatility"),
        ).collect()[0]

        return {
            "symbol": symbol,
            "avg_ma_short": summary.avg_ma_short,
            "avg_ma_long": summary.avg_ma_long,
            "avg_ma_momentum": summary.avg_ma_momentum,
            "avg_ma_trend_strength": summary.avg_ma_trend_strength,
            "golden_crosses": summary.golden_crosses,
            "death_crosses": summary.death_crosses,
            "max_price": summary.max_price,
            "min_price": summary.min_price,
            "momentum_volatility": summary.momentum_volatility,
        }


def create_ma_features(spark: SparkSession) -> MAFeatures:
    """
    Factory function to create MAFeatures instance.

    Args:
        spark: PySpark session

    Returns:
        Configured MAFeatures instance
    """
    return MAFeatures(spark)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    # Create Spark session
    spark = (
        SparkSession.builder.appName("MAFeaturesTest")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    # Create MA features calculator
    ma_features = create_ma_features(spark)

    try:
        # Calculate MA features from Delta table
        result = ma_features.calculate_ma_features_from_delta(
            table_path="data/ohlcv",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path="data/ma_features",
        )

        # Get summary
        summary = ma_features.get_ma_summary(result)
        print(f"MA Summary: {summary}")

        # Get symbol-specific summary
        for symbol in ["AAPL", "MSFT"]:
            symbol_summary = ma_features.get_symbol_ma_summary(result, symbol)
            print(f"{symbol} MA Summary: {symbol_summary}")

        # Detect signals
        signals = ma_features.detect_ma_signals(result)
        print(f"Detected {signals.filter(col('ma_bullish_signal')).count()} bullish signals")
        print(f"Detected {signals.filter(col('ma_bearish_signal')).count()} bearish signals")
        print(f"Detected {signals.filter(col('ma_trend_following_signal')).count()} trend following signals")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        spark.stop()
