"""
Advance/Decline (A/D) Features for Breadth/Thrust Signals POC

Implements A/D Issues and A/D Volume calculations for market breadth analysis.
These are fundamental indicators for measuring market participation and momentum.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, sum as spark_sum, count, lag, lead, 
    window, expr, row_number, rank, dense_rank,
    date_format, year, month, dayofmonth
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    LongType, TimestampType, BooleanType
)

from features.common.config import get_config
from features.common.io import read_delta, write_delta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ADFeatures:
    """
    Advance/Decline (A/D) Features Calculator.
    
    Calculates market breadth indicators based on advancing vs declining stocks:
    - A/D Issues Ratio: Ratio of advancing to declining stocks
    - A/D Volume Ratio: Volume-weighted A/D ratio
    - A/D Line: Cumulative A/D difference
    - A/D Momentum: Rate of change in A/D indicators
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize ADFeatures calculator.
        
        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()
        
        # A/D calculation parameters
        self.min_volume_threshold = float(self.config.get("AD_MIN_VOLUME", 1000))
        self.min_price_threshold = float(self.config.get("AD_MIN_PRICE", 1.0))
        self.lookback_period = int(self.config.get("AD_LOOKBACK_PERIOD", 20))
        
        # Feature schema
        self.ad_schema = StructType([
            StructField("date", TimestampType(), False),
            StructField("ad_issues_ratio", DoubleType(), True),
            StructField("ad_volume_ratio", DoubleType(), True),
            StructField("ad_line", DoubleType(), True),
            StructField("ad_momentum", DoubleType(), True),
            StructField("advancing_stocks", LongType(), True),
            StructField("declining_stocks", LongType(), True),
            StructField("unchanged_stocks", LongType(), True),
            StructField("total_volume", LongType(), True),
            StructField("advancing_volume", LongType(), True),
            StructField("declining_volume", LongType(), True),
            StructField("ad_strength", DoubleType(), True),
            StructField("ad_breadth", DoubleType(), True),
            StructField("computed_at", TimestampType(), False)
        ])
        
        logger.info("ADFeatures initialized")
    
    def calculate_ad_features(
        self, 
        df: DataFrame,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> DataFrame:
        """
        Calculate A/D features for given data.
        
        Args:
            df: DataFrame with OHLCV data
            symbols: List of symbols to include (optional)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            
        Returns:
            DataFrame with A/D features
        """
        logger.info("Calculating A/D features")
        
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
        
        # Calculate daily A/D metrics
        daily_ad = self._calculate_daily_ad(df)
        
        # Calculate A/D line (cumulative)
        ad_line = self._calculate_ad_line(daily_ad)
        
        # Calculate momentum indicators
        ad_momentum = self._calculate_ad_momentum(ad_line)
        
        # Add computed timestamp
        result = ad_momentum.withColumn("computed_at", expr("current_timestamp()"))
        
        logger.info(f"A/D features calculated for {result.count()} days")
        return result
    
    def _filter_quality_data(self, df: DataFrame) -> DataFrame:
        """
        Filter out low-quality data points.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        return df.filter(
            (col("volume") >= self.min_volume_threshold) &
            (col("close") >= self.min_price_threshold) &
            (col("close").isNotNull()) &
            (col("volume").isNotNull())
        )
    
    def _calculate_daily_ad(self, df: DataFrame) -> DataFrame:
        """
        Calculate daily A/D metrics.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with daily A/D metrics
        """
        # Calculate price changes
        window_spec = Window.partitionBy("symbol").orderBy("date")
        
        df_with_changes = df.withColumn(
            "price_change_pct", 
            (col("close") - lag("close", 1).over(window_spec)) / lag("close", 1).over(window_spec)
        )
        
        # Classify stocks as advancing, declining, or unchanged
        df_classified = df_with_changes.withColumn(
            "stock_status",
            when(col("price_change_pct") > 0.001, "advancing")
            .when(col("price_change_pct") < -0.001, "declining")
            .otherwise("unchanged")
        )
        
        # Aggregate by date
        daily_metrics = df_classified.groupBy("date").agg(
            # Count stocks by status
            spark_sum(when(col("stock_status") == "advancing", 1).otherwise(0)).alias("advancing_stocks"),
            spark_sum(when(col("stock_status") == "declining", 1).otherwise(0)).alias("declining_stocks"),
            spark_sum(when(col("stock_status") == "unchanged", 1).otherwise(0)).alias("unchanged_stocks"),
            
            # Volume metrics
            spark_sum("volume").alias("total_volume"),
            spark_sum(when(col("stock_status") == "advancing", col("volume")).otherwise(0)).alias("advancing_volume"),
            spark_sum(when(col("stock_status") == "declining", col("volume")).otherwise(0)).alias("declining_volume")
        )
        
        # Calculate ratios
        daily_ad = daily_metrics.withColumn(
            "ad_issues_ratio",
            when(col("declining_stocks") > 0, col("advancing_stocks") / col("declining_stocks")).otherwise(0.0)
        ).withColumn(
            "ad_volume_ratio",
            when(col("declining_volume") > 0, col("advancing_volume") / col("declining_volume")).otherwise(0.0)
        ).withColumn(
            "ad_strength",
            (col("advancing_stocks") - col("declining_stocks")) / (col("advancing_stocks") + col("declining_stocks"))
        ).withColumn(
            "ad_breadth",
            col("advancing_stocks") / (col("advancing_stocks") + col("declining_stocks") + col("unchanged_stocks"))
        )
        
        return daily_ad.orderBy("date")
    
    def _calculate_ad_line(self, daily_ad: DataFrame) -> DataFrame:
        """
        Calculate A/D line (cumulative A/D difference).
        
        Args:
            daily_ad: Daily A/D metrics DataFrame
            
        Returns:
            DataFrame with A/D line
        """
        # Calculate daily A/D difference
        ad_diff = daily_ad.withColumn(
            "ad_difference",
            col("advancing_stocks") - col("declining_stocks")
        )
        
        # Calculate cumulative A/D line
        window_spec = Window.orderBy("date")
        ad_line = ad_diff.withColumn(
            "ad_line",
            spark_sum("ad_difference").over(window_spec)
        )
        
        return ad_line
    
    def _calculate_ad_momentum(self, ad_line: DataFrame) -> DataFrame:
        """
        Calculate A/D momentum indicators.
        
        Args:
            ad_line: DataFrame with A/D line
            
        Returns:
            DataFrame with momentum indicators
        """
        # Calculate momentum (rate of change)
        window_spec = Window.orderBy("date")
        
        ad_momentum = ad_line.withColumn(
            "ad_momentum",
            (col("ad_line") - lag("ad_line", self.lookback_period).over(window_spec)) / self.lookback_period
        )
        
        return ad_momentum
    
    def calculate_ad_features_from_delta(
        self,
        table_path: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> DataFrame:
        """
        Calculate A/D features from Delta Lake table.
        
        Args:
            table_path: Path to Delta table with OHLCV data
            symbols: List of symbols to include
            start_date: Start date filter
            end_date: End date filter
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame with A/D features
        """
        logger.info(f"Loading data from {table_path}")
        
        # Read from Delta Lake
        df = read_delta(self.spark, table_path)
        
        # Calculate features
        ad_features = self.calculate_ad_features(df, symbols, start_date, end_date)
        
        # Save to Delta Lake if output path provided
        if output_path:
            logger.info(f"Saving A/D features to {output_path}")
            write_delta(
                df=ad_features,
                path=output_path,
                partition_cols=["year", "month"],
                mode="append"
            )
        
        return ad_features
    
    def get_ad_summary(self, ad_features: DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for A/D features.
        
        Args:
            ad_features: DataFrame with A/D features
            
        Returns:
            Dictionary with summary statistics
        """
        summary = ad_features.agg(
            spark_sum("advancing_stocks").alias("total_advancing"),
            spark_sum("declining_stocks").alias("total_declining"),
            spark_sum("total_volume").alias("total_volume"),
            spark_sum("advancing_volume").alias("total_advancing_volume"),
            spark_sum("declining_volume").alias("total_declining_volume"),
            expr("avg(ad_issues_ratio)").alias("avg_ad_issues_ratio"),
            expr("avg(ad_volume_ratio)").alias("avg_ad_volume_ratio"),
            expr("avg(ad_strength)").alias("avg_ad_strength"),
            expr("avg(ad_breadth)").alias("avg_ad_breadth"),
            expr("stddev(ad_momentum)").alias("ad_momentum_volatility")
        ).collect()[0]
        
        return {
            "total_advancing": summary.total_advancing,
            "total_declining": summary.total_declining,
            "total_volume": summary.total_volume,
            "total_advancing_volume": summary.total_advancing_volume,
            "total_declining_volume": summary.total_declining_volume,
            "avg_ad_issues_ratio": summary.avg_ad_issues_ratio,
            "avg_ad_volume_ratio": summary.avg_ad_volume_ratio,
            "avg_ad_strength": summary.avg_ad_strength,
            "avg_ad_breadth": summary.avg_ad_breadth,
            "ad_momentum_volatility": summary.ad_momentum_volatility
        }
    
    def detect_ad_signals(self, ad_features: DataFrame) -> DataFrame:
        """
        Detect A/D-based trading signals.
        
        Args:
            ad_features: DataFrame with A/D features
            
        Returns:
            DataFrame with signals
        """
        # Define signal conditions
        signals = ad_features.withColumn(
            "ad_bullish_signal",
            (col("ad_issues_ratio") > 2.0) & (col("ad_volume_ratio") > 1.5) & (col("ad_momentum") > 0)
        ).withColumn(
            "ad_bearish_signal",
            (col("ad_issues_ratio") < 0.5) & (col("ad_volume_ratio") < 0.67) & (col("ad_momentum") < 0)
        ).withColumn(
            "ad_thrust_signal",
            (col("ad_issues_ratio") > 3.0) & (col("ad_volume_ratio") > 2.0) & (col("ad_momentum") > 5)
        ).withColumn(
            "ad_divergence_signal",
            (col("ad_issues_ratio") > 1.5) & (col("ad_volume_ratio") < 0.8)
        )
        
        return signals
    
    def calculate_ad_rolling_metrics(
        self, 
        ad_features: DataFrame, 
        window_days: int = 20
    ) -> DataFrame:
        """
        Calculate rolling A/D metrics.
        
        Args:
            ad_features: DataFrame with A/D features
            window_days: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        window_spec = Window.orderBy("date").rowsBetween(-window_days, 0)
        
        rolling_metrics = ad_features.withColumn(
            "rolling_ad_issues_ratio",
            expr(f"avg(ad_issues_ratio) over (order by date rows between {window_days} preceding and current row)")
        ).withColumn(
            "rolling_ad_volume_ratio",
            expr(f"avg(ad_volume_ratio) over (order by date rows between {window_days} preceding and current row)")
        ).withColumn(
            "rolling_ad_strength",
            expr(f"avg(ad_strength) over (order by date rows between {window_days} preceding and current row)")
        ).withColumn(
            "rolling_ad_momentum",
            expr(f"avg(ad_momentum) over (order by date rows between {window_days} preceding and current row)")
        )
        
        return rolling_metrics


def create_ad_features(spark: SparkSession) -> ADFeatures:
    """
    Factory function to create ADFeatures instance.
    
    Args:
        spark: PySpark session
        
    Returns:
        Configured ADFeatures instance
    """
    return ADFeatures(spark)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ADFeaturesTest") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Create A/D features calculator
    ad_features = create_ad_features(spark)
    
    try:
        # Calculate A/D features from Delta table
        result = ad_features.calculate_ad_features_from_delta(
            table_path="data/ohlcv",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path="data/ad_features"
        )
        
        # Get summary
        summary = ad_features.get_ad_summary(result)
        print(f"A/D Summary: {summary}")
        
        # Detect signals
        signals = ad_features.detect_ad_signals(result)
        print(f"Detected {signals.filter(col('ad_bullish_signal')).count()} bullish signals")
        print(f"Detected {signals.filter(col('ad_bearish_signal')).count()} bearish signals")
        print(f"Detected {signals.filter(col('ad_thrust_signal')).count()} thrust signals")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        spark.stop()
