"""
Signal Scoring for Breadth/Thrust Signals POC

Implements composite scoring system that combines all breadth indicators:
- Z-score normalization for feature standardization
- Winsorization for outlier handling
- Weighted combination of indicators
- 0-100 scaling for final signals
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, sum as spark_sum, avg, stddev, min as spark_min, max as spark_max,
    window, expr, row_number, rank, dense_rank, lit, udf
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


class SignalScoring:
    """
    Signal Scoring Engine.
    
    Combines multiple breadth indicators into unified trading signals:
    - Z-score normalization for feature standardization
    - Winsorization for robust outlier handling
    - Configurable indicator weights
    - 0-100 scaling for interpretable signals
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize SignalScoring engine.
        
        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()
        
        # Scoring parameters
        self.winsorization_percentile = float(self.config.get("SCORING_WINSORIZATION", 0.05))  # 5%
        self.z_score_threshold = float(self.config.get("SCORING_Z_THRESHOLD", 3.0))
        self.min_data_points = int(self.config.get("SCORING_MIN_DATA", 50))
        
        # Default indicator weights (can be overridden)
        self.indicator_weights = {
            "ad_issues_ratio": float(self.config.get("WEIGHT_AD_ISSUES", 0.25)),
            "ad_volume_ratio": float(self.config.get("WEIGHT_AD_VOLUME", 0.20)),
            "ad_momentum": float(self.config.get("WEIGHT_AD_MOMENTUM", 0.15)),
            "ma_momentum": float(self.config.get("WEIGHT_MA_MOMENTUM", 0.15)),
            "ma_trend_strength": float(self.config.get("WEIGHT_MA_TREND", 0.10)),
            "mcclellan_oscillator": float(self.config.get("WEIGHT_MCCLELLAN", 0.10)),
            "zbt_confidence": float(self.config.get("WEIGHT_ZBT", 0.05))
        }
        
        # Feature schema
        self.scoring_schema = StructType([
            StructField("date", TimestampType(), False),
            StructField("ad_issues_ratio_raw", DoubleType(), True),
            StructField("ad_volume_ratio_raw", DoubleType(), True),
            StructField("ad_momentum_raw", DoubleType(), True),
            StructField("ma_momentum_raw", DoubleType(), True),
            StructField("ma_trend_strength_raw", DoubleType(), True),
            StructField("mcclellan_oscillator_raw", DoubleType(), True),
            StructField("zbt_confidence_raw", DoubleType(), True),
            StructField("ad_issues_ratio_normalized", DoubleType(), True),
            StructField("ad_volume_ratio_normalized", DoubleType(), True),
            StructField("ad_momentum_normalized", DoubleType(), True),
            StructField("ma_momentum_normalized", DoubleType(), True),
            StructField("ma_trend_strength_normalized", DoubleType(), True),
            StructField("mcclellan_oscillator_normalized", DoubleType(), True),
            StructField("zbt_confidence_normalized", DoubleType(), True),
            StructField("composite_score_raw", DoubleType(), True),
            StructField("composite_score_0_100", DoubleType(), True),
            StructField("signal_strength", StringType(), True),
            StructField("signal_direction", StringType(), True),
            StructField("confidence_score", DoubleType(), True),
            StructField("computed_at", TimestampType(), False)
        ])
        
        logger.info("SignalScoring initialized")
    
    def calculate_composite_score(
        self,
        ad_features: Optional[DataFrame] = None,
        ma_features: Optional[DataFrame] = None,
        mcclellan_features: Optional[DataFrame] = None,
        zbt_features: Optional[DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> DataFrame:
        """
        Calculate composite score from all feature sets.
        
        Args:
            ad_features: A/D features DataFrame
            ma_features: Moving average features DataFrame
            mcclellan_features: McClellan oscillator DataFrame
            zbt_features: ZBT features DataFrame
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with composite scores
        """
        logger.info("Calculating composite signal scores")
        
        # Load features from Delta if not provided
        if ad_features is None:
            ad_features = self._load_ad_features(start_date, end_date)
        
        if ma_features is None:
            ma_features = self._load_ma_features(start_date, end_date)
        
        if mcclellan_features is None:
            mcclellan_features = self._load_mcclellan_features(start_date, end_date)
        
        if zbt_features is None:
            zbt_features = self._load_zbt_features(start_date, end_date)
        
        # Join all features by date
        combined_features = self._join_features(
            ad_features, ma_features, mcclellan_features, zbt_features
        )
        
        # Normalize features
        normalized_features = self._normalize_features(combined_features)
        
        # Calculate composite score
        composite_score = self._calculate_weighted_score(normalized_features)
        
        # Generate signal interpretations
        final_signals = self._generate_signal_interpretations(composite_score)
        
        # Add computed timestamp
        result = final_signals.withColumn("computed_at", expr("current_timestamp()"))
        
        logger.info(f"Composite scores calculated for {result.count()} days")
        return result
    
    def _load_ad_features(self, start_date: Optional[str], end_date: Optional[str]) -> DataFrame:
        """Load A/D features from Delta Lake."""
        try:
            df = read_delta(self.spark, "data/ad_features")
            if start_date:
                df = df.filter(col("date") >= start_date)
            if end_date:
                df = df.filter(col("date") <= end_date)
            return df
        except Exception as e:
            logger.warning(f"Could not load A/D features: {e}")
            return self.spark.createDataFrame([], schema=self._get_ad_schema())
    
    def _load_ma_features(self, start_date: Optional[str], end_date: Optional[str]) -> DataFrame:
        """Load MA features from Delta Lake."""
        try:
            df = read_delta(self.spark, "data/ma_features")
            if start_date:
                df = df.filter(col("date") >= start_date)
            if end_date:
                df = df.filter(col("date") <= end_date)
            return df
        except Exception as e:
            logger.warning(f"Could not load MA features: {e}")
            return self.spark.createDataFrame([], schema=self._get_ma_schema())
    
    def _load_mcclellan_features(self, start_date: Optional[str], end_date: Optional[str]) -> DataFrame:
        """Load McClellan features from Delta Lake."""
        try:
            df = read_delta(self.spark, "data/mcclellan")
            if start_date:
                df = df.filter(col("date") >= start_date)
            if end_date:
                df = df.filter(col("date") <= end_date)
            return df
        except Exception as e:
            logger.warning(f"Could not load McClellan features: {e}")
            return self.spark.createDataFrame([], schema=self._get_mcclellan_schema())
    
    def _load_zbt_features(self, start_date: Optional[str], end_date: Optional[str]) -> DataFrame:
        """Load ZBT features from Delta Lake."""
        try:
            df = read_delta(self.spark, "data/zbt")
            if start_date:
                df = df.filter(col("date") >= start_date)
            if end_date:
                df = df.filter(col("date") <= end_date)
            return df
        except Exception as e:
            logger.warning(f"Could not load ZBT features: {e}")
            return self.spark.createDataFrame([], schema=self._get_zbt_schema())
    
    def _get_ad_schema(self) -> StructType:
        """Get A/D features schema."""
        return StructType([
            StructField("date", TimestampType(), False),
            StructField("ad_issues_ratio", DoubleType(), True),
            StructField("ad_volume_ratio", DoubleType(), True),
            StructField("ad_momentum", DoubleType(), True)
        ])
    
    def _get_ma_schema(self) -> StructType:
        """Get MA features schema."""
        return StructType([
            StructField("date", TimestampType(), False),
            StructField("ma_momentum", DoubleType(), True),
            StructField("ma_trend_strength", DoubleType(), True)
        ])
    
    def _get_mcclellan_schema(self) -> StructType:
        """Get McClellan features schema."""
        return StructType([
            StructField("date", TimestampType(), False),
            StructField("mcclellan_oscillator", DoubleType(), True)
        ])
    
    def _get_zbt_schema(self) -> StructType:
        """Get ZBT features schema."""
        return StructType([
            StructField("date", TimestampType(), False),
            StructField("zbt_confidence", DoubleType(), True)
        ])
    
    def _join_features(
        self,
        ad_features: DataFrame,
        ma_features: DataFrame,
        mcclellan_features: DataFrame,
        zbt_features: DataFrame
    ) -> DataFrame:
        """
        Join all feature sets by date.
        
        Args:
            ad_features: A/D features
            ma_features: MA features
            mcclellan_features: McClellan features
            zbt_features: ZBT features
            
        Returns:
            Joined DataFrame
        """
        # Start with A/D features
        combined = ad_features.select(
            "date",
            col("ad_issues_ratio").alias("ad_issues_ratio_raw"),
            col("ad_volume_ratio").alias("ad_volume_ratio_raw"),
            col("ad_momentum").alias("ad_momentum_raw")
        )
        
        # Join with MA features
        if ma_features.count() > 0:
            combined = combined.join(
                ma_features.select(
                    "date",
                    col("ma_momentum").alias("ma_momentum_raw"),
                    col("ma_trend_strength").alias("ma_trend_strength_raw")
                ),
                "date",
                "left"
            )
        else:
            combined = combined.withColumn("ma_momentum_raw", lit(None)) \
                             .withColumn("ma_trend_strength_raw", lit(None))
        
        # Join with McClellan features
        if mcclellan_features.count() > 0:
            combined = combined.join(
                mcclellan_features.select(
                    "date",
                    col("mcclellan_oscillator").alias("mcclellan_oscillator_raw")
                ),
                "date",
                "left"
            )
        else:
            combined = combined.withColumn("mcclellan_oscillator_raw", lit(None))
        
        # Join with ZBT features
        if zbt_features.count() > 0:
            combined = combined.join(
                zbt_features.select(
                    "date",
                    col("zbt_confidence").alias("zbt_confidence_raw")
                ),
                "date",
                "left"
            )
        else:
            combined = combined.withColumn("zbt_confidence_raw", lit(None))
        
        return combined.orderBy("date")
    
    def _normalize_features(self, features: DataFrame) -> DataFrame:
        """
        Normalize features using Z-score and winsorization.
        
        Args:
            features: Raw features DataFrame
            
        Returns:
            DataFrame with normalized features
        """
        # Calculate statistics for normalization
        stats = features.agg(
            avg("ad_issues_ratio_raw").alias("ad_issues_mean"),
            stddev("ad_issues_ratio_raw").alias("ad_issues_std"),
            avg("ad_volume_ratio_raw").alias("ad_volume_mean"),
            stddev("ad_volume_ratio_raw").alias("ad_volume_std"),
            avg("ad_momentum_raw").alias("ad_momentum_mean"),
            stddev("ad_momentum_raw").alias("ad_momentum_std"),
            avg("ma_momentum_raw").alias("ma_momentum_mean"),
            stddev("ma_momentum_raw").alias("ma_momentum_std"),
            avg("ma_trend_strength_raw").alias("ma_trend_mean"),
            stddev("ma_trend_strength_raw").alias("ma_trend_std"),
            avg("mcclellan_oscillator_raw").alias("mcclellan_mean"),
            stddev("mcclellan_oscillator_raw").alias("mcclellan_std"),
            avg("zbt_confidence_raw").alias("zbt_confidence_mean"),
            stddev("zbt_confidence_raw").alias("zbt_confidence_std")
        ).collect()[0]
        
        # Apply Z-score normalization with winsorization
        normalized = features.withColumn(
            "ad_issues_ratio_normalized",
            self._winsorize_and_normalize(
                col("ad_issues_ratio_raw"), 
                stats.ad_issues_mean, 
                stats.ad_issues_std
            )
        ).withColumn(
            "ad_volume_ratio_normalized",
            self._winsorize_and_normalize(
                col("ad_volume_ratio_raw"), 
                stats.ad_volume_mean, 
                stats.ad_volume_std
            )
        ).withColumn(
            "ad_momentum_normalized",
            self._winsorize_and_normalize(
                col("ad_momentum_raw"), 
                stats.ad_momentum_mean, 
                stats.ad_momentum_std
            )
        ).withColumn(
            "ma_momentum_normalized",
            self._winsorize_and_normalize(
                col("ma_momentum_raw"), 
                stats.ma_momentum_mean, 
                stats.ma_momentum_std
            )
        ).withColumn(
            "ma_trend_strength_normalized",
            self._winsorize_and_normalize(
                col("ma_trend_strength_raw"), 
                stats.ma_trend_mean, 
                stats.ma_trend_std
            )
        ).withColumn(
            "mcclellan_oscillator_normalized",
            self._winsorize_and_normalize(
                col("mcclellan_oscillator_raw"), 
                stats.mcclellan_mean, 
                stats.mcclellan_std
            )
        ).withColumn(
            "zbt_confidence_normalized",
            self._winsorize_and_normalize(
                col("zbt_confidence_raw"), 
                stats.zbt_confidence_mean, 
                stats.zbt_confidence_std
            )
        )
        
        return normalized
    
    def _winsorize_and_normalize(self, column, mean_val, std_val):
        """
        Apply winsorization and Z-score normalization to a column.
        
        Args:
            column: Column to normalize
            mean_val: Mean value for normalization
            std_val: Standard deviation for normalization
            
        Returns:
            Normalized column expression
        """
        if std_val is None or std_val == 0:
            return lit(0.0)
        
        # Winsorize at specified percentile
        winsorized = when(
            column > mean_val + self.z_score_threshold * std_val,
            mean_val + self.z_score_threshold * std_val
        ).when(
            column < mean_val - self.z_score_threshold * std_val,
            mean_val - self.z_score_threshold * std_val
        ).otherwise(column)
        
        # Z-score normalization
        return (winsorized - mean_val) / std_val
    
    def _calculate_weighted_score(self, normalized_features: DataFrame) -> DataFrame:
        """
        Calculate weighted composite score.
        
        Args:
            normalized_features: DataFrame with normalized features
            
        Returns:
            DataFrame with composite scores
        """
        # Calculate weighted sum
        weighted_score = normalized_features.withColumn(
            "composite_score_raw",
            (col("ad_issues_ratio_normalized") * self.indicator_weights["ad_issues_ratio"]) +
            (col("ad_volume_ratio_normalized") * self.indicator_weights["ad_volume_ratio"]) +
            (col("ad_momentum_normalized") * self.indicator_weights["ad_momentum"]) +
            (col("ma_momentum_normalized") * self.indicator_weights["ma_momentum"]) +
            (col("ma_trend_strength_normalized") * self.indicator_weights["ma_trend_strength"]) +
            (col("mcclellan_oscillator_normalized") * self.indicator_weights["mcclellan_oscillator"]) +
            (col("zbt_confidence_normalized") * self.indicator_weights["zbt_confidence"])
        )
        
        # Scale to 0-100 range
        # First, get min/max for scaling
        stats = weighted_score.agg(
            min("composite_score_raw").alias("min_score"),
            max("composite_score_raw").alias("max_score")
        ).collect()[0]
        
        score_range = stats.max_score - stats.min_score
        
        if score_range == 0:
            # If all scores are the same, set to 50
            final_score = weighted_score.withColumn(
                "composite_score_0_100",
                lit(50.0)
            )
        else:
            # Scale to 0-100
            final_score = weighted_score.withColumn(
                "composite_score_0_100",
                ((col("composite_score_raw") - stats.min_score) / score_range) * 100
            )
        
        return final_score
    
    def _generate_signal_interpretations(self, composite_score: DataFrame) -> DataFrame:
        """
        Generate signal interpretations and confidence scores.
        
        Args:
            composite_score: DataFrame with composite scores
            
        Returns:
            DataFrame with signal interpretations
        """
        # Generate signal strength categories
        signal_strength = composite_score.withColumn(
            "signal_strength",
            when(col("composite_score_0_100") >= 80, "very_strong")
            .when(col("composite_score_0_100") >= 60, "strong")
            .when(col("composite_score_0_100") >= 40, "moderate")
            .when(col("composite_score_0_100") >= 20, "weak")
            .otherwise("very_weak")
        )
        
        # Generate signal direction
        signal_direction = signal_strength.withColumn(
            "signal_direction",
            when(col("composite_score_0_100") >= 60, "bullish")
            .when(col("composite_score_0_100") <= 40, "bearish")
            .otherwise("neutral")
        )
        
        # Calculate confidence score based on feature availability and score consistency
        confidence_score = signal_direction.withColumn(
            "confidence_score",
            self._calculate_confidence_score()
        )
        
        return confidence_score
    
    def _calculate_confidence_score(self):
        """
        Calculate confidence score based on feature availability and consistency.
        
        Returns:
            Confidence score expression
        """
        # Count available features
        available_features = (
            when(col("ad_issues_ratio_raw").isNotNull(), 1).otherwise(0) +
            when(col("ad_volume_ratio_raw").isNotNull(), 1).otherwise(0) +
            when(col("ad_momentum_raw").isNotNull(), 1).otherwise(0) +
            when(col("ma_momentum_raw").isNotNull(), 1).otherwise(0) +
            when(col("ma_trend_strength_raw").isNotNull(), 1).otherwise(0) +
            when(col("mcclellan_oscillator_raw").isNotNull(), 1).otherwise(0) +
            when(col("zbt_confidence_raw").isNotNull(), 1).otherwise(0)
        )
        
        # Base confidence on feature availability (0-70 points)
        feature_confidence = (available_features / 7.0) * 70
        
        # Additional confidence based on score extremity (0-30 points)
        score_confidence = when(
            col("composite_score_0_100") >= 80, 30
        ).when(
            col("composite_score_0_100") >= 60, 20
        ).when(
            col("composite_score_0_100") <= 20, 30
        ).when(
            col("composite_score_0_100") <= 40, 20
        ).otherwise(10)
        
        return feature_confidence + score_confidence
    
    def calculate_scores_from_delta(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> DataFrame:
        """
        Calculate composite scores from Delta Lake tables.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            output_path: Path to save results (optional)
            
        Returns:
            DataFrame with composite scores
        """
        logger.info("Calculating composite scores from Delta Lake")
        
        # Calculate composite scores
        composite_scores = self.calculate_composite_score(
            start_date=start_date,
            end_date=end_date
        )
        
        # Save to Delta Lake if output path provided
        if output_path:
            logger.info(f"Saving composite scores to {output_path}")
            write_delta(
                df=composite_scores,
                path=output_path,
                partition_cols=["year", "month"],
                mode="append"
            )
        
        return composite_scores
    
    def get_scoring_summary(self, composite_scores: DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for composite scores.
        
        Args:
            composite_scores: DataFrame with composite scores
            
        Returns:
            Dictionary with summary statistics
        """
        summary = composite_scores.agg(
            expr("avg(composite_score_0_100)").alias("avg_score"),
            expr("min(composite_score_0_100)").alias("min_score"),
            expr("max(composite_score_0_100)").alias("max_score"),
            expr("stddev(composite_score_0_100)").alias("score_volatility"),
            expr("avg(confidence_score)").alias("avg_confidence"),
            spark_sum(when(col("signal_direction") == "bullish", 1).otherwise(0)).alias("bullish_days"),
            spark_sum(when(col("signal_direction") == "bearish", 1).otherwise(0)).alias("bearish_days"),
            spark_sum(when(col("signal_direction") == "neutral", 1).otherwise(0)).alias("neutral_days"),
            spark_sum(when(col("signal_strength") == "very_strong", 1).otherwise(0)).alias("very_strong_signals"),
            spark_sum(when(col("signal_strength") == "strong", 1).otherwise(0)).alias("strong_signals")
        ).collect()[0]
        
        return {
            "avg_score": summary.avg_score,
            "min_score": summary.min_score,
            "max_score": summary.max_score,
            "score_volatility": summary.score_volatility,
            "avg_confidence": summary.avg_confidence,
            "bullish_days": summary.bullish_days,
            "bearish_days": summary.bearish_days,
            "neutral_days": summary.neutral_days,
            "very_strong_signals": summary.very_strong_signals,
            "strong_signals": summary.strong_signals
        }
    
    def detect_trading_signals(self, composite_scores: DataFrame) -> DataFrame:
        """
        Detect trading signals based on composite scores.
        
        Args:
            composite_scores: DataFrame with composite scores
            
        Returns:
            DataFrame with trading signals
        """
        # Define trading signal conditions
        signals = composite_scores.withColumn(
            "buy_signal",
            (col("signal_direction") == "bullish") & 
            (col("signal_strength").isin(["strong", "very_strong"])) & 
            (col("confidence_score") >= 70)
        ).withColumn(
            "sell_signal",
            (col("signal_direction") == "bearish") & 
            (col("signal_strength").isin(["strong", "very_strong"])) & 
            (col("confidence_score") >= 70)
        ).withColumn(
            "hold_signal",
            (col("signal_direction") == "neutral") | 
            (col("confidence_score") < 50)
        ).withColumn(
            "strong_buy_signal",
            (col("signal_direction") == "bullish") & 
            (col("signal_strength") == "very_strong") & 
            (col("confidence_score") >= 80)
        ).withColumn(
            "strong_sell_signal",
            (col("signal_direction") == "bearish") & 
            (col("signal_strength") == "very_strong") & 
            (col("confidence_score") >= 80)
        )
        
        return signals


def create_signal_scoring(spark: SparkSession) -> SignalScoring:
    """
    Factory function to create SignalScoring instance.
    
    Args:
        spark: PySpark session
        
    Returns:
        Configured SignalScoring instance
    """
    return SignalScoring(spark)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("SignalScoringTest") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Create signal scoring engine
    scoring = create_signal_scoring(spark)
    
    try:
        # Calculate composite scores from Delta tables
        result = scoring.calculate_scores_from_delta(
            start_date="2024-01-01",
            end_date="2024-12-31",
            output_path="data/composite_scores"
        )
        
        # Get summary
        summary = scoring.get_scoring_summary(result)
        print(f"Scoring Summary: {summary}")
        
        # Detect trading signals
        signals = scoring.detect_trading_signals(result)
        print(f"Detected {signals.filter(col('buy_signal')).count()} buy signals")
        print(f"Detected {signals.filter(col('sell_signal')).count()} sell signals")
        print(f"Detected {signals.filter(col('strong_buy_signal')).count()} strong buy signals")
        print(f"Detected {signals.filter(col('strong_sell_signal')).count()} strong sell signals")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        spark.stop()
