"""
Signal Generator for Breadth/Thrust Signals POC

Main orchestrator for signal generation process:
- Coordinates feature calculation from all indicators
- Manages composite scoring and signal generation
- Handles signal persistence and storage
- Provides real-time signal generation capabilities
"""

import logging
import json
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, sum as spark_sum, avg, stddev, min as spark_min, max as spark_max,
    window, expr, row_number, rank, dense_rank, lit, udf, current_timestamp
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    LongType, TimestampType, BooleanType
)

from features.common.config import get_config
from features.common.io import read_delta, write_delta, upsert_delta

# Import feature calculators
from features.ad_features import create_ad_features
from features.ma_features import create_ma_features
from features.mcclellan import create_mcclellan_oscillator
from features.zbt import create_zbt

# Import scoring engine
from model.scoring import create_signal_scoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Main Signal Generator for Breadth/Thrust Signals.
    
    Orchestrates the complete signal generation pipeline:
    - Calculates all breadth indicators
    - Combines features into composite scores
    - Generates trading signals
    - Manages signal persistence and storage
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize SignalGenerator.
        
        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()
        
        # Initialize feature calculators (skip if Delta Lake not available)
        try:
            self.ad_features = create_ad_features(spark)
            self.ma_features = create_ma_features(spark)
            self.mcclellan = create_mcclellan_oscillator(spark)
            self.zbt = create_zbt(spark)
            self.scoring = create_signal_scoring(spark)
            self.features_available = True
        except Exception as e:
            logger.warning(f"Feature modules not available (Delta Lake issue): {e}")
            self.features_available = False
        
        # Signal generation parameters
        self.min_confidence_threshold = float(self.config.get("SIGNAL_MIN_CONFIDENCE", 70.0))
        self.signal_strength_threshold = self.config.get("SIGNAL_STRENGTH_THRESHOLD", "strong")
        self.real_time_mode = bool(self.config.get("SIGNAL_REAL_TIME", False))
        
        # Signal schema
        self.signal_schema = StructType([
            StructField("date", TimestampType(), False),
            StructField("composite_score_0_100", DoubleType(), True),
            StructField("signal_direction", StringType(), True),
            StructField("signal_strength", StringType(), True),
            StructField("confidence_score", DoubleType(), True),
            StructField("buy_signal", BooleanType(), True),
            StructField("sell_signal", BooleanType(), True),
            StructField("hold_signal", BooleanType(), True),
            StructField("strong_buy_signal", BooleanType(), True),
            StructField("strong_sell_signal", BooleanType(), True),
            StructField("signal_metadata", StringType(), True),
            StructField("generated_at", TimestampType(), False)
        ])
        
        logger.info("SignalGenerator initialized")
    
    def generate_signals(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_recalculate: bool = False,
        save_features: bool = True,
        save_signals: bool = True
    ) -> Dict[str, Any]:
        """
        Generate complete trading signals for the specified period.
        
        Args:
            symbols: List of symbols to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            force_recalculate: Force recalculation of all features
            save_features: Save calculated features to Delta Lake
            save_signals: Save generated signals to Delta Lake
            
        Returns:
            Dictionary with generation results and statistics
        """
        logger.info(f"Generating signals for period {start_date} to {end_date}")
        start_time = datetime.now()
        
        try:
            # Step 1: Calculate all features
            feature_results = self._calculate_all_features(
                symbols, start_date, end_date, force_recalculate, save_features
            )
            
            # Step 2: Generate composite scores
            scoring_results = self._generate_composite_scores(
                start_date, end_date, save_signals
            )
            
            # Step 3: Generate final trading signals
            signal_results = self._generate_trading_signals(scoring_results)
            
            # Compile results
            generation_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                "success": True,
                "generation_time_seconds": generation_time,
                "feature_results": feature_results,
                "scoring_results": scoring_results,
                "signal_results": signal_results,
                "total_signals_generated": signal_results.get("total_signals", 0),
                "buy_signals": signal_results.get("buy_signals", 0),
                "sell_signals": signal_results.get("sell_signals", 0),
                "strong_signals": signal_results.get("strong_signals", 0)
            }
            
            logger.info(f"Signal generation completed in {generation_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in signal generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "generation_time_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    def _calculate_all_features(
        self,
        symbols: Optional[List[str]],
        start_date: Optional[str],
        end_date: Optional[str],
        force_recalculate: bool,
        save_features: bool
    ) -> Dict[str, Any]:
        """
        Calculate all breadth indicators.
        
        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            force_recalculate: Force recalculation
            save_features: Save to Delta Lake
            
        Returns:
            Dictionary with feature calculation results
        """
        logger.info("Calculating all breadth indicators")
        
        results = {}
        
        # Calculate A/D features
        try:
            ad_result = self.ad_features.calculate_ad_features_from_delta(
                table_path="data/ohlcv",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                output_path="data/ad_features" if save_features else None
            )
            results["ad_features"] = {
                "success": True,
                "records": ad_result.count(),
                "summary": self.ad_features.get_ad_summary(ad_result)
            }
        except Exception as e:
            logger.error(f"Error calculating A/D features: {e}")
            results["ad_features"] = {"success": False, "error": str(e)}
        
        # Calculate MA features
        try:
            ma_result = self.ma_features.calculate_ma_features_from_delta(
                table_path="data/ohlcv",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                output_path="data/ma_features" if save_features else None
            )
            results["ma_features"] = {
                "success": True,
                "records": ma_result.count(),
                "summary": self.ma_features.get_ma_summary(ma_result)
            }
        except Exception as e:
            logger.error(f"Error calculating MA features: {e}")
            results["ma_features"] = {"success": False, "error": str(e)}
        
        # Calculate McClellan oscillator
        try:
            mcclellan_result = self.mcclellan.calculate_mcclellan_from_delta(
                table_path="data/ohlcv",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                output_path="data/mcclellan" if save_features else None
            )
            results["mcclellan"] = {
                "success": True,
                "records": mcclellan_result.count(),
                "summary": self.mcclellan.get_mcclellan_summary(mcclellan_result)
            }
        except Exception as e:
            logger.error(f"Error calculating McClellan oscillator: {e}")
            results["mcclellan"] = {"success": False, "error": str(e)}
        
        # Calculate ZBT
        try:
            zbt_result = self.zbt.calculate_zbt_from_delta(
                table_path="data/ohlcv",
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                output_path="data/zbt" if save_features else None
            )
            results["zbt"] = {
                "success": True,
                "records": zbt_result.count(),
                "summary": self.zbt.get_zbt_summary(zbt_result)
            }
        except Exception as e:
            logger.error(f"Error calculating ZBT: {e}")
            results["zbt"] = {"success": False, "error": str(e)}
        
        return results
    
    def _generate_composite_scores(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        save_signals: bool
    ) -> Dict[str, Any]:
        """
        Generate composite scores from all features.
        
        Args:
            start_date: Start date
            end_date: End date
            save_signals: Save to Delta Lake
            
        Returns:
            Dictionary with scoring results
        """
        logger.info("Generating composite scores")
        
        try:
            # Check if advanced features are available
            if not hasattr(self, 'features_available') or not self.features_available:
                logger.info("Using simple signal generation (advanced features not available)")
                return self._generate_simple_composite_scores(symbols, start_date, end_date)
            
            # Load OHLCV data directly from MinIO instead of Delta Lake
            import boto3
            import io
            
            s3_client = boto3.client(
                's3',
                endpoint_url='http://minio:9000',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
                region_name='us-east-1'
            )
            
            # Load data for the specified symbols and date range
            all_data = []
            symbols_to_process = symbols if symbols is not None else ["AAPL", "MSFT", "GOOGL"]
            
            for symbol in symbols_to_process:
                try:
                    # Try to load the specific date range file
                    key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                    response = s3_client.get_object(Bucket='breadthflow', Key=key)
                    parquet_content = response['Body'].read()
                    df = pd.read_parquet(io.BytesIO(parquet_content))
                    df['symbol'] = symbol
                    all_data.append(df)
                    logger.info(f"Loaded data for {symbol}: {len(df)} records")
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {e}")
                    continue
            
            if not all_data:
                return {"success": False, "error": "No OHLCV data found for any symbols"}
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Convert to Spark DataFrame
            composite_scores = self.spark.createDataFrame(combined_df)
            
            # Generate simple signals based on price movement
            signals_df = self._generate_simple_signals(composite_scores)
            
            return {
                "success": True,
                "records": signals_df.count(),
                "summary": f"Generated signals for {len(symbols_to_process)} symbols",
                "composite_scores": signals_df
            }
            
        except Exception as e:
            logger.error(f"Error generating composite scores: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_simple_signals(self, ohlcv_df: DataFrame) -> DataFrame:
        """
        Generate simple trading signals based on OHLCV data.
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trading signals
        """
        from pyspark.sql.functions import col, when, lit, current_timestamp
        
        # Convert to pandas for easier processing
        pandas_df = ohlcv_df.toPandas()
        
        signals_data = []
        
        for symbol in pandas_df['symbol'].unique():
            symbol_data = pandas_df[pandas_df['symbol'] == symbol].copy()
            
            if len(symbol_data) == 0:
                continue
                
            # Sort by date
            symbol_data = symbol_data.sort_values('date')
            
            # Calculate simple technical indicators
            symbol_data['price_change'] = symbol_data['close'].pct_change()
            symbol_data['volume_change'] = symbol_data['volume'].pct_change()
            
            # Generate signals for the latest date only
            latest_data = symbol_data.iloc[-1]
            
            # Simple signal logic based on price and volume
            price_change = latest_data['price_change']
            volume_change = latest_data['volume_change']
            
            if price_change > 0.02 and volume_change > 0.1:  # Strong positive movement
                signal_type = 'buy'
                confidence = 85.0
                strength = 'strong'
            elif price_change > 0.01:  # Moderate positive movement
                signal_type = 'buy'
                confidence = 70.0
                strength = 'medium'
            elif price_change < -0.02 and volume_change > 0.1:  # Strong negative movement
                signal_type = 'sell'
                confidence = 85.0
                strength = 'strong'
            elif price_change < -0.01:  # Moderate negative movement
                signal_type = 'sell'
                confidence = 70.0
                strength = 'medium'
            else:  # No clear direction
                signal_type = 'hold'
                confidence = 60.0
                strength = 'weak'
            
            # Create signal record
            signal_record = {
                'symbol': symbol,
                'date': latest_data['date'],
                'signal_type': signal_type,
                'confidence': confidence,
                'strength': strength,
                'composite_score': confidence,  # Use confidence as composite score
                'price_change': price_change,
                'volume_change': volume_change,
                'close': latest_data['close'],
                'volume': latest_data['volume'],
                'generated_at': datetime.now()
            }
            
            signals_data.append(signal_record)
        
        # Convert back to Spark DataFrame
        signals_df = self.spark.createDataFrame(signals_data)
        
        # Add signal boolean columns
        signals_df = signals_df.withColumn('buy_signal', when(col('signal_type') == 'buy', lit(True)).otherwise(lit(False))) \
                              .withColumn('sell_signal', when(col('signal_type') == 'sell', lit(True)).otherwise(lit(False))) \
                              .withColumn('hold_signal', when(col('signal_type') == 'hold', lit(True)).otherwise(lit(False))) \
                              .withColumn('strong_buy_signal', when((col('signal_type') == 'buy') & (col('strength') == 'strong'), lit(True)).otherwise(lit(False))) \
                              .withColumn('strong_sell_signal', when((col('signal_type') == 'sell') & (col('strength') == 'strong'), lit(True)).otherwise(lit(False))) \
                              .withColumn('confidence_score', col('confidence')) \
                              .withColumn('composite_score_0_100', col('composite_score')) \
                              .withColumn('signal_strength', col('strength')) \
                              .withColumn('signal_direction', col('signal_type'))
        
        return signals_df
    
    def _generate_simple_composite_scores(self, symbols: Optional[List[str]], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Generate simple composite scores directly from OHLCV data.
        
        Args:
            symbols: List of symbols to process
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with scoring results
        """
        try:
            import boto3
            import io
            
            s3_client = boto3.client(
                's3',
                endpoint_url='http://minio:9000',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
                region_name='us-east-1'
            )
            
            # Load data for the specified symbols and date range
            all_data = []
            symbols_to_process = symbols if symbols is not None else ["AAPL", "MSFT", "GOOGL"]
            
            for symbol in symbols_to_process:
                try:
                    # Try to load the specific date range file
                    key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                    response = s3_client.get_object(Bucket='breadthflow', Key=key)
                    parquet_content = response['Body'].read()
                    df = pd.read_parquet(io.BytesIO(parquet_content))
                    df['symbol'] = symbol
                    all_data.append(df)
                    logger.info(f"Loaded data for {symbol}: {len(df)} records")
                except Exception as e:
                    logger.warning(f"Could not load data for {symbol}: {e}")
                    continue
            
            if not all_data:
                return {"success": False, "error": "No OHLCV data found for any symbols"}
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Convert to Spark DataFrame
            composite_scores = self.spark.createDataFrame(combined_df)
            
            # Generate simple signals based on price movement
            signals_df = self._generate_simple_signals(composite_scores)
            
            return {
                "success": True,
                "records": signals_df.count(),
                "summary": f"Generated simple signals for {len(symbols_to_process)} symbols",
                "composite_scores": signals_df
            }
            
        except Exception as e:
            logger.error(f"Error generating simple composite scores: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_trading_signals(self, scoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final trading signals from composite scores.
        
        Args:
            scoring_results: Results from composite scoring
            
        Returns:
            Dictionary with signal results
        """
        logger.info("Generating trading signals")
        
        if not scoring_results.get("success", False):
            return {"success": False, "error": "No composite scores available"}
        
        try:
            composite_scores = scoring_results["composite_scores"]
            
            # Detect trading signals
            trading_signals = self.scoring.detect_trading_signals(composite_scores)
            
            # Save signals directly to MinIO instead of Delta Lake
            try:
                import boto3
                import io
                from datetime import datetime
                
                # Convert Spark DataFrame to pandas for MinIO storage
                signals_pandas = trading_signals.toPandas()
                
                # Create MinIO client
                s3_client = boto3.client(
                    's3',
                    endpoint_url='http://minio:9000',
                    aws_access_key_id='minioadmin',
                    aws_secret_access_key='minioadmin',
                    region_name='us-east-1'
                )
                
                # Save as Parquet
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                parquet_key = f"trading_signals/signals_{timestamp}.parquet"
                
                buffer = io.BytesIO()
                signals_pandas.to_parquet(buffer, index=False)
                buffer.seek(0)
                
                s3_client.put_object(
                    Bucket='breadthflow',
                    Key=parquet_key,
                    Body=buffer.getvalue()
                )
                
                # Save as JSON for easier dashboard consumption
                json_key = f"trading_signals/signals_{timestamp}.json"
                
                # Convert to JSON-friendly format
                signals_json = []
                for _, row in signals_pandas.iterrows():
                    signal_record = {
                        'symbol': row.get('symbol', 'UNKNOWN'),
                        'date': row.get('date', '').strftime('%Y-%m-%d') if hasattr(row.get('date', ''), 'strftime') else str(row.get('date', '')),
                        'signal_type': 'buy' if row.get('buy_signal', False) else ('sell' if row.get('sell_signal', False) else 'hold'),
                        'confidence': float(row.get('confidence_score', 0)),
                        'strength': row.get('signal_strength', 'medium'),
                        'composite_score': float(row.get('composite_score_0_100', 0)),
                        'generated_at': datetime.now().isoformat()
                    }
                    signals_json.append(signal_record)
                
                s3_client.put_object(
                    Bucket='breadthflow',
                    Key=json_key,
                    Body=json.dumps(signals_json, indent=2),
                    ContentType='application/json'
                )
                
                logger.info(f"Signals saved to MinIO: {parquet_key} and {json_key}")
                
            except Exception as save_error:
                logger.error(f"Error saving signals to MinIO: {save_error}")
                # Continue without saving - the signals are still generated
            
            # Count signal types
            signal_counts = trading_signals.agg(
                spark_sum(when(col("buy_signal"), 1).otherwise(0)).alias("buy_signals"),
                spark_sum(when(col("sell_signal"), 1).otherwise(0)).alias("sell_signals"),
                spark_sum(when(col("strong_buy_signal"), 1).otherwise(0)).alias("strong_buy_signals"),
                spark_sum(when(col("strong_sell_signal"), 1).otherwise(0)).alias("strong_sell_signals"),
                spark_sum(when(col("hold_signal"), 1).otherwise(0)).alias("hold_signals")
            ).collect()[0]
            
            return {
                "success": True,
                "total_signals": trading_signals.count(),
                "buy_signals": signal_counts.buy_signals,
                "sell_signals": signal_counts.sell_signals,
                "strong_buy_signals": signal_counts.strong_buy_signals,
                "strong_sell_signals": signal_counts.strong_sell_signals,
                "hold_signals": signal_counts.hold_signals,
                "strong_signals": signal_counts.strong_buy_signals + signal_counts.strong_sell_signals
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_signals_for_date(self, date: str, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate signals for a specific date (real-time mode).
        
        Args:
            date: Date to generate signals for (YYYY-MM-DD)
            symbols: List of symbols to analyze
            
        Returns:
            Dictionary with signal results for the date
        """
        logger.info(f"Generating signals for date: {date}")
        
        try:
            # Generate signals for the specific date
            results = self.generate_signals(
                symbols=symbols,
                start_date=date,
                end_date=date,
                force_recalculate=True,
                save_features=False,
                save_signals=True
            )
            
            if results["success"]:
                # Get the specific signal for the date
                signals_df = read_delta(self.spark, "data/trading_signals")
                date_signal = signals_df.filter(col("date") == date)
                
                if date_signal.count() > 0:
                    signal_row = date_signal.collect()[0]
                    return {
                        "success": True,
                        "date": date,
                        "composite_score": signal_row.composite_score_0_100,
                        "signal_direction": signal_row.signal_direction,
                        "signal_strength": signal_row.signal_strength,
                        "confidence_score": signal_row.confidence_score,
                        "buy_signal": signal_row.buy_signal,
                        "sell_signal": signal_row.sell_signal,
                        "strong_buy_signal": signal_row.strong_buy_signal,
                        "strong_sell_signal": signal_row.strong_sell_signal
                    }
                else:
                    return {"success": False, "error": f"No signal found for date {date}"}
            else:
                return results
                
        except Exception as e:
            logger.error(f"Error generating signals for date {date}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_latest_signals(self, limit: int = 10) -> DataFrame:
        """
        Get the latest trading signals.
        
        Args:
            limit: Number of latest signals to retrieve
            
        Returns:
            DataFrame with latest signals
        """
        try:
            signals_df = read_delta(self.spark, "data/trading_signals")
            return signals_df.orderBy(col("date").desc()).limit(limit)
        except Exception as e:
            logger.error(f"Error retrieving latest signals: {e}")
            return self.spark.createDataFrame([], schema=self.signal_schema)
    
    def get_signals_summary(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of generated signals.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Dictionary with signal summary
        """
        try:
            signals_df = read_delta(self.spark, "data/trading_signals")
            
            if start_date:
                signals_df = signals_df.filter(col("date") >= start_date)
            if end_date:
                signals_df = signals_df.filter(col("date") <= end_date)
            
            summary = signals_df.agg(
                count("*").alias("total_signals"),
                expr("avg(composite_score_0_100)").alias("avg_score"),
                expr("avg(confidence_score)").alias("avg_confidence"),
                spark_sum(when(col("buy_signal"), 1).otherwise(0)).alias("buy_signals"),
                spark_sum(when(col("sell_signal"), 1).otherwise(0)).alias("sell_signals"),
                spark_sum(when(col("strong_buy_signal"), 1).otherwise(0)).alias("strong_buy_signals"),
                spark_sum(when(col("strong_sell_signal"), 1).otherwise(0)).alias("strong_sell_signals"),
                spark_sum(when(col("signal_direction") == "bullish", 1).otherwise(0)).alias("bullish_days"),
                spark_sum(when(col("signal_direction") == "bearish", 1).otherwise(0)).alias("bearish_days"),
                spark_sum(when(col("signal_direction") == "neutral", 1).otherwise(0)).alias("neutral_days")
            ).collect()[0]
            
            return {
                "total_signals": summary.total_signals,
                "avg_score": summary.avg_score,
                "avg_confidence": summary.avg_confidence,
                "buy_signals": summary.buy_signals,
                "sell_signals": summary.sell_signals,
                "strong_buy_signals": summary.strong_buy_signals,
                "strong_sell_signals": summary.strong_sell_signals,
                "bullish_days": summary.bullish_days,
                "bearish_days": summary.bearish_days,
                "neutral_days": summary.neutral_days
            }
            
        except Exception as e:
            logger.error(f"Error getting signals summary: {e}")
            return {"error": str(e)}
    
    def search_signals(
        self,
        signal_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_score: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> DataFrame:
        """
        Search for signals based on criteria.
        
        Args:
            signal_type: Type of signal (buy, sell, strong_buy, strong_sell)
            min_confidence: Minimum confidence score
            min_score: Minimum composite score
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching signals
        """
        try:
            signals_df = read_delta(self.spark, "data/trading_signals")
            
            # Apply filters
            if signal_type:
                if signal_type == "buy":
                    signals_df = signals_df.filter(col("buy_signal"))
                elif signal_type == "sell":
                    signals_df = signals_df.filter(col("sell_signal"))
                elif signal_type == "strong_buy":
                    signals_df = signals_df.filter(col("strong_buy_signal"))
                elif signal_type == "strong_sell":
                    signals_df = signals_df.filter(col("strong_sell_signal"))
            
            if min_confidence is not None:
                signals_df = signals_df.filter(col("confidence_score") >= min_confidence)
            
            if min_score is not None:
                signals_df = signals_df.filter(col("composite_score_0_100") >= min_score)
            
            if start_date:
                signals_df = signals_df.filter(col("date") >= start_date)
            
            if end_date:
                signals_df = signals_df.filter(col("date") <= end_date)
            
            return signals_df.orderBy(col("date").desc()).limit(limit)
            
        except Exception as e:
            logger.error(f"Error searching signals: {e}")
            return self.spark.createDataFrame([], schema=self.signal_schema)


def create_signal_generator(spark: SparkSession) -> SignalGenerator:
    """
    Factory function to create SignalGenerator instance.
    
    Args:
        spark: PySpark session
        
    Returns:
        Configured SignalGenerator instance
    """
    return SignalGenerator(spark)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("SignalGeneratorTest") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Create signal generator
    generator = create_signal_generator(spark)
    
    try:
        # Generate signals for a period
        results = generator.generate_signals(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            start_date="2024-01-01",
            end_date="2024-12-31"
        )
        
        print(f"Signal Generation Results: {results}")
        
        # Get latest signals
        latest_signals = generator.get_latest_signals(5)
        print(f"Latest signals: {latest_signals.count()}")
        
        # Get signals summary
        summary = generator.get_signals_summary()
        print(f"Signals Summary: {summary}")
        
        # Search for strong buy signals
        strong_buys = generator.search_signals(
            signal_type="strong_buy",
            min_confidence=80.0,
            limit=10
        )
        print(f"Strong buy signals: {strong_buys.count()}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        spark.stop()
