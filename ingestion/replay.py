"""
Replay Manager for Breadth/Thrust Signals POC

Implements historical data replay from Delta Lake to Kafka
for real-time simulation and testing of the signal generation pipeline.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Iterator
from dataclasses import dataclass, asdict

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, to_timestamp, date_format, struct, to_json, 
    row_number, window, expr, lit
)
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

from kafka import KafkaProducer
from kafka.errors import KafkaError

from features.common.config import get_config
from features.common.io import read_delta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for data replay."""
    speed_multiplier: float = 60.0  # 1 minute = 1 second
    batch_size: int = 1000  # Records per batch
    topic_name: str = "quotes_replay"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbols: Optional[List[str]] = None
    include_metadata: bool = True


class ReplayManager:
    """
    Manages historical data replay from Delta Lake to Kafka.
    
    Features:
    - Configurable replay speed (real-time simulation)
    - Batch processing for performance
    - Progress tracking and monitoring
    - Error handling and recovery
    - Metadata injection for tracking
    """
    
    def __init__(self, spark: SparkSession, kafka_producer: Optional[KafkaProducer] = None):
        """
        Initialize ReplayManager.
        
        Args:
            spark: PySpark session
            kafka_producer: Kafka producer instance (optional)
        """
        self.spark = spark
        self.config = get_config()
        
        # Initialize Kafka producer if not provided
        if kafka_producer is None:
            self.kafka_producer = self._create_kafka_producer()
        else:
            self.kafka_producer = kafka_producer
        
        # Replay state
        self.is_replaying = False
        self.current_batch = 0
        self.total_records = 0
        self.processed_records = 0
        self.start_time = None
        self.end_time = None
        
        logger.info("ReplayManager initialized")
    
    def _create_kafka_producer(self) -> KafkaProducer:
        """Create Kafka producer with configuration."""
        kafka_config = self.config.get_kafka_config()
        
        producer_config = {
            'bootstrap_servers': kafka_config.get('bootstrap_servers', 'localhost:9092'),
            'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 10,
            'buffer_memory': 33554432,
        }
        
        return KafkaProducer(**producer_config)
    
    def load_historical_data(
        self, 
        table_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Load historical data from Delta Lake for replay.
        
        Args:
            table_path: Path to Delta table
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            symbols: List of symbols to filter
            
        Returns:
            DataFrame with historical data
        """
        logger.info(f"Loading historical data from {table_path}")
        
        # Read from Delta Lake
        df = read_delta(self.spark, table_path)
        
        # Apply filters
        if start_date:
            df = df.filter(col("date") >= start_date)
            logger.info(f"Filtered data from {start_date}")
        
        if end_date:
            df = df.filter(col("date") <= end_date)
            logger.info(f"Filtered data to {end_date}")
        
        if symbols:
            df = df.filter(col("symbol").isin(symbols))
            logger.info(f"Filtered to {len(symbols)} symbols")
        
        # Sort by date for chronological replay
        df = df.orderBy("date", "symbol")
        
        # Add replay metadata
        df = df.withColumn("replay_timestamp", col("date")) \
               .withColumn("original_date", col("date"))
        
        total_records = df.count()
        logger.info(f"Loaded {total_records} records for replay")
        
        return df
    
    def replay_data(
        self, 
        df: DataFrame, 
        config: ReplayConfig
    ) -> Dict[str, Any]:
        """
        Replay historical data to Kafka with configurable speed.
        
        Args:
            df: DataFrame with historical data
            config: Replay configuration
            
        Returns:
            Dictionary with replay results
        """
        if self.is_replaying:
            raise RuntimeError("Replay already in progress")
        
        self.is_replaying = True
        self.start_time = datetime.now()
        self.current_batch = 0
        self.total_records = df.count()
        self.processed_records = 0
        
        logger.info(f"Starting replay of {self.total_records} records")
        logger.info(f"Speed multiplier: {config.speed_multiplier}x")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Kafka topic: {config.topic_name}")
        
        try:
            # Convert to pandas for easier processing
            pandas_df = df.toPandas()
            
            # Group by date for chronological replay
            pandas_df = pandas_df.sort_values(['date', 'symbol'])
            
            # Process in batches
            batch_results = []
            for batch_start in range(0, len(pandas_df), config.batch_size):
                batch_end = min(batch_start + config.batch_size, len(pandas_df))
                batch_df = pandas_df.iloc[batch_start:batch_end]
                
                batch_result = self._process_batch(batch_df, config)
                batch_results.append(batch_result)
                
                self.current_batch += 1
                self.processed_records += len(batch_df)
                
                # Progress logging
                if self.current_batch % 10 == 0:
                    progress = (self.processed_records / self.total_records) * 100
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    logger.info(f"Progress: {progress:.1f}% ({self.processed_records}/{self.total_records}) - {elapsed:.1f}s elapsed")
                
                # Simulate real-time delay
                if config.speed_multiplier > 0:
                    time.sleep(1.0 / config.speed_multiplier)
            
            self.end_time = datetime.now()
            total_duration = (self.end_time - self.start_time).total_seconds()
            
            # Compile results
            result = {
                "success": True,
                "total_records": self.total_records,
                "processed_records": self.processed_records,
                "total_batches": self.current_batch,
                "duration_seconds": total_duration,
                "speed_multiplier": config.speed_multiplier,
                "effective_speed": self.total_records / total_duration if total_duration > 0 else 0,
                "kafka_topic": config.topic_name,
                "batch_results": batch_results
            }
            
            logger.info(f"Replay completed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during replay: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_records": self.processed_records,
                "current_batch": self.current_batch
            }
        finally:
            self.is_replaying = False
    
    def _process_batch(
        self, 
        batch_df: pd.DataFrame, 
        config: ReplayConfig
    ) -> Dict[str, Any]:
        """
        Process a batch of records and send to Kafka.
        
        Args:
            batch_df: Batch of records
            config: Replay configuration
            
        Returns:
            Batch processing results
        """
        batch_start_time = time.time()
        successful_sends = 0
        failed_sends = 0
        
        for _, row in batch_df.iterrows():
            try:
                # Prepare message
                message = self._prepare_message(row, config)
                
                # Send to Kafka
                future = self.kafka_producer.send(
                    topic=config.topic_name,
                    key=row['symbol'],
                    value=message
                )
                
                # Wait for send confirmation
                record_metadata = future.get(timeout=10)
                successful_sends += 1
                
            except Exception as e:
                logger.error(f"Failed to send message for {row['symbol']}: {str(e)}")
                failed_sends += 1
        
        batch_duration = time.time() - batch_start_time
        
        return {
            "batch_number": self.current_batch,
            "records_processed": len(batch_df),
            "successful_sends": successful_sends,
            "failed_sends": failed_sends,
            "duration_seconds": batch_duration
        }
    
    def _prepare_message(self, row: pd.Series, config: ReplayConfig) -> Dict[str, Any]:
        """
        Prepare message for Kafka.
        
        Args:
            row: DataFrame row
            config: Replay configuration
            
        Returns:
            Message dictionary
        """
        message = {
            "symbol": row['symbol'],
            "date": row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
            "open": float(row['open']) if pd.notna(row['open']) else None,
            "high": float(row['high']) if pd.notna(row['high']) else None,
            "low": float(row['low']) if pd.notna(row['low']) else None,
            "close": float(row['close']) if pd.notna(row['close']) else None,
            "volume": int(row['volume']) if pd.notna(row['volume']) else None,
            "adj_close": float(row['adj_close']) if pd.notna(row['adj_close']) else None,
            "replay_timestamp": datetime.now().isoformat()
        }
        
        if config.include_metadata:
            message["metadata"] = {
                "original_date": row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                "data_quality_score": float(row['data_quality_score']) if 'data_quality_score' in row and pd.notna(row['data_quality_score']) else None,
                "fetched_at": row['fetched_at'].isoformat() if 'fetched_at' in row and hasattr(row['fetched_at'], 'isoformat') else None,
                "replay_batch": self.current_batch,
                "replay_config": {
                    "speed_multiplier": config.speed_multiplier,
                    "topic_name": config.topic_name
                }
            }
        
        return message
    
    def get_replay_status(self) -> Dict[str, Any]:
        """
        Get current replay status.
        
        Returns:
            Dictionary with replay status
        """
        if not self.is_replaying:
            return {"status": "idle"}
        
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        progress = (self.processed_records / self.total_records) * 100 if self.total_records > 0 else 0
        
        return {
            "status": "replaying",
            "progress_percent": progress,
            "processed_records": self.processed_records,
            "total_records": self.total_records,
            "current_batch": self.current_batch,
            "elapsed_seconds": elapsed,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }
    
    def stop_replay(self) -> bool:
        """
        Stop the current replay.
        
        Returns:
            True if replay was stopped, False if not running
        """
        if not self.is_replaying:
            return False
        
        logger.info("Stopping replay...")
        self.is_replaying = False
        return True
    
    def create_replay_stream(
        self, 
        df: DataFrame, 
        config: ReplayConfig
    ) -> Iterator[Dict[str, Any]]:
        """
        Create a streaming iterator for replay data.
        
        Args:
            df: DataFrame with historical data
            config: Replay configuration
            
        Yields:
            Message dictionaries for Kafka
        """
        pandas_df = df.toPandas().sort_values(['date', 'symbol'])
        
        for _, row in pandas_df.iterrows():
            if not self.is_replaying:
                break
            
            message = self._prepare_message(row, config)
            yield message
            
            # Simulate real-time delay
            if config.speed_multiplier > 0:
                time.sleep(1.0 / config.speed_multiplier)
    
    def cleanup(self):
        """Clean up resources."""
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
            logger.info("Kafka producer closed")


def create_replay_manager(spark: SparkSession) -> ReplayManager:
    """
    Factory function to create ReplayManager instance.
    
    Args:
        spark: PySpark session
        
    Returns:
        Configured ReplayManager instance
    """
    return ReplayManager(spark)


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the replay functionality
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ReplayManagerTest") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Create replay manager
    replay_manager = create_replay_manager(spark)
    
    # Example configuration
    config = ReplayConfig(
        speed_multiplier=60.0,  # 1 minute = 1 second
        batch_size=100,
        topic_name="quotes_replay",
        start_date="2024-01-01",
        end_date="2024-01-31",
        symbols=["AAPL", "MSFT", "GOOGL"]
    )
    
    try:
        # Load data (assuming it exists)
        df = replay_manager.load_historical_data(
            table_path="data/ohlcv",
            start_date=config.start_date,
            end_date=config.end_date,
            symbols=config.symbols
        )
        
        # Start replay
        result = replay_manager.replay_data(df, config)
        print(f"Replay result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        replay_manager.cleanup()
        spark.stop()
