"""
Spark Structured Streaming Pipeline for BreadthFlow

Implements continuous data processing using Spark Structured Streaming
with micro-batch intervals, automatic fault tolerance, and real-time processing.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import uuid

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, date_format, struct, to_json, 
    window, expr, lit, current_timestamp, from_json
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    LongType, TimestampType, BooleanType
)
from pyspark.sql.streaming import DataStreamWriter

from cli.kibana_enhanced_bf import DualLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkStreamingPipeline:
    """
    Spark Structured Streaming Pipeline for continuous data processing.
    
    Features:
    - Continuous processing with micro-batch intervals
    - Automatic fault tolerance and recovery
    - Real-time data processing
    - Built-in scheduling (no need for cron or custom threading)
    - Runs until explicitly stopped
    """
    
    def __init__(self, spark: SparkSession = None):
        """Initialize the streaming pipeline."""
        print("🔍 DEBUG: Starting SparkStreamingPipeline.__init__")
        
        if spark is None:
            # Get existing Spark session
            from pyspark.sql import SparkSession
            print("🔍 DEBUG: Getting existing Spark session...")
            self.spark = SparkSession.getActiveSession()
            if self.spark is None:
                # Create new session if none exists
                print("🔍 DEBUG: Creating new Spark session...")
                self.spark = SparkSession.builder \
                    .appName("BreadthFlow-Streaming-Pipeline") \
                    .master("local[*]") \
                    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
                    .config("spark.hadoop.fs.defaultFS", "file:///") \
                    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
                    .getOrCreate()
                print("✅ DEBUG: New Spark session created")
            else:
                print("✅ DEBUG: Using existing Spark session")
        else:
            self.spark = spark
            print("✅ DEBUG: Using provided Spark session")
            
        self.is_running = False
        self.query = None
        self.pipeline_id = str(uuid.uuid4())
        print(f"🔍 DEBUG: Pipeline ID created: {self.pipeline_id}")
        
        # Create logger for pipeline
        try:
            print("🔍 DEBUG: Creating DualLogger...")
            self.dual_logger = DualLogger(
                self.pipeline_id, 
                f"spark_streaming_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print("✅ DEBUG: DualLogger created successfully")
            print("✅ DEBUG: SparkStreamingPipeline initialized")
        except Exception as e:
            print(f"❌ DEBUG: Failed to create DualLogger: {str(e)}")
            import traceback
            print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
            raise
    
    def start_continuous_pipeline(
        self, 
        symbols: List[str], 
        timeframe: str = "1day",
        interval_seconds: int = 300,  # Default 5 minutes for safety
        data_source: str = "yfinance"
    ):
        """
        Start continuous pipeline using Spark Structured Streaming.
        
        Args:
            symbols: List of symbols to process
            timeframe: Data timeframe (1min, 5min, 15min, 1hour, 1day)
            interval_seconds: Micro-batch interval in seconds
            data_source: Data source (yfinance, etc.)
        """
        if self.is_running:
            return "❌ Pipeline is already running"
        
        self.is_running = True
        self.dual_logger.log("INFO", f"🚀 Starting Spark Structured Streaming Pipeline")
        self.dual_logger.log("INFO", f"📊 Symbols: {', '.join(symbols)}")
        self.dual_logger.log("INFO", f"⏰ Timeframe: {timeframe}")
        self.dual_logger.log("INFO", f"⏱️ Interval: {interval_seconds} seconds")
        
        try:
            # Create streaming DataFrame that generates data every interval
            self.dual_logger.log("INFO", "🔍 Step 1: About to call _create_streaming_source...")
            streaming_df = self._create_streaming_source(interval_seconds)
            self.dual_logger.log("INFO", "✅ Step 1: Streaming source created successfully")
            
            # Process the streaming data
            self.dual_logger.log("INFO", "🔍 Step 2: Processing streaming data...")
            query = self._process_streaming_data(
                streaming_df, symbols, timeframe, data_source, interval_seconds
            )
            self.dual_logger.log("INFO", "✅ Step 2: Streaming data processing completed")
            
            self.query = query
            self.dual_logger.log("INFO", "✅ Spark Structured Streaming Pipeline started successfully")
            
            return f"🚀 Continuous pipeline started with {interval_seconds}s intervals\n📊 Symbols: {', '.join(symbols)}\n⏰ Timeframe: {timeframe}\n🔄 Will run until stopped"
            
        except Exception as e:
            self.is_running = False
            self.dual_logger.log("ERROR", f"❌ Failed to start pipeline: {str(e)}")
            import traceback
            self.dual_logger.log("ERROR", f"❌ Full traceback: {traceback.format_exc()}")
            return f"❌ Failed to start pipeline: {str(e)}"
    
    def _create_streaming_source(self, interval_seconds: int):
        """Create a streaming source that generates triggers every interval."""
        # Create a simple streaming DataFrame using rate source
        # Use basic options to avoid calculation issues
        self.dual_logger.log("DEBUG", "🔍 Creating rate streaming source...")
        streaming_df = self.spark.readStream \
            .format("rate") \
            .option("rowsPerSecond", "1") \
            .load()
        self.dual_logger.log("DEBUG", "✅ Rate streaming source created")
        
        return streaming_df
    
    def _process_streaming_data(
        self, 
        streaming_df, 
        symbols: List[str], 
        timeframe: str, 
        data_source: str,
        interval_seconds: int
    ):
        """Process streaming data with data fetch, signals, and backtest using backpressure controls."""
        
        self.dual_logger.log("DEBUG", "🔍 Processing streaming data - adding columns...")
        
        # Add timestamp for processing
        # Note: We'll use batch_id from foreachBatch instead of monotonically_increasing_id()
        processed_df = streaming_df.withColumn(
            "processing_time", current_timestamp()
        )
        
        self.dual_logger.log("DEBUG", "✅ Columns added successfully")
        
        # Initialize backpressure tracking
        self._current_batch_id = None
        self._last_completed_batch = None
        self._batch_start_time = None
        
        # Define the processing logic for each micro-batch with backpressure
        def process_batch_with_backpressure(batch_df, batch_id):
            """Process each micro-batch with backpressure controls."""
            if batch_df.count() == 0:
                return
            
            # Check if previous batch is still running
            if (self._current_batch_id is not None and 
                self._current_batch_id != self._last_completed_batch):
                
                self.dual_logger.log("WARN", f"⏳ Previous batch {self._current_batch_id} still running, skipping batch {batch_id}")
                return  # Skip this batch - backpressure in action
            
            # Check if batch is taking too long (timeout after 5 minutes)
            if (self._batch_start_time and 
                time.time() - self._batch_start_time > 300):  # 5 minutes timeout
                
                self.dual_logger.log("WARN", f"⏰ Batch {self._current_batch_id} taking too long, forcing completion")
                self._last_completed_batch = self._current_batch_id
            
            # Start processing this batch
            self._current_batch_id = batch_id
            self._batch_start_time = time.time()
            cycle_id = batch_id  # Use batch_id as cycle_id instead of pipeline_cycle column
            
            self.dual_logger.log("INFO", f"🔄 Processing batch {batch_id}, cycle {cycle_id}")
            
            try:
                # Step 1: Data Fetch
                self.dual_logger.log("INFO", f"📥 Step 1: Data Fetch (Cycle {cycle_id})")
                self._execute_data_fetch(symbols, timeframe, data_source)
                
                # Step 2: Signal Generation
                self.dual_logger.log("INFO", f"📊 Step 2: Signal Generation (Cycle {cycle_id})")
                self._execute_signal_generation(symbols, timeframe)
                
                # Step 3: Backtesting
                self.dual_logger.log("INFO", f"📈 Step 3: Backtesting (Cycle {cycle_id})")
                self._execute_backtest(symbols, timeframe)
                
                # Mark batch as complete
                self._last_completed_batch = batch_id
                self.dual_logger.log("INFO", f"✅ Cycle {cycle_id} completed successfully")
                
            except Exception as e:
                self.dual_logger.log("ERROR", f"❌ Cycle {cycle_id} failed: {str(e)}")
                # Mark as complete even if failed to allow next batch to proceed
                self._last_completed_batch = batch_id
        
        # Start the streaming query with backpressure controls
        self.dual_logger.log("DEBUG", "🔍 Starting streaming query...")
        self.dual_logger.log("DEBUG", "🔍 Setting up writeStream...")
        
        query = processed_df.writeStream \
            .foreachBatch(process_batch_with_backpressure) \
            .outputMode("append") \
            .option("maxOffsetsPerTrigger", 1) \
            .option("maxRatePerPartition", 1) \
            .option("spark.sql.streaming.minBatchesToRetain", 1) \
            .option("checkpointLocation", "/tmp/streaming-checkpoint")
        
        self.dual_logger.log("DEBUG", "🔍 Setting trigger...")
        query = query.trigger(processingTime="300 seconds")
        
        self.dual_logger.log("DEBUG", "🔍 Starting query...")
        query = query.start()
        
        self.dual_logger.log("DEBUG", "✅ Streaming query started successfully")
        return query
    
    def _execute_data_fetch(self, symbols: List[str], timeframe: str, data_source: str):
        """Execute data fetch step."""
        import subprocess
        
        symbols_str = ",".join(symbols)
        
        # Calculate required data window based on timeframe and lookback period
        lookback_period = self._get_lookback_period_for_timeframe(timeframe)
        required_days = self._calculate_required_days(timeframe, lookback_period)
        
        start_date = (datetime.now() - timedelta(days=required_days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        cmd = [
            'python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 
            'data', 'fetch',
            '--symbols', symbols_str,
            '--start-date', start_date,
            '--end-date', end_date,
            '--timeframe', timeframe,
            '--data-source', data_source
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"Data fetch failed: {result.stderr}")
    
    def _get_lookback_period_for_timeframe(self, timeframe: str) -> int:
        """Get the lookback period for a given timeframe."""
        # Default lookback periods based on timeframe_agnostic_signals.py
        lookback_periods = {
            '1day': 50,
            '1hour': 50,
            '15min': 32,
            '5min': 24,
            '1min': 20
        }
        return lookback_periods.get(timeframe, 50)
    
    def _calculate_required_days(self, timeframe: str, lookback_period: int) -> int:
        """Calculate required days of data based on timeframe and lookback period."""
        # Simple approach: use 2x the lookback period to ensure enough data
        # This works for all timeframes and is much simpler
        
        if timeframe == '1day':
            # For daily data, we need 2x lookback_period days
            return lookback_period * 2
        elif timeframe == '1hour':
            # For hourly data, we need 2x lookback_period days
            # (markets are open ~6.5 hours per day, so this gives plenty of buffer)
            return lookback_period * 2
        elif timeframe in ['15min', '5min', '1min']:
            # For intraday timeframes, use 1.5x lookback_period days
            # This accounts for market hours and gives sufficient buffer
            return int(lookback_period * 1.5)
        else:
            # Default fallback: 2x lookback_period
            return lookback_period * 2
    
    def _execute_signal_generation(self, symbols: List[str], timeframe: str):
        """Execute signal generation step."""
        import subprocess
        
        symbols_str = ",".join(symbols)
        
        # Use the same dynamic window calculation as data fetch
        lookback_period = self._get_lookback_period_for_timeframe(timeframe)
        required_days = self._calculate_required_days(timeframe, lookback_period)
        
        start_date = (datetime.now() - timedelta(days=required_days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        cmd = [
            'python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 
            'signals', 'generate',
            '--symbols', symbols_str,
            '--start-date', start_date,
            '--end-date', end_date,
            '--timeframe', timeframe
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"Signal generation failed: {result.stderr}")
    
    def _execute_backtest(self, symbols: List[str], timeframe: str):
        """Execute backtest step."""
        import subprocess
        
        symbols_str = ",".join(symbols)
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        cmd = [
            'python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 
            'backtest', 'run',
            '--symbols', symbols_str,
            '--from-date', start_date,
            '--to-date', end_date,
            '--timeframe', timeframe,
            '--initial-capital', '100000'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise Exception(f"Backtest failed: {result.stderr}")
    
    def stop_pipeline(self):
        """Stop the streaming pipeline."""
        if not self.is_running or self.query is None:
            return "❌ No pipeline is running"
        
        try:
            self.query.stop()
            self.is_running = False
            self.dual_logger.log("INFO", "🛑 Spark Structured Streaming Pipeline stopped")
            self.dual_logger.complete('completed')
            return "🛑 Pipeline stopped successfully"
            
        except Exception as e:
            self.dual_logger.log("ERROR", f"❌ Failed to stop pipeline: {str(e)}")
            return f"❌ Failed to stop pipeline: {str(e)}"
    
    def get_status(self):
        """Get pipeline status."""
        if not self.is_running:
            return {
                "state": "stopped",
                "pipeline_id": self.pipeline_id,
                "is_running": False
            }
        
        return {
            "state": "running",
            "pipeline_id": self.pipeline_id,
            "is_running": True,
            "query_status": self.query.status if self.query else None
        }


def main():
    """Main function to run the streaming pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spark Structured Streaming Pipeline")
    parser.add_argument("--symbols", default="AAPL,MSFT", help="Comma-separated symbols")
    parser.add_argument("--timeframe", default="1day", help="Data timeframe")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    parser.add_argument("--data-source", default="yfinance", help="Data source")
    parser.add_argument("--action", choices=["start", "stop", "status"], default="start", help="Action to perform")
    
    args = parser.parse_args()
    
    # Initialize Spark
    print("🔍 DEBUG: Starting Spark session creation...")
    try:
        spark = SparkSession.builder \
            .appName("BreadthFlow-Streaming-Pipeline") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/streaming-checkpoint") \
            .getOrCreate()
        print("✅ DEBUG: Spark session created successfully")
    except Exception as e:
        print(f"❌ DEBUG: Failed to create Spark session: {str(e)}")
        import traceback
        print(f"❌ DEBUG: Full traceback: {traceback.format_exc()}")
        raise
    
    # Create pipeline
    pipeline = SparkStreamingPipeline(spark)
    pipeline.dual_logger.log("INFO", f"🚀 Starting Spark Structured Streaming Pipeline")
    time.sleep(5)
    pipeline.dual_logger.log("INFO", f"🚀 slept 5 seconds before starting Spark Structured Streaming Pipeline")
    if args.action == "start":
        symbols = [s.strip() for s in args.symbols.split(",")]
        result = pipeline.start_continuous_pipeline(
            symbols=symbols,
            timeframe=args.timeframe,
            interval_seconds=args.interval,
            data_source=args.data_source
        )
        print(result)
        
        # Keep the pipeline running
        try:
            while pipeline.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping pipeline...")
            pipeline.stop_pipeline()
    
    elif args.action == "stop":
        result = pipeline.stop_pipeline()
        print(result)
    
    elif args.action == "status":
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))
    
    spark.stop()


if __name__ == "__main__":
    main()
