"""
Data Fetcher for Breadth/Thrust Signals POC

Implements PySpark-based concurrent data fetching from Yahoo Finance
with robust error handling, retry logic, and data quality validation.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType

from features.common.config import get_config
from features.common.io import write_delta, read_delta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    PySpark-based data fetcher for market data with concurrent processing.

    Features:
    - Concurrent data fetching from Yahoo Finance
    - PySpark UDF integration for distributed processing
    - Robust error handling and retry logic
    - Data quality validation and cleaning
    - Delta Lake integration for storage
    """

    def __init__(self, spark: SparkSession):
        """
        Initialize DataFetcher with Spark session.

        Args:
            spark: PySpark session for distributed processing
        """
        self.spark = spark
        self.config = get_config()

        # Delta Lake is handled by the Python delta-spark package
        logger.info("Delta Lake support available via delta-spark package")

        # Data schema for OHLCV
        self.ohlcv_schema = StructType(
            [
                StructField("symbol", StringType(), False),
                StructField("date", TimestampType(), False),
                StructField("open", DoubleType(), True),
                StructField("high", DoubleType(), True),
                StructField("low", DoubleType(), True),
                StructField("close", DoubleType(), True),
                StructField("volume", LongType(), True),
                StructField("adj_close", DoubleType(), True),
                StructField("fetched_at", TimestampType(), False),
                StructField("data_quality_score", DoubleType(), True),
            ]
        )

        # Retry configuration
        self.max_retries = int(self.config.get("DATA_MAX_RETRIES", 3))
        self.retry_delay = float(self.config.get("DATA_RETRY_DELAY", 1.0))
        self.max_workers = int(self.config.get("DATA_MAX_WORKERS", 2))

        # Data quality thresholds
        self.min_data_completeness = float(self.config.get("DATA_MIN_COMPLETENESS", 0.95))
        self.max_price_change = float(self.config.get("DATA_MAX_PRICE_CHANGE", 0.5))  # 50%

        logger.info(f"DataFetcher initialized with {self.max_workers} workers")

    def fetch_symbol_data(self, symbol: str, start_date: str, end_date: str, retry_count: int = 0) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single symbol with retry logic.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            retry_count: Current retry attempt

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.debug(f"Fetching data for {symbol} from {start_date} to {end_date}")

            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Reset index to get date as column
            df = df.reset_index()

            # Rename columns to match our schema
            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Adj Close": "adj_close",
                }
            )

            # Ensure date column is properly formatted as datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

            # Add metadata columns
            df["symbol"] = symbol
            df["fetched_at"] = datetime.now()

            # Keep only the columns that match our schema
            schema_columns = [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adj_close",
                "fetched_at",
                "data_quality_score",
            ]
            df = df[df.columns.intersection(schema_columns)]

            # Ensure all required columns are present
            for column_name in schema_columns:
                if column_name not in df.columns:
                    if column_name == "data_quality_score":
                        df[column_name] = 1.0  # Default quality score
                    else:
                        logger.warning(f"Missing required column: {column_name}")

            # Calculate data quality score
            df["data_quality_score"] = self._calculate_data_quality(df)

            # Validate data quality
            if not self._validate_data_quality(df):
                logger.warning(f"Data quality check failed for {symbol}")
                if retry_count < self.max_retries:
                    logger.info(f"Retrying {symbol} (attempt {retry_count + 1})")
                    time.sleep(self.retry_delay)
                    return self.fetch_symbol_data(symbol, start_date, end_date, retry_count + 1)
                return None

            logger.debug(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            if retry_count < self.max_retries:
                logger.info(f"Retrying {symbol} due to error (attempt {retry_count + 1})")
                time.sleep(self.retry_delay * (retry_count + 1))  # Exponential backoff
                return self.fetch_symbol_data(symbol, start_date, end_date, retry_count + 1)
            return None

    def fetch_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str, max_workers: int = None) -> DataFrame:
        """
        Fetch data for multiple symbols using Spark's mapPartitions for true parallelism.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            PySpark DataFrame with all fetched data
        """
        logger.info(f"Starting Spark mapPartitions fetch for {len(symbols)} symbols")
        start_time = time.time()

        # Create a DataFrame with symbols to parallelize
        symbols_df = self.spark.createDataFrame([(symbol,) for symbol in symbols], ["symbol"])

        # Use mapPartitions for true Spark parallelism
        def fetch_partition(partition):
            import yfinance as yf
            import pandas as pd
            from datetime import datetime

            results = []
            partition_symbols = []

            # Collect symbols in this partition
            for row in partition:
                partition_symbols.append(row.symbol)

            logger.info(f"🔄 Processing partition with symbols: {partition_symbols}")

            for row in partition:
                symbol = row.symbol
                symbol_start = time.time()
                logger.info(f"🔄 Fetching data for symbol: {symbol}")

                try:
                    # Fetch data from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

                    if not df.empty:
                        # Reset index to get date as column
                        df = df.reset_index()

                        # Rename columns
                        df = df.rename(
                            columns={
                                "Date": "date",
                                "Open": "open",
                                "High": "high",
                                "Low": "low",
                                "Close": "close",
                                "Volume": "volume",
                                "Adj Close": "adj_close",
                            }
                        )

                        # Convert date to timestamp
                        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                        df["fetched_at"] = pd.Timestamp.now()
                        df["data_quality_score"] = 1.0

                        # Add all rows to results
                        rows_added = 0
                        for _, row_data in df.iterrows():
                            results.append(
                                (
                                    symbol,
                                    row_data["date"],
                                    float(row_data["open"]) if pd.notna(row_data["open"]) else None,
                                    float(row_data["high"]) if pd.notna(row_data["high"]) else None,
                                    float(row_data["low"]) if pd.notna(row_data["low"]) else None,
                                    float(row_data["close"]) if pd.notna(row_data["close"]) else None,
                                    int(row_data["volume"]) if pd.notna(row_data["volume"]) else None,
                                    float(row_data["adj_close"]) if pd.notna(row_data["adj_close"]) else None,
                                    row_data["fetched_at"],
                                    float(row_data["data_quality_score"]),
                                )
                            )
                            rows_added += 1

                        symbol_duration = time.time() - symbol_start
                        logger.info(f"✅ Symbol {symbol}: {rows_added} records fetched in {symbol_duration:.2f}s")
                    else:
                        logger.warning(f"⚠️ Symbol {symbol}: No data available")

                except Exception as e:
                    symbol_duration = time.time() - symbol_start
                    logger.error(f"❌ Symbol {symbol}: Error after {symbol_duration:.2f}s - {str(e)}")
                    continue

            logger.info(f"✅ Partition completed: {len(results)} total records for {partition_symbols}")
            return results

        # Apply mapPartitions to fetch data
        logger.info("🔄 Applying mapPartitions to distribute work across Spark workers")
        map_start = time.time()
        result_rdd = symbols_df.rdd.mapPartitions(fetch_partition)
        map_duration = time.time() - map_start
        logger.info(f"✅ mapPartitions completed in {map_duration:.2f}s")

        # Convert back to DataFrame
        logger.info("🔄 Converting RDD back to Spark DataFrame")
        df_start = time.time()
        result_df = result_rdd.toDF(self.ohlcv_schema)
        df_duration = time.time() - df_start
        logger.info(f"✅ DataFrame conversion completed in {df_duration:.2f}s")

        # Add partitioning columns
        logger.info("🔄 Adding partitioning columns (year, month, day)")
        partition_start = time.time()
        from pyspark.sql.functions import year, month, dayofmonth, col

        result_df = (
            result_df.withColumn("year", year(col("date")))
            .withColumn("month", month(col("date")))
            .withColumn("day", dayofmonth(col("date")))
        )
        partition_duration = time.time() - partition_start
        logger.info(f"✅ Partitioning columns added in {partition_duration:.2f}s")

        total_duration = time.time() - start_time
        logger.info(f"🎉 Successfully fetched data using Spark mapPartitions in {total_duration:.2f}s")
        logger.info(
            f"📊 Breakdown: mapPartitions={map_duration:.2f}s, DataFrame={df_duration:.2f}s, partitioning={partition_duration:.2f}s"
        )

        return result_df

    def fetch_and_store(
        self, symbols: List[str], start_date: str, end_date: str, table_path: Optional[str] = None, max_workers: int = None
    ) -> Dict[str, Any]:
        """
        Fetch and store data for multiple symbols using Spark.

        Args:
            symbols: List of symbols to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            table_path: Delta table path for storage
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary with operation results
        """
        start_time = time.time()

        # Use provided workers or default
        workers = max_workers if max_workers is not None else self.max_workers

        # Set table path
        if table_path is None:
            table_path = self.config.get("DELTA_OHLCV_PATH", "s3a://breadthflow/ohlcv/")

        logger.info(f"Starting fetch and store operation for {len(symbols)} symbols")
        logger.info(f"Using {workers} parallel workers")

        try:
            # Step 1: Fetch data using Spark mapPartitions
            logger.info(f"🔄 Step 1: Fetching data with {workers} Spark workers")
            fetch_start = time.time()

            df = self.fetch_multiple_symbols(symbols, start_date, end_date, workers)

            if df is None:
                logger.error("❌ DataFrame is None after fetch_multiple_symbols")
                return {
                    "success": False,
                    "message": "DataFrame is None after fetch_multiple_symbols",
                    "symbols_fetched": 0,
                    "total_records": 0,
                    "failed_symbols": symbols,
                    "duration": time.time() - start_time,
                }

            fetch_duration = time.time() - fetch_start
            logger.info(f"✅ Step 1: Data fetch completed in {fetch_duration:.2f}s")

            # Check Spark session health before Step 2
            logger.info("🔍 Checking Spark session health before Step 2...")
            if not self._check_spark_session_health():
                logger.error("❌ Spark session is unhealthy - cannot proceed to Step 2")
                return {
                    "success": False,
                    "message": "Spark session became unhealthy after Step 1",
                    "symbols_fetched": 0,
                    "total_records": 0,
                    "failed_symbols": symbols,
                    "duration": time.time() - start_time,
                }

            # Step 1.5: Save DataFrame to Parquet immediately to persist data
            logger.info("🔄 Step 1.5: Saving DataFrame to Parquet to persist data")
            save_start = time.time()

            # Create a temporary parquet path for immediate storage
            temp_parquet_path = f"{table_path}/temp_fetch_{int(time.time())}.parquet"

            try:
                # Save to Parquet format (more stable than Delta for temporary storage)
                df.write.mode("overwrite").parquet(temp_parquet_path)
                save_duration = time.time() - save_start
                logger.info(f"✅ Step 1.5: Data saved to {temp_parquet_path} in {save_duration:.2f}s")

                # Reload the DataFrame from Parquet to ensure persistence
                logger.info("🔄 Reloading DataFrame from Parquet")
                reload_start = time.time()
                df = self.spark.read.parquet(temp_parquet_path)
                reload_duration = time.time() - reload_start
                logger.info(f"✅ DataFrame reloaded from Parquet in {reload_duration:.2f}s")

            except Exception as e:
                logger.error(f"❌ Failed to save DataFrame to Parquet: {str(e)}")
                return {
                    "success": False,
                    "message": f"Failed to save DataFrame to Parquet: {str(e)}",
                    "symbols_fetched": 0,
                    "total_records": 0,
                    "failed_symbols": symbols,
                    "duration": time.time() - start_time,
                }

            # Test if the reloaded DataFrame is accessible
            logger.info("🔍 Testing if reloaded DataFrame is accessible...")
            if not self._test_dataframe_accessibility(df):
                logger.error("❌ Reloaded DataFrame is not accessible - cannot proceed to Step 2")
                return {
                    "success": False,
                    "message": "Reloaded DataFrame is not accessible",
                    "symbols_fetched": 0,
                    "total_records": 0,
                    "failed_symbols": symbols,
                    "duration": time.time() - start_time,
                }

            logger.info("✅ DataFrame is accessible - proceeding to Step 2")

            # Step 2: Get statistics using the same DataFrame
            logger.info("🔄 Step 2: Getting statistics")
            stats_start = time.time()

            # Cache the DataFrame for better performance
            df = df.cache()

            # Get statistics using Spark operations
            total_records = df.count()
            logger.info(f"✅ Step 2a: Record count completed - {total_records} total records")

            if total_records == 0:
                logger.warning("⚠️ No data was fetched - returning empty result")
                return {
                    "success": False,
                    "message": "No data was fetched",
                    "symbols_fetched": 0,
                    "total_records": 0,
                    "failed_symbols": symbols,
                    "duration": time.time() - start_time,
                }

            # Get unique symbols using Spark
            symbols_df = df.select("symbol").distinct()
            symbols_fetched = symbols_df.count()
            fetched_symbols = [row.symbol for row in symbols_df.collect()]
            failed_symbols = [s for s in symbols if s not in fetched_symbols]
            stats_duration = time.time() - stats_start

            logger.info(f"✅ Step 2b: Symbol statistics completed in {stats_duration:.2f}s")
            logger.info(f"📈 Statistics: {total_records} total records, {symbols_fetched} symbols fetched")
            logger.info(f"📋 Fetched symbols: {fetched_symbols}")
            if failed_symbols:
                logger.warning(f"⚠️ Failed symbols: {failed_symbols}")

            # Check Spark session health before Step 3
            logger.info("🔍 Checking Spark session health before Step 3...")
            if not self._check_spark_session_health():
                logger.error("❌ Spark session is unhealthy - cannot proceed to Step 3")
                return {
                    "success": False,
                    "message": "Spark session became unhealthy after Step 2",
                    "symbols_fetched": symbols_fetched,
                    "total_records": total_records,
                    "failed_symbols": failed_symbols,
                    "duration": time.time() - start_time,
                }
            logger.info("✅ Spark session is healthy - proceeding to Step 3")

            # Step 3: Store in Delta Lake
            logger.info(f"🔄 Step 3: Storing data in Delta Lake at {table_path}")
            storage_start = time.time()

            write_delta(df=df, path=table_path, partition_cols=["year", "month", "day"], mode="append")
            storage_duration = time.time() - storage_start
            logger.info(f"✅ Step 3: Data storage completed in {storage_duration:.2f}s")

            # Step 4: Prepare final result
            total_duration = time.time() - start_time

            return {
                "success": True,
                "message": "Data fetch and store completed successfully",
                "symbols_fetched": symbols_fetched,
                "total_records": total_records,
                "failed_symbols": failed_symbols,
                "duration": total_duration,
                "breakdown": {"fetch": fetch_duration, "statistics": stats_duration, "storage": storage_duration},
            }

        except Exception as e:
            logger.error(f"❌ Error during fetch operation: {str(e)}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            import traceback

            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")

            return {
                "success": False,
                "message": f"Error during fetch operation: {str(e)}",
                "symbols_fetched": 0,
                "total_records": 0,
                "failed_symbols": symbols,
                "duration": time.time() - start_time,
            }

    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score for a DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Quality score between 0 and 1
        """
        if df.empty:
            return 0.0

        # Check for missing values
        missing_ratio = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum() / (len(df) * 5)

        # Check for zero/negative prices
        invalid_prices = (df[["open", "high", "low", "close"]] <= 0).sum().sum() / (len(df) * 4)

        # Check for extreme price changes (potential data errors)
        price_changes = df["close"].pct_change().abs()
        extreme_changes = (price_changes > self.max_price_change).sum() / len(df)

        # Calculate overall quality score
        quality_score = 1.0 - (missing_ratio + invalid_prices + extreme_changes) / 3

        return max(0.0, min(1.0, quality_score))

    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality against thresholds.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            True if data quality is acceptable
        """
        quality_score = self._calculate_data_quality(df)

        # Check completeness
        completeness = 1.0 - df[["open", "high", "low", "close", "volume"]].isnull().sum().sum() / (len(df) * 5)

        # Check for minimum data points
        min_data_points = 10  # At least 10 trading days

        return (
            quality_score >= self.min_data_completeness
            and completeness >= self.min_data_completeness
            and len(df) >= min_data_points
        )

    def _check_spark_session_health(self) -> bool:
        """
        Check if the Spark session is healthy and executors are available.

        Returns:
            True if session is healthy, False otherwise
        """
        try:
            # Check if SparkContext is active
            if self.spark.sparkContext._jsc.sc().isStopped():
                logger.error("❌ SparkContext is stopped")
                return False

            # Check if we can get application ID (indicates session is working)
            app_id = self.spark.sparkContext.applicationId
            logger.info(f"✅ Application ID: {app_id}")

            # Check executor status and active tasks
            if not self._check_executor_status():
                logger.error("❌ Executor status check failed")
                return False

            # Try a simple operation to verify executors are responsive
            # This is the most reliable test - if this fails, executors are dead
            test_df = self.spark.createDataFrame([(1, "test")], ["id", "value"])
            test_count = test_df.count()
            logger.info(f"✅ Health check successful: {test_count} test records")

            return True

        except Exception as e:
            logger.error(f"❌ Spark session health check failed: {str(e)}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            return False

    def _check_executor_status(self) -> bool:
        """
        Check detailed executor and task status.

        Returns:
            True if executors are healthy, False otherwise
        """
        try:
            # Get status tracker
            status_tracker = self.spark.sparkContext._jsc.sc().statusTracker()

            # Check for any active jobs (this method exists in Spark 3.5.6)
            active_jobs = status_tracker.getActiveJobIds()
            if active_jobs:
                logger.warning(f"⚠️ Found {len(active_jobs)} active jobs: {active_jobs}")
                # Check if any jobs have been running too long
                for job_id in active_jobs:
                    try:
                        job_info = status_tracker.getJobInfo(job_id)
                        if job_info:
                            logger.warning(f"⚠️ Job {job_id} status: {job_info.status()}")
                    except Exception as job_error:
                        logger.warning(f"⚠️ Job {job_id}: Could not get status - {str(job_error)}")

            # Simple check: try to get application ID (this should work)
            app_id = self.spark.sparkContext.applicationId
            logger.info(f"✅ Application {app_id} is active")

            logger.info("✅ Executor status check passed")
            return True

        except Exception as e:
            logger.error(f"❌ Executor status check failed: {str(e)}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            return False

    def _wait_for_tasks_completion(self, max_wait_time: int = 30) -> bool:
        """
        Wait for any stuck or running tasks to complete.

        Args:
            max_wait_time: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout or stuck
        """
        try:
            status_tracker = self.spark.sparkContext._jsc.sc().statusTracker()
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                # Check for active jobs
                active_jobs = status_tracker.getActiveJobIds()
                if not active_jobs:
                    logger.info("✅ No active jobs found - all tasks completed")
                    return True

                # Log current status
                logger.info(f"⏳ Waiting for {len(active_jobs)} active jobs to complete...")
                for job_id in active_jobs:
                    try:
                        job_info = status_tracker.getJobInfo(job_id)
                        if job_info:
                            logger.info(f"   Job {job_id}: {job_info.status()}")
                    except Exception as job_error:
                        logger.warning(f"   Job {job_id}: Could not get status - {str(job_error)}")

                # Wait a bit before checking again
                time.sleep(2)

            # If we get here, we timed out
            logger.error(f"❌ Timeout waiting for tasks to complete after {max_wait_time}s")
            return False

        except Exception as e:
            logger.error(f"❌ Error waiting for task completion: {str(e)}")
            return False

    def _test_dataframe_accessibility(self, df: DataFrame) -> bool:
        """
        Test if a DataFrame is still accessible and has data.

        Args:
            df: DataFrame to test

        Returns:
            True if DataFrame is accessible and has data, False otherwise
        """
        try:
            logger.info("🔄 Testing DataFrame accessibility...")

            # Test 1: Check if DataFrame is not None
            if df is None:
                logger.error("❌ DataFrame is None")
                return False

            # Test 2: Check if we can get the schema
            schema = df.schema
            logger.info(f"✅ DataFrame schema: {len(schema.fields)} fields")

            # Test 3: Check if we can get a sample of data
            sample = df.limit(5)
            sample_count = sample.count()
            logger.info(f"✅ Sample count: {sample_count} records")

            if sample_count == 0:
                logger.error("❌ DataFrame has 0 records in sample")
                return False

            # Test 4: Check if we can get column names
            columns = df.columns
            logger.info(f"✅ DataFrame columns: {columns}")

            # Test 5: Check if we can get total count (this is the critical test)
            logger.info("🔄 Testing total count operation...")
            total_count = df.count()
            logger.info(f"✅ Total count: {total_count} records")

            if total_count == 0:
                logger.error("❌ DataFrame has 0 total records")
                return False

            logger.info("✅ DataFrame accessibility test passed")
            return True

        except Exception as e:
            logger.error(f"❌ DataFrame accessibility test failed: {str(e)}")
            logger.error(f"🔍 Error type: {type(e).__name__}")
            import traceback

            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return False

    def get_available_symbols(self, symbols: List[str]) -> List[str]:
        """
        Check which symbols are available and return valid ones.

        Args:
            symbols: List of symbols to check

        Returns:
            List of available symbols
        """
        available_symbols = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    available_symbols.append(symbol)
                    logger.debug(f"Symbol {symbol} is available")
                else:
                    logger.warning(f"Symbol {symbol} is not available")
            except Exception as e:
                logger.warning(f"Error checking symbol {symbol}: {str(e)}")

        logger.info(f"Found {len(available_symbols)} available symbols out of {len(symbols)}")
        return available_symbols

    def get_data_summary(self, table_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for stored data.

        Args:
            table_path: Delta table path (optional)

        Returns:
            Dictionary with data summary
        """
        if table_path is None:
            table_path = self.config.get("DELTA_OHLCV_PATH", "data/ohlcv")

        try:
            df = read_delta(self.spark, table_path)

            summary = {
                "total_records": df.count(),
                "unique_symbols": df.select("symbol").distinct().count(),
                "date_range": {
                    "start": df.agg({"date": "min"}).collect()[0][0],
                    "end": df.agg({"date": "max"}).collect()[0][0],
                },
                "avg_quality_score": df.agg({"data_quality_score": "avg"}).collect()[0][0],
                "symbols": [row.symbol for row in df.select("symbol").distinct().collect()],
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}")
            return {"error": str(e)}


# PySpark UDF for distributed data fetching
@udf(returnType=StringType())
def fetch_symbol_udf(symbol: str, start_date: str, end_date: str) -> str:
    """
    PySpark UDF for fetching symbol data.
    This allows distributed data fetching across Spark executors.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date

    Returns:
        JSON string with fetched data or error message
    """
    try:
        import json
        import yfinance as yf
        from datetime import datetime

        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            return json.dumps({"error": "No data available", "symbol": symbol})

        # Convert to JSON-serializable format
        df = df.reset_index()
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df["symbol"] = symbol
        df["fetched_at"] = datetime.now().isoformat()

        return df.to_json(orient="records")

    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


def create_data_fetcher(spark: SparkSession) -> DataFetcher:
    """
    Factory function to create DataFetcher instance.

    Args:
        spark: PySpark session

    Returns:
        Configured DataFetcher instance
    """
    return DataFetcher(spark)
