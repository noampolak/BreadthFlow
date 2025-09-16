"""
Data Ingestion Service for ML Pipeline

Handles data fetching from various sources and storing in MinIO
with proper organization for ML training.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_timestamp, dayofmonth, month, year

logger = logging.getLogger(__name__)


class DataIngestionService:
    """
    Service for ingesting market data and storing in MinIO for ML training.

    Features:
    - Fetches data from Yahoo Finance
    - Stores data in MinIO with proper partitioning
    - Supports multiple symbol lists
    - Handles data quality validation
    - Provides data lineage tracking
    """

    def __init__(self, spark: SparkSession, minio_client=None):
        """
        Initialize the data ingestion service.

        Args:
            spark: Spark session for data processing
            minio_client: MinIO client for object storage
        """
        self.spark = spark
        self.minio_client = minio_client
        self.bucket_name = "breadthflow-ml-data"

        # Data schema for OHLCV
        self.ohlcv_schema = {
            "symbol": "string",
            "date": "timestamp",
            "open": "double",
            "high": "double",
            "low": "double",
            "close": "double",
            "volume": "long",
            "adj_close": "double",
            "fetched_at": "timestamp",
            "data_quality_score": "double",
        }

        logger.info("DataIngestionService initialized")

    def load_symbol_lists(self, symbols_file: str = "/data/symbols.json") -> Dict[str, List[str]]:
        """
        Load symbol lists from JSON file.

        Args:
            symbols_file: Path to symbols JSON file

        Returns:
            Dictionary of symbol lists
        """
        try:
            with open(symbols_file, "r") as f:
                data = json.load(f)

            symbol_lists = {}
            for key, value in data["symbols"].items():
                symbol_lists[key] = value["symbols"]

            logger.info(f"Loaded {len(symbol_lists)} symbol lists from {symbols_file}")
            return symbol_lists

        except Exception as e:
            logger.error(f"Error loading symbol lists: {str(e)}")
            return {}

    def fetch_symbol_data(self, symbol: str, start_date: str, end_date: str, retry_count: int = 0) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Stock symbol
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

            # Ensure date column is properly formatted
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

            # Add metadata columns
            df["symbol"] = symbol
            df["fetched_at"] = datetime.now()
            df["data_quality_score"] = self._calculate_data_quality(df)

            logger.debug(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            if retry_count < 3:
                logger.info(f"Retrying {symbol} (attempt {retry_count + 1})")
                return self.fetch_symbol_data(symbol, start_date, end_date, retry_count + 1)
            return None

    def fetch_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str) -> DataFrame:
        """
        Fetch data for multiple symbols using Spark.

        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            PySpark DataFrame with all fetched data
        """
        logger.info(f"Starting data fetch for {len(symbols)} symbols")
        start_time = datetime.now()

        # Create a DataFrame with symbols to parallelize
        symbols_df = self.spark.createDataFrame([(symbol,) for symbol in symbols], ["symbol"])

        def fetch_partition(partition):
            """Fetch data for symbols in this partition."""
            from datetime import datetime

            import pandas as pd
            import yfinance as yf

            results = []

            for row in partition:
                symbol = row.symbol
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

                        logger.info(f"✅ Fetched {len(df)} records for {symbol}")
                    else:
                        logger.warning(f"⚠️ No data for {symbol}")

                except Exception as e:
                    logger.error(f"❌ Error fetching {symbol}: {str(e)}")
                    continue

            return results

        # Apply mapPartitions to fetch data
        result_rdd = symbols_df.rdd.mapPartitions(fetch_partition)

        # Convert back to DataFrame
        from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType, TimestampType

        schema = StructType(
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

        result_df = result_rdd.toDF(schema)

        # Add partitioning columns
        result_df = (
            result_df.withColumn("year", year(col("date")))
            .withColumn("month", month(col("date")))
            .withColumn("day", dayofmonth(col("date")))
        )

        duration = datetime.now() - start_time
        logger.info(f"✅ Data fetch completed in {duration.total_seconds():.2f}s")

        return result_df

    def store_data_in_minio(self, df: DataFrame, data_type: str = "ohlcv", symbol_list: str = "default") -> Dict[str, Any]:
        """
        Store DataFrame in MinIO with proper organization.

        Args:
            df: DataFrame to store
            data_type: Type of data (ohlcv, features, etc.)
            symbol_list: Name of symbol list used

        Returns:
            Dictionary with storage results
        """
        try:
            # Create path structure: data_type/symbol_list/year/month/day/
            timestamp = datetime.now()
            base_path = f"{data_type}/{symbol_list}/{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}"

            # Convert DataFrame to Parquet format
            parquet_data = df.toPandas()

            # Store in MinIO
            if self.minio_client:
                # Create bucket if it doesn't exist
                if not self.minio_client.bucket_exists(self.bucket_name):
                    self.minio_client.make_bucket(self.bucket_name)

                # Store as Parquet file
                file_path = f"{base_path}/data_{timestamp.strftime('%H%M%S')}.parquet"
                parquet_bytes = parquet_data.to_parquet()

                self.minio_client.put_object(self.bucket_name, file_path, parquet_bytes, len(parquet_bytes))

                logger.info(f"✅ Data stored in MinIO: {file_path}")

                return {"success": True, "file_path": file_path, "records": len(parquet_data), "bucket": self.bucket_name}
            else:
                # Fallback to local storage
                local_path = f"data/processed/{base_path}"
                Path(local_path).mkdir(parents=True, exist_ok=True)

                file_path = f"{local_path}/data_{timestamp.strftime('%H%M%S')}.parquet"
                parquet_data.to_parquet(file_path)

                logger.info(f"✅ Data stored locally: {file_path}")

                return {"success": True, "file_path": file_path, "records": len(parquet_data), "storage_type": "local"}

        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
            return {"success": False, "error": str(e)}

    def ingest_symbol_list(self, symbol_list_name: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Ingest data for a specific symbol list.

        Args:
            symbol_list_name: Name of symbol list to ingest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dictionary with ingestion results
        """
        try:
            # Load symbol lists
            symbol_lists = self.load_symbol_lists()

            if symbol_list_name not in symbol_lists:
                return {"success": False, "error": f"Symbol list '{symbol_list_name}' not found"}

            symbols = symbol_lists[symbol_list_name]
            logger.info(f"Starting ingestion for {len(symbols)} symbols in '{symbol_list_name}'")

            # Fetch data
            df = self.fetch_multiple_symbols(symbols, start_date, end_date)

            if df.count() == 0:
                return {"success": False, "error": "No data fetched"}

            # Store data
            storage_result = self.store_data_in_minio(df, "ohlcv", symbol_list_name)

            if not storage_result["success"]:
                return storage_result

            # Get statistics
            total_records = df.count()
            unique_symbols = df.select("symbol").distinct().count()

            return {
                "success": True,
                "symbol_list": symbol_list_name,
                "symbols_requested": len(symbols),
                "symbols_fetched": unique_symbols,
                "total_records": total_records,
                "storage": storage_result,
            }

        except Exception as e:
            logger.error(f"Error ingesting symbol list: {str(e)}")
            return {"success": False, "error": str(e)}

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

        # Check for extreme price changes
        price_changes = df["close"].pct_change().abs()
        extreme_changes = (price_changes > 0.5).sum() / len(df)  # 50% change threshold

        # Calculate overall quality score
        quality_score = 1.0 - (missing_ratio + invalid_prices + extreme_changes) / 3

        return max(0.0, min(1.0, quality_score))

    def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get status of data ingestion service.

        Returns:
            Dictionary with service status
        """
        try:
            # Check Spark session
            spark_status = "healthy" if self.spark.sparkContext._jsc.sc().isStopped() == False else "stopped"

            # Check MinIO connection
            minio_status = "connected" if self.minio_client else "not_configured"

            return {
                "spark_status": spark_status,
                "minio_status": minio_status,
                "bucket_name": self.bucket_name,
                "schema": self.ohlcv_schema,
            }

        except Exception as e:
            return {"error": str(e), "spark_status": "error", "minio_status": "error"}
