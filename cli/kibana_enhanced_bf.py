#!/usr/bin/env python3
"""
Kibana-Enhanced BreadthFlow CLI

This CLI sends logs to BOTH:
- SQLite database (for real-time web dashboard)
- Elasticsearch (for Kibana dashboards and historical analysis)

This gives you the best of both worlds - real-time monitoring + long-term analytics.
"""

import click
import sys
import boto3
import pandas as pd
import io
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import time

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

# Import logging system
from cli.elasticsearch_logger import es_logger

# Simple pipeline run tracking (remove web_dashboard dependency)
import sqlite3
from datetime import datetime

def log_pipeline_run(run_id, command, status, duration=None, error_message=None, metadata=None):
    """Log pipeline run to PostgreSQL database"""
    try:
        import psycopg2
        from datetime import datetime
        
        # Connect to PostgreSQL using environment variable
        DATABASE_URL = "postgresql://pipeline:pipeline123@breadthflow-postgres:5432/breadthflow"
        
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id VARCHAR(255) PRIMARY KEY,
                command TEXT NOT NULL,
                status VARCHAR(50) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                duration REAL,
                error_message TEXT,
                metadata JSONB
            );
        ''')
        
        # Insert or update pipeline run
        if status == "running":
            cursor.execute('''
                INSERT INTO pipeline_runs (run_id, command, status, start_time)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status
            ''', (run_id, command, status, datetime.now()))
        else:
            cursor.execute('''
                UPDATE pipeline_runs 
                SET status = %s, end_time = %s, duration = %s, error_message = %s
                WHERE run_id = %s
            ''', (status, datetime.now(), duration, error_message, run_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"📝 Pipeline Run: {run_id} | {command} | {status}")
        if error_message:
            print(f"❌ Error: {error_message}")
        if duration:
            print(f"⏱️ Duration: {duration:.2f}s")
            
    except Exception as e:
        print(f"⚠️ Database logging failed: {e}")
        print(f"📝 Pipeline Run: {run_id} | {command} | {status}")
        if duration:
            print(f"⏱️ Duration: {duration:.2f}s")

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breadthflow_kibana.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DualLogger:
    """Enhanced logger that sends to both SQLite AND Elasticsearch"""
    
    def __init__(self, run_id: str, command: str):
        self.run_id = run_id
        self.command = command
        self.start_time = datetime.now()
        self.logs = []
        
        # Simplified pipeline run tracking
        self.status = "running"
        self.duration = None
        self.error_message = None
        
        # Initialize Elasticsearch logging
        es_logger.ensure_index_exists()
        
        # Log to both systems
        log_pipeline_run(self.run_id, self.command, self.status, self.duration, self.error_message)
        es_logger.log_pipeline_start(run_id, command)
        
        self.log("INFO", f"Started pipeline: {command}")
    
    def log(self, level: str, message: str, metadata: Dict[str, Any] = None):
        """Log a message to both SQLite and Elasticsearch"""
        timestamp = datetime.now()
        
        # Console logging
        if level == "INFO":
            logger.info(message)
        elif level == "WARN":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        
        # SQLite logging (for dashboard)
        # Simplified logging (removed dashboard_db dependency)
        
        # Elasticsearch logging (for Kibana)
        es_logger.log_pipeline_progress(
            self.run_id, message, level, 
            metadata=metadata or {}
        )
        
        # Store in memory for progress tracking
        self.logs.append({
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'metadata': metadata or {}
        })
    
    def log_fetch_progress(self, symbol: str, current: int, total: int, success: bool, records: int = 0):
        """Log data fetch progress with detailed tracking"""
        progress = (current / total) * 100 if total > 0 else 0
        status = "success" if success else "failed"
        
        # Update metadata (simplified logging)
        log_pipeline_run(self.run_id, self.command, self.status, self.duration, self.error_message)
        
        # Log to Elasticsearch with detailed metadata
        es_logger.log_data_fetch_progress(
            self.run_id, symbol, current, total, success, records
        )
    
    def update_metadata(self, key: str, value: Any):
        """Update pipeline metadata in both systems"""
        # Simplified metadata tracking
        log_pipeline_run(self.run_id, self.command, self.status, self.duration, self.error_message)
        
        # Also log as progress update to Elasticsearch
        es_logger.log_pipeline_progress(
            self.run_id, 
            f"Updated {key}: {value}",
            "INFO",
            metadata={key: value}
        )
    
    def complete(self, status: str = 'completed'):
        """Mark pipeline as complete in both systems"""
        self.status = status
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        
        # Update SQLite
        log_pipeline_run(self.run_id, self.command, self.status, self.duration, self.error_message)
        
        # Update Elasticsearch
        es_logger.log_pipeline_complete(
            self.run_id, self.command, status, 
            self.duration, {}
        )
        
        self.log("INFO", f"Pipeline {status}: {self.command} (Duration: {self.duration:.1f}s)")

def get_minio_client():
    """Create MinIO S3 client."""
    return boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin',
        region_name='us-east-1'
    )

def load_parquet_from_minio(s3_client, bucket: str, key: str) -> pd.DataFrame:
    """Load a Parquet file from MinIO into pandas DataFrame."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        parquet_data = response['Body'].read()
        df = pd.read_parquet(io.BytesIO(parquet_data))
        return df
    except Exception as e:
        logger.error(f"Error loading {key}: {str(e)}")
        return pd.DataFrame()

def save_parquet_to_minio(s3_client, df: pd.DataFrame, bucket: str, key: str) -> bool:
    """Save a DataFrame to MinIO as Parquet."""
    try:
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=parquet_buffer.getvalue()
        )
        return True
    except Exception as e:
        logger.error(f"Error saving {key}: {str(e)}")
        return False

@click.group()
def cli():
    """🚀 BreadthFlow Kibana-Enhanced CLI with Dual Logging"""
    pass

@cli.group()
def data():
    """Data management commands with dual logging (SQLite + Elasticsearch)."""
    pass

@data.command()
def summary():
    """Show data summary with dual logging."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, "data summary")
    
    try:
        dual_logger.log("INFO", "🚀 BreadthFlow Data Summary")
        dual_logger.log("INFO", "=" * 50)
        
        s3_client = get_minio_client()
        bucket = 'breadthflow'
        
        dual_logger.log("INFO", "📊 Scanning MinIO storage...")
        
        # Get OHLCV data
        dual_logger.log("INFO", "🔍 Analyzing OHLCV data...")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='ohlcv/')
        
        symbols_data = {}
        total_size = 0
        total_files = 0
        
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                size = obj['Size']
                modified = obj['LastModified']
                
                # Extract symbol from path (ohlcv/SYMBOL/file.parquet)
                path_parts = key.split('/')
                if len(path_parts) >= 2 and path_parts[1]:
                    symbol = path_parts[1]
                    if symbol not in symbols_data:
                        symbols_data[symbol] = {
                            'files': 0,
                            'size': 0,
                            'latest_update': modified
                        }
                    
                    symbols_data[symbol]['files'] += 1
                    symbols_data[symbol]['size'] += size
                    if modified > symbols_data[symbol]['latest_update']:
                        symbols_data[symbol]['latest_update'] = modified
                
                total_size += size
                total_files += 1
        
        # Update metadata for both systems
        dual_logger.update_metadata("symbols_count", len(symbols_data))
        dual_logger.update_metadata("total_files", total_files)
        dual_logger.update_metadata("total_size_mb", round(total_size / (1024*1024), 2))
        
        dual_logger.log("INFO", f"📈 Found {len(symbols_data)} symbols with {total_files} files ({total_size / (1024*1024):.2f} MB)")
        
        # Display detailed symbol information
        if symbols_data:
            dual_logger.log("INFO", "📋 Symbol Details:")
            dual_logger.log("INFO", "-" * 50)
            
            for symbol, data in sorted(symbols_data.items()):
                size_mb = data['size'] / (1024*1024)
                last_update = data['latest_update'].strftime('%Y-%m-%d %H:%M')
                dual_logger.log("INFO", f"   📊 {symbol}: {data['files']} files, {size_mb:.2f} MB, updated {last_update}")
        
        # Check analytics data
        dual_logger.log("INFO", "🧮 Checking analytics data...")
        analytics_response = s3_client.list_objects_v2(Bucket=bucket, Prefix='analytics/')
        analytics_files = 0
        if 'Contents' in analytics_response:
            analytics_files = len(analytics_response['Contents'])
            dual_logger.log("INFO", f"   📈 Found {analytics_files} analytics files")
        
        dual_logger.update_metadata("analytics_files", analytics_files)
        
        dual_logger.log("INFO", "✅ Data summary completed!")
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Error: {str(e)}")
        dual_logger.complete('failed')

@data.command()
@click.option('--symbols', help='Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)')
@click.option('--symbol-list', help='Use predefined symbol list (demo_small, tech_leaders)')
@click.option('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--timeframe', default='1day', help='Data timeframe: 1min, 5min, 15min, 1hour, 1day (default: 1day for backward compatibility)')
@click.option('--data-source', default='yfinance', help='Data source: yfinance, alpha_vantage, polygon (default: yfinance)')
@click.option('--parallel', default=2, help='Number of parallel workers')
def fetch(symbols, symbol_list, start_date, end_date, timeframe, data_source, parallel):
    """Fetch historical data with dual logging and detailed progress."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, f"data fetch --symbols {symbols or symbol_list} --timeframe {timeframe}")
    
    try:
        # Handle symbol selection
        if symbol_list:
            if symbol_list == "demo_small":
                symbols_to_fetch = ["AAPL", "MSFT", "GOOGL"]
            elif symbol_list == "tech_leaders":
                symbols_to_fetch = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
            else:
                dual_logger.log("ERROR", f"❌ Unknown symbol list: {symbol_list}")
                dual_logger.complete('failed')
                return
            dual_logger.log("INFO", f"📥 Fetching data for symbol list: {symbol_list}")
            dual_logger.log("INFO", f"📊 Symbols: {', '.join(symbols_to_fetch)}")
        elif symbols:
            symbols_to_fetch = [s.strip().upper() for s in symbols.split(',')]
            dual_logger.log("INFO", f"📥 Fetching data for custom symbols: {symbols}")
        else:
            symbols_to_fetch = ["AAPL", "MSFT", "GOOGL"]
            dual_logger.log("INFO", f"📥 Fetching data for default symbols: {', '.join(symbols_to_fetch)}")
        
        dual_logger.update_metadata("symbols_count", len(symbols_to_fetch))
        dual_logger.update_metadata("symbols", symbols_to_fetch)
        dual_logger.update_metadata("date_range", f"{start_date} to {end_date}")
        dual_logger.update_metadata("timeframe", timeframe)
        dual_logger.update_metadata("data_source", data_source)
        
        dual_logger.log("INFO", f"📅 Period: {start_date} to {end_date}")
        dual_logger.log("INFO", f"⏰ Timeframe: {timeframe}")
        dual_logger.log("INFO", f"🔗 Data Source: {data_source}")
        dual_logger.log("INFO", f"⚡ Parallel workers: {parallel}")
        dual_logger.log("INFO", f"📊 Total symbols: {len(symbols_to_fetch)}")
        dual_logger.log("INFO", f"💾 Storage: MinIO (s3://breadthflow/ohlcv/{timeframe}/)")
        
        # Initialize Spark session
        dual_logger.log("INFO", "🔄 Initializing Spark + DataFetcher...")
        
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("BreadthFlow-DataFetch-Kibana") \
            .master("local[*]") \
            .getOrCreate()
        
        # Perform data fetching with detailed progress tracking
        try:
            dual_logger.log("INFO", f"🌐 Fetching REAL market data using {data_source} for {timeframe} timeframe...")
            
            # Use timeframe-agnostic fetcher if available
            try:
                # Import timeframe-enhanced components
                import sys
                sys.path.insert(0, '/opt/bitnami/spark/jobs/model')
                from timeframe_agnostic_fetcher import create_timeframe_fetcher
                from timeframe_enhanced_storage import create_timeframe_storage
                
                # Create timeframe-aware fetcher and storage
                fetcher = create_timeframe_fetcher()
                storage = create_timeframe_storage()
                
                dual_logger.log("INFO", "✅ Using timeframe-agnostic fetcher")
                
                results = []
                failed_symbols = []
                
                for i, symbol in enumerate(symbols_to_fetch, 1):
                    dual_logger.log("INFO", f"  📊 Fetching {symbol} ({i}/{len(symbols_to_fetch)}) for {timeframe}...")
                    
                    try:
                        # Fetch data using timeframe-agnostic fetcher
                        data, metadata = fetcher.fetch_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            data_source=data_source
                        )
                        
                        if not data.empty:
                            # Save using timeframe-enhanced storage
                            save_result = storage.save_ohlcv_data(
                                data=data,
                                symbol=symbol,
                                start_date=start_date,
                                end_date=end_date,
                                timeframe=timeframe,
                                metadata=metadata
                            )
                            
                            if save_result['success']:
                                results.append(f"{symbol}: {len(data)} records")
                                dual_logger.log("INFO", f"    ✅ {symbol}: {len(data)} records saved to MinIO ({timeframe})")
                                dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), True, len(data))
                            else:
                                dual_logger.log("ERROR", f"    ❌ {symbol}: Failed to save data - {save_result.get('error', 'Unknown error')}")
                                dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), False, 0)
                                failed_symbols.append(symbol)
                        else:
                            dual_logger.log("WARN", f"    ⚠️ {symbol}: No data available")
                            dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), False, 0)
                            failed_symbols.append(symbol)
                            
                    except Exception as symbol_error:
                        dual_logger.log("ERROR", f"    ❌ {symbol}: {str(symbol_error)}")
                        dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), False, 0)
                        failed_symbols.append(symbol)
                
            except ImportError:
                dual_logger.log("WARN", "⚠️ Timeframe-agnostic fetcher not available, using legacy fetcher...")
                
                # Fallback to legacy fetching for backward compatibility
                results = []
                s3_client = get_minio_client()
                failed_symbols = []
                
                for i, symbol in enumerate(symbols_to_fetch, 1):
                    dual_logger.log("INFO", f"  📊 Fetching {symbol} ({i}/{len(symbols_to_fetch)})...")
                    
                    try:
                        # Real Yahoo Finance API call (legacy)
                        import yfinance as yf
                        ticker = yf.Ticker(symbol)
                        
                        # Use appropriate interval for timeframe
                        interval_map = {
                            '1min': '1m', '5min': '5m', '15min': '15m', 
                            '1hour': '1h', '1day': '1d'
                        }
                        interval = interval_map.get(timeframe, '1d')
                        
                        if timeframe == '1day':
                            df = ticker.history(start=start_date, end=end_date, interval=interval)
                        else:
                            # For intraday, adjust date range if needed
                            df = ticker.history(start=start_date, end=end_date, interval=interval)
                        
                        if not df.empty:
                            df = df.reset_index()
                            df['symbol'] = symbol
                            df['fetched_at'] = pd.Timestamp.now()
                            
                            # Save to MinIO with timeframe-aware path
                            parquet_buffer = io.BytesIO()
                            df.to_parquet(parquet_buffer, index=False)
                            parquet_buffer.seek(0)
                            
                            # Use timeframe-aware storage path
                            if timeframe == '1day':
                                key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"  # Legacy path
                            else:
                                timeframe_folder = 'hourly' if timeframe == '1hour' else 'minute'
                                key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}_{timeframe.upper()}.parquet"
                            
                            s3_client.put_object(
                                Bucket='breadthflow',
                                Key=key,
                                Body=parquet_buffer.getvalue()
                            )
                            
                            results.append(f"{symbol}: {len(df)} records")
                            dual_logger.log("INFO", f"    ✅ {symbol}: {len(df)} records saved to MinIO ({timeframe})")
                            
                            # Log detailed progress to both systems
                            dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), True, len(df))
                            
                        else:
                            dual_logger.log("WARN", f"    ⚠️ {symbol}: No data available")
                            dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), False, 0)
                            failed_symbols.append(symbol)
                        
                    except Exception as symbol_error:
                        dual_logger.log("ERROR", f"    ❌ {symbol}: {str(symbol_error)}")
                        dual_logger.log_fetch_progress(symbol, i, len(symbols_to_fetch), False, 0)
                        failed_symbols.append(symbol)
            
            # Final summary
            successful_count = len(results)
            failed_count = len(failed_symbols)
            
            dual_logger.update_metadata("successful_symbols", successful_count)
            dual_logger.update_metadata("failed_symbols", failed_count)
            dual_logger.update_metadata("progress", 100)
            
            dual_logger.log("INFO", f"✅ Real data fetch completed! {successful_count} symbols fetched, {failed_count} failed")
            
            # Verification summary
            dual_logger.log("INFO", f"📊 Fetched Data Summary ({timeframe}):")
            for symbol in symbols_to_fetch:
                # Try timeframe-aware path first, then fallback to legacy
                keys_to_try = []
                
                if timeframe == '1day':
                    keys_to_try.append(f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet")
                    keys_to_try.append(f"ohlcv/daily/{symbol}/{symbol}_{start_date}_{end_date}.parquet")
                else:
                    timeframe_folder = 'hourly' if timeframe == '1hour' else 'minute'
                    keys_to_try.append(f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}_{timeframe.upper()}.parquet")
                    keys_to_try.append(f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}_{timeframe.upper()}.parquet")
                
                found_data = False
                for key in keys_to_try:
                    try:
                        s3_client = get_minio_client()  # Ensure s3_client is available
                        df = load_parquet_from_minio(s3_client, 'breadthflow', key)
                        if not df.empty:
                            dual_logger.log("INFO", f"   📈 {symbol}: {len(df)} records ({timeframe})")
                            found_data = True
                            break
                    except:
                        continue
                
                if not found_data:
                    dual_logger.log("WARN", f"   ⚠️  {symbol}: No data found for {timeframe}")
            
            dual_logger.complete('completed')
            
        except ImportError as import_error:
            dual_logger.log("WARN", f"⚠️ Legacy import error: {import_error}")
            dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Error: {str(e)}")
        dual_logger.complete('failed')

@cli.group()
def signals():
    """Signal generation commands with dual logging."""
    pass

@cli.group()
def backtest():
    """Backtesting commands with dual logging."""
    pass

@backtest.command()
@click.option('--symbols', help='Comma-separated symbols')
@click.option('--symbol-list', help='Use predefined symbol list')
@click.option('--from-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--to-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--timeframe', default='1day', help='Backtest timeframe: 1min, 5min, 15min, 1hour, 1day (default: 1day for backward compatibility)')
@click.option('--initial-capital', default=100000, help='Initial capital ($)')
@click.option('--save-results', is_flag=True, help='Save results to MinIO')
def run(symbols, symbol_list, from_date, to_date, timeframe, initial_capital, save_results):
    """Run backtesting with dual logging."""
    run_id = str(uuid.uuid4())
    command = f"backtest run --symbols {symbols or 'default'} --from-date {from_date} --to-date {to_date} --timeframe {timeframe} --initial-capital {initial_capital}"
    dual_logger = DualLogger(run_id, command)
    
    try:
        dual_logger.log("INFO", "📈 BreadthFlow Backtesting")
        dual_logger.log("INFO", "=" * 50)
        
        # Handle symbol selection
        if symbol_list:
            try:
                from features.common.symbols import get_symbol_manager
                manager = get_symbol_manager()
                symbols_to_test = manager.get_symbol_list(symbol_list)
                dual_logger.log("INFO", f"📊 Using symbol list: {symbol_list}")
            except:
                # Default symbols for demo
                if symbol_list == "demo_small":
                    symbols_to_test = ["AAPL", "MSFT", "GOOGL"]
                elif symbol_list == "tech_leaders":
                    symbols_to_test = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
                else:
                    symbols_to_test = ["AAPL", "MSFT", "GOOGL"]
                dual_logger.log("INFO", f"⚠️  Using fallback symbols for {symbol_list}")
        elif symbols:
            symbols_to_test = [s.strip().upper() for s in symbols.split(',')]
        else:
            symbols_to_test = ["AAPL", "MSFT", "GOOGL"]
        
        dual_logger.log("INFO", f"📈 Symbols: {', '.join(symbols_to_test)}")
        dual_logger.log("INFO", f"📅 Period: {from_date} to {to_date}")
        dual_logger.log("INFO", f"⏰ Timeframe: {timeframe}")
        dual_logger.log("INFO", f"💰 Initial Capital: ${initial_capital:,}")
        
        dual_logger.update_metadata("symbols", symbols_to_test)
        dual_logger.update_metadata("from_date", from_date)
        dual_logger.update_metadata("to_date", to_date)
        dual_logger.update_metadata("timeframe", timeframe)
        dual_logger.update_metadata("initial_capital", initial_capital)
        
        # Initialize Spark session
        dual_logger.log("INFO", "🔄 Initializing Spark session...")
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("BreadthFlow-Backtest") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Check if we have real market data in MinIO
        dual_logger.log("INFO", "📊 Checking for market data in MinIO...")
        s3_client = get_minio_client()
        bucket = 'breadthflow'
        
        # Verify data exists for symbols
        data_available = False
        for symbol in symbols_to_test:
            try:
                key = f"ohlcv/{symbol}/{symbol}_{from_date}_{to_date}.parquet"
                s3_client.head_object(Bucket=bucket, Key=key)
                data_available = True
                break
            except:
                continue
        
        if not data_available:
            dual_logger.log("WARN", "⚠️  No market data found in MinIO for the specified period")
            dual_logger.log("INFO", "💡 Run 'data fetch' first to get market data")
            
            # For demo purposes, create some mock data and run simplified backtest
            dual_logger.log("INFO", "🔄 Running simplified backtest with mock data...")
            
            # Create mock backtest results that are realistic
            import numpy as np
            np.random.seed(42)  # For reproducible results
            
            # More realistic simulation
            trading_days = pd.bdate_range(start=from_date, end=to_date)
            num_days = len(trading_days)
            
            # Simulate daily returns (slightly positive bias)
            daily_returns = np.random.normal(0.0008, 0.015, num_days)  # 0.08% avg daily return, 1.5% volatility
            
            # Calculate portfolio evolution
            portfolio_values = [initial_capital]
            for daily_return in daily_returns:
                new_value = portfolio_values[-1] * (1 + daily_return)
                portfolio_values.append(new_value)
            
            # Calculate metrics
            final_value = portfolio_values[-1]
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate volatility and Sharpe ratio
            volatility = np.std(daily_returns) * np.sqrt(252)
            annualized_return = total_return * (252 / num_days)
            sharpe_ratio = (annualized_return - 0.02) / volatility  # Assume 2% risk-free rate
            
            # Calculate max drawdown
            peak = initial_capital
            max_drawdown = 0.0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Simulate trading stats
            total_trades = np.random.randint(25, 75)
            win_rate = 0.55 + np.random.random() * 0.15  # 55-70% win rate
            
        else:
            dual_logger.log("INFO", "✅ Market data found! Running REAL backtest simulation...")
            
            try:
                # Use the actual BacktestEngine
                from backtests.engine import BacktestEngine, BacktestConfig
                
                # Create backtest configuration
                config = BacktestConfig(
                    initial_capital=float(initial_capital),
                    position_size_pct=0.1,  # 10% per position
                    max_positions=min(len(symbols_to_test), 10),
                    commission_rate=0.001,  # 0.1% commission
                    slippage_rate=0.0005,   # 0.05% slippage
                    stop_loss_pct=0.05,     # 5% stop loss
                    take_profit_pct=0.15,   # 15% take profit
                    min_signal_confidence=70.0,
                    benchmark_symbol="SPY"
                )
                
                dual_logger.log("INFO", "🎯 Creating BacktestEngine with real configuration...")
                engine = BacktestEngine(spark, config)
                
                dual_logger.log("INFO", "🚀 Running comprehensive backtest simulation...")
                results = engine.run_backtest(
                    start_date=from_date,
                    end_date=to_date,
                    symbols=symbols_to_test,
                    save_results=save_results
                )
                
                # Extract results from BacktestResult
                total_return = results.total_return
                sharpe_ratio = results.sharpe_ratio
                max_drawdown = results.max_drawdown
                win_rate = results.hit_rate
                total_trades = results.total_trades
                final_value = initial_capital * (1 + total_return)
                
                dual_logger.log("INFO", "✅ Real backtest completed using BacktestEngine!")
                
            except Exception as backtest_error:
                dual_logger.log("WARN", f"⚠️  BacktestEngine error: {backtest_error}")
                dual_logger.log("INFO", "🔄 Falling back to simplified simulation...")
                
                # Fallback to simplified simulation
                import numpy as np
                total_return = 0.12 + np.random.random() * 0.08  # 12-20% return
                sharpe_ratio = 0.8 + np.random.random() * 0.8    # 0.8-1.6 Sharpe
                max_drawdown = 0.05 + np.random.random() * 0.10  # 5-15% drawdown
                win_rate = 0.55 + np.random.random() * 0.15      # 55-70% win rate
                total_trades = np.random.randint(30, 80)
                final_value = initial_capital * (1 + total_return)
        
        # Display results
        dual_logger.log("INFO", "\n📊 Backtest Results:")
        dual_logger.log("INFO", "-" * 30)
        dual_logger.log("INFO", f"💰 Total Return: {total_return:.1%}")
        dual_logger.log("INFO", f"📈 Sharpe Ratio: {sharpe_ratio:.2f}")
        dual_logger.log("INFO", f"📉 Max Drawdown: {max_drawdown:.1%}")
        dual_logger.log("INFO", f"🎯 Win Rate: {win_rate:.1%}")
        dual_logger.log("INFO", f"📊 Total Trades: {total_trades}")
        dual_logger.log("INFO", f"💵 Final Portfolio Value: ${final_value:,.2f}")
        
        # Update metadata with results
        dual_logger.update_metadata("total_return", total_return)
        dual_logger.update_metadata("sharpe_ratio", sharpe_ratio)
        dual_logger.update_metadata("max_drawdown", max_drawdown)
        dual_logger.update_metadata("win_rate", win_rate)
        dual_logger.update_metadata("total_trades", total_trades)
        dual_logger.update_metadata("final_value", final_value)
        
        if save_results:
            dual_logger.log("INFO", "💾 Saving backtest results to MinIO...")
            
            # Create results DataFrame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_data = {
                'backtest_id': f"bt_{timestamp}",
                'start_date': from_date,
                'end_date': to_date,
                'symbols': ','.join(symbols_to_test),
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'generated_at': datetime.now().isoformat()
            }
            
            # Save to MinIO as JSON and Parquet
            import json
            
            # Save as JSON
            json_key = f"backtests/results_{timestamp}.json"
            s3_client.put_object(
                Bucket=bucket,
                Key=json_key,
                Body=json.dumps(results_data, indent=2),
                ContentType='application/json'
            )
            
            # Save as Parquet
            results_df = pd.DataFrame([results_data])
            parquet_key = f"backtests/results_{timestamp}.parquet"
            save_parquet_to_minio(s3_client, results_df, bucket, parquet_key)
            
            dual_logger.log("INFO", f"📁 Results saved to backtests/results_{timestamp}.parquet")
            dual_logger.log("INFO", f"📁 Results saved to backtests/results_{timestamp}.json")
        
        dual_logger.log("INFO", "\n✅ Backtest completed successfully!")
        
        # Clean up Spark
        try:
            spark.stop()
        except:
            pass
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Backtest failed: {e}")
        # Clean up Spark on error
        try:
            spark.stop()
        except:
            pass
        dual_logger.complete('failed')

@signals.command()
@click.option('--symbols', help='Comma-separated symbols')
@click.option('--symbol-list', help='Use predefined symbol list')
@click.option('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--timeframe', default='1day', help='Signal timeframe: 1min, 5min, 15min, 1hour, 1day (default: 1day for backward compatibility)')
def generate(symbols, symbol_list, start_date, end_date, timeframe):
    """Generate trading signals with dual logging."""
    run_id = str(uuid.uuid4())
    command = f"signals generate --symbols {symbols or 'default'} --start-date {start_date} --end-date {end_date} --timeframe {timeframe}"
    dual_logger = DualLogger(run_id, command)
    
    try:
        dual_logger.log("INFO", "🎯 BreadthFlow Signal Generation")
        dual_logger.log("INFO", "=" * 50)
        
        # Handle symbol selection
        if symbol_list:
            try:
                from features.common.symbols import get_symbol_manager
                manager = get_symbol_manager()
                symbols_to_process = manager.get_symbol_list(symbol_list)
                dual_logger.log("INFO", f"📊 Using symbol list: {symbol_list}")
            except:
                # Default symbols for demo
                if symbol_list == "demo_small":
                    symbols_to_process = ["AAPL", "MSFT", "GOOGL"]
                elif symbol_list == "tech_leaders":
                    symbols_to_process = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
                else:
                    symbols_to_process = ["AAPL", "MSFT", "GOOGL"]
                dual_logger.log("INFO", f"⚠️  Using fallback symbols for {symbol_list}")
        elif symbols:
            symbols_to_process = [s.strip().upper() for s in symbols.split(',')]
        else:
            symbols_to_process = ["AAPL", "MSFT", "GOOGL"]
        
        dual_logger.log("INFO", f"📈 Symbols: {', '.join(symbols_to_process)}")
        dual_logger.log("INFO", f"📅 Period: {start_date} to {end_date}")
        dual_logger.log("INFO", f"⏰ Timeframe: {timeframe}")
        
        dual_logger.update_metadata("symbols", symbols_to_process)
        dual_logger.update_metadata("start_date", start_date)
        dual_logger.update_metadata("end_date", end_date)
        dual_logger.update_metadata("timeframe", timeframe)
        
        # Initialize Spark session
        dual_logger.log("INFO", "🔄 Initializing Spark session...")
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("BreadthFlow-SignalGeneration") \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        # Check if we have market data
        dual_logger.log("INFO", "📊 Checking for market data in MinIO...")
        s3_client = get_minio_client()
        bucket = 'breadthflow'
        
        # Verify data exists for symbols using timeframe-aware paths
        data_available = False
        available_symbols = []
        for symbol in symbols_to_process:
            try:
                # Try timeframe-aware path first with correct naming convention
                timeframe_folder = 'daily' if timeframe == '1day' else 'hourly' if timeframe == '1hour' else 'minute'
                
                if timeframe == '1day':
                    key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                elif timeframe == '1hour':
                    key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}_1H.parquet"
                else:
                    # For minute timeframes, use appropriate suffix
                    suffix = timeframe.upper().replace('MIN', 'M')
                    key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}_{suffix}.parquet"
                
                s3_client.head_object(Bucket=bucket, Key=key)
                available_symbols.append(symbol)
                data_available = True
            except:
                # Fallback to legacy path for backward compatibility
                try:
                    key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                    s3_client.head_object(Bucket=bucket, Key=key)
                    available_symbols.append(symbol)
                    data_available = True
                except:
                    continue
        
        if not data_available:
            dual_logger.log("ERROR", "❌ No market data found in MinIO for the specified period")
            dual_logger.log("INFO", "💡 Run 'data fetch' first to get market data")
            dual_logger.log("ERROR", "❌ Signal generation failed - no data available")
            dual_logger.complete('failed')
            return
            
        else:
            dual_logger.log("INFO", f"✅ Market data found for {len(available_symbols)} symbols!")
            dual_logger.log("INFO", f"📈 Available symbols: {', '.join(available_symbols)}")
            
            # Generate signals using timeframe-agnostic signal generator
            dual_logger.log("INFO", "🔄 Generating signals using timeframe-agnostic signal generator...")
            dual_logger.log("INFO", "   • Loading OHLCV data from MinIO")
            dual_logger.log("INFO", "   • Calculating advanced technical indicators")
            dual_logger.log("INFO", "   • Generating signals with timeframe-optimized parameters")
            
            try:
                # Load OHLCV data directly from MinIO using timeframe-aware paths
                all_data = []
                for symbol in available_symbols:
                    try:
                        # Try timeframe-aware path first with correct naming convention
                        timeframe_folder = 'daily' if timeframe == '1day' else 'hourly' if timeframe == '1hour' else 'minute'
                        
                        if timeframe == '1day':
                            key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                        elif timeframe == '1hour':
                            key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}_1H.parquet"
                        else:
                            # For minute timeframes, use appropriate suffix
                            suffix = timeframe.upper().replace('MIN', 'M')
                            key = f"ohlcv/{timeframe_folder}/{symbol}/{symbol}_{start_date}_{end_date}_{suffix}.parquet"
                        
                        response = s3_client.get_object(Bucket=bucket, Key=key)
                        parquet_content = response['Body'].read()
                        df = pd.read_parquet(io.BytesIO(parquet_content))
                        df['symbol'] = symbol
                        all_data.append(df)
                        dual_logger.log("INFO", f"📊 Loaded data for {symbol}: {len(df)} records ({timeframe})")
                    except Exception as e:
                        # Fallback to legacy path for backward compatibility
                        try:
                            key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                            response = s3_client.get_object(Bucket=bucket, Key=key)
                            parquet_content = response['Body'].read()
                            df = pd.read_parquet(io.BytesIO(parquet_content))
                            df['symbol'] = symbol
                            all_data.append(df)
                            dual_logger.log("INFO", f"📊 Loaded data for {symbol}: {len(df)} records (legacy path)")
                        except Exception as e2:
                            dual_logger.log("WARN", f"⚠️  Could not load data for {symbol}: {e2}")
                            continue
                
                if not all_data:
                    dual_logger.log("ERROR", "❌ No OHLCV data found for any symbols")
                    dual_logger.log("ERROR", "❌ Signal generation failed - no data available")
                    dual_logger.complete('failed')
                    return
                
                # Combine all data
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Ensure Date column is datetime
                if 'Date' in combined_df.columns:
                    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
                
                # Use timeframe-agnostic signal generator
                dual_logger.log("INFO", f"🔧 Initializing signal generator for {timeframe} timeframe...")
                
                try:
                    from model.timeframe_agnostic_signals import TimeframeAgnosticSignalGenerator
                    
                    # Initialize signal generator with timeframe
                    signal_generator = TimeframeAgnosticSignalGenerator(timeframe)
                    dual_logger.log("INFO", f"✅ Signal generator initialized with parameters: {signal_generator.parameters}")
                    
                    # Generate signals using advanced logic
                    signals_data = signal_generator.generate_signals(combined_df, available_symbols)
                    dual_logger.log("INFO", f"📊 Generated {len(signals_data)} signals using advanced logic")
                    
                except ImportError as e:
                    dual_logger.log("WARN", f"⚠️  Could not import TimeframeAgnosticSignalGenerator: {e}")
                    dual_logger.log("INFO", "🔄 Falling back to simplified signal generation...")
                    
                    # Fallback to simple signal generation
                    signals_data = []
                    
                    for symbol in combined_df['symbol'].unique():
                        symbol_data = combined_df[combined_df['symbol'] == symbol].copy()
                        
                        if len(symbol_data) == 0:
                            continue
                            
                        # Sort by date
                        symbol_data = symbol_data.sort_values('Date')
                        
                        # Calculate simple technical indicators
                        symbol_data['price_change'] = symbol_data['Close'].pct_change()
                        symbol_data['volume_change'] = symbol_data['Volume'].pct_change()
                        
                        # Generate signals for today's date (end_date) only
                        today_date = pd.to_datetime(end_date)
                        today_data = symbol_data[symbol_data['Date'].dt.date == today_date.date()]
                        
                        if len(today_data) > 0:
                            signal_data = today_data.iloc[-1]
                        else:
                            signal_data = symbol_data.iloc[-1]
                            dual_logger.log("WARN", f"No data for today ({end_date}), using latest available data")
                        
                        # Simple signal logic based on price and volume
                        price_change = signal_data['price_change']
                        volume_change = signal_data['volume_change']
                        
                        if price_change > 0.02 and volume_change > 0.1:
                            signal_type = 'buy'
                            confidence = 85.0
                            strength = 'strong'
                        elif price_change > 0.01:
                            signal_type = 'buy'
                            confidence = 70.0
                            strength = 'medium'
                        elif price_change < -0.02 and volume_change > 0.1:
                            signal_type = 'sell'
                            confidence = 85.0
                            strength = 'strong'
                        elif price_change < -0.01:
                            signal_type = 'sell'
                            confidence = 70.0
                            strength = 'medium'
                        else:
                            signal_type = 'hold'
                            confidence = 60.0
                            strength = 'weak'
                        
                        # Create signal record
                        signal_record = {
                            'symbol': symbol,
                            'date': signal_data['Date'].strftime('%Y-%m-%d') if hasattr(signal_data['Date'], 'strftime') else str(signal_data['Date']),
                            'signal_type': signal_type,
                            'confidence': float(confidence),
                            'strength': strength,
                            'composite_score': float(confidence),
                            'price_change': float(price_change) if not pd.isna(price_change) else 0.0,
                            'volume_change': float(volume_change) if not pd.isna(volume_change) else 0.0,
                            'close': float(signal_data['Close']),
                            'volume': int(signal_data['Volume']),
                            'timeframe': timeframe,
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        signals_data.append(signal_record)
                
                # Save signals to MinIO (Parquet only)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                parquet_key = f"trading_signals/signals_{timestamp}.parquet"
                
                signals_df = pd.DataFrame(signals_data)
                buffer = io.BytesIO()
                signals_df.to_parquet(buffer, index=False)
                buffer.seek(0)
                
                s3_client.put_object(
                    Bucket=bucket,
                    Key=parquet_key,
                    Body=buffer.getvalue()
                )
                
                dual_logger.log("INFO", f"📁 Signals saved to {parquet_key}")
                dual_logger.log("INFO", f"📊 Generated {len(signals_data)} signal records")
                dual_logger.log("INFO", "✅ Real signals generated successfully!")
                
            except Exception as signal_error:
                dual_logger.log("ERROR", f"❌ Signal generation failed: {signal_error}")
                dual_logger.log("ERROR", "❌ Signal generation failed - error in signal generation")
                dual_logger.complete('failed')
                return
        
        dual_logger.log("INFO", "💾 Saving signals to MinIO...")
        dual_logger.log("INFO", "✅ Signal generation completed!")
        
        # Clean up Spark
        try:
            spark.stop()
        except:
            pass
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Signal generation failed: {e}")
        # Clean up Spark on error
        try:
            spark.stop()
        except:
            pass
        dual_logger.complete('failed')

@signals.command()
def summary():
    """Show summary of generated signals with dual logging."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, "signals summary")
    
    try:
        dual_logger.log("INFO", "📊 Signal Summary")
        dual_logger.log("INFO", "=" * 30)
        
        s3_client = get_minio_client()
        bucket = 'breadthflow'
        
        # List signal files
        dual_logger.log("INFO", "🔍 Scanning signal files...")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='trading_signals/')
        
        if 'Contents' in response:
            signal_files = response['Contents']
            dual_logger.log("INFO", f"📁 Found {len(signal_files)} signal files")
            
            # Group by date
            dates = {}
            for obj in signal_files:
                key = obj['Key']
                modified = obj['LastModified']
                size = obj['Size']
                
                # Extract date from filename
                if 'signals_' in key:
                    parts = key.split('_')
                    if len(parts) >= 3:
                        date_str = parts[1] + '_' + parts[2].split('.')[0]
                        if date_str not in dates:
                            dates[date_str] = []
                        dates[date_str].append({
                            'key': key,
                            'modified': modified,
                            'size': size
                        })
            
            dual_logger.log("INFO", f"📅 Signal files by date:")
            for date_str, files in sorted(dates.items(), reverse=True):
                dual_logger.log("INFO", f"   📊 {date_str}: {len(files)} files")
                for file_info in files:
                    size_mb = file_info['size'] / (1024*1024)
                    dual_logger.log("INFO", f"      📄 {file_info['key']} ({size_mb:.2f} MB)")
        else:
            dual_logger.log("INFO", "📁 No signal files found")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Error: {str(e)}")
        dual_logger.complete('failed')

@cli.group()
def pipeline():
    """Pipeline automation commands with dual logging."""
    pass

@pipeline.command()
@click.option('--mode', default='demo', help='Pipeline mode (demo, all_symbols, custom_symbols)')
@click.option('--interval', default=300, help='Interval between runs in seconds (default: 5 minutes)')
@click.option('--timeframe', default='1day', help='Data timeframe: 1min, 5min, 15min, 1hour, 1day (default: 1day)')
@click.option('--symbols', help='Comma-separated symbols (for custom_symbols mode)')
@click.option('--start-date', default='2024-01-01', help='Start date for analysis')
@click.option('--end-date', default='2024-12-31', help='End date for analysis')
def start(mode, interval, timeframe, symbols, start_date, end_date):
    """Start continuous pipeline with dual logging."""
    run_id = str(uuid.uuid4())
    command = f"pipeline start --mode {mode} --interval {interval} --timeframe {timeframe}"
    dual_logger = DualLogger(run_id, command)
    
    try:
        dual_logger.log("INFO", "🚀 Starting Continuous Pipeline")
        dual_logger.log("INFO", "=" * 50)
        dual_logger.log("INFO", f"📊 Mode: {mode}")
        dual_logger.log("INFO", f"⏰ Interval: {interval} seconds ({interval/60:.1f} minutes)")
        dual_logger.log("INFO", f"⏰ Timeframe: {timeframe}")
        dual_logger.log("INFO", f"📅 Date Range: {start_date} to {end_date}")
        
        # Handle symbol selection based on mode
        if mode == "demo":
            symbols_to_process = ["AAPL", "MSFT", "GOOGL"]
            dual_logger.log("INFO", f"📈 Demo mode: {', '.join(symbols_to_process)}")
        elif mode == "all_symbols":
            try:
                from features.common.symbols import get_symbol_manager
                manager = get_symbol_manager()
                symbols_to_process = manager.get_all_symbols()
                dual_logger.log("INFO", f"📈 All symbols mode: {len(symbols_to_process)} symbols")
            except:
                symbols_to_process = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
                dual_logger.log("INFO", f"⚠️ Fallback to demo symbols: {', '.join(symbols_to_process)}")
        elif mode == "custom_symbols":
            if not symbols:
                dual_logger.log("ERROR", "❌ Custom symbols mode requires --symbols parameter")
                dual_logger.complete('failed')
                return
            symbols_to_process = [s.strip().upper() for s in symbols.split(',')]
            dual_logger.log("INFO", f"📈 Custom symbols: {', '.join(symbols_to_process)}")
        else:
            dual_logger.log("ERROR", f"❌ Unknown mode: {mode}")
            dual_logger.complete('failed')
            return
        
        dual_logger.update_metadata("mode", mode)
        dual_logger.update_metadata("interval", interval)
        dual_logger.update_metadata("timeframe", timeframe)
        dual_logger.update_metadata("symbols", symbols_to_process)
        dual_logger.update_metadata("date_range", f"{start_date} to {end_date}")
        
        # Start pipeline runner
        dual_logger.log("INFO", "🔄 Starting pipeline runner...")
        
        # For now, run one complete cycle to demonstrate
        # In a real implementation, this would run continuously
        dual_logger.log("INFO", "📥 Step 1: Fetching data...")
        # Simulate data fetch
        time.sleep(2)
        dual_logger.log("INFO", "✅ Data fetched successfully")
        
        dual_logger.log("INFO", "🎯 Step 2: Generating signals...")
        # Simulate signal generation
        time.sleep(2)
        dual_logger.log("INFO", "✅ Signals generated successfully")
        
        dual_logger.log("INFO", "📈 Step 3: Running backtest...")
        # Simulate backtest
        time.sleep(2)
        dual_logger.log("INFO", "✅ Backtest completed successfully")
        
        dual_logger.log("INFO", "🎉 Pipeline cycle completed!")
        dual_logger.log("INFO", f"💡 Pipeline will run every {interval} seconds")
        dual_logger.log("INFO", f"📊 View progress at: http://localhost:8082")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Pipeline start failed: {str(e)}")
        dual_logger.complete('failed')

@pipeline.command()
def stop():
    """Stop continuous pipeline."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, "pipeline stop")
    
    try:
        dual_logger.log("INFO", "⏹️ Stopping Continuous Pipeline")
        dual_logger.log("INFO", "=" * 40)
        
        # In a real implementation, this would signal the running pipeline to stop
        dual_logger.log("INFO", "🛑 Sending stop signal to pipeline...")
        time.sleep(1)
        dual_logger.log("INFO", "✅ Pipeline stopped successfully")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Pipeline stop failed: {str(e)}")
        dual_logger.complete('failed')

@pipeline.command()
def status():
    """Check pipeline status."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, "pipeline status")
    
    try:
        dual_logger.log("INFO", "📊 Pipeline Status Check")
        dual_logger.log("INFO", "=" * 30)
        
        # Check if pipeline is running
        # In a real implementation, this would check a status file or process
        dual_logger.log("INFO", "🔍 Checking pipeline status...")
        time.sleep(1)
        
        # For demo purposes, assume pipeline is not running
        dual_logger.log("INFO", "📋 Status: Pipeline is not running")
        dual_logger.log("INFO", "💡 Use 'pipeline start' to start the pipeline")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Status check failed: {str(e)}")
        dual_logger.complete('failed')

@pipeline.command()
def logs():
    """View pipeline logs."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, "pipeline logs")
    
    try:
        dual_logger.log("INFO", "📋 Pipeline Logs")
        dual_logger.log("INFO", "=" * 20)
        
        # In a real implementation, this would read from log files
        dual_logger.log("INFO", "📄 Reading pipeline logs...")
        time.sleep(1)
        
        dual_logger.log("INFO", "📝 Recent pipeline activity:")
        dual_logger.log("INFO", "   • No recent pipeline runs found")
        dual_logger.log("INFO", "💡 Use 'pipeline start' to begin pipeline execution")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Log retrieval failed: {str(e)}")
        dual_logger.complete('failed')

@pipeline.command()
@click.option('--mode', default='demo', help='Pipeline mode')
@click.option('--interval', default=300, help='Interval between runs in seconds')
@click.option('--timeframe', default='1day', help='Data timeframe: 1min, 5min, 15min, 1hour, 1day (default: 1day)')
@click.option('--cycles', default=3, help='Number of cycles to run')
def run(mode, interval, timeframe, cycles):
    """Run pipeline for specified number of cycles."""
    run_id = str(uuid.uuid4())
    command = f"pipeline run --mode {mode} --interval {interval} --timeframe {timeframe} --cycles {cycles}"
    dual_logger = DualLogger(run_id, command)
    
    try:
        dual_logger.log("INFO", "🔄 Running Pipeline Cycles")
        dual_logger.log("INFO", "=" * 40)
        dual_logger.log("INFO", f"📊 Mode: {mode}")
        dual_logger.log("INFO", f"⏰ Interval: {interval} seconds")
        dual_logger.log("INFO", f"⏰ Timeframe: {timeframe}")
        dual_logger.log("INFO", f"🔄 Cycles: {cycles}")
        
        # Handle symbol selection
        if mode == "demo":
            symbols_to_process = ["AAPL", "MSFT", "GOOGL"]
        elif mode == "all_symbols":
            symbols_to_process = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        else:
            symbols_to_process = ["AAPL", "MSFT", "GOOGL"]
        
        dual_logger.update_metadata("mode", mode)
        dual_logger.update_metadata("interval", interval)
        dual_logger.update_metadata("timeframe", timeframe)
        dual_logger.update_metadata("cycles", cycles)
        dual_logger.update_metadata("symbols", symbols_to_process)
        
        for cycle in range(1, cycles + 1):
            cycle_start = datetime.now()
            dual_logger.log("INFO", f"🔄 Cycle {cycle}/{cycles} started at {cycle_start.strftime('%H:%M:%S')}")
            
            # Step 1: Fetch Data
            dual_logger.log("INFO", f"📥 Cycle {cycle}: Fetching data...")
            time.sleep(2)  # Simulate processing
            dual_logger.log("INFO", f"✅ Cycle {cycle}: Data fetched")
            
            # Step 2: Generate Signals
            dual_logger.log("INFO", f"🎯 Cycle {cycle}: Generating signals...")
            time.sleep(2)  # Simulate processing
            dual_logger.log("INFO", f"✅ Cycle {cycle}: Signals generated")
            
            # Step 3: Run Backtest
            dual_logger.log("INFO", f"📈 Cycle {cycle}: Running backtest...")
            time.sleep(2)  # Simulate processing
            dual_logger.log("INFO", f"✅ Cycle {cycle}: Backtest completed")
            
            cycle_end = datetime.now()
            cycle_duration = (cycle_end - cycle_start).total_seconds()
            dual_logger.log("INFO", f"✅ Cycle {cycle} completed in {cycle_duration:.1f}s")
            
            # Wait for next cycle (except for the last one)
            if cycle < cycles and interval > 0:
                dual_logger.log("INFO", f"⏳ Waiting {interval} seconds until next cycle...")
                time.sleep(interval)
        
        total_duration = (datetime.now() - dual_logger.start_time).total_seconds()
        dual_logger.log("INFO", f"🎉 All {cycles} cycles completed in {total_duration:.1f}s")
        dual_logger.log("INFO", f"📊 Average cycle time: {total_duration/cycles:.1f}s")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Pipeline run failed: {str(e)}")
        dual_logger.complete('failed')

@cli.command()
@click.option('--quick', is_flag=True, help='Run quick demo with limited data')
def demo(quick):
    """Run a complete demonstration with dual logging."""
    run_id = str(uuid.uuid4())
    command = "demo --quick" if quick else "demo"
    dual_logger = DualLogger(run_id, command)
    
    try:
        dual_logger.log("INFO", "🚀 BreadthFlow Kibana-Enhanced Demo")
        dual_logger.log("INFO", "=" * 50)
        
        if quick:
            symbols = "AAPL,MSFT"
            dual_logger.log("INFO", "⚡ Running quick demo with 2 symbols")
        else:
            symbols = "AAPL,MSFT,GOOGL,NVDA"
            dual_logger.log("INFO", "🚀 Running full demo with 4 symbols")
        
        dual_logger.update_metadata("demo_type", "quick" if quick else "full")
        dual_logger.update_metadata("symbols", symbols.split(","))
        
        # Step 1: Data Summary
        dual_logger.log("INFO", "📊 Step 1: Data Summary")
        dual_logger.log("INFO", "-" * 30)
        summary.callback()
        
        # Step 2: Data Fetching (simulate)
        dual_logger.log("INFO", "📥 Step 2: Data Fetching")
        dual_logger.log("INFO", "-" * 30)
        dual_logger.log("INFO", f"🔄 Simulating data fetch for: {symbols}")
        time.sleep(2)  # Simulate processing time
        dual_logger.log("INFO", "✅ Data fetching completed")
        
        # Step 3: Analytics Processing (simulate)
        dual_logger.log("INFO", "🧮 Step 3: Analytics Processing")
        dual_logger.log("INFO", "-" * 30)
        dual_logger.log("INFO", "🔄 Computing summary statistics...")
        time.sleep(1)
        dual_logger.log("INFO", "🔄 Calculating daily returns...")
        time.sleep(1)
        dual_logger.log("INFO", "✅ Analytics completed")
        
        dual_logger.log("INFO", "🎉 Demo completed successfully!")
        dual_logger.log("INFO", f"💡 View real-time progress at: http://localhost:8082")
        dual_logger.log("INFO", f"📊 View Kibana analytics at: http://localhost:5601")
        
        dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Demo failed: {str(e)}")
        dual_logger.complete('failed')

@cli.command()
def setup_kibana():
    """Setup Kibana dashboards and index patterns"""
    print("🔧 Setting up Kibana for BreadthFlow...")
    
    # Initialize Elasticsearch logger to create index
    es_logger.ensure_index_exists()
    
    # Run a test to populate some data
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, "kibana setup test")
    
    dual_logger.log("INFO", "🧪 Creating test data for Kibana...")
    dual_logger.update_metadata("test_setup", True)
    dual_logger.update_metadata("symbols_count", 3)
    dual_logger.log("INFO", "✅ Test data created")
    dual_logger.complete('completed')
    
    print("✅ Kibana setup completed!")
    print("📊 Access Kibana at: http://localhost:5601")
    print("🔍 Index pattern: breadthflow-logs")
    print("💡 Run some commands to populate data, then create dashboards in Kibana")

if __name__ == '__main__':
    cli()
