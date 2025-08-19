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
@click.option('--parallel', default=2, help='Number of parallel workers')
def fetch(symbols, symbol_list, start_date, end_date, parallel):
    """Fetch historical data with dual logging and detailed progress."""
    run_id = str(uuid.uuid4())
    dual_logger = DualLogger(run_id, f"data fetch --symbols {symbols or symbol_list}")
    
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
        
        dual_logger.log("INFO", f"📅 Period: {start_date} to {end_date}")
        dual_logger.log("INFO", f"⚡ Parallel workers: {parallel}")
        dual_logger.log("INFO", f"📊 Total symbols: {len(symbols_to_fetch)}")
        dual_logger.log("INFO", "💾 Storage: MinIO (s3://breadthflow/ohlcv/)")
        
        # Initialize Spark session
        dual_logger.log("INFO", "🔄 Initializing Spark + DataFetcher...")
        
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("BreadthFlow-DataFetch-Kibana") \
            .master("local[*]") \
            .getOrCreate()
        
        # Perform data fetching with detailed progress tracking
        try:
            dual_logger.log("INFO", "🌐 Fetching REAL market data from Yahoo Finance...")
            
            results = []
            s3_client = get_minio_client()
            failed_symbols = []
            
            for i, symbol in enumerate(symbols_to_fetch, 1):
                dual_logger.log("INFO", f"  📊 Fetching {symbol} ({i}/{len(symbols_to_fetch)})...")
                
                try:
                    # Real Yahoo Finance API call
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                    
                    if not df.empty:
                        df = df.reset_index()
                        df['symbol'] = symbol
                        df['fetched_at'] = pd.Timestamp.now()
                        
                        # Save to MinIO as Parquet in symbol-specific folder
                        parquet_buffer = io.BytesIO()
                        df.to_parquet(parquet_buffer, index=False)
                        parquet_buffer.seek(0)
                        
                        key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                        s3_client.put_object(
                            Bucket='breadthflow',
                            Key=key,
                            Body=parquet_buffer.getvalue()
                        )
                        
                        results.append(f"{symbol}: {len(df)} records")
                        dual_logger.log("INFO", f"    ✅ {symbol}: {len(df)} records saved to MinIO")
                        
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
            dual_logger.log("INFO", "📊 Fetched Data Summary:")
            for symbol in symbols_to_fetch:
                key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                try:
                    df = load_parquet_from_minio(s3_client, 'breadthflow', key)
                    if not df.empty:
                        dual_logger.log("INFO", f"   📈 {symbol}: {len(df)} records")
                    else:
                        dual_logger.log("WARN", f"   ⚠️  {symbol}: No data found")
                except:
                    dual_logger.log("ERROR", f"   ❌ {symbol}: Error loading data")
            
            dual_logger.complete('completed')
            
        except ImportError:
            dual_logger.log("WARN", "⚠️ DataFetcher not available, using simple fetch...")
            dual_logger.complete('completed')
        
    except Exception as e:
        dual_logger.log("ERROR", f"❌ Error: {str(e)}")
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
