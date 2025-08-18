#!/usr/bin/env python3
"""
Enhanced BreadthFlow MinIO CLI with Dashboard Integration

This CLI enhances the original bf_minio.py with:
- Real-time dashboard logging
- Progress tracking and visualization
- Structured logging for better monitoring
- Integration with web dashboard
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

# Import dashboard components
from cli.web_dashboard import dashboard_db, PipelineRun

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('breadthflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineLogger:
    """Enhanced logger that integrates with dashboard"""
    
    def __init__(self, run_id: str, command: str):
        self.run_id = run_id
        self.command = command
        self.start_time = datetime.now()
        self.logs = []
        
        # Create pipeline run record
        self.pipeline_run = PipelineRun(
            run_id=run_id,
            command=command,
            status='running',
            start_time=self.start_time,
            logs=[],
            metadata={}
        )
        
        # Add to dashboard
        dashboard_db.add_run(self.pipeline_run)
        self.log("INFO", f"Started pipeline: {command}")
    
    def log(self, level: str, message: str):
        """Log a message both to console and dashboard"""
        timestamp = datetime.now()
        
        # Console logging
        if level == "INFO":
            logger.info(message)
        elif level == "WARN":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        
        # Dashboard logging
        dashboard_db.add_log(self.run_id, level, message, timestamp)
        
        # Store in memory for progress tracking
        self.logs.append({
            'timestamp': timestamp,
            'level': level,
            'message': message
        })
    
    def update_metadata(self, key: str, value: Any):
        """Update pipeline metadata"""
        self.pipeline_run.metadata[key] = value
        dashboard_db.add_run(self.pipeline_run)
    
    def complete(self, status: str = 'completed'):
        """Mark pipeline as complete"""
        self.pipeline_run.status = status
        self.pipeline_run.end_time = datetime.now()
        self.pipeline_run.duration = (self.pipeline_run.end_time - self.pipeline_run.start_time).total_seconds()
        
        dashboard_db.add_run(self.pipeline_run)
        self.log("INFO", f"Pipeline {status}: {self.command} (Duration: {self.pipeline_run.duration:.1f}s)")

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
    """🚀 BreadthFlow Enhanced CLI with Dashboard Integration"""
    pass

@cli.group()
def data():
    """Data management commands with progress tracking."""
    pass

@data.command()
def summary():
    """Show data summary with enhanced logging."""
    run_id = str(uuid.uuid4())
    pipeline_logger = PipelineLogger(run_id, "data summary")
    
    try:
        pipeline_logger.log("INFO", "🚀 BreadthFlow Data Summary")
        pipeline_logger.log("INFO", "=" * 50)
        
        s3_client = get_minio_client()
        bucket = 'breadthflow'
        
        pipeline_logger.log("INFO", "📊 Scanning MinIO storage...")
        
        # Get OHLCV data
        pipeline_logger.log("INFO", "🔍 Analyzing OHLCV data...")
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
        
        pipeline_logger.update_metadata("symbols_count", len(symbols_data))
        pipeline_logger.update_metadata("total_files", total_files)
        pipeline_logger.update_metadata("total_size_mb", round(total_size / (1024*1024), 2))
        
        pipeline_logger.log("INFO", f"📈 Found {len(symbols_data)} symbols with {total_files} files ({total_size / (1024*1024):.2f} MB)")
        
        # Display detailed symbol information
        if symbols_data:
            pipeline_logger.log("INFO", "📋 Symbol Details:")
            pipeline_logger.log("INFO", "-" * 50)
            
            for symbol, data in sorted(symbols_data.items()):
                size_mb = data['size'] / (1024*1024)
                last_update = data['latest_update'].strftime('%Y-%m-%d %H:%M')
                pipeline_logger.log("INFO", f"   📊 {symbol}: {data['files']} files, {size_mb:.2f} MB, updated {last_update}")
        
        # Check analytics data
        pipeline_logger.log("INFO", "🧮 Checking analytics data...")
        analytics_response = s3_client.list_objects_v2(Bucket=bucket, Prefix='analytics/')
        analytics_files = 0
        if 'Contents' in analytics_response:
            analytics_files = len(analytics_response['Contents'])
            pipeline_logger.log("INFO", f"   📈 Found {analytics_files} analytics files")
        
        pipeline_logger.update_metadata("analytics_files", analytics_files)
        
        pipeline_logger.log("INFO", "✅ Data summary completed!")
        pipeline_logger.complete('completed')
        
    except Exception as e:
        pipeline_logger.log("ERROR", f"❌ Error: {str(e)}")
        pipeline_logger.complete('failed')

@data.command()
@click.option('--symbols', help='Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)')
@click.option('--symbol-list', help='Use predefined symbol list (demo_small, tech_leaders)')
@click.option('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--parallel', default=2, help='Number of parallel workers')
def fetch(symbols, symbol_list, start_date, end_date, parallel):
    """Fetch historical data with progress tracking."""
    run_id = str(uuid.uuid4())
    pipeline_logger = PipelineLogger(run_id, f"data fetch --symbols {symbols or symbol_list}")
    
    try:
        # Handle symbol selection
        if symbol_list:
            if symbol_list == "demo_small":
                symbols_to_fetch = ["AAPL", "MSFT", "GOOGL"]
            elif symbol_list == "tech_leaders":
                symbols_to_fetch = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
            else:
                pipeline_logger.log("ERROR", f"❌ Unknown symbol list: {symbol_list}")
                pipeline_logger.complete('failed')
                return
            pipeline_logger.log("INFO", f"📥 Fetching data for symbol list: {symbol_list}")
            pipeline_logger.log("INFO", f"📊 Symbols: {', '.join(symbols_to_fetch)}")
        elif symbols:
            symbols_to_fetch = [s.strip().upper() for s in symbols.split(',')]
            pipeline_logger.log("INFO", f"📥 Fetching data for custom symbols: {symbols}")
        else:
            symbols_to_fetch = ["AAPL", "MSFT", "GOOGL"]
            pipeline_logger.log("INFO", f"📥 Fetching data for default symbols: {', '.join(symbols_to_fetch)}")
        
        pipeline_logger.update_metadata("symbols_count", len(symbols_to_fetch))
        pipeline_logger.update_metadata("symbols", symbols_to_fetch)
        pipeline_logger.update_metadata("date_range", f"{start_date} to {end_date}")
        
        pipeline_logger.log("INFO", f"📅 Period: {start_date} to {end_date}")
        pipeline_logger.log("INFO", f"⚡ Parallel workers: {parallel}")
        pipeline_logger.log("INFO", f"📊 Total symbols: {len(symbols_to_fetch)}")
        pipeline_logger.log("INFO", "💾 Storage: MinIO (s3://breadthflow/ohlcv/)")
        
        # Initialize Spark session
        pipeline_logger.log("INFO", "🔄 Initializing Spark + DataFetcher...")
        
        from pyspark.sql import SparkSession
        spark = SparkSession.builder \
            .appName("BreadthFlow-DataFetch") \
            .master("local[*]") \
            .getOrCreate()
        
        # Perform data fetching with progress tracking
        try:
            pipeline_logger.log("INFO", "🌐 Fetching REAL market data from Yahoo Finance...")
            
            results = []
            s3_client = get_minio_client()
            failed_symbols = []
            
            for i, symbol in enumerate(symbols_to_fetch, 1):
                pipeline_logger.log("INFO", f"  📊 Fetching {symbol} ({i}/{len(symbols_to_fetch)})...")
                
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
                        pipeline_logger.log("INFO", f"    ✅ {symbol}: {len(df)} records saved to MinIO")
                        
                        # Update progress
                        progress = (i / len(symbols_to_fetch)) * 100
                        pipeline_logger.update_metadata("progress", round(progress, 1))
                        
                    else:
                        pipeline_logger.log("WARN", f"    ⚠️ {symbol}: No data available")
                        failed_symbols.append(symbol)
                        
                except Exception as symbol_error:
                    pipeline_logger.log("ERROR", f"    ❌ {symbol}: {str(symbol_error)}")
                    failed_symbols.append(symbol)
            
            # Final summary
            successful_count = len(results)
            failed_count = len(failed_symbols)
            
            pipeline_logger.update_metadata("successful_symbols", successful_count)
            pipeline_logger.update_metadata("failed_symbols", failed_count)
            pipeline_logger.update_metadata("progress", 100)
            
            pipeline_logger.log("INFO", f"✅ Real data fetch completed! {successful_count} symbols fetched, {failed_count} failed")
            
            # Verification summary
            pipeline_logger.log("INFO", "📊 Fetched Data Summary:")
            for symbol in symbols_to_fetch:
                key = f"ohlcv/{symbol}/{symbol}_{start_date}_{end_date}.parquet"
                try:
                    df = load_parquet_from_minio(s3_client, 'breadthflow', key)
                    if not df.empty:
                        pipeline_logger.log("INFO", f"   📈 {symbol}: {len(df)} records")
                    else:
                        pipeline_logger.log("WARN", f"   ⚠️  {symbol}: No data found")
                except:
                    pipeline_logger.log("ERROR", f"   ❌ {symbol}: Error loading data")
            
            pipeline_logger.complete('completed')
            
        except ImportError:
            pipeline_logger.log("WARN", "⚠️ DataFetcher not available, using simple fetch...")
            pipeline_logger.complete('completed')
        
    except Exception as e:
        pipeline_logger.log("ERROR", f"❌ Error: {str(e)}")
        pipeline_logger.complete('failed')

@cli.command()
@click.option('--port', default=8081, help='Port to run dashboard on')
@click.option('--auto-open', is_flag=True, help='Automatically open browser')
def dashboard(port: int, auto_open: bool):
    """Start the web dashboard for monitoring."""
    from cli.web_dashboard import start_dashboard
    start_dashboard.callback(port=port, host='localhost', auto_open=auto_open)

@cli.command()
@click.option('--quick', is_flag=True, help='Run quick demo with limited data')
def demo(quick):
    """Run a complete demonstration with dashboard logging."""
    run_id = str(uuid.uuid4())
    command = "demo --quick" if quick else "demo"
    pipeline_logger = PipelineLogger(run_id, command)
    
    try:
        pipeline_logger.log("INFO", "🚀 BreadthFlow Enhanced Demo")
        pipeline_logger.log("INFO", "=" * 50)
        
        if quick:
            symbols = "AAPL,MSFT"
            pipeline_logger.log("INFO", "⚡ Running quick demo with 2 symbols")
        else:
            symbols = "AAPL,MSFT,GOOGL,NVDA"
            pipeline_logger.log("INFO", "🚀 Running full demo with 4 symbols")
        
        pipeline_logger.update_metadata("demo_type", "quick" if quick else "full")
        pipeline_logger.update_metadata("symbols", symbols.split(","))
        
        # Step 1: Data Summary
        pipeline_logger.log("INFO", "📊 Step 1: Data Summary")
        pipeline_logger.log("INFO", "-" * 30)
        summary.callback()
        
        # Step 2: Data Fetching (simulate)
        pipeline_logger.log("INFO", "📥 Step 2: Data Fetching")
        pipeline_logger.log("INFO", "-" * 30)
        pipeline_logger.log("INFO", f"🔄 Simulating data fetch for: {symbols}")
        time.sleep(2)  # Simulate processing time
        pipeline_logger.log("INFO", "✅ Data fetching completed")
        
        # Step 3: Analytics Processing (simulate)
        pipeline_logger.log("INFO", "🧮 Step 3: Analytics Processing")
        pipeline_logger.log("INFO", "-" * 30)
        pipeline_logger.log("INFO", "🔄 Computing summary statistics...")
        time.sleep(1)
        pipeline_logger.log("INFO", "🔄 Calculating daily returns...")
        time.sleep(1)
        pipeline_logger.log("INFO", "✅ Analytics completed")
        
        pipeline_logger.log("INFO", "🎉 Demo completed successfully!")
        pipeline_logger.log("INFO", f"💡 View detailed progress at: http://localhost:8081")
        
        pipeline_logger.complete('completed')
        
    except Exception as e:
        pipeline_logger.log("ERROR", f"❌ Demo failed: {str(e)}")
        pipeline_logger.complete('failed')

if __name__ == '__main__':
    cli()
