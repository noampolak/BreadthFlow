#!/usr/bin/env python3
"""
BreadthFlow Container CLI

This CLI runs entirely inside the Spark master container, eliminating host-container communication issues.
"""

import click
import subprocess
import time
import requests
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

# Add the jobs directory to Python path
sys.path.insert(0, '/opt/bitnami/spark/jobs')

# Set Spark environment variables
os.environ['SPARK_LOCAL_DIRS'] = '/tmp/spark-local'
os.environ['SPARK_CONF_DIR'] = '/opt/bitnami/spark/conf'
os.environ['SPARK_HOME'] = '/opt/bitnami/spark'
os.environ['HOME'] = '/opt/bitnami/spark'
os.environ['USER'] = '1001'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session(app_name: str):
    """
    Create a Spark session inside the container.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession
    
    # Try to get existing session first
    try:
        spark = SparkSession.builder.getOrCreate()
        if spark is not None:
            return spark
    except:
        pass
    
    # Create new Spark session inside container with proper Delta Lake and S3 support
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
        .config("spark.executor.memory", "1g") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.memory", "1g") \
        .getOrCreate()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """BreadthFlow Container CLI - Runs entirely inside Spark master container."""
    pass


@cli.group()
def infra():
    """Infrastructure management commands."""
    pass


@infra.command()
def status():
    """Check status of infrastructure services."""
    click.echo("📊 Infrastructure Status")
    click.echo("=" * 40)
    
    # Check Spark cluster status
    try:
        response = requests.get("http://localhost:8080/json/", timeout=5)
        data = response.json()
        
        click.echo(f"✅ Spark Master: Healthy")
        click.echo(f"📊 Workers: {data.get('aliveworkers', 0)}")
        click.echo(f"💾 Memory: {data.get('memory', 0)}MB")
        click.echo(f"🖥️  Cores: {data.get('cores', 0)}")
        
        # Show worker details
        workers = data.get('workers', [])
        for i, worker in enumerate(workers, 1):
            click.echo(f"  Worker {i}: {worker.get('state', 'UNKNOWN')} - {worker.get('host', 'unknown')}")
            
    except Exception as e:
        click.echo(f"❌ Spark Master: {e}")
    
    # Check MinIO
    try:
        response = requests.get("http://minio:9000/minio/health/live", timeout=5)
        click.echo("✅ MinIO: Healthy")
    except Exception as e:
        click.echo(f"❌ MinIO: {e}")
    
    # Show service URLs
    click.echo("\n🌐 Service URLs:")
    click.echo("  • Spark UI: http://localhost:8080")
    click.echo("  • Spark History: http://localhost:18080")
    click.echo("  • MinIO Console: http://localhost:9001")


@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option('--symbols', help='Comma-separated symbols')
@click.option('--symbol-list', help='Use predefined symbol list (e.g., demo_small, sp500, tech_leaders)')
@click.option('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--parallel', default=2, help='Number of parallel workers')
@click.option('--table-path', help='Delta table path for storage')
def fetch(symbols, symbol_list, start_date, end_date, parallel, table_path):
    """Fetch historical data for symbols using container-native Spark."""
    
    # Handle symbol selection
    if symbol_list:
        try:
            from features.common.symbols import get_symbol_manager
            manager = get_symbol_manager()
            symbols_to_fetch = manager.get_symbol_list(symbol_list)
            click.echo(f"📥 Fetching data for symbol list: {symbol_list}")
            click.echo(f"📊 Symbols: {', '.join(symbols_to_fetch[:5])}{'...' if len(symbols_to_fetch) > 5 else ''}")
        except Exception as e:
            click.echo(f"❌ Error loading symbol list '{symbol_list}': {e}")
            return
    elif symbols:
        symbols_to_fetch = [s.strip() for s in symbols.split(',')]
        click.echo(f"📥 Fetching data for custom symbols: {symbols}")
    else:
        # Default to demo symbols
        from features.common.symbols import get_demo_symbols
        symbols_to_fetch = get_demo_symbols("small")
        click.echo(f"📥 Fetching data for default demo symbols: {', '.join(symbols_to_fetch)}")
    
    click.echo(f"📅 Period: {start_date} to {end_date}")
    click.echo(f"⚡ Parallel workers: {parallel}")
    click.echo(f"📊 Total symbols: {len(symbols_to_fetch)}")
    
    if table_path:
        click.echo(f"💾 Storage: {table_path}")
    else:
        click.echo("💾 Storage: MinIO (s3://breadthflow/ohlcv/)")
    
    try:
        # Create Spark session inside container
        click.echo("🔄 Creating Spark session...")
        spark = create_spark_session("DataFetcher")
        click.echo("✅ Spark session created successfully")
        
        # Import and create data fetcher
        from ingestion.data_fetcher import create_data_fetcher
        fetcher = create_data_fetcher(spark)
        
        # Check available symbols
        click.echo("🔍 Checking symbol availability...")
        available_symbols = fetcher.get_available_symbols(symbols_to_fetch)
        click.echo(f"✅ Found {len(available_symbols)} available symbols")
        
        if not available_symbols:
            click.echo("❌ No symbols are available")
            return
        
        # Fetch and store data
        click.echo("🚀 Starting data fetch...")
        result = fetcher.fetch_and_store(
            symbols=available_symbols,
            start_date=start_date,
            end_date=end_date,
            table_path=table_path,
            max_workers=parallel
        )
        
        if result["success"]:
            click.echo(f"✅ Successfully fetched {result['symbols_fetched']} symbols")
            click.echo(f"📊 Total records: {result['total_records']}")
            click.echo(f"⏱️  Duration: {result['duration']:.2f}s")
            if result.get('failed_symbols'):
                click.echo(f"⚠️  Failed symbols: {result['failed_symbols']}")
        else:
            click.echo(f"❌ Fetch failed: {result['message']}")
            
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
    finally:
        try:
            spark.stop()
            click.echo("✅ Spark session stopped")
        except:
            pass


@data.command()
def summary():
    """Show summary of stored data."""
    click.echo("📊 Data Summary")
    
    try:
        # Create Spark session inside container
        spark = create_spark_session("DataSummary")
        
        # Import and create data fetcher
        from ingestion.data_fetcher import create_data_fetcher
        fetcher = create_data_fetcher(spark)
        
        # Get summary
        summary = fetcher.get_data_summary()
        
        if "error" in summary:
            click.echo(f"❌ Error: {summary['error']}")
            return
        
        click.echo(f"📊 Total records: {summary['total_records']:,}")
        click.echo(f"📈 Unique symbols: {summary['unique_symbols']}")
        click.echo(f"📅 Date range: {summary['date_range']}")
        click.echo(f"💾 Storage location: {summary['storage_location']}")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
    finally:
        try:
            spark.stop()
        except:
            pass


if __name__ == "__main__":
    cli()
