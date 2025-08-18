#!/usr/bin/env python3
"""
Breadth/Thrust Signals POC - Command Line Interface

A comprehensive CLI for managing the entire system using Python and Click.
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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the new Spark job submitter
from cli.spark_job_submitter import SparkJobSubmitter


def create_spark_session(app_name: str):
    """
    Create a Spark session that connects to the Docker Spark cluster.
    
    Args:
        app_name: Name of the Spark application
        
    Returns:
        Configured SparkSession
    """
    from pyspark.sql import SparkSession
    import socket
    
    # Get host machine's IP address for executors to connect to driver
    # Use the actual network interface IP, not hostname resolution
    import subprocess
    try:
        # Get the actual network IP that Docker containers can reach
        result = subprocess.run(['ifconfig', 'en0'], capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'inet ' in line:
                host_ip = line.strip().split()[1]
                break
        else:
            # Fallback to hostname resolution
            host_ip = socket.gethostbyname(socket.gethostname())
    except:
        # Fallback to hostname resolution
        host_ip = socket.gethostbyname(socket.gethostname())
    
    # We're on host, connect to Docker Spark cluster with Delta Lake and S3 support
    return SparkSession.builder \
        .appName(app_name) \
        .master("spark://localhost:7077") \
        .config("spark.driver.host", host_ip) \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.executor.memory", "512m") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memoryOverhead", "256m") \
        .config("spark.driver.memoryOverhead", "256m") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.submit.deployMode", "cluster") \
        .config("spark.driver.extraClassPath", "/opt/bitnami/spark/jars/*") \
        .config("spark.executor.extraClassPath", "/opt/bitnami/spark/jars/*") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .config("spark.storage.blockManagerSlaveTimeoutMs", "800000") \
        .config("spark.rpc.askTimeout", "800s") \
        .config("spark.rpc.lookupTimeout", "800s") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.speculation", "false") \
        .config("spark.pyspark.python", "python3") \
        .config("spark.pyspark.driver.python", "python3") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .config("spark.sql.adaptive.skewJoin.enabled", "false") \
        .getOrCreate()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Breadth/Thrust Signals POC - Real-time market breadth analysis system.
    
    Built with PySpark, Kafka, Delta Lake, and modern big data technologies.
    """
    pass


# ============================================================================
# Infrastructure Management Commands
# ============================================================================

@cli.group()
def infra():
    """Infrastructure management commands."""
    pass


@infra.command()
@click.option('--wait', default=30, help='Seconds to wait for services to start')
def start(wait):
    """Start all infrastructure services."""
    click.echo("🚀 Starting Breadth/Thrust Signals Infrastructure")
    
    # Check if Docker Compose file exists
    docker_compose_file = Path("infra/docker-compose.yml")
    if not docker_compose_file.exists():
        click.echo("❌ Docker Compose file not found!")
        click.echo("Please ensure infra/docker-compose.yml exists")
        return
    
    try:
        # Start services
        click.echo("📦 Starting Docker services...")
        subprocess.run(
            ["docker", "compose", "-f", "infra/docker-compose.yml", "up", "-d"],
            check=True
        )
        click.echo("✅ Docker services started successfully")
        
        # Wait for services
        if wait > 0:
            click.echo(f"⏳ Waiting {wait} seconds for services to start...")
            time.sleep(wait)
        
        # Health check
        click.echo("🏥 Performing health checks...")
        health_status = check_infrastructure_health()
        
        if health_status['all_healthy']:
            click.echo("🎉 All services are healthy! Infrastructure is ready.")
            show_service_urls()
        else:
            click.echo(f"⚠️  {health_status['healthy_count']}/{health_status['total_count']} services are healthy")
            click.echo("Check logs with: bf infra logs")
            
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to start Docker services: {e}")
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")


@infra.command()
def stop():
    """Stop all infrastructure services."""
    click.echo("🛑 Stopping infrastructure services...")
    try:
        subprocess.run(
            ["docker", "compose", "-f", "infra/docker-compose.yml", "down"],
            check=True
        )
        click.echo("✅ Infrastructure stopped successfully")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to stop services: {e}")


@infra.command()
def restart():
    """Restart all infrastructure services."""
    click.echo("🔄 Restarting infrastructure services...")
    try:
        # Stop
        subprocess.run(
            ["docker", "compose", "-f", "infra/docker-compose.yml", "down"],
            check=True
        )
        click.echo("✅ Services stopped")
        
        # Start
        subprocess.run(
            ["docker", "compose", "-f", "infra/docker-compose.yml", "up", "-d"],
            check=True
        )
        click.echo("✅ Services started")
        
        # Health check
        time.sleep(30)
        health_status = check_infrastructure_health()
        if health_status['all_healthy']:
            click.echo("🎉 All services are healthy!")
        else:
            click.echo(f"⚠️  {health_status['healthy_count']}/{health_status['total_count']} services are healthy")
            
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to restart services: {e}")


@infra.command()
def status():
    """Check status of infrastructure services."""
    click.echo("📊 Infrastructure Status")
    click.echo("=" * 40)
    
    # Show service URLs
    show_service_urls()
    click.echo()
    
    # Check health
    health_status = check_infrastructure_health()
    click.echo(f"Health: {health_status['healthy_count']}/{health_status['total_count']} services healthy")
    
    # Show Docker status
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", "infra/docker-compose.yml", "ps"],
            capture_output=True, text=True, check=True
        )
        click.echo("\nDocker Service Status:")
        click.echo(result.stdout)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to get Docker status: {e}")


@infra.command()
@click.option('--follow', '-f', is_flag=True, help='Follow logs')
def logs(follow):
    """Show infrastructure service logs."""
    try:
        cmd = ["docker", "compose", "-f", "infra/docker-compose.yml", "logs"]
        if follow:
            cmd.append("-f")
        subprocess.run(cmd)
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to show logs: {e}")


@infra.command()
def health():
    """Perform detailed health checks."""
    click.echo("🏥 Performing detailed health checks...")
    health_status = check_infrastructure_health()
    
    if health_status['all_healthy']:
        click.echo("🎉 All services are healthy!")
    else:
        click.echo(f"⚠️  {health_status['healthy_count']}/{health_status['total_count']} services are healthy")
        for service, status in health_status['services'].items():
            icon = "✅" if status else "❌"
            click.echo(f"  {icon} {service}")


# ============================================================================
# Data Management Commands
# ============================================================================

@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
def summary():
    """Show summary of stored data."""
    click.echo("📊 Data Summary")
    
    try:
        # Try to use local PySpark first
        try:
            # Import here to avoid circular imports
            from ingestion.data_fetcher import create_data_fetcher
            
            # Create Spark session - connect to Docker cluster
            spark = create_spark_session("DataSummary")
            
            # Create data fetcher
            fetcher = create_data_fetcher(spark)
            
            # Get summary
            summary = fetcher.get_data_summary()
            
            if "error" in summary:
                click.echo(f"❌ Error: {summary['error']}")
                return
            
            click.echo(f"📊 Total records: {summary['total_records']:,}")
            click.echo(f"📈 Unique symbols: {summary['unique_symbols']}")
            click.echo(f"📅 Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            click.echo(f"⭐ Average quality score: {summary['avg_quality_score']:.3f}")
            
            if summary['symbols']:
                click.echo(f"📋 Symbols: {', '.join(summary['symbols'][:10])}")
                if len(summary['symbols']) > 10:
                    click.echo(f"   ... and {len(summary['symbols']) - 10} more")
            
            spark.stop()
            
        except Exception as local_error:
            click.echo(f"❌ PySpark error: {str(local_error)}")
            click.echo("💡 Make sure your Spark cluster is running: docker-compose up -d")
            return
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@data.command()
@click.option('--symbols', help='Comma-separated symbols')
@click.option('--symbol-list', help='Use predefined symbol list (e.g., demo_small, sp500, tech_leaders)')
@click.option('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--parallel', default=2, help='Number of parallel workers')
@click.option('--table-path', help='Delta table path for storage')
def fetch(symbols, symbol_list, start_date, end_date, parallel, table_path):
    """Fetch historical data for symbols using proper Spark infrastructure."""
    
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
        # Use the new Spark job submitter for proper infrastructure
        click.echo("🔄 Initializing Spark job submitter...")
        submitter = SparkJobSubmitter()
        
        # Check cluster status
        click.echo("🔍 Checking Spark cluster status...")
        if not submitter.check_cluster_status():
            click.echo("❌ Spark cluster is not healthy. Please check infrastructure status.")
            return
        
        worker_count = submitter.get_worker_count()
        click.echo(f"✅ Spark cluster is healthy with {worker_count} workers")
        
        # Limit parallel workers to available Spark workers
        actual_workers = min(parallel, worker_count)
        if actual_workers != parallel:
            click.echo(f"⚠️  Limiting parallel workers to {actual_workers} (available Spark workers)")
        
        # Submit the job to the cluster
        click.echo("🚀 Submitting data fetch job to Spark cluster...")
        result = submitter.submit_data_fetch_job(
            symbols=symbols_to_fetch,
            start_date=start_date,
            end_date=end_date,
            max_workers=actual_workers
        )
        
        click.echo("✅ Job submitted successfully to Spark cluster")
        click.echo("📊 Check Spark UI at http://localhost:8080 for job progress")
        click.echo("📊 Check Spark History at http://localhost:18080 for completed jobs")
        
    except Exception as e:
        click.echo(f"❌ Error submitting job: {str(e)}")
        click.echo("🔍 Please check that the Spark cluster is running: poetry run bf infra status")


@data.command()
def symbols():
    """List available symbol lists and their information."""
    click.echo("📊 Available Symbol Lists")
    click.echo("=" * 40)
    
    try:
        from features.common.symbols import get_symbol_manager
        manager = get_symbol_manager()
        
        # Get summary
        summary = manager.get_symbol_list_summary()
        
        click.echo(f"📋 Total Lists: {summary['total_lists']}")
        click.echo(f"📝 Description: {summary['metadata']['description']}")
        click.echo(f"🕒 Last Updated: {summary['metadata']['last_updated']}")
        
        click.echo("\n📊 Symbol Lists:")
        for list_name, list_info in summary['lists'].items():
            click.echo(f"  • {list_name}: {list_info['name']}")
            click.echo(f"    Description: {list_info['description']}")
            click.echo(f"    Symbols: {list_info['symbol_count']}")
            if list_info['symbol_count'] <= 10:
                click.echo(f"    List: {', '.join(list_info['symbols'])}")
            else:
                click.echo(f"    Sample: {', '.join(list_info['symbols'][:5])}...")
            click.echo()
        
        click.echo("💡 Recommendations:")
        for use_case, list_name in summary['recommendations'].items():
            click.echo(f"  • {use_case}: {list_name}")
        
        click.echo("\n🔧 Usage Examples:")
        click.echo("  • poetry run bf data fetch --symbol-list demo_small")
        click.echo("  • poetry run bf data fetch --symbol-list sp500")
        click.echo("  • poetry run bf data fetch --symbol-list tech_leaders")
        click.echo("  • poetry run bf data fetch --symbols AAPL,MSFT,GOOGL")
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@data.command()
@click.option('--speed', default=60, help='Replay speed multiplier')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--symbols', help='Comma-separated symbols to replay')
@click.option('--topic', default='quotes_replay', help='Kafka topic')
@click.option('--table-path', help='Delta table path')
def replay(speed, start_date, end_date, symbols, topic, table_path):
    """Start data replay from Delta to Kafka."""
    click.echo(f"🔄 Starting data replay")
    click.echo(f"⚡ Speed: {speed}x")
    click.echo(f"📡 Topic: {topic}")
    
    if start_date:
        click.echo(f"📅 Start date: {start_date}")
    if end_date:
        click.echo(f"📅 End date: {end_date}")
    if symbols:
        click.echo(f"📊 Symbols: {symbols}")
    
    try:
        # Import here to avoid circular imports
        from ingestion.replay import create_replay_manager, ReplayConfig
        
        # Create Spark session - connect to Docker cluster
        spark = create_spark_session("ReplayManager")
        
        # Create replay manager
        replay_manager = create_replay_manager(spark)
        
        # Parse symbols if provided
        symbol_list = None
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
        
        # Set table path
        if table_path is None:
            table_path = "data/ohlcv"
        
        # Load historical data
        click.echo("📂 Loading historical data...")
        df = replay_manager.load_historical_data(
            table_path=table_path,
            start_date=start_date,
            end_date=end_date,
            symbols=symbol_list
        )
        
        if df.count() == 0:
            click.echo("❌ No data found for replay")
            return
        
        # Configure replay
        config = ReplayConfig(
            speed_multiplier=float(speed),
            topic_name=topic,
            start_date=start_date,
            end_date=end_date,
            symbols=symbol_list
        )
        
        # Start replay
        click.echo("🚀 Starting replay...")
        result = replay_manager.replay_data(df, config)
        
        if result["success"]:
            click.echo(f"✅ Replay completed successfully")
            click.echo(f"📊 Records processed: {result['processed_records']}")
            click.echo(f"⏱️  Duration: {result['duration_seconds']:.2f}s")
            click.echo(f"⚡ Effective speed: {result['effective_speed']:.1f} records/sec")
        else:
            click.echo(f"❌ Replay failed: {result.get('error', 'Unknown error')}")
        
        # Cleanup
        replay_manager.cleanup()
        spark.stop()
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        click.echo("💡 Make sure infrastructure is running: poetry run bf infra start")


# ============================================================================
# Signal Generation Commands
# ============================================================================

@cli.group()
def signals():
    """Signal generation commands."""
    pass


@signals.command()
@click.option('--date', help='Date to generate signals for (YYYY-MM-DD)')
@click.option('--start-date', help='Start date for signal generation (YYYY-MM-DD)')
@click.option('--end-date', help='End date for signal generation (YYYY-MM-DD)')
@click.option('--symbols', help='Comma-separated symbols to analyze')
@click.option('--symbol-list', help='Use predefined symbol list (e.g., demo_small, sp500, tech_leaders)')
@click.option('--force-recalculate', is_flag=True, help='Force recalculation of all features')
def generate(date, start_date, end_date, symbols, symbol_list, force_recalculate):
    """Generate trading signals."""
    click.echo("🎯 Generating trading signals")
    
    if date:
        click.echo(f"📅 Date: {date}")
    elif start_date and end_date:
        click.echo(f"📅 Period: {start_date} to {end_date}")
    else:
        click.echo("📅 Date: Latest available")
    
    # Handle symbol selection
    if symbol_list:
        try:
            from features.common.symbols import get_symbol_manager
            manager = get_symbol_manager()
            symbols_to_analyze = manager.get_symbol_list(symbol_list)
            click.echo(f"📊 Using symbol list: {symbol_list}")
            click.echo(f"📈 Symbols: {', '.join(symbols_to_analyze[:5])}{'...' if len(symbols_to_analyze) > 5 else ''}")
        except Exception as e:
            click.echo(f"❌ Error loading symbol list '{symbol_list}': {e}")
            return
    elif symbols:
        symbols_to_analyze = [s.strip() for s in symbols.split(',')]
        click.echo(f"📊 Custom symbols: {symbols}")
    else:
        symbols_to_analyze = None
        click.echo("📊 Using all available symbols")
    
    if force_recalculate:
        click.echo("🔄 Force recalculation enabled")
    
    try:
        # Import here to avoid circular imports
        from model.signal_generator import create_signal_generator
        
        # Create Spark session - connect to Docker cluster
        spark = create_spark_session("SignalGenerator")
        
        # Create signal generator
        generator = create_signal_generator(spark)
        
        # Generate signals
        if date:
            # Generate for specific date
            result = generator.generate_signals_for_date(date, symbols_to_analyze)
        else:
            # Generate for period
            result = generator.generate_signals(
                symbols=symbols_to_analyze,
                start_date=start_date,
                end_date=end_date,
                force_recalculate=force_recalculate
            )
        
        if result["success"]:
            click.echo("✅ Signal generation completed successfully")
            
            if "generation_time_seconds" in result:
                click.echo(f"⏱️  Generation time: {result['generation_time_seconds']:.2f}s")
            
            if "total_signals_generated" in result:
                click.echo(f"📊 Total signals: {result['total_signals_generated']}")
                click.echo(f"🟢 Buy signals: {result['buy_signals']}")
                click.echo(f"🔴 Sell signals: {result['sell_signals']}")
                click.echo(f"⭐ Strong signals: {result['strong_signals']}")
            
            if "composite_score" in result:
                click.echo(f"📈 Composite score: {result['composite_score']:.2f}")
                click.echo(f"🎯 Signal direction: {result['signal_direction']}")
                click.echo(f"💪 Signal strength: {result['signal_strength']}")
                click.echo(f"🎯 Confidence: {result['confidence_score']:.2f}")
        else:
            click.echo(f"❌ Signal generation failed: {result.get('error', 'Unknown error')}")
        
        spark.stop()
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        click.echo("💡 Make sure infrastructure is running: poetry run bf infra start")


@signals.command()
def summary():
    """Show summary of generated signals."""
    click.echo("📊 Signal Summary")
    
    try:
        # Import here to avoid circular imports
        from model.signal_generator import create_signal_generator
        
        # Create Spark session - connect to Docker cluster
        spark = create_spark_session("SignalSummary")
        
        # Create signal generator
        generator = create_signal_generator(spark)
        
        # Get summary
        summary = generator.get_signals_summary()
        
        if "error" in summary:
            click.echo(f"❌ Error: {summary['error']}")
            return
        
        click.echo(f"📊 Total signals: {summary['total_signals']:,}")
        click.echo(f"📈 Average score: {summary['avg_score']:.2f}")
        click.echo(f"🎯 Average confidence: {summary['avg_confidence']:.2f}")
        click.echo(f"🟢 Buy signals: {summary['buy_signals']}")
        click.echo(f"🔴 Sell signals: {summary['sell_signals']}")
        click.echo(f"⭐ Strong buy signals: {summary['strong_buy_signals']}")
        click.echo(f"⭐ Strong sell signals: {summary['strong_sell_signals']}")
        click.echo(f"📈 Bullish days: {summary['bullish_days']}")
        click.echo(f"📉 Bearish days: {summary['bearish_days']}")
        click.echo(f"➡️  Neutral days: {summary['neutral_days']}")
        
        spark.stop()
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")


@signals.command()
@click.option('--query', required=True, help='Search query')
@click.option('--index', default='breadth_signals', help='Elasticsearch index')
def search(query, index):
    """Search signals using Elasticsearch."""
    click.echo(f"🔍 Searching signals")
    click.echo(f"📝 Query: {query}")
    click.echo(f"📊 Index: {index}")
    
    # This will be implemented in the search module
    click.echo("🔄 Signal search not yet implemented")
    click.echo("Run: python -m streaming.elasticsearch_sink")


# ============================================================================
# Backtesting Commands
# ============================================================================

@cli.group()
def backtest():
    """Backtesting commands."""
    pass


@backtest.command()
@click.option('--from-date', required=True, help='Start date (YYYY-MM-DD)')
@click.option('--to-date', required=True, help='End date (YYYY-MM-DD)')
@click.option('--symbols', help='Comma-separated symbols to trade')
@click.option('--symbol-list', help='Use predefined symbol list (e.g., demo_small, sp500, tech_leaders)')
@click.option('--initial-capital', default=100000.0, help='Initial capital amount')
@click.option('--position-size', default=0.1, help='Position size as percentage of capital')
@click.option('--max-positions', default=10, help='Maximum number of concurrent positions')
@click.option('--commission-rate', default=0.001, help='Commission rate (e.g., 0.001 for 0.1%)')
@click.option('--slippage-rate', default=0.0005, help='Slippage rate (e.g., 0.0005 for 0.05%)')
@click.option('--save-results', is_flag=True, help='Save results to Delta Lake')
def run(from_date, to_date, symbols, symbol_list, initial_capital, position_size, max_positions, commission_rate, slippage_rate, save_results):
    """Run backtesting analysis."""
    click.echo(f"📊 Running backtest")
    click.echo(f"📅 Period: {from_date} to {to_date}")
    click.echo(f"💰 Initial capital: ${initial_capital:,.2f}")
    click.echo(f"📈 Position size: {position_size:.1%}")
    click.echo(f"🔢 Max positions: {max_positions}")
    click.echo(f"💸 Commission rate: {commission_rate:.3%}")
    click.echo(f"📉 Slippage rate: {slippage_rate:.3%}")
    
    # Handle symbol selection
    if symbol_list:
        try:
            from features.common.symbols import get_symbol_manager
            manager = get_symbol_manager()
            symbols_to_trade = manager.get_symbol_list(symbol_list)
            click.echo(f"📊 Using symbol list: {symbol_list}")
            click.echo(f"📈 Symbols: {', '.join(symbols_to_trade[:5])}{'...' if len(symbols_to_trade) > 5 else ''}")
        except Exception as e:
            click.echo(f"❌ Error loading symbol list '{symbol_list}': {e}")
            return
    elif symbols:
        symbols_to_trade = [s.strip() for s in symbols.split(',')]
        click.echo(f"📊 Custom symbols: {symbols}")
    else:
        symbols_to_trade = None
        click.echo("📊 Using all available symbols")
    
    try:
        # Import here to avoid circular imports
        from backtests.engine import create_backtest_engine, BacktestConfig
        
        # Create Spark session - connect to Docker cluster
        spark = create_spark_session("BacktestEngine")
        
        # Create backtest configuration
        config = BacktestConfig(
            initial_capital=initial_capital,
            position_size_pct=position_size,
            max_positions=max_positions,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate
        )
        
        # Create backtest engine
        engine = create_backtest_engine(spark, config)
        
        # Run backtest
        results = engine.run_backtest(
            start_date=from_date,
            end_date=to_date,
            symbols=symbols_to_trade,
            save_results=save_results
        )
        
        if results:
            click.echo("✅ Backtest completed successfully")
            click.echo(f"📊 Total Return: {results.total_return:.2%}")
            click.echo(f"📈 Annualized Return: {results.annualized_return:.2%}")
            click.echo(f"📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
            click.echo(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
            click.echo(f"🎯 Hit Rate: {results.hit_rate:.2%}")
            click.echo(f"💰 Total Trades: {results.total_trades}")
            click.echo(f"📊 Winning Trades: {results.winning_trades}")
            click.echo(f"📉 Losing Trades: {results.losing_trades}")
            click.echo(f"💵 Average Win: ${results.avg_win:,.2f}")
            click.echo(f"💸 Average Loss: ${results.avg_loss:,.2f}")
            click.echo(f"📈 Profit Factor: {results.profit_factor:.2f}")
            click.echo(f"🎯 Signal Accuracy: {results.signal_accuracy:.2%}")
            
            if save_results:
                click.echo(f"💾 Results saved to backtests/out/results_{from_date}_{to_date}")
        else:
            click.echo("❌ Backtest failed")
        
        spark.stop()
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        click.echo("💡 Make sure infrastructure is running: poetry run bf infra start")


@backtest.command()
@click.option('--results-path', help='Path to backtest results in Delta Lake')
@click.option('--start-date', help='Start date filter for analysis')
@click.option('--end-date', help='End date filter for analysis')
def analyze(results_path, start_date, end_date):
    """Analyze backtest results."""
    click.echo("📈 Analyzing backtest results")
    
    if results_path:
        click.echo(f"📄 Results path: {results_path}")
    else:
        click.echo("📄 Using latest backtest results")
    
    if start_date and end_date:
        click.echo(f"📅 Period: {start_date} to {end_date}")
    
    try:
        # Import here to avoid circular imports
        from backtests.metrics import create_performance_metrics
        from features.common.io import read_delta
        
        # Create Spark session - connect to Docker cluster
        spark = create_spark_session("BacktestAnalysis")
        
        # Load backtest results
        if results_path:
            results_df = read_delta(spark, results_path)
        else:
            # Try to find latest results
            import glob
            import os
            result_files = glob.glob("backtests/out/results_*")
            if result_files:
                latest_result = max(result_files, key=os.path.getctime)
                results_df = read_delta(spark, latest_result)
                click.echo(f"📄 Found latest results: {latest_result}")
            else:
                click.echo("❌ No backtest results found")
                return
        
        # Apply date filters if provided
        if start_date:
            results_df = results_df.filter(col("date") >= start_date)
        if end_date:
            results_df = results_df.filter(col("date") <= end_date)
        
        # Convert to pandas for analysis
        results_pdf = results_df.toPandas()
        
        if results_pdf.empty:
            click.echo("❌ No results found for the specified criteria")
            return
        
        # Calculate performance metrics
        metrics_calc = create_performance_metrics()
        
        # Extract returns from portfolio values
        portfolio_values = results_pdf["portfolio_value"].tolist()
        returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(daily_return)
        
        # Calculate comprehensive metrics
        metrics = metrics_calc.calculate_all_metrics(
            returns=returns,
            portfolio_values=portfolio_values
        )
        
        # Generate and display report
        report = metrics_calc.generate_performance_report(
            metrics, 
            "Breadth/Thrust Strategy Backtest"
        )
        click.echo(report)
        
        # Additional insights
        click.echo("🔍 Additional Insights:")
        click.echo(f"📊 Total Days: {len(results_pdf)}")
        click.echo(f"📈 Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
        click.echo(f"💰 Peak Portfolio Value: ${max(portfolio_values):,.2f}")
        click.echo(f"📉 Trough Portfolio Value: ${min(portfolio_values):,.2f}")
        
        # Signal analysis
        if "signals_processed" in results_pdf.columns:
            avg_signals = results_pdf["signals_processed"].mean()
            avg_acted = results_pdf["signals_acted_upon"].mean()
            click.echo(f"🎯 Average Signals per Day: {avg_signals:.1f}")
            click.echo(f"⚡ Average Signals Acted Upon: {avg_acted:.1f}")
        
        spark.stop()
        
    except Exception as e:
        click.echo(f"❌ Error: {str(e)}")
        click.echo("💡 Make sure backtest results exist: poetry run bf backtest run")


# ============================================================================
# Development Commands
# ============================================================================

@cli.group()
def dev():
    """Development commands."""
    pass


@dev.command()
def install():
    """Install Python dependencies using Poetry."""
    click.echo("📦 Installing Python dependencies with Poetry...")
    try:
        subprocess.run(["poetry", "install"], check=True)
        click.echo("✅ Dependencies installed successfully")
        click.echo("💡 Use 'poetry shell' to activate the virtual environment")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Failed to install dependencies: {e}")
        click.echo("💡 Make sure Poetry is installed: https://python-poetry.org/docs/#installation")


@dev.command()
def setup():
    """Setup environment and configuration."""
    click.echo("⚙️  Setting up environment...")
    
    # Copy environment file
    env_example = Path("env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        click.echo("✅ Environment file created (.env)")
        click.echo("📝 Please edit .env with your configuration")
    else:
        click.echo("ℹ️  Environment file already exists")


@dev.command()
def test():
    """Run tests using Poetry."""
    click.echo("🧪 Running tests...")
    try:
        subprocess.run(["poetry", "run", "pytest"], check=True)
        click.echo("✅ Tests completed successfully")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Tests failed: {e}")


@dev.command()
def lint():
    """Run linting checks using Poetry."""
    click.echo("🔍 Running linting checks...")
    try:
        subprocess.run(["poetry", "run", "flake8", ".", "--count", "--exit-zero", "--max-complexity=10", "--max-line-length=127"], check=True)
        click.echo("✅ Linting passed")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Linting failed: {e}")


@dev.command()
def format():
    """Format code with black using Poetry."""
    click.echo("🎨 Formatting code...")
    try:
        subprocess.run(["poetry", "run", "black", ".", "--line-length=127"], check=True)
        click.echo("✅ Code formatted successfully")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Formatting failed: {e}")


@dev.command()
def clean():
    """Clean up temporary files and containers."""
    click.echo("🧹 Cleaning up...")
    
    try:
        # Stop containers
        subprocess.run(["docker", "compose", "-f", "infra/docker-compose.yml", "down", "-v"], check=True)
        click.echo("✅ Containers stopped")
        
        # Clean Docker
        subprocess.run(["docker", "system", "prune", "-f"], check=True)
        click.echo("✅ Docker cleaned")
        
        # Clean Python cache
        subprocess.run(["find", ".", "-type", "d", "-name", "__pycache__, -exec", "rm", "-rf", "{}", "+"], check=True)
        subprocess.run(["find", ".", "-type", "f", "-name", "*.pyc", "-delete"], check=True)
        click.echo("✅ Python cache cleaned")
        
        # Remove checkpoints
        import shutil
        if Path(".checkpoints").exists():
            shutil.rmtree(".checkpoints")
            click.echo("✅ Checkpoints cleaned")
        
        click.echo("🎉 Cleanup completed")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Cleanup failed: {e}")


# ============================================================================
# Demo Command
# ============================================================================

@cli.command()
@click.option('--skip-infra', is_flag=True, help='Skip infrastructure checks')
@click.option('--quick', is_flag=True, help='Run quick demo with limited data')
@click.option('--continuous', is_flag=True, help='Run continuous pipeline mode')
def demo(skip_infra, quick, continuous):
    """Run a complete demonstration of the Breadth/Thrust Signals system."""
    click.echo("🎯 Breadth/Thrust Signals POC Demo")
    click.echo("=" * 50)
    
    # Check infrastructure
    if not skip_infra:
        click.echo("🔧 Checking infrastructure...")
        health = check_infrastructure_health()
        
        if not health["all_healthy"]:
            click.echo("❌ Infrastructure not ready. Starting services...")
            subprocess.run(["docker-compose", "-f", "infra/docker-compose.yml", "up", "-d"])
            time.sleep(30)
            health = check_infrastructure_health()
            
            if not health["all_healthy"]:
                click.echo("❌ Failed to start infrastructure")
                click.echo("💡 Try: poetry run bf infra start")
                return
        
        click.echo("✅ Infrastructure ready")
    else:
        click.echo("⏭️  Skipping infrastructure checks")
    
    # Demo configuration
    if quick:
        symbol_list = "demo_small"
        start_date = "2024-06-01"
        end_date = "2024-12-31"
        initial_capital = 50000
        click.echo("⚡ Running quick demo with limited data")
    else:
        symbol_list = "demo_medium"
        start_date = "2024-01-01"
        end_date = "2024-12-31"
        initial_capital = 100000
        click.echo("🚀 Running full demo")
    
    if continuous:
        click.echo("🔄 Running in continuous pipeline mode")
        click.echo("📊 Pipeline will run continuously until stopped")
        click.echo("⏹️  Press Ctrl+C to stop the pipeline")
    
    try:
        # Step 1: Data Summary
        click.echo("\n📊 Step 1: Data Summary")
        click.echo("-" * 30)
        subprocess.run(["poetry", "run", "bf", "data", "summary"], check=True)
        
        # Step 2: Fetch Sample Data
        click.echo("\n📥 Step 2: Fetch Sample Data")
        click.echo("-" * 30)
        subprocess.run([
            "poetry", "run", "bf", "data", "fetch",
            "--symbol-list", symbol_list,
            "--start-date", start_date,
            "--end-date", end_date
        ], check=True)
        
        # Step 3: Generate Signals
        click.echo("\n🎯 Step 3: Generate Signals")
        click.echo("-" * 30)
        subprocess.run([
            "poetry", "run", "bf", "signals", "generate",
            "--start-date", start_date,
            "--end-date", end_date,
            "--symbol-list", symbol_list
        ], check=True)
        
        # Step 4: Signal Summary
        click.echo("\n📊 Step 4: Signal Summary")
        click.echo("-" * 30)
        subprocess.run(["poetry", "run", "bf", "signals", "summary"], check=True)
        
        # Step 5: Run Backtest
        click.echo("\n📈 Step 5: Run Backtest")
        click.echo("-" * 30)
        subprocess.run([
            "poetry", "run", "bf", "backtest", "run",
            "--from-date", start_date,
            "--to-date", end_date,
            "--symbol-list", symbol_list,
            "--initial-capital", str(initial_capital),
            "--save-results"
        ], check=True)
        
        # Step 6: Analyze Results
        click.echo("\n📊 Step 6: Analyze Results")
        click.echo("-" * 30)
        subprocess.run(["poetry", "run", "bf", "backtest", "analyze"], check=True)
        
        # Demo completion
        click.echo("\n🎉 Demo completed successfully!")
        click.echo("=" * 50)
        
        # Show next steps
        click.echo("💡 Next Steps:")
        click.echo("   • Check web interfaces for detailed analysis")
        click.echo("   • Explore different time periods and symbols")
        click.echo("   • Adjust backtest parameters for optimization")
        click.echo("   • Run individual commands for specific analysis")
        
        # Show service URLs
        click.echo("\n🌐 Web Interfaces:")
        show_service_urls()
        
        # Show available commands
        click.echo("\n🔧 Available Commands:")
        click.echo("   • poetry run bf data fetch --help")
        click.echo("   • poetry run bf signals generate --help")
        click.echo("   • poetry run bf backtest run --help")
        click.echo("   • poetry run bf --help")
        
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Demo step failed: {e}")
        click.echo("💡 Check the error message above and try running individual commands")
    except Exception as e:
        click.echo(f"❌ Demo failed: {e}")
        click.echo("💡 Make sure all dependencies are installed: poetry install")


@cli.command()
@click.option('--symbol-list', default='demo_small', help='Symbol list to use')
@click.option('--interval', default=300, help='Interval between runs in seconds (default: 5 minutes)')
@click.option('--start-date', default='2024-01-01', help='Start date for analysis')
@click.option('--end-date', default='2024-12-31', help='End date for analysis')
@click.option('--auto-start-infra', is_flag=True, help='Automatically start infrastructure')
def pipeline(symbol_list, interval, start_date, end_date, auto_start_infra):
    """Run continuous pipeline mode - fetches data, generates signals, and runs backtests continuously."""
    click.echo("🔄 Starting Continuous Pipeline Mode")
    click.echo("=" * 50)
    click.echo(f"📊 Symbol List: {symbol_list}")
    click.echo(f"⏰ Interval: {interval} seconds ({interval/60:.1f} minutes)")
    click.echo(f"📅 Date Range: {start_date} to {end_date}")
    click.echo(f"🏗️  Auto-start Infrastructure: {auto_start_infra}")
    click.echo()
    click.echo("🔄 Pipeline will run continuously until stopped")
    click.echo("⏹️  Press Ctrl+C to stop the pipeline")
    click.echo()
    
    # Start infrastructure if requested
    if auto_start_infra:
        click.echo("🚀 Starting infrastructure...")
        try:
            subprocess.run(
                ["docker", "compose", "-f", "infra/docker-compose.yml", "up", "-d"],
                check=True
            )
            click.echo("✅ Infrastructure started")
            time.sleep(30)  # Wait for services to be ready
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Failed to start infrastructure: {e}")
            return
    
    # Check infrastructure health
    health_status = check_infrastructure_health()
    if not health_status['all_healthy']:
        click.echo("❌ Infrastructure not healthy. Please start it first:")
        click.echo("   poetry run bf infra start")
        return
    
    click.echo("✅ Infrastructure is healthy")
    
    run_count = 0
    start_time = datetime.now()
    
    try:
        while True:
            run_count += 1
            run_start = datetime.now()
            
            click.echo(f"\n🔄 Pipeline Run #{run_count}")
            click.echo(f"⏰ Started at: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
            click.echo("-" * 40)
            
            try:
                # Step 1: Fetch Data
                click.echo("📥 Step 1: Fetching data...")
                fetch_result = subprocess.run([
                    "poetry", "run", "bf", "data", "fetch",
                    "--symbol-list", symbol_list,
                    "--start-date", start_date,
                    "--end-date", end_date
                ], capture_output=True, text=True, check=True)
                click.echo("✅ Data fetched successfully")
                
                # Step 2: Generate Signals
                click.echo("🎯 Step 2: Generating signals...")
                signal_result = subprocess.run([
                    "poetry", "run", "bf", "signals", "generate",
                    "--symbol-list", symbol_list,
                    "--start-date", start_date,
                    "--end-date", end_date
                ], capture_output=True, text=True, check=True)
                click.echo("✅ Signals generated successfully")
                
                # Step 3: Run Backtest
                click.echo("📈 Step 3: Running backtest...")
                backtest_result = subprocess.run([
                    "poetry", "run", "bf", "backtest", "run",
                    "--symbol-list", symbol_list,
                    "--from-date", start_date,
                    "--to-date", end_date,
                    "--save-results"
                ], capture_output=True, text=True, check=True)
                click.echo("✅ Backtest completed successfully")
                
                # Calculate run time
                run_end = datetime.now()
                run_duration = (run_end - run_start).total_seconds()
                
                click.echo(f"✅ Pipeline run #{run_count} completed in {run_duration:.1f}s")
                
                # Show summary
                total_duration = (datetime.now() - start_time).total_seconds()
                avg_duration = total_duration / run_count
                click.echo(f"📊 Total runs: {run_count}, Avg duration: {avg_duration:.1f}s")
                
                # Wait for next run
                if interval > 0:
                    click.echo(f"⏳ Waiting {interval} seconds until next run...")
                    click.echo(f"🕐 Next run at: {(datetime.now() + timedelta(seconds=interval)).strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(interval)
                else:
                    click.echo("🔄 Running immediately (no interval)")
                
            except subprocess.CalledProcessError as e:
                click.echo(f"❌ Pipeline run #{run_count} failed: {e}")
                click.echo(f"📄 Error output: {e.stderr}")
                click.echo("🔄 Continuing with next run...")
                time.sleep(interval)
            except Exception as e:
                click.echo(f"❌ Unexpected error in run #{run_count}: {e}")
                click.echo("🔄 Continuing with next run...")
                time.sleep(interval)
                
    except KeyboardInterrupt:
        click.echo("\n⏹️  Pipeline stopped by user")
        click.echo(f"📊 Total runs completed: {run_count}")
        click.echo(f"⏱️  Total time: {(datetime.now() - start_time).total_seconds():.1f}s")
        
        if auto_start_infra:
            click.echo("🛑 Stopping infrastructure...")
            try:
                subprocess.run(
                    ["docker", "compose", "-f", "infra/docker-compose.yml", "down"],
                    check=True
                )
                click.echo("✅ Infrastructure stopped")
            except subprocess.CalledProcessError as e:
                click.echo(f"❌ Failed to stop infrastructure: {e}")


@cli.command()
@click.option('--symbol-list', default='demo_small', help='Symbol list to use')
@click.option('--speed', default=60, help='Replay speed multiplier (default: 60x)')
@click.option('--auto-start-infra', is_flag=True, help='Automatically start infrastructure')
def stream(symbol_list, speed, auto_start_infra):
    """Run streaming mode - continuously replays data and generates real-time signals."""
    click.echo("🌊 Starting Streaming Mode")
    click.echo("=" * 50)
    click.echo(f"📊 Symbol List: {symbol_list}")
    click.echo(f"⚡ Speed: {speed}x (real-time)")
    click.echo(f"🏗️  Auto-start Infrastructure: {auto_start_infra}")
    click.echo()
    click.echo("🌊 Streaming will run continuously until stopped")
    click.echo("⏹️  Press Ctrl+C to stop the stream")
    click.echo()
    
    # Start infrastructure if requested
    if auto_start_infra:
        click.echo("🚀 Starting infrastructure...")
        try:
            subprocess.run(
                ["docker", "compose", "-f", "infra/docker-compose.yml", "up", "-d"],
                check=True
            )
            click.echo("✅ Infrastructure started")
            time.sleep(30)  # Wait for services to be ready
        except subprocess.CalledProcessError as e:
            click.echo(f"❌ Failed to start infrastructure: {e}")
            return
    
    # Check infrastructure health
    health_status = check_infrastructure_health()
    if not health_status['all_healthy']:
        click.echo("❌ Infrastructure not healthy. Please start it first:")
        click.echo("   poetry run bf infra start")
        return
    
    click.echo("✅ Infrastructure is healthy")
    
    try:
        # Start data replay in background
        click.echo("📡 Starting data replay...")
        replay_process = subprocess.Popen([
            "poetry", "run", "bf", "data", "replay",
            "--symbol-list", symbol_list,
            "--speed", str(speed)
        ])
        
        click.echo("✅ Data replay started")
        click.echo("🌊 Streaming mode active - data is flowing through the system")
        click.echo("📊 Check web interfaces for real-time monitoring:")
        show_service_urls()
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(10)  # Check every 10 seconds
                if replay_process.poll() is not None:
                    click.echo("❌ Data replay process stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            click.echo("\n⏹️  Stopping stream...")
            
    except Exception as e:
        click.echo(f"❌ Error in streaming mode: {e}")
    finally:
        # Clean up
        if 'replay_process' in locals():
            replay_process.terminate()
            click.echo("✅ Data replay stopped")
        
        if auto_start_infra:
            click.echo("🛑 Stopping infrastructure...")
            try:
                subprocess.run(
                    ["docker", "compose", "-f", "infra/docker-compose.yml", "down"],
                    check=True
                )
                click.echo("✅ Infrastructure stopped")
            except subprocess.CalledProcessError as e:
                click.echo(f"❌ Failed to stop infrastructure: {e}")


@cli.command()
@click.option('--symbol-list', default='demo_small', help='Symbol list to use')
@click.option('--interval', default=300, help='Interval between runs in seconds')
def enhanced(symbol_list, interval):
    """Run enhanced pipeline with Airflow-like features (task history, retries)."""
    click.echo("🔄 Enhanced Pipeline Mode")
    click.echo("=" * 50)
    click.echo(f"📊 Symbol List: {symbol_list}")
    click.echo(f"⏰ Interval: {interval} seconds")
    click.echo()
    click.echo("🔄 Pipeline will run continuously until stopped")
    click.echo("⏹️  Press Ctrl+C to stop the pipeline")
    click.echo()
    
    # Import and run enhanced pipeline
    try:
        from cli.enhanced_pipeline import enhanced_pipeline
        enhanced_pipeline.callback(symbol_list, interval)  # Fixed: only pass 2 arguments
    except ImportError:
        click.echo("❌ Enhanced pipeline not available. Please ensure cli/enhanced_pipeline.py exists.")
    except Exception as e:
        click.echo(f"❌ Error running enhanced pipeline: {e}")


@cli.command()
@click.option('--port', default=8081, help='Dashboard port (default: 8081)')
@click.option('--host', default='localhost', help='Dashboard host (default: localhost)')
def dashboard(port, host):
    """Start the pipeline dashboard web interface."""
    click.echo("🌐 Starting Pipeline Dashboard")
    click.echo("=" * 40)
    click.echo(f"📍 URL: http://{host}:{port}")
    click.echo(f"📊 Features: Pipeline history, task monitoring, run statistics")
    click.echo()
    click.echo("🌐 Dashboard will run until stopped")
    click.echo("⏹️  Press Ctrl+C to stop the dashboard")
    click.echo()
    
    try:
        from cli.enhanced_pipeline import start_dashboard_server
        start_dashboard_server(host, port)
    except ImportError:
        click.echo("❌ Dashboard not available. Please ensure cli/enhanced_pipeline.py exists.")
    except Exception as e:
        click.echo(f"❌ Error starting dashboard: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def check_infrastructure_health() -> dict:
    """Check health of all infrastructure services."""
    services = [
        ("http://localhost:8080", "Spark UI"),
        ("http://localhost:9000/minio/health/live", "MinIO"),
        ("http://localhost:9200/_cluster/health", "Elasticsearch"),
        ("http://localhost:5601/api/status", "Kibana"),
    ]
    
    results = {}
    healthy_count = 0
    
    for url, name in services:
        try:
            response = requests.get(url, timeout=5)
            is_healthy = response.status_code == 200
            results[name] = is_healthy
            if is_healthy:
                healthy_count += 1
        except requests.exceptions.RequestException:
            results[name] = False
    
    return {
        'services': results,
        'healthy_count': healthy_count,
        'total_count': len(services),
        'all_healthy': healthy_count == len(services)
    }


def show_service_urls():
    """Show service URLs."""
    click.echo("\n📋 Service URLs:")
    click.echo("  • Spark UI: http://localhost:8080")
    click.echo("  • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)")
    click.echo("  • Kibana: http://localhost:5601")
    click.echo("  • Elasticsearch: http://localhost:9200")
    click.echo("  • Kafka: localhost:9092")


if __name__ == '__main__':
    cli()
