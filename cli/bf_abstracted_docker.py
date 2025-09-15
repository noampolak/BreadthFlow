#!/usr/bin/env python3
"""
BreadthFlow Abstracted CLI - Docker Integration

This CLI integrates the new abstraction system with the existing Docker infrastructure.
It follows the same pattern as kibana_enhanced_bf.py but uses the new workflow manager.
"""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import click

# Add the jobs directory to Python path (Docker container path)
sys.path.insert(0, "/opt/bitnami/spark/jobs")

# Add the new module directories to the path
sys.path.extend(
    [
        "/opt/bitnami/spark/jobs/model/registry",
        "/opt/bitnami/spark/jobs/model/config",
        "/opt/bitnami/spark/jobs/model/logging",
        "/opt/bitnami/spark/jobs/model/data",
        "/opt/bitnami/spark/jobs/model/data/resources",
        "/opt/bitnami/spark/jobs/model/data/sources",
        "/opt/bitnami/spark/jobs/model/signals",
        "/opt/bitnami/spark/jobs/model/signals/components",
        "/opt/bitnami/spark/jobs/model/signals/strategies",
        "/opt/bitnami/spark/jobs/model/backtesting",
        "/opt/bitnami/spark/jobs/model/backtesting/execution",
        "/opt/bitnami/spark/jobs/model/backtesting/risk",
        "/opt/bitnami/spark/jobs/model/backtesting/analytics",
        "/opt/bitnami/spark/jobs/model/backtesting/engines",
        "/opt/bitnami/spark/jobs/model/orchestration",
    ]
)

# Import logging system
import logging

# Import the dashboard integration
from cli.dashboard_integration import (
    fetch_data_async,
    generate_signals_async,
    get_dashboard_integration,
    get_pipeline_status_async,
    get_system_health_sync,
    run_backtest_async,
    start_pipeline_async,
    stop_pipeline_async,
)

es_logger = logging.getLogger(__name__)

# Simple pipeline run tracking (same as kibana_enhanced_bf.py)
import sqlite3
from datetime import datetime


def log_pipeline_run(run_id, command, status, duration=None, error_message=None, metadata=None):
    """Log pipeline run to PostgreSQL database"""
    try:
        from datetime import datetime

        import psycopg2

        # Connect to PostgreSQL using environment variable
        DATABASE_URL = "postgresql://pipeline:pipeline123@breadthflow-postgres:5432/breadthflow"

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute(
            """
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
        """
        )

        # Insert or update pipeline run
        if status == "running":
            cursor.execute(
                """
                INSERT INTO pipeline_runs (run_id, command, status, start_time)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status
            """,
                (run_id, command, status, datetime.now()),
            )
        else:
            cursor.execute(
                """
                UPDATE pipeline_runs 
                SET status = %s, end_time = %s, duration = %s, error_message = %s
                WHERE run_id = %s
            """,
                (status, datetime.now(), duration, error_message, run_id),
            )

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("breadthflow_abstracted.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """BreadthFlow Abstracted CLI - Docker Integration"""
    pass


@cli.command()
@click.option("--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated list of symbols")
@click.option("--timeframe", default="1day", help="Timeframe (1min, 5min, 15min, 1hour, 1day)")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--data-source", default="yfinance", help="Data source (yfinance, alpha_vantage, polygon)")
def data_fetch(symbols, timeframe, start_date, end_date, data_source):
    """Fetch data using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = f"data_fetch --symbols {symbols} --timeframe {timeframe} --data-source {data_source}"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Convert to async execution
        result = asyncio.run(
            fetch_data_async(
                symbols=symbols.split(","),
                timeframe=timeframe,
                start_date=start_date or "2024-01-01",
                end_date=end_date or "2024-01-31",
                data_source=data_source,
            )
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ Data fetch completed successfully")
            print(f"📊 Fetched data for {len(symbols.split(','))} symbols")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Data fetch failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Data fetch failed: {e}")


@cli.command()
@click.option("--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated list of symbols")
@click.option("--timeframe", default="1day", help="Timeframe (1min, 5min, 15min, 1hour, 1day)")
@click.option("--strategy", default="technical", help="Signal strategy (technical, fundamental, sentiment)")
def signals_generate(symbols, timeframe, strategy):
    """Generate signals using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = f"signals_generate --symbols {symbols} --timeframe {timeframe} --strategy {strategy}"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Convert to async execution
        result = asyncio.run(
            generate_signals_async(
                symbols=symbols.split(","), start_date="2024-01-01", end_date="2024-01-31", timeframe=timeframe
            )
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ Signal generation completed successfully")
            print(f"📊 Generated signals for {len(symbols.split(','))} symbols")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Signal generation failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Signal generation failed: {e}")


@cli.command()
@click.option("--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated list of symbols")
@click.option("--timeframe", default="1day", help="Timeframe (1min, 5min, 15min, 1hour, 1day)")
@click.option("--start-date", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", help="End date (YYYY-MM-DD)")
@click.option("--initial-capital", default=100000, help="Initial capital")
def backtest_run(symbols, timeframe, start_date, end_date, initial_capital):
    """Run backtesting using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = f"backtest_run --symbols {symbols} --timeframe {timeframe} --initial-capital {initial_capital}"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Convert to async execution
        result = asyncio.run(
            run_backtest_async(
                symbols=symbols.split(","),
                from_date=start_date or "2024-01-01",
                to_date=end_date or "2024-01-31",
                timeframe=timeframe,
                initial_capital=initial_capital,
            )
        )

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ Backtesting completed successfully")
            print(f"📊 Backtested {len(symbols.split(','))} symbols")
            print(f"💰 Initial Capital: ${initial_capital:,}")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Backtesting failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Backtesting failed: {e}")


@cli.command()
@click.option("--mode", default="demo", help="Pipeline mode (demo, small, medium, full)")
def pipeline_start(mode):
    """Start pipeline using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = f"pipeline_start --mode {mode}"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Convert to async execution
        result = asyncio.run(start_pipeline_async(mode=mode))

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ Pipeline started successfully")
            print(f"🎮 Mode: {mode}")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Pipeline start failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Pipeline start failed: {e}")


@cli.command()
def pipeline_stop():
    """Stop pipeline using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = "pipeline_stop"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Convert to async execution
        result = asyncio.run(stop_pipeline_async())

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ Pipeline stopped successfully")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Pipeline stop failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Pipeline stop failed: {e}")


@cli.command()
def pipeline_status():
    """Get pipeline status using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = "pipeline_status"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Convert to async execution
        result = asyncio.run(get_pipeline_status_async())

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ Pipeline status retrieved successfully")
            status = result.get("status", {})
            print(f"📊 Status: {status.get('status', 'Unknown')}")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Pipeline status failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Pipeline status failed: {e}")


@cli.command()
def health():
    """Get system health using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = "health"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        # Get system health
        result = get_system_health_sync()

        duration = (datetime.now() - start_time).total_seconds()

        if result.get("success"):
            log_pipeline_run(run_id, command, "completed", duration)
            print(f"✅ System health check completed")
            health = result.get("health", {})
            print(f"🏥 Overall Health: {health.get('overall_health', 'Unknown')}")
            print(f"⏱️ Duration: {duration:.2f}s")
        else:
            error_msg = result.get("error", "Unknown error")
            log_pipeline_run(run_id, command, "failed", duration, error_msg)
            print(f"❌ Health check failed: {error_msg}")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Health check failed: {e}")


@cli.command()
def demo():
    """Run a complete demo using the new abstraction system"""
    run_id = str(uuid.uuid4())
    command = "demo"

    try:
        log_pipeline_run(run_id, command, "running")
        start_time = datetime.now()

        print("🚀 Starting BreadthFlow Abstracted Demo...")

        # Step 1: Data Fetch
        print("\n📊 Step 1: Fetching Data...")
        data_result = asyncio.run(
            fetch_data_async(
                symbols=["AAPL", "MSFT"],
                timeframe="1day",
                start_date="2024-01-01",
                end_date="2024-01-31",
                data_source="yfinance",
            )
        )

        if not data_result.get("success"):
            raise Exception(f"Data fetch failed: {data_result.get('error')}")

        # Step 2: Signal Generation
        print("\n🎯 Step 2: Generating Signals...")
        signal_result = asyncio.run(generate_signals_async(symbols=["AAPL", "MSFT"], timeframe="1day", strategy="technical"))

        if not signal_result.get("success"):
            raise Exception(f"Signal generation failed: {signal_result.get('error')}")

        # Step 3: Backtesting
        print("\n📈 Step 3: Running Backtest...")
        backtest_result = asyncio.run(
            run_backtest_async(
                symbols=["AAPL", "MSFT"],
                timeframe="1day",
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_capital=100000,
            )
        )

        if not backtest_result.get("success"):
            raise Exception(f"Backtesting failed: {backtest_result.get('error')}")

        # Step 4: System Health
        print("\n🏥 Step 4: Checking System Health...")
        health_result = get_system_health_sync()

        duration = (datetime.now() - start_time).total_seconds()

        log_pipeline_run(run_id, command, "completed", duration)
        print(f"\n✅ Demo completed successfully!")
        print(f"📊 Data Fetch: ✅")
        print(f"🎯 Signal Generation: ✅")
        print(f"📈 Backtesting: ✅")
        print(f"🏥 System Health: ✅")
        print(f"⏱️ Total Duration: {duration:.2f}s")

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        log_pipeline_run(run_id, command, "failed", duration, str(e))
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    cli()
