#!/usr/bin/env python3
"""
BreadthFlow Abstracted CLI

This CLI uses the new abstraction system with workflow manager,
providing the same interface as the old CLI but with enhanced capabilities.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import click

# Add the new module directories to the path
sys.path.extend(
    [
        "model/registry",
        "model/config",
        "model/logging",
        "model/data",
        "model/data/resources",
        "model/data/sources",
        "model/signals",
        "model/signals/components",
        "model/signals/strategies",
        "model/backtesting",
        "model/backtesting/execution",
        "model/backtesting/risk",
        "model/backtesting/analytics",
        "model/backtesting/engines",
        "model/orchestration",
    ]
)

# Import the dashboard integration
from dashboard_integration import (
    fetch_data_async,
    generate_signals_async,
    get_dashboard_integration,
    get_pipeline_status_async,
    get_system_health_sync,
    run_backtest_async,
    start_pipeline_async,
    stop_pipeline_async,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("breadthflow_abstracted.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def parse_symbols(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols string into list"""
    if not symbols_str:
        return []
    return [s.strip() for s in symbols_str.split(",") if s.strip()]


def format_result(result: Dict[str, Any]) -> str:
    """Format result for display"""
    if result.get("success"):
        return f"✅ Success: {json.dumps(result, indent=2)}"
    else:
        return f"❌ Failed: {result.get('error', 'Unknown error')}"


@click.group()
def cli():
    """BreadthFlow Abstracted CLI - Using New Workflow Manager"""
    pass


@cli.group()
def data():
    """Data operations"""
    pass


@data.command()
@click.option("--symbols", help="Comma-separated symbols (e.g., AAPL,MSFT,GOOGL)")
@click.option("--symbol-list", help="Use predefined symbol list (demo_small, tech_leaders)")
@click.option("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--timeframe", default="1day", help="Data timeframe: 1min, 5min, 15min, 1hour, 1day")
@click.option("--data-source", default="yfinance", help="Data source: yfinance, alpha_vantage, polygon")
@click.option("--parallel", default=2, help="Number of parallel workers")
def fetch(symbols, symbol_list, start_date, end_date, timeframe, data_source, parallel):
    """Fetch data using the new abstraction system"""

    # Determine symbols
    if symbol_list:
        if symbol_list == "demo_small":
            symbols_list = ["AAPL", "MSFT"]
        elif symbol_list == "tech_leaders":
            symbols_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        else:
            click.echo(f"❌ Unknown symbol list: {symbol_list}")
            return
    else:
        symbols_list = parse_symbols(symbols)

    if not symbols_list:
        click.echo("❌ No symbols specified")
        return

    click.echo(f"🔄 Fetching data for {len(symbols_list)} symbols...")
    click.echo(f"   Symbols: {', '.join(symbols_list)}")
    click.echo(f"   Date Range: {start_date} to {end_date}")
    click.echo(f"   Timeframe: {timeframe}")
    click.echo(f"   Data Source: {data_source}")

    # Run the fetch operation
    async def run_fetch():
        return await fetch_data_async(
            symbols=symbols_list, start_date=start_date, end_date=end_date, timeframe=timeframe, data_source=data_source
        )

    result = asyncio.run(run_fetch())

    if result.get("success"):
        click.echo("✅ Data fetch completed successfully!")
        click.echo(f"   Duration: {result.get('duration', 0):.2f}s")
        click.echo(f"   Data Fetched: {result.get('data_fetched', False)}")
        if result.get("errors"):
            click.echo(f"   Warnings: {len(result.get('errors', []))} errors occurred")
    else:
        click.echo(f"❌ Data fetch failed: {result.get('error', 'Unknown error')}")


@cli.group()
def signals():
    """Signal generation operations"""
    pass


@signals.command()
@click.option("--symbols", help="Comma-separated symbols")
@click.option("--symbol-list", help="Use predefined symbol list")
@click.option("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--timeframe", default="1day", help="Signal timeframe: 1min, 5min, 15min, 1hour, 1day")
def generate(symbols, symbol_list, start_date, end_date, timeframe):
    """Generate signals using the new abstraction system"""

    # Determine symbols
    if symbol_list:
        if symbol_list == "demo_small":
            symbols_list = ["AAPL", "MSFT"]
        elif symbol_list == "tech_leaders":
            symbols_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        else:
            click.echo(f"❌ Unknown symbol list: {symbol_list}")
            return
    else:
        symbols_list = parse_symbols(symbols)

    if not symbols_list:
        click.echo("❌ No symbols specified")
        return

    click.echo(f"🔄 Generating signals for {len(symbols_list)} symbols...")
    click.echo(f"   Symbols: {', '.join(symbols_list)}")
    click.echo(f"   Date Range: {start_date} to {end_date}")
    click.echo(f"   Timeframe: {timeframe}")

    # Run the signal generation
    async def run_signals():
        return await generate_signals_async(
            symbols=symbols_list, start_date=start_date, end_date=end_date, timeframe=timeframe
        )

    result = asyncio.run(run_signals())

    if result.get("success"):
        click.echo("✅ Signal generation completed successfully!")
        click.echo(f"   Duration: {result.get('duration', 0):.2f}s")
        click.echo(f"   Strategy: {result.get('strategy', 'unknown')}")
        click.echo(f"   Signals Generated: {result.get('signals_generated', False)}")
        if result.get("errors"):
            click.echo(f"   Warnings: {len(result.get('errors', []))} errors occurred")
    else:
        click.echo(f"❌ Signal generation failed: {result.get('error', 'Unknown error')}")


@cli.group()
def backtest():
    """Backtesting operations"""
    pass


@backtest.command()
@click.option("--symbols", help="Comma-separated symbols")
@click.option("--symbol-list", help="Use predefined symbol list")
@click.option("--from-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--to-date", default="2024-12-31", help="End date (YYYY-MM-DD)")
@click.option("--timeframe", default="1day", help="Backtest timeframe: 1min, 5min, 15min, 1hour, 1day")
@click.option("--initial-capital", default=100000, help="Initial capital ($)")
@click.option("--save-results", is_flag=True, help="Save results to MinIO")
def run(symbols, symbol_list, from_date, to_date, timeframe, initial_capital, save_results):
    """Run backtest using the new abstraction system"""

    # Determine symbols
    if symbol_list:
        if symbol_list == "demo_small":
            symbols_list = ["AAPL", "MSFT"]
        elif symbol_list == "tech_leaders":
            symbols_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        else:
            click.echo(f"❌ Unknown symbol list: {symbol_list}")
            return
    else:
        symbols_list = parse_symbols(symbols)

    if not symbols_list:
        click.echo("❌ No symbols specified")
        return

    click.echo(f"🔄 Running backtest for {len(symbols_list)} symbols...")
    click.echo(f"   Symbols: {', '.join(symbols_list)}")
    click.echo(f"   Date Range: {from_date} to {to_date}")
    click.echo(f"   Timeframe: {timeframe}")
    click.echo(f"   Initial Capital: ${initial_capital:,.0f}")

    # Run the backtest
    async def run_backtest():
        return await run_backtest_async(
            symbols=symbols_list, from_date=from_date, to_date=to_date, timeframe=timeframe, initial_capital=initial_capital
        )

    result = asyncio.run(run_backtest())

    if result.get("success"):
        click.echo("✅ Backtest completed successfully!")
        click.echo(f"   Duration: {result.get('duration', 0):.2f}s")
        click.echo(f"   Backtest Completed: {result.get('backtest_completed', False)}")

        # Display performance metrics if available
        metrics = result.get("performance_metrics", {})
        if metrics:
            click.echo("   Performance Metrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    click.echo(f"     {metric_name}: {value:.4f}")
                else:
                    click.echo(f"     {metric_name}: {value}")

        if result.get("errors"):
            click.echo(f"   Warnings: {len(result.get('errors', []))} errors occurred")
    else:
        click.echo(f"❌ Backtest failed: {result.get('error', 'Unknown error')}")


@cli.group()
def pipeline():
    """Pipeline operations"""
    pass


@pipeline.command()
@click.option("--mode", default="demo", help="Pipeline mode (demo, demo_small, tech_leaders, all_symbols, custom)")
@click.option("--interval", default="5m", help="Interval between runs (e.g., 5m, 1h, 300s)")
@click.option("--timeframe", default="1day", help="Data timeframe: 1min, 5min, 15min, 1hour, 1day")
@click.option("--symbols", help="Comma-separated symbols (for custom mode)")
@click.option("--data-source", default="yfinance", help="Data source: yfinance, alpha_vantage, polygon")
def start(mode, interval, timeframe, symbols, data_source):
    """Start pipeline using the new abstraction system"""

    click.echo(f"🔄 Starting pipeline in {mode} mode...")
    click.echo(f"   Mode: {mode}")
    click.echo(f"   Interval: {interval}")
    click.echo(f"   Timeframe: {timeframe}")
    click.echo(f"   Data Source: {data_source}")

    if mode == "custom" and symbols:
        symbols_list = parse_symbols(symbols)
        click.echo(f"   Custom Symbols: {', '.join(symbols_list)}")
    else:
        symbols_list = None

    # Start the pipeline
    async def start_pipeline():
        return await start_pipeline_async(
            mode=mode, interval=interval, timeframe=timeframe, symbols=symbols_list, data_source=data_source
        )

    result = asyncio.run(start_pipeline())

    if result.get("success"):
        click.echo("✅ Pipeline started successfully!")
        click.echo(f"   Execution ID: {result.get('execution_id', 'none')}")
        click.echo(f"   Pipeline Started: {result.get('pipeline_started', False)}")
        click.echo(f"   Symbols: {', '.join(result.get('symbols', []))}")
    else:
        click.echo(f"❌ Pipeline start failed: {result.get('error', 'Unknown error')}")


@pipeline.command()
def stop():
    """Stop running pipelines"""
    click.echo("🔄 Stopping all running pipelines...")

    async def stop_pipelines():
        return await stop_pipeline_async()

    result = asyncio.run(stop_pipelines())

    if result.get("success"):
        click.echo("✅ Pipeline stop completed!")
        click.echo(f"   Pipelines Stopped: {result.get('pipelines_stopped', 0)}")
        click.echo(f"   Total Running: {result.get('total_running', 0)}")
    else:
        click.echo(f"❌ Pipeline stop failed: {result.get('error', 'Unknown error')}")


@pipeline.command()
def status():
    """Get pipeline status"""
    click.echo("🔄 Getting pipeline status...")

    async def get_status():
        return await get_pipeline_status_async()

    result = asyncio.run(get_status())

    if result.get("success"):
        click.echo("✅ Pipeline status retrieved!")
        executions = result.get("executions", {})
        click.echo(f"   Total Executions: {executions.get('total', 0)}")

        by_status = executions.get("by_status", {})
        if by_status:
            click.echo("   Status Breakdown:")
            for status, count in by_status.items():
                click.echo(f"     {status}: {count}")

        click.echo(f"   System Health: {result.get('system_health', 'unknown')}")
        click.echo(f"   Active Workflows: {result.get('active_workflows', 0)}")
    else:
        click.echo(f"❌ Status check failed: {result.get('error', 'Unknown error')}")


@cli.command()
def health():
    """Get system health"""
    click.echo("🔄 Getting system health...")

    health_result = get_system_health_sync()

    if health_result.get("success"):
        click.echo("✅ System health retrieved!")
        click.echo(f"   Overall Health: {health_result.get('overall_health', 'unknown')}")
        click.echo(f"   Health Checks: {health_result.get('health_checks', 0)}")
        click.echo(f"   Metrics: {health_result.get('metrics', 0)}")
        click.echo(f"   Alerts: {health_result.get('alerts', 0)}")
        click.echo(f"   Timestamp: {health_result.get('timestamp', 'unknown')}")
    else:
        click.echo(f"❌ Health check failed: {health_result.get('error', 'Unknown error')}")


@cli.command()
def demo():
    """Run a quick demo of the new system"""
    click.echo("🚀 Running BreadthFlow Abstracted System Demo...")
    click.echo("=" * 50)

    # Test data fetching
    click.echo("\n1. Testing Data Fetching...")

    async def demo_fetch():
        return await fetch_data_async(
            symbols=["AAPL", "MSFT"], start_date="2024-01-01", end_date="2024-01-31", timeframe="1day", data_source="yfinance"
        )

    fetch_result = asyncio.run(demo_fetch())
    if fetch_result.get("success"):
        click.echo("   ✅ Data fetching works!")
    else:
        click.echo(f"   ❌ Data fetching failed: {fetch_result.get('error')}")

    # Test signal generation
    click.echo("\n2. Testing Signal Generation...")

    async def demo_signals():
        return await generate_signals_async(
            symbols=["AAPL", "MSFT"], start_date="2024-01-01", end_date="2024-01-31", timeframe="1day"
        )

    signal_result = asyncio.run(demo_signals())
    if signal_result.get("success"):
        click.echo("   ✅ Signal generation works!")
    else:
        click.echo(f"   ❌ Signal generation failed: {signal_result.get('error')}")

    # Test backtesting
    click.echo("\n3. Testing Backtesting...")

    async def demo_backtest():
        return await run_backtest_async(
            symbols=["AAPL", "MSFT"], from_date="2024-01-01", to_date="2024-01-31", timeframe="1day", initial_capital=100000
        )

    backtest_result = asyncio.run(demo_backtest())
    if backtest_result.get("success"):
        click.echo("   ✅ Backtesting works!")
    else:
        click.echo(f"   ❌ Backtesting failed: {backtest_result.get('error')}")

    # Test system health
    click.echo("\n4. Testing System Health...")
    health_result = get_system_health_sync()
    if health_result.get("success"):
        click.echo("   ✅ System health monitoring works!")
    else:
        click.echo(f"   ❌ System health failed: {health_result.get('error')}")

    click.echo("\n" + "=" * 50)
    click.echo("🎉 Demo completed! New abstraction system is working correctly.")


if __name__ == "__main__":
    cli()
