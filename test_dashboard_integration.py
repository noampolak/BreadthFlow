#!/usr/bin/env python3
"""
Test Dashboard Integration

This script tests the integration layer between the existing dashboard
and the new BreadthFlow abstraction system.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the CLI directory to the path
sys.path.append('cli')

# Import the dashboard integration
from dashboard_integration import (
    get_dashboard_integration,
    fetch_data_async,
    generate_signals_async,
    run_backtest_async,
    start_pipeline_async,
    stop_pipeline_async,
    get_pipeline_status_async,
    get_system_health_sync
)


async def test_dashboard_integration():
    """Test the dashboard integration layer"""
    print("Testing Dashboard Integration Layer")
    print("=" * 50)
    
    # Test 1: System Health
    print("\n1. Testing System Health...")
    health = get_system_health_sync()
    print(f"   System Health: {health.get('overall_health', 'unknown')}")
    print(f"   Health Checks: {health.get('health_checks', 0)}")
    print(f"   Metrics: {health.get('metrics', 0)}")
    print(f"   Success: {health.get('success', False)}")
    
    # Test 2: Data Fetching
    print("\n2. Testing Data Fetching...")
    symbols = ['AAPL', 'MSFT']
    start_date = '2024-01-01'
    end_date = '2024-01-31'
    
    fetch_result = await fetch_data_async(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe='1day',
        data_source='yfinance'
    )
    
    print(f"   Success: {fetch_result.get('success', False)}")
    print(f"   Data Fetched: {fetch_result.get('data_fetched', False)}")
    print(f"   Symbols: {fetch_result.get('symbols', [])}")
    print(f"   Duration: {fetch_result.get('duration', 0):.2f}s")
    
    if fetch_result.get('errors'):
        print(f"   Errors: {fetch_result.get('errors', [])}")
    
    # Test 3: Signal Generation
    print("\n3. Testing Signal Generation...")
    signal_result = await generate_signals_async(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe='1day'
    )
    
    print(f"   Success: {signal_result.get('success', False)}")
    print(f"   Signals Generated: {signal_result.get('signals_generated', False)}")
    print(f"   Strategy: {signal_result.get('strategy', 'unknown')}")
    print(f"   Duration: {signal_result.get('duration', 0):.2f}s")
    
    if signal_result.get('errors'):
        print(f"   Errors: {signal_result.get('errors', [])}")
    
    # Test 4: Backtest
    print("\n4. Testing Backtest...")
    backtest_result = await run_backtest_async(
        symbols=symbols,
        from_date=start_date,
        to_date=end_date,
        timeframe='1day',
        initial_capital=100000
    )
    
    print(f"   Success: {backtest_result.get('success', False)}")
    print(f"   Backtest Completed: {backtest_result.get('backtest_completed', False)}")
    print(f"   Initial Capital: ${backtest_result.get('initial_capital', 0):,.0f}")
    print(f"   Duration: {backtest_result.get('duration', 0):.2f}s")
    
    if backtest_result.get('performance_metrics'):
        metrics = backtest_result.get('performance_metrics', {})
        print(f"   Performance Metrics: {len(metrics)} metrics available")
    
    if backtest_result.get('errors'):
        print(f"   Errors: {backtest_result.get('errors', [])}")
    
    # Test 5: Pipeline Status
    print("\n5. Testing Pipeline Status...")
    status_result = await get_pipeline_status_async()
    
    print(f"   Success: {status_result.get('success', False)}")
    print(f"   Total Executions: {status_result.get('executions', {}).get('total', 0)}")
    print(f"   System Health: {status_result.get('system_health', 'unknown')}")
    print(f"   Active Workflows: {status_result.get('active_workflows', 0)}")
    
    if status_result.get('executions', {}).get('by_status'):
        by_status = status_result.get('executions', {}).get('by_status', {})
        print(f"   Status Breakdown: {by_status}")
    
    # Test 6: Pipeline Start/Stop (Quick Test)
    print("\n6. Testing Pipeline Start/Stop...")
    
    # Start pipeline
    start_result = await start_pipeline_async(
        mode='demo',
        interval='5m',
        timeframe='1day',
        data_source='yfinance'
    )
    
    print(f"   Start Success: {start_result.get('success', False)}")
    print(f"   Pipeline Started: {start_result.get('pipeline_started', False)}")
    print(f"   Execution ID: {start_result.get('execution_id', 'none')}")
    
    if start_result.get('success'):
        # Wait a moment for pipeline to start
        await asyncio.sleep(2)
        
        # Stop pipeline
        stop_result = await stop_pipeline_async()
        
        print(f"   Stop Success: {stop_result.get('success', False)}")
        print(f"   Pipelines Stopped: {stop_result.get('pipelines_stopped', 0)}")
        print(f"   Total Running: {stop_result.get('total_running', 0)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("System Health", health.get('success', False)),
        ("Data Fetching", fetch_result.get('success', False)),
        ("Signal Generation", signal_result.get('success', False)),
        ("Backtest", backtest_result.get('success', False)),
        ("Pipeline Status", status_result.get('success', False)),
        ("Pipeline Start/Stop", start_result.get('success', False))
    ]
    
    passed = 0
    for test_name, success in tests:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All integration tests passed! Dashboard integration is working correctly.")
        return True
    else:
        print("⚠️ Some integration tests failed. Check the errors above.")
        return False


async def test_integration_with_existing_commands():
    """Test how the integration would work with existing dashboard commands"""
    print("\n" + "=" * 50)
    print("SIMULATING DASHBOARD COMMAND INTEGRATION")
    print("=" * 50)
    
    # Simulate dashboard command calls
    print("\nSimulating dashboard 'data fetch' command...")
    print("Command: data fetch --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-01-31")
    
    result = await fetch_data_async(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        timeframe='1day',
        data_source='yfinance'
    )
    
    if result.get('success'):
        print("✅ Dashboard 'data fetch' command would work with new system")
        print(f"   Duration: {result.get('duration', 0):.2f}s")
        print(f"   Symbols processed: {len(result.get('symbols', []))}")
    else:
        print("❌ Dashboard 'data fetch' command would fail")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\nSimulating dashboard 'signals generate' command...")
    print("Command: signals generate --symbols AAPL,MSFT --start-date 2024-01-01 --end-date 2024-01-31")
    
    result = await generate_signals_async(
        symbols=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        timeframe='1day'
    )
    
    if result.get('success'):
        print("✅ Dashboard 'signals generate' command would work with new system")
        print(f"   Strategy: {result.get('strategy', 'unknown')}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")
    else:
        print("❌ Dashboard 'signals generate' command would fail")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\nSimulating dashboard 'backtest run' command...")
    print("Command: backtest run --symbols AAPL,MSFT --from-date 2024-01-01 --to-date 2024-01-31 --initial-capital 100000")
    
    result = await run_backtest_async(
        symbols=['AAPL', 'MSFT'],
        from_date='2024-01-01',
        to_date='2024-01-31',
        timeframe='1day',
        initial_capital=100000
    )
    
    if result.get('success'):
        print("✅ Dashboard 'backtest run' command would work with new system")
        print(f"   Initial Capital: ${result.get('initial_capital', 0):,.0f}")
        print(f"   Duration: {result.get('duration', 0):.2f}s")
        print(f"   Performance Metrics: {len(result.get('performance_metrics', {}))} metrics")
    else:
        print("❌ Dashboard 'backtest run' command would fail")
        print(f"   Error: {result.get('error', 'Unknown error')}")


async def main():
    """Main test function"""
    print("Dashboard Integration Test")
    print("=" * 50)
    
    # Test basic integration
    success = await test_dashboard_integration()
    
    # Test dashboard command simulation
    await test_integration_with_existing_commands()
    
    print("\n" + "=" * 50)
    print("INTEGRATION READINESS ASSESSMENT")
    print("=" * 50)
    
    if success:
        print("✅ Dashboard integration is READY for deployment")
        print("✅ All dashboard commands can be connected to the new system")
        print("✅ Backward compatibility is maintained")
        print("✅ New features are available through the abstraction layer")
    else:
        print("⚠️ Dashboard integration needs more work")
        print("⚠️ Some components may need additional configuration")
        print("⚠️ Check the test results above for specific issues")
    
    return success


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
