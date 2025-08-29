#!/usr/bin/env python3
"""
Dashboard Connector

This script provides functions to connect the existing dashboard
to the new BreadthFlow abstraction system.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import the dashboard integration
from cli.dashboard_integration import (
    get_dashboard_integration,
    fetch_data_async,
    generate_signals_async,
    run_backtest_async,
    start_pipeline_async,
    stop_pipeline_async,
    get_pipeline_status_async,
    get_system_health_sync
)

logger = logging.getLogger(__name__)


class DashboardConnector:
    """
    Connector class to bridge the existing dashboard with the new abstraction system
    """
    
    def __init__(self):
        """Initialize the dashboard connector"""
        self.integration = get_dashboard_integration()
        logger.info("Dashboard connector initialized")
    
    async def execute_data_fetch(self, symbols: str, start_date: str, end_date: str, 
                                timeframe: str = '1day', data_source: str = 'yfinance') -> Dict[str, Any]:
        """
        Execute data fetch command (dashboard compatible)
        
        Args:
            symbols: Comma-separated symbols string
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            data_source: Data source
            
        Returns:
            Dictionary with results
        """
        try:
            # Parse symbols
            symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]
            
            if not symbols_list:
                return {
                    'success': False,
                    'error': 'No symbols specified'
                }
            
            # Execute fetch
            result = await fetch_data_async(
                symbols=symbols_list,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                data_source=data_source
            )
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'data_fetch',
                'symbols': symbols_list,
                'timeframe': timeframe,
                'data_source': data_source,
                'duration': result.get('duration', 0),
                'data_fetched': result.get('data_fetched', False),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return {
                'success': False,
                'command': 'data_fetch',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_signal_generation(self, symbols: str, start_date: str, end_date: str,
                                      timeframe: str = '1day') -> Dict[str, Any]:
        """
        Execute signal generation command (dashboard compatible)
        
        Args:
            symbols: Comma-separated symbols string
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Signal timeframe
            
        Returns:
            Dictionary with results
        """
        try:
            # Parse symbols
            symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]
            
            if not symbols_list:
                return {
                    'success': False,
                    'error': 'No symbols specified'
                }
            
            # Execute signal generation
            result = await generate_signals_async(
                symbols=symbols_list,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'signal_generation',
                'symbols': symbols_list,
                'timeframe': timeframe,
                'strategy': result.get('strategy', 'unknown'),
                'duration': result.get('duration', 0),
                'signals_generated': result.get('signals_generated', False),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {
                'success': False,
                'command': 'signal_generation',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_backtest(self, symbols: str, from_date: str, to_date: str,
                             timeframe: str = '1day', initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Execute backtest command (dashboard compatible)
        
        Args:
            symbols: Comma-separated symbols string
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timeframe: Backtest timeframe
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary with results
        """
        try:
            # Parse symbols
            symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]
            
            if not symbols_list:
                return {
                    'success': False,
                    'error': 'No symbols specified'
                }
            
            # Execute backtest
            result = await run_backtest_async(
                symbols=symbols_list,
                from_date=from_date,
                to_date=to_date,
                timeframe=timeframe,
                initial_capital=initial_capital
            )
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'backtest',
                'symbols': symbols_list,
                'timeframe': timeframe,
                'initial_capital': initial_capital,
                'duration': result.get('duration', 0),
                'backtest_completed': result.get('backtest_completed', False),
                'performance_metrics': result.get('performance_metrics', {}),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {
                'success': False,
                'command': 'backtest',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_pipeline_start(self, mode: str = 'demo', interval: str = '5m',
                                   timeframe: str = '1day', symbols: Optional[str] = None,
                                   data_source: str = 'yfinance') -> Dict[str, Any]:
        """
        Execute pipeline start command (dashboard compatible)
        
        Args:
            mode: Pipeline mode
            interval: Interval between runs
            timeframe: Data timeframe
            symbols: Comma-separated symbols (for custom mode)
            data_source: Data source
            
        Returns:
            Dictionary with results
        """
        try:
            symbols_list = None
            if mode == 'custom' and symbols:
                symbols_list = [s.strip() for s in symbols.split(',') if s.strip()]
            
            # Execute pipeline start
            result = await start_pipeline_async(
                mode=mode,
                interval=interval,
                timeframe=timeframe,
                symbols=symbols_list,
                data_source=data_source
            )
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'pipeline_start',
                'mode': mode,
                'interval': interval,
                'timeframe': timeframe,
                'data_source': data_source,
                'execution_id': result.get('execution_id'),
                'pipeline_started': result.get('pipeline_started', False),
                'symbols': result.get('symbols', []),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"Pipeline start failed: {e}")
            return {
                'success': False,
                'command': 'pipeline_start',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_pipeline_stop(self) -> Dict[str, Any]:
        """
        Execute pipeline stop command (dashboard compatible)
        
        Returns:
            Dictionary with results
        """
        try:
            # Execute pipeline stop
            result = await stop_pipeline_async()
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'pipeline_stop',
                'pipelines_stopped': result.get('pipelines_stopped', 0),
                'total_running': result.get('total_running', 0),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"Pipeline stop failed: {e}")
            return {
                'success': False,
                'command': 'pipeline_stop',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def execute_pipeline_status(self) -> Dict[str, Any]:
        """
        Execute pipeline status command (dashboard compatible)
        
        Returns:
            Dictionary with results
        """
        try:
            # Execute pipeline status
            result = await get_pipeline_status_async()
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'pipeline_status',
                'executions': result.get('executions', {}),
                'system_health': result.get('system_health', 'unknown'),
                'active_workflows': result.get('active_workflows', 0),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"Pipeline status failed: {e}")
            return {
                'success': False,
                'command': 'pipeline_status',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def execute_system_health(self) -> Dict[str, Any]:
        """
        Execute system health command (dashboard compatible)
        
        Returns:
            Dictionary with results
        """
        try:
            # Execute system health
            result = get_system_health_sync()
            
            # Format result for dashboard
            return {
                'success': result.get('success', False),
                'command': 'system_health',
                'overall_health': result.get('overall_health', 'unknown'),
                'health_checks': result.get('health_checks', 0),
                'metrics': result.get('metrics', 0),
                'alerts': result.get('alerts', 0),
                'timestamp': datetime.now().isoformat(),
                'error': result.get('error') if not result.get('success') else None
            }
            
        except Exception as e:
            logger.error(f"System health failed: {e}")
            return {
                'success': False,
                'command': 'system_health',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global connector instance
_dashboard_connector = None

def get_dashboard_connector() -> DashboardConnector:
    """Get the global dashboard connector instance"""
    global _dashboard_connector
    if _dashboard_connector is None:
        _dashboard_connector = DashboardConnector()
    return _dashboard_connector


# Convenience functions for easy integration
async def execute_command(command: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a dashboard command using the new abstraction system
    
    Args:
        command: Command to execute (data_fetch, signal_generation, backtest, etc.)
        **kwargs: Command-specific parameters
        
    Returns:
        Dictionary with command results
    """
    connector = get_dashboard_connector()
    
    if command == 'data_fetch':
        return await connector.execute_data_fetch(**kwargs)
    elif command == 'signal_generation':
        return await connector.execute_signal_generation(**kwargs)
    elif command == 'backtest':
        return await connector.execute_backtest(**kwargs)
    elif command == 'pipeline_start':
        return await connector.execute_pipeline_start(**kwargs)
    elif command == 'pipeline_stop':
        return await connector.execute_pipeline_stop()
    elif command == 'pipeline_status':
        return await connector.execute_pipeline_status()
    elif command == 'system_health':
        return connector.execute_system_health()
    else:
        return {
            'success': False,
            'error': f'Unknown command: {command}',
            'timestamp': datetime.now().isoformat()
        }


# Example usage functions
async def demo_dashboard_commands():
    """Demo all dashboard commands"""
    print("🚀 Demo Dashboard Commands with New Abstraction System")
    print("=" * 60)
    
    commands = [
        {
            'name': 'Data Fetch',
            'command': 'data_fetch',
            'params': {
                'symbols': 'AAPL,MSFT',
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'timeframe': '1day',
                'data_source': 'yfinance'
            }
        },
        {
            'name': 'Signal Generation',
            'command': 'signal_generation',
            'params': {
                'symbols': 'AAPL,MSFT',
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'timeframe': '1day'
            }
        },
        {
            'name': 'Backtest',
            'command': 'backtest',
            'params': {
                'symbols': 'AAPL,MSFT',
                'from_date': '2024-01-01',
                'to_date': '2024-01-31',
                'timeframe': '1day',
                'initial_capital': 100000
            }
        },
        {
            'name': 'Pipeline Start',
            'command': 'pipeline_start',
            'params': {
                'mode': 'demo',
                'interval': '5m',
                'timeframe': '1day',
                'data_source': 'yfinance'
            }
        },
        {
            'name': 'Pipeline Status',
            'command': 'pipeline_status',
            'params': {}
        },
        {
            'name': 'System Health',
            'command': 'system_health',
            'params': {}
        }
    ]
    
    for cmd in commands:
        print(f"\n📋 {cmd['name']}...")
        result = await execute_command(cmd['command'], **cmd['params'])
        
        if result.get('success'):
            print(f"   ✅ Success!")
            print(f"   Duration: {result.get('duration', 0):.2f}s")
            if result.get('command') == 'backtest' and result.get('performance_metrics'):
                print(f"   Performance Metrics: {len(result.get('performance_metrics', {}))} metrics")
        else:
            print(f"   ❌ Failed: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print("🎉 Dashboard command demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_dashboard_commands())
