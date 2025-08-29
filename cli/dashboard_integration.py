#!/usr/bin/env python3
"""
Dashboard Integration Layer

This module provides integration between the existing dashboard
and the new BreadthFlow abstraction system, allowing for gradual migration.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Add the jobs directory to Python path (Docker container path)
sys.path.insert(0, '/opt/bitnami/spark/jobs')

# Add the new module directories to the path
sys.path.extend([
    '/opt/bitnami/spark/jobs/model/registry',
    '/opt/bitnami/spark/jobs/model/config',
    '/opt/bitnami/spark/jobs/model/logging',
    '/opt/bitnami/spark/jobs/model/data',
    '/opt/bitnami/spark/jobs/model/data/resources',
    '/opt/bitnami/spark/jobs/model/data/sources',
    '/opt/bitnami/spark/jobs/model/signals',
    '/opt/bitnami/spark/jobs/model/signals/components',
    '/opt/bitnami/spark/jobs/model/signals/strategies',
    '/opt/bitnami/spark/jobs/model/backtesting',
    '/opt/bitnami/spark/jobs/model/backtesting/execution',
    '/opt/bitnami/spark/jobs/model/backtesting/risk',
    '/opt/bitnami/spark/jobs/model/backtesting/analytics',
    '/opt/bitnami/spark/jobs/model/backtesting/engines',
    '/opt/bitnami/spark/jobs/model/orchestration'
])

# Import new abstraction components
from model.orchestration.pipeline_orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResult
from model.orchestration.workflow_manager import WorkflowManager, WorkflowDefinition, WorkflowStep, WorkflowStatus
from model.orchestration.system_monitor import SystemMonitor, HealthStatus

# Import supporting components
from model.registry.component_registry import ComponentRegistry
from model.config.configuration_manager import ConfigurationManager
from model.logging.error_handler import ErrorHandler
from model.logging.enhanced_logger import EnhancedLogger
from model.data.resources.data_resources import DataResource, STOCK_PRICE, REVENUE, MARKET_CAP
from model.signals.signal_config import SignalConfig
from model.backtesting.backtest_config import BacktestConfig

logger = logging.getLogger(__name__)


class DashboardIntegration:
    """
    Integration layer between existing dashboard and new abstraction system
    
    This class provides adapter methods that translate dashboard commands
    to the new abstraction system while maintaining backward compatibility.
    """
    
    def __init__(self):
        """Initialize the integration layer"""
        self.orchestrator = PipelineOrchestrator()
        self.workflow_manager = WorkflowManager()
        self.system_monitor = SystemMonitor(update_interval=30)
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        logger.info("Dashboard integration layer initialized")
    
    async def fetch_data(self, symbols: List[str], start_date: str, end_date: str, 
                        timeframe: str = '1day', data_source: str = 'yfinance') -> Dict[str, Any]:
        """
        Fetch data using the new abstraction system
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            data_source: Data source to use
            
        Returns:
            Dictionary with fetch results
        """
        try:
            # Convert date strings to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Create pipeline configuration
            config = PipelineConfig(
                pipeline_id=f"fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbols=symbols,
                start_date=start_dt,
                end_date=end_dt,
                data_resources=[STOCK_PRICE],  # Default to stock price data
                signal_config=SignalConfig(
                    strategy_name="technical_analysis",
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    parameters={"rsi_period": 14}
                ),
                backtest_config=BacktestConfig(
                    name="default_backtest",
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    initial_capital=100000
                ),
                enable_training=False
            )
            
            # Run the data fetching pipeline
            result = await self.orchestrator.run_pipeline(config)
            
            return {
                'success': result.success,
                'data_fetched': result.data_fetch_result is not None,
                'symbols': symbols,
                'timeframe': timeframe,
                'data_source': data_source,
                'duration': result.performance_metrics.get('total_duration_seconds', 0),
                'errors': result.errors
            }
            
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbols': symbols,
                'timeframe': timeframe,
                'data_source': data_source
            }
    
    async def generate_signals(self, symbols: List[str], start_date: str, end_date: str,
                              timeframe: str = '1day') -> Dict[str, Any]:
        """
        Generate signals using the new abstraction system
        
        Args:
            symbols: List of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Signal timeframe
            
        Returns:
            Dictionary with signal generation results
        """
        try:
            # Convert date strings to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Create pipeline configuration
            config = PipelineConfig(
                pipeline_id=f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbols=symbols,
                start_date=start_dt,
                end_date=end_dt,
                data_resources=[STOCK_PRICE],
                signal_config=SignalConfig(
                    strategy_name="technical_analysis",
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    parameters={
                        "rsi_period": 14,
                        "macd_fast": 12,
                        "macd_slow": 26,
                        "bb_period": 20
                    }
                ),
                backtest_config=BacktestConfig(
                    name="default_backtest",
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    initial_capital=100000
                ),
                enable_training=False
            )
            
            # Run the signal generation pipeline
            result = await self.orchestrator.run_pipeline(config)
            
            return {
                'success': result.success,
                'signals_generated': result.signal_result is not None,
                'symbols': symbols,
                'timeframe': timeframe,
                'strategy': config.signal_config.strategy_name,
                'duration': result.performance_metrics.get('total_duration_seconds', 0),
                'errors': result.errors
            }
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbols': symbols,
                'timeframe': timeframe
            }
    
    async def run_backtest(self, symbols: List[str], from_date: str, to_date: str,
                          timeframe: str = '1day', initial_capital: float = 100000) -> Dict[str, Any]:
        """
        Run backtest using the new abstraction system
        
        Args:
            symbols: List of symbols
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timeframe: Backtest timeframe
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Convert date strings to datetime objects
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')
            
            # Create pipeline configuration
            config = PipelineConfig(
                pipeline_id=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbols=symbols,
                start_date=start_dt,
                end_date=end_dt,
                data_resources=[STOCK_PRICE],
                signal_config=SignalConfig(
                    strategy_name="technical_analysis",
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    parameters={"rsi_period": 14}
                ),
                backtest_config=BacktestConfig(
                    name="backtest_run",
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    initial_capital=initial_capital,
                    commission_rate=0.001,
                    slippage_rate=0.0005
                ),
                enable_training=False
            )
            
            # Run the complete pipeline (data + signals + backtest)
            result = await self.orchestrator.run_pipeline(config)
            
            return {
                'success': result.success,
                'backtest_completed': result.backtest_result is not None,
                'symbols': symbols,
                'timeframe': timeframe,
                'initial_capital': initial_capital,
                'duration': result.performance_metrics.get('total_duration_seconds', 0),
                'performance_metrics': result.backtest_result.get('metrics', {}) if result.backtest_result else {},
                'errors': result.errors
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbols': symbols,
                'timeframe': timeframe,
                'initial_capital': initial_capital
            }
    
    async def start_pipeline(self, mode: str = 'demo', interval: str = '5m',
                           timeframe: str = '1day', symbols: Optional[List[str]] = None,
                           data_source: str = 'yfinance') -> Dict[str, Any]:
        """
        Start a pipeline using the new abstraction system
        
        Args:
            mode: Pipeline mode (demo, custom)
            interval: Interval between runs
            timeframe: Data timeframe
            symbols: List of symbols (for custom mode)
            data_source: Data source to use
            
        Returns:
            Dictionary with pipeline start results
        """
        try:
            # Determine symbols based on mode
            if mode == 'demo':
                pipeline_symbols = ['AAPL', 'MSFT', 'GOOGL']
            elif mode == 'custom' and symbols:
                pipeline_symbols = symbols
            else:
                pipeline_symbols = ['AAPL', 'MSFT']  # Default fallback
            
            # Create workflow for continuous pipeline
            async def pipeline_step(**kwargs):
                """Single pipeline execution step"""
                config = PipelineConfig(
                    pipeline_id=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbols=pipeline_symbols,
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now(),
                    data_resources=[STOCK_PRICE],
                    signal_config=SignalConfig(
                        strategy_name="technical_analysis",
                        symbols=pipeline_symbols,
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        parameters={"rsi_period": 14}
                    ),
                    backtest_config=BacktestConfig(
                        name="pipeline_backtest",
                        symbols=pipeline_symbols,
                        start_date=datetime.now() - timedelta(days=30),
                        end_date=datetime.now(),
                        initial_capital=100000
                    ),
                    enable_training=False
                )
                
                result = await self.orchestrator.run_pipeline(config)
                return {
                    'pipeline_id': config.pipeline_id,
                    'success': result.success,
                    'duration': result.performance_metrics.get('total_duration_seconds', 0),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Create workflow step
            step = WorkflowStep(
                step_id="pipeline_execution",
                name="Pipeline Execution",
                description="Execute complete pipeline (data + signals + backtest)",
                function=pipeline_step,
                timeout_seconds=300
            )
            
            # Create workflow
            workflow = WorkflowDefinition(
                workflow_id=f"continuous_pipeline_{mode}",
                name=f"Continuous Pipeline - {mode}",
                description=f"Continuous pipeline execution in {mode} mode",
                version="1.0.0",
                steps=[step],
                max_concurrent_executions=1
            )
            
            # Register and start workflow
            self.workflow_manager.register_workflow(workflow)
            execution_id = await self.workflow_manager.execute_workflow(f"continuous_pipeline_{mode}")
            
            return {
                'success': True,
                'pipeline_started': True,
                'execution_id': execution_id,
                'mode': mode,
                'symbols': pipeline_symbols,
                'timeframe': timeframe,
                'data_source': data_source,
                'interval': interval
            }
            
        except Exception as e:
            logger.error(f"Pipeline start failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'mode': mode,
                'timeframe': timeframe
            }
    
    async def stop_pipeline(self) -> Dict[str, Any]:
        """
        Stop running pipelines
        
        Returns:
            Dictionary with stop results
        """
        try:
            # Get running executions
            running_executions = self.workflow_manager.list_executions(status=WorkflowStatus.RUNNING)
            
            stopped_count = 0
            for execution in running_executions:
                if self.workflow_manager.cancel_execution(execution.execution_id):
                    stopped_count += 1
            
            return {
                'success': True,
                'pipelines_stopped': stopped_count,
                'total_running': len(running_executions)
            }
            
        except Exception as e:
            logger.error(f"Pipeline stop failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline status
        
        Returns:
            Dictionary with pipeline status
        """
        try:
            # Get all executions
            all_executions = self.workflow_manager.list_executions()
            
            # Count by status
            status_counts = {}
            for execution in all_executions:
                status = execution.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Get system status
            system_status = self.system_monitor.get_system_status()
            
            return {
                'success': True,
                'executions': {
                    'total': len(all_executions),
                    'by_status': status_counts
                },
                'system_health': system_status.overall_health.value if system_status else 'unknown',
                'active_workflows': len(self.workflow_manager.list_workflows())
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health information
        
        Returns:
            Dictionary with system health
        """
        try:
            status = self.system_monitor.get_system_status()
            
            if status:
                return {
                    'success': True,
                    'overall_health': status.overall_health.value,
                    'health_checks': len(status.health_checks),
                    'metrics': len(status.metrics),
                    'alerts': len(status.alerts),
                    'timestamp': status.timestamp.isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'System status not available'
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Global integration instance
_dashboard_integration = None

def get_dashboard_integration() -> DashboardIntegration:
    """Get the global dashboard integration instance"""
    global _dashboard_integration
    if _dashboard_integration is None:
        _dashboard_integration = DashboardIntegration()
    return _dashboard_integration


# Convenience functions for easy integration
async def fetch_data_async(symbols: List[str], start_date: str, end_date: str, 
                          timeframe: str = '1day', data_source: str = 'yfinance') -> Dict[str, Any]:
    """Async wrapper for data fetching"""
    integration = get_dashboard_integration()
    return await integration.fetch_data(symbols, start_date, end_date, timeframe, data_source)

async def generate_signals_async(symbols: List[str], start_date: str, end_date: str,
                                timeframe: str = '1day') -> Dict[str, Any]:
    """Async wrapper for signal generation"""
    integration = get_dashboard_integration()
    return await integration.generate_signals(symbols, start_date, end_date, timeframe)

async def run_backtest_async(symbols: List[str], from_date: str, to_date: str,
                            timeframe: str = '1day', initial_capital: float = 100000) -> Dict[str, Any]:
    """Async wrapper for backtest execution"""
    integration = get_dashboard_integration()
    return await integration.run_backtest(symbols, from_date, to_date, timeframe, initial_capital)

async def start_pipeline_async(mode: str = 'demo', interval: str = '5m',
                              timeframe: str = '1day', symbols: Optional[List[str]] = None,
                              data_source: str = 'yfinance') -> Dict[str, Any]:
    """Async wrapper for pipeline start"""
    integration = get_dashboard_integration()
    return await integration.start_pipeline(mode, interval, timeframe, symbols, data_source)

async def stop_pipeline_async() -> Dict[str, Any]:
    """Async wrapper for pipeline stop"""
    integration = get_dashboard_integration()
    return await integration.stop_pipeline()

async def get_pipeline_status_async() -> Dict[str, Any]:
    """Async wrapper for pipeline status"""
    integration = get_dashboard_integration()
    return await integration.get_pipeline_status()

def get_system_health_sync() -> Dict[str, Any]:
    """Sync wrapper for system health"""
    integration = get_dashboard_integration()
    return integration.get_system_health()
