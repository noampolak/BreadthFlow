#!/usr/bin/env python3
"""
Test Script for Phase 5: Full System Integration
===============================================

This script tests the complete orchestration system including:
- Pipeline Orchestrator
- Workflow Manager  
- System Monitor
- Integration between all components
"""

import sys
import os
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the new module directories to the path
sys.path.extend([
    'model/registry',
    'model/config', 
    'model/logging',
    'model/data',
    'model/data/resources',
    'model/data/sources',
    'model/signals',
    'model/signals/components',
    'model/signals/strategies',
    'model/backtesting',
    'model/backtesting/execution',
    'model/backtesting/risk',
    'model/backtesting/analytics',
    'model/backtesting/engines',
    'model/orchestration'
])

# Import orchestration components
from pipeline_orchestrator import PipelineOrchestrator, PipelineConfig, PipelineResult
from workflow_manager import WorkflowManager, WorkflowDefinition, WorkflowStep, WorkflowStatus
from system_monitor import SystemMonitor, HealthStatus, MetricType

# Import supporting components
from component_registry import ComponentRegistry
from configuration_manager import ConfigurationManager
from error_handler import ErrorHandler
from enhanced_logger import EnhancedLogger
from data_resources import DataResource, STOCK_PRICE, REVENUE, MARKET_CAP
from signal_config import SignalConfig
from backtest_config import BacktestConfig


async def test_pipeline_orchestrator():
    """Test the pipeline orchestrator"""
    print("\n=== Testing Pipeline Orchestrator ===")
    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator()
        print("✓ Pipeline orchestrator initialized")
        
        # Test system status
        status = orchestrator.get_system_status()
        print(f"✓ System status retrieved: {status['component_registry']['total_components']} components")
        
        # Test component listing
        components = orchestrator.list_available_components()
        print(f"✓ Available components: {list(components.keys())}")
        
        # Create a test pipeline configuration
        config = PipelineConfig(
            pipeline_id="test_pipeline_001",
            symbols=["AAPL", "MSFT"],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            data_resources=[STOCK_PRICE, REVENUE],
            signal_config=SignalConfig(
                strategy_name="technical_analysis",
                parameters={"rsi_period": 14, "macd_fast": 12, "macd_slow": 26}
            ),
            backtest_config=BacktestConfig(
                initial_capital=100000,
                commission_rate=0.001,
                slippage_rate=0.0005
            ),
            enable_training=False
        )
        
        print("✓ Pipeline configuration created")
        
        # Note: Full pipeline execution would require actual data sources
        # For now, we just test the configuration and initialization
        print("✓ Pipeline orchestrator test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline orchestrator test failed: {e}")
        return False


def test_workflow_manager():
    """Test the workflow manager"""
    print("\n=== Testing Workflow Manager ===")
    
    try:
        # Initialize workflow manager
        workflow_manager = WorkflowManager()
        print("✓ Workflow manager initialized")
        
        # Define test workflow steps
        def step1_function(**kwargs):
            print("  Executing step 1...")
            time.sleep(0.1)
            return {"result": "step1_completed", "data": [1, 2, 3]}
        
        def step2_function(**kwargs):
            print("  Executing step 2...")
            time.sleep(0.1)
            return {"result": "step2_completed", "data": [4, 5, 6]}
        
        def step3_function(**kwargs):
            print("  Executing step 3...")
            time.sleep(0.1)
            return {"result": "step3_completed", "data": [7, 8, 9]}
        
        # Create workflow steps
        step1 = WorkflowStep(
            step_id="step1",
            name="Data Preparation",
            description="Prepare data for processing",
            function=step1_function,
            timeout_seconds=60,
            retry_attempts=2
        )
        
        step2 = WorkflowStep(
            step_id="step2", 
            name="Data Processing",
            description="Process the prepared data",
            function=step2_function,
            dependencies=["step1"],
            timeout_seconds=60,
            retry_attempts=2
        )
        
        step3 = WorkflowStep(
            step_id="step3",
            name="Data Analysis", 
            description="Analyze the processed data",
            function=step3_function,
            dependencies=["step2"],
            timeout_seconds=60,
            retry_attempts=2
        )
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id="test_workflow_001",
            name="Test Data Pipeline",
            description="A test workflow for data processing",
            version="1.0.0",
            steps=[step1, step2, step3],
            max_concurrent_executions=1,
            timeout_seconds=300
        )
        
        # Register workflow
        success = workflow_manager.register_workflow(workflow)
        if success:
            print("✓ Workflow registered successfully")
        else:
            print("✗ Failed to register workflow")
            return False
        
        # List workflows
        workflows = workflow_manager.list_workflows()
        print(f"✓ Registered workflows: {len(workflows)}")
        
        # Test workflow validation
        workflow_def = workflow_manager.get_workflow_definition("test_workflow_001")
        if workflow_def:
            print(f"✓ Retrieved workflow: {workflow_def.name}")
        else:
            print("✗ Failed to retrieve workflow")
            return False
        
        print("✓ Workflow manager test completed")
        return True
        
    except Exception as e:
        print(f"✗ Workflow manager test failed: {e}")
        return False


async def test_workflow_execution():
    """Test workflow execution"""
    print("\n=== Testing Workflow Execution ===")
    
    try:
        # Initialize workflow manager
        workflow_manager = WorkflowManager()
        
        # Define async test functions
        async def async_step1(**kwargs):
            print("  Executing async step 1...")
            await asyncio.sleep(0.1)
            return {"result": "async_step1_completed"}
        
        async def async_step2(**kwargs):
            print("  Executing async step 2...")
            await asyncio.sleep(0.1)
            return {"result": "async_step2_completed"}
        
        # Create workflow steps
        step1 = WorkflowStep(
            step_id="async_step1",
            name="Async Step 1",
            description="First async step",
            function=async_step1,
            timeout_seconds=60
        )
        
        step2 = WorkflowStep(
            step_id="async_step2",
            name="Async Step 2", 
            description="Second async step",
            function=async_step2,
            dependencies=["async_step1"],
            timeout_seconds=60
        )
        
        # Create workflow
        workflow = WorkflowDefinition(
            workflow_id="async_test_workflow",
            name="Async Test Workflow",
            description="Test workflow with async steps",
            version="1.0.0",
            steps=[step1, step2],
            max_concurrent_executions=1
        )
        
        # Register workflow
        workflow_manager.register_workflow(workflow)
        print("✓ Async workflow registered")
        
        # Execute workflow
        execution_id = await workflow_manager.execute_workflow("async_test_workflow")
        print(f"✓ Workflow execution started: {execution_id}")
        
        # Wait for completion
        max_wait = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            execution = workflow_manager.get_execution_status(execution_id)
            if execution and execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                break
            await asyncio.sleep(0.5)
        
        # Check final status
        execution = workflow_manager.get_execution_status(execution_id)
        if execution:
            print(f"✓ Workflow execution completed with status: {execution.status}")
            print(f"  Steps completed: {sum(1 for status in execution.steps.values() if status.value == 'completed')}")
            print(f"  Results: {len(execution.results)}")
            print(f"  Errors: {len(execution.errors)}")
            
            if execution.status == WorkflowStatus.COMPLETED:
                print("✓ Workflow execution test successful")
                return True
            else:
                print(f"✗ Workflow execution failed: {execution.errors}")
                return False
        else:
            print("✗ Could not retrieve execution status")
            return False
            
    except Exception as e:
        print(f"✗ Workflow execution test failed: {e}")
        return False


def test_system_monitor():
    """Test the system monitor"""
    print("\n=== Testing System Monitor ===")
    
    try:
        # Initialize system monitor
        monitor = SystemMonitor(update_interval=5)
        print("✓ System monitor initialized")
        
        # Start monitoring
        monitor.start_monitoring()
        print("✓ System monitoring started")
        
        # Wait a bit for initial data collection
        time.sleep(3)
        
        # Get system status
        status = monitor.get_system_status()
        if status:
            print(f"✓ System status retrieved - Overall health: {status.overall_health.value}")
            print(f"  Health checks: {len(status.health_checks)}")
            print(f"  Metrics: {len(status.metrics)}")
            print(f"  Alerts: {len(status.alerts)}")
        else:
            print("✗ Could not retrieve system status")
        
        # Get metrics
        metrics = monitor.get_metrics(time_range=timedelta(minutes=5))
        print(f"✓ Retrieved {len(metrics)} metrics")
        
        # Get health history
        health_checks = monitor.get_health_history(time_range=timedelta(minutes=5))
        print(f"✓ Retrieved {len(health_checks)} health checks")
        
        # Get performance summary
        performance = monitor.get_performance_summary()
        print(f"✓ Performance summary: {len(performance)} metrics")
        
        # Test alert threshold
        monitor.set_alert_threshold("cpu_usage_percent", 50.0)
        print("✓ Alert threshold set")
        
        # Get alerts
        alerts = monitor.get_alerts()
        print(f"✓ Current alerts: {len(alerts)}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("✓ System monitoring stopped")
        
        print("✓ System monitor test completed")
        return True
        
    except Exception as e:
        print(f"✗ System monitor test failed: {e}")
        return False


async def test_integration():
    """Test integration between all orchestration components"""
    print("\n=== Testing Component Integration ===")
    
    try:
        # Initialize all components
        orchestrator = PipelineOrchestrator()
        workflow_manager = WorkflowManager()
        monitor = SystemMonitor(update_interval=10)
        
        print("✓ All orchestration components initialized")
        
        # Start monitoring
        monitor.start_monitoring()
        print("✓ System monitoring started")
        
        # Create a simple workflow that uses the orchestrator
        async def test_orchestrator_step(**kwargs):
            print("  Testing orchestrator integration...")
            
            # Test orchestrator functionality
            status = orchestrator.get_system_status()
            components = orchestrator.list_available_components()
            
            return {
                "orchestrator_status": status,
                "available_components": components,
                "test_result": "integration_successful"
            }
        
        # Create workflow step
        step = WorkflowStep(
            step_id="orchestrator_test",
            name="Orchestrator Integration Test",
            description="Test integration with pipeline orchestrator",
            function=test_orchestrator_step,
            timeout_seconds=60
        )
        
        # Create workflow
        workflow = WorkflowDefinition(
            workflow_id="integration_test_workflow",
            name="Integration Test Workflow",
            description="Test workflow for component integration",
            version="1.0.0",
            steps=[step],
            max_concurrent_executions=1
        )
        
        # Register and execute workflow
        workflow_manager.register_workflow(workflow)
        execution_id = await workflow_manager.execute_workflow("integration_test_workflow")
        
        # Wait for completion
        await asyncio.sleep(2)
        
        # Check results
        execution = workflow_manager.get_execution_status(execution_id)
        if execution and execution.status == WorkflowStatus.COMPLETED:
            print("✓ Integration workflow completed successfully")
            
            # Check orchestrator integration results
            if "orchestrator_test" in execution.results:
                result = execution.results["orchestrator_test"]
                print(f"✓ Orchestrator integration test result: {result.get('test_result')}")
            
            # Check system monitor integration
            status = monitor.get_system_status()
            if status:
                print(f"✓ System monitor integration - Health: {status.overall_health.value}")
            
            print("✓ Component integration test completed successfully")
            return True
        else:
            print(f"✗ Integration workflow failed: {execution.errors if execution else 'No execution found'}")
            return False
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
    finally:
        # Clean up
        if 'monitor' in locals():
            monitor.stop_monitoring()


async def main():
    """Main test function"""
    print("Phase 5: Full System Integration Test")
    print("=" * 50)
    
    results = []
    
    # Test individual components
    results.append(await test_pipeline_orchestrator())
    results.append(test_workflow_manager())
    results.append(await test_workflow_execution())
    results.append(test_system_monitor())
    results.append(await test_integration())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Pipeline Orchestrator",
        "Workflow Manager", 
        "Workflow Execution",
        "System Monitor",
        "Component Integration"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{i+1}. {name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 5 tests passed! The orchestration system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
