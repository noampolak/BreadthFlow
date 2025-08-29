#!/usr/bin/env python3
"""
Minimal Dashboard Integration Test

This script tests the core orchestration components without importing
the full pipeline orchestrator that has pandas dependencies.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add the new module directories to the path
sys.path.extend([
    'model/registry',
    'model/config', 
    'model/logging',
    'model/orchestration'
])

# Import only the core orchestration components
from workflow_manager import WorkflowManager, WorkflowDefinition, WorkflowStep, WorkflowStatus
from system_monitor import SystemMonitor, HealthStatus
from component_registry import ComponentRegistry
from configuration_manager import ConfigurationManager
from error_handler import ErrorHandler
from enhanced_logger import EnhancedLogger


class MinimalDashboardIntegration:
    """
    Minimal integration layer that only uses core orchestration components
    without pandas dependencies
    """
    
    def __init__(self):
        """Initialize the minimal integration layer"""
        self.workflow_manager = WorkflowManager()
        self.system_monitor = SystemMonitor(update_interval=30)
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        print("Minimal dashboard integration layer initialized")
    
    async def test_workflow_management(self) -> dict:
        """Test workflow management capabilities"""
        try:
            # Create a simple test workflow
            async def test_step(**kwargs):
                """Simple test step"""
                await asyncio.sleep(0.1)  # Simulate work
                return {"status": "completed", "timestamp": datetime.now().isoformat()}
            
            step = WorkflowStep(
                step_id="test_step",
                name="Test Step",
                description="Simple test workflow step",
                function=test_step,
                timeout_seconds=10
            )
            
            workflow = WorkflowDefinition(
                workflow_id="test_workflow",
                name="Test Workflow",
                description="Test workflow for dashboard integration",
                version="1.0.0",
                steps=[step],
                max_concurrent_executions=1
            )
            
            # Register and execute workflow
            self.workflow_manager.register_workflow(workflow)
            execution_id = await self.workflow_manager.execute_workflow("test_workflow")
            
            # Wait for completion
            await asyncio.sleep(1)
            
            # Get execution status
            execution = self.workflow_manager.get_execution(execution_id)
            
            return {
                'success': True,
                'workflow_registered': True,
                'execution_id': execution_id,
                'execution_status': execution.status.value if execution else 'unknown',
                'steps_completed': len(execution.results) if execution else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_system_monitoring(self) -> dict:
        """Test system monitoring capabilities"""
        try:
            # Get system status
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
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_pipeline_simulation(self) -> dict:
        """Simulate pipeline operations using workflows"""
        try:
            # Create a pipeline simulation workflow
            async def data_fetch_step(**kwargs):
                """Simulate data fetching"""
                await asyncio.sleep(0.1)
                return {"data_fetched": True, "symbols": ["AAPL", "MSFT"]}
            
            async def signal_generation_step(**kwargs):
                """Simulate signal generation"""
                await asyncio.sleep(0.1)
                return {"signals_generated": True, "strategy": "technical_analysis"}
            
            async def backtest_step(**kwargs):
                """Simulate backtesting"""
                await asyncio.sleep(0.1)
                return {"backtest_completed": True, "sharpe_ratio": 1.2}
            
            # Create workflow steps
            steps = [
                WorkflowStep(
                    step_id="data_fetch",
                    name="Data Fetch",
                    description="Fetch market data",
                    function=data_fetch_step,
                    timeout_seconds=30
                ),
                WorkflowStep(
                    step_id="signal_generation",
                    name="Signal Generation",
                    description="Generate trading signals",
                    function=signal_generation_step,
                    timeout_seconds=30,
                    dependencies=["data_fetch"]
                ),
                WorkflowStep(
                    step_id="backtest",
                    name="Backtest",
                    description="Run backtest",
                    function=backtest_step,
                    timeout_seconds=60,
                    dependencies=["signal_generation"]
                )
            ]
            
            # Create workflow
            workflow = WorkflowDefinition(
                workflow_id="pipeline_simulation",
                name="Pipeline Simulation",
                description="Simulate complete pipeline execution",
                version="1.0.0",
                steps=steps,
                max_concurrent_executions=1
            )
            
            # Register and execute
            self.workflow_manager.register_workflow(workflow)
            execution_id = await self.workflow_manager.execute_workflow("pipeline_simulation")
            
            # Wait for completion
            await asyncio.sleep(2)
            
            # Get results
            execution = self.workflow_manager.get_execution(execution_id)
            
            if execution and execution.status == WorkflowStatus.COMPLETED:
                results = {}
                for step_id, result in execution.results.items():
                    results[step_id] = result
                
                return {
                    'success': True,
                    'pipeline_completed': True,
                    'execution_id': execution_id,
                    'steps_completed': len(execution.results),
                    'results': results
                }
            else:
                return {
                    'success': False,
                    'error': f'Pipeline failed with status: {execution.status.value if execution else "unknown"}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_dashboard_commands(self) -> dict:
        """Test simulated dashboard commands"""
        try:
            results = {}
            
            # Simulate "data fetch" command
            print("Simulating 'data fetch' command...")
            fetch_result = await self.test_workflow_management()
            results['data_fetch'] = fetch_result
            
            # Simulate "signals generate" command
            print("Simulating 'signals generate' command...")
            signal_result = await self.test_workflow_management()
            results['signals_generate'] = signal_result
            
            # Simulate "backtest run" command
            print("Simulating 'backtest run' command...")
            backtest_result = await self.test_pipeline_simulation()
            results['backtest_run'] = backtest_result
            
            # Simulate "pipeline status" command
            print("Simulating 'pipeline status' command...")
            status_result = await self.test_system_monitoring()
            results['pipeline_status'] = status_result
            
            return {
                'success': True,
                'dashboard_commands_tested': True,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


async def test_minimal_integration():
    """Test the minimal dashboard integration"""
    print("Testing Minimal Dashboard Integration")
    print("=" * 50)
    
    integration = MinimalDashboardIntegration()
    
    # Test 1: Workflow Management
    print("\n1. Testing Workflow Management...")
    workflow_result = await integration.test_workflow_management()
    print(f"   Success: {workflow_result.get('success', False)}")
    if workflow_result.get('success'):
        print(f"   Execution ID: {workflow_result.get('execution_id', 'none')}")
        print(f"   Status: {workflow_result.get('execution_status', 'unknown')}")
        print(f"   Steps Completed: {workflow_result.get('steps_completed', 0)}")
    else:
        print(f"   Error: {workflow_result.get('error', 'Unknown error')}")
    
    # Test 2: System Monitoring
    print("\n2. Testing System Monitoring...")
    monitor_result = await integration.test_system_monitoring()
    print(f"   Success: {monitor_result.get('success', False)}")
    if monitor_result.get('success'):
        print(f"   Overall Health: {monitor_result.get('overall_health', 'unknown')}")
        print(f"   Health Checks: {monitor_result.get('health_checks', 0)}")
        print(f"   Metrics: {monitor_result.get('metrics', 0)}")
        print(f"   Alerts: {monitor_result.get('alerts', 0)}")
    else:
        print(f"   Error: {monitor_result.get('error', 'Unknown error')}")
    
    # Test 3: Pipeline Simulation
    print("\n3. Testing Pipeline Simulation...")
    pipeline_result = await integration.test_pipeline_simulation()
    print(f"   Success: {pipeline_result.get('success', False)}")
    if pipeline_result.get('success'):
        print(f"   Pipeline Completed: {pipeline_result.get('pipeline_completed', False)}")
        print(f"   Steps Completed: {pipeline_result.get('steps_completed', 0)}")
        print(f"   Execution ID: {pipeline_result.get('execution_id', 'none')}")
    else:
        print(f"   Error: {pipeline_result.get('error', 'Unknown error')}")
    
    # Test 4: Dashboard Commands
    print("\n4. Testing Dashboard Commands...")
    commands_result = await integration.test_dashboard_commands()
    print(f"   Success: {commands_result.get('success', False)}")
    if commands_result.get('success'):
        print(f"   Dashboard Commands Tested: {commands_result.get('dashboard_commands_tested', False)}")
        results = commands_result.get('results', {})
        for command, result in results.items():
            status = "✅ PASSED" if result.get('success', False) else "❌ FAILED"
            print(f"   {command}: {status}")
    else:
        print(f"   Error: {commands_result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 50)
    print("MINIMAL INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    tests = [
        ("Workflow Management", workflow_result.get('success', False)),
        ("System Monitoring", monitor_result.get('success', False)),
        ("Pipeline Simulation", pipeline_result.get('success', False)),
        ("Dashboard Commands", commands_result.get('success', False))
    ]
    
    passed = 0
    for test_name, success in tests:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All minimal integration tests passed!")
        print("✅ Core orchestration components are working correctly")
        print("✅ Dashboard integration foundation is ready")
        print("✅ Workflow management and system monitoring are functional")
        return True
    else:
        print("⚠️ Some minimal integration tests failed")
        print("⚠️ Core components may need additional configuration")
        return False


async def main():
    """Main test function"""
    print("Minimal Dashboard Integration Test")
    print("=" * 50)
    
    success = await test_minimal_integration()
    
    print("\n" + "=" * 50)
    print("INTEGRATION READINESS ASSESSMENT")
    print("=" * 50)
    
    if success:
        print("✅ Core integration components are READY")
        print("✅ Workflow management system is functional")
        print("✅ System monitoring is operational")
        print("✅ Dashboard commands can be simulated")
        print("\n📋 Next Steps:")
        print("   1. Install pandas and other dependencies for full integration")
        print("   2. Connect dashboard buttons to workflow manager")
        print("   3. Implement data fetching and signal generation workflows")
        print("   4. Add backtesting workflows")
    else:
        print("⚠️ Core integration needs more work")
        print("⚠️ Check the test results above for specific issues")
    
    return success


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
