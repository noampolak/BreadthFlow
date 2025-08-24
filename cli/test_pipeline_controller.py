#!/usr/bin/env python3
"""
Test script for Pipeline Controller
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline_controller import PipelineController

def test_pipeline_controller():
    """Test the pipeline controller functionality"""
    print("🧪 Testing Pipeline Controller...")
    
    controller = PipelineController()
    
    # Test 1: Check if pipeline is running
    print("\n1. Testing is_pipeline_running()...")
    is_running = controller.is_pipeline_running()
    print(f"   Pipeline running: {is_running}")
    
    # Test 2: Get pipeline status
    print("\n2. Testing get_pipeline_status()...")
    status = controller.get_pipeline_status()
    print(f"   Status: {status}")
    
    # Test 3: Get recent pipeline runs
    print("\n3. Testing get_recent_pipeline_runs()...")
    runs = controller.get_recent_pipeline_runs(days=2)
    print(f"   Found {len(runs)} recent runs")
    for run in runs[:3]:  # Show first 3 runs
        print(f"   - {run['run_id']}: {run['status']} ({run['start_time']})")
    
    # Test 4: Test start pipeline (should fail if already running)
    print("\n4. Testing start_pipeline()...")
    config = {
        'mode': 'demo',
        'interval': '5m',
        'timeframe': '1day',
        'symbols': 'AAPL,MSFT',
        'data_source': 'yfinance'
    }
    result = controller.start_pipeline(config)
    print(f"   Start result: {result['success']} - {result.get('message', result.get('error', 'No message'))}")
    
    # Test 5: Test stop pipeline (should fail if not running)
    print("\n5. Testing stop_pipeline()...")
    result = controller.stop_pipeline()
    print(f"   Stop result: {result['success']} - {result.get('message', result.get('error', 'No message'))}")
    
    print("\n✅ Pipeline Controller tests completed!")

if __name__ == "__main__":
    test_pipeline_controller()
