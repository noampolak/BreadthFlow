#!/usr/bin/env python3
"""
Test Docker Integration

This script tests the new abstraction system integration with the Docker infrastructure.
It simulates how the dashboard would call the new CLI commands.
"""

import sys
import os
import subprocess
import json
from datetime import datetime

def test_docker_command(command):
    """Test a command in the Docker container"""
    print(f"\n🧪 Testing: {command}")
    print("=" * 60)
    
    try:
        # Execute command in spark-master container
        result = subprocess.run([
            'docker', 'exec', 'spark-master', 
            'python3', '/opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py'
        ] + command.split(), 
        capture_output=True, text=True, timeout=60)
        
        print(f"✅ Command executed successfully")
        print(f"📤 Output:")
        print(result.stdout)
        
        if result.stderr:
            print(f"⚠️ Warnings/Errors:")
            print(result.stderr)
            
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after 60 seconds")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        print(f"📤 Output: {e.stdout}")
        print(f"❌ Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Testing BreadthFlow Docker Integration")
    print("=" * 60)
    
    # Check if containers are running
    print("🔍 Checking if containers are running...")
    try:
        result = subprocess.run(['docker', 'ps', '--filter', 'name=spark-master'], 
                              capture_output=True, text=True)
        if 'spark-master' not in result.stdout:
            print("❌ spark-master container is not running!")
            print("Please start the infrastructure first:")
            print("  ./scripts/start_infrastructure.sh")
            return False
        print("✅ spark-master container is running")
    except Exception as e:
        print(f"❌ Error checking containers: {e}")
        return False
    
    # Test commands
    tests = [
        "health",
        "demo",
        "data-fetch --symbols AAPL,MSFT --timeframe 1day",
        "signals-generate --symbols AAPL,MSFT --timeframe 1day",
        "backtest-run --symbols AAPL,MSFT --timeframe 1day --initial-capital 100000",
        "pipeline-start --mode demo",
        "pipeline-status",
        "pipeline-stop"
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test_docker_command(test):
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Docker integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
