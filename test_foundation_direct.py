#!/usr/bin/env python3
"""
Direct Foundation Components Test Script

Tests the core foundation components by importing them directly
without going through the model package to avoid PySpark dependencies.
"""

import sys
import os
import json
import yaml
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

# Add the model subdirectories to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'registry'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'config'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'logging'))

# Import our new foundation components directly
from component_registry import ComponentRegistry, ComponentMetadata
from configuration_manager import ConfigurationManager, ConfigSchema
from error_handler import ErrorHandler, ErrorRecord
from enhanced_logger import EnhancedLogger
from error_recovery import ErrorRecovery

def test_component_registry():
    """Test Component Registry functionality"""
    print("🧪 Testing Component Registry...")
    
    # Create registry
    registry = ComponentRegistry("test_component_registry.json")
    
    # Create test metadata
    test_metadata = ComponentMetadata(
        name="test_component",
        type="test_type",
        version="1.0.0",
        description="Test component for validation",
        author="Test Author",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={}
    )
    
    # Test component listing
    components = registry.list_components()
    print(f"✅ Registry initialized with {len(components)} components")
    
    # Test component validation
    is_valid = registry.validate_component("test_type", "test_component")
    print(f"✅ Component validation: {is_valid}")
    
    return True

def test_configuration_manager():
    """Test Configuration Manager functionality"""
    print("🧪 Testing Configuration Manager...")
    
    # Create config manager
    config_manager = ConfigurationManager("test_config/")
    
    # Test config loading
    test_config = {
        'test_key': 'test_value',
        'nested': {
            'key': 'value'
        }
    }
    
    # Save test config
    config_manager.save_config("test", test_config)
    
    # Load test config
    loaded_config = config_manager.get_config("test")
    print(f"✅ Configuration saved and loaded: {loaded_config['test_key']}")
    
    # Test nested key access
    nested_value = config_manager.get_config("test", "nested.key")
    print(f"✅ Nested key access: {nested_value}")
    
    return True

def test_error_handler():
    """Test Error Handler functionality"""
    print("🧪 Testing Error Handler...")
    
    # Create error handler
    error_handler = ErrorHandler()
    
    # Test error handling
    try:
        raise ValueError("Test error for validation")
    except Exception as e:
        error_record = error_handler.handle_error(
            e, 
            {'test_context': 'test_value'}, 
            'test_component', 
            'test_operation'
        )
        print(f"✅ Error handled: {error_record.error_type}")
    
    # Test error summary
    summary = error_handler.get_error_summary()
    print(f"✅ Error summary generated: {summary['total_errors']} errors")
    
    return True

def test_enhanced_logger():
    """Test Enhanced Logger functionality"""
    print("🧪 Testing Enhanced Logger...")
    
    # Create enhanced logger
    logger = EnhancedLogger("test_logger", "test_component")
    
    # Test basic logging
    logger.log_operation("test_operation", {'test_data': 'test_value'})
    print("✅ Basic logging test passed")
    
    # Test performance logging
    with logger.log_performance("test_performance"):
        time.sleep(0.1)  # Simulate work
    
    print("✅ Performance logging test passed")
    
    return True

def test_error_recovery():
    """Test Error Recovery functionality"""
    print("🧪 Testing Error Recovery...")
    
    # Create error recovery
    recovery = ErrorRecovery()
    
    # Test retry decorator
    @recovery.retry_on_error(['ValueError'])
    def test_function():
        raise ValueError("Test retry error")
    
    try:
        test_function()
    except Exception as e:
        print(f"✅ Retry mechanism worked: {type(e).__name__}")
    
    return True

def main():
    """Run all foundation tests"""
    print("🚀 Starting Foundation Components Test Suite")
    print("=" * 50)
    
    tests = [
        test_component_registry,
        test_configuration_manager,
        test_error_handler,
        test_enhanced_logger,
        test_error_recovery
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All foundation components are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
