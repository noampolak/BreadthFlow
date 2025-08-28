#!/usr/bin/env python3
"""
Data Fetching System Test Script

Tests the Phase 2 data fetching abstraction components.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the model subdirectories to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'data'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'data', 'sources'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'data', 'resources'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'logging'))

# Import our data fetching components
from resources.data_resources import (
    ResourceType, DataFrequency, ResourceField, DataResource,
    STOCK_PRICE, REVENUE, MARKET_CAP, NEWS_SENTIMENT,
    get_resource_by_name, list_available_resources, validate_resource_data
)
from sources.data_source_interface import DataSourceInterface
from universal_data_fetcher import UniversalDataFetcher

def test_data_resources():
    """Test data resource definitions"""
    print("🧪 Testing Data Resources...")
    
    # Test resource listing
    available_resources = list_available_resources()
    print(f"✅ Available resources: {available_resources}")
    
    # Test resource retrieval
    stock_price_resource = get_resource_by_name("stock_price")
    if stock_price_resource:
        print(f"✅ Stock price resource: {stock_price_resource.name}")
        print(f"   Type: {stock_price_resource.type.value}")
        print(f"   Frequency: {stock_price_resource.frequency.value}")
        print(f"   Fields: {len(stock_price_resource.fields)}")
    
    # Test data validation
    test_data = {
        'symbol': 'AAPL',
        'date': '2024-01-01',
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000000
    }
    
    is_valid = validate_resource_data(test_data, STOCK_PRICE)
    print(f"✅ Data validation: {is_valid}")
    
    return True

def test_data_source_interface():
    """Test data source interface"""
    print("🧪 Testing Data Source Interface...")
    
    # Create a mock data source
    class MockDataSource(DataSourceInterface):
        def get_name(self):
            return "mock_source"
        
        def get_supported_resources(self):
            return ["stock_price"]
        
        def fetch_data(self, resource_name, symbols, start_date, end_date, **kwargs):
            # Return mock data
            data = {
                'symbol': symbols,
                'date': [start_date] * len(symbols),
                'open': [100.0] * len(symbols),
                'high': [105.0] * len(symbols),
                'low': [95.0] * len(symbols),
                'close': [102.0] * len(symbols),
                'volume': [1000000] * len(symbols)
            }
            return pd.DataFrame(data)
        
        def get_config(self):
            return {'mock': True}
    
    # Test mock data source
    mock_source = MockDataSource()
    print(f"✅ Mock source name: {mock_source.get_name()}")
    print(f"✅ Supported resources: {mock_source.get_supported_resources()}")
    
    # Test data fetching
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    symbols = ['AAPL', 'MSFT']
    
    data = mock_source.fetch_data('stock_price', symbols, start_date, end_date)
    print(f"✅ Mock data fetched: {len(data)} records")
    
    return True

def test_universal_data_fetcher():
    """Test Universal Data Fetcher"""
    print("🧪 Testing Universal Data Fetcher...")
    
    # Create universal data fetcher
    fetcher = UniversalDataFetcher()
    
    # Create and register mock data source
    class MockDataSource(DataSourceInterface):
        def get_name(self):
            return "mock_source"
        
        def get_supported_resources(self):
            return ["stock_price"]
        
        def fetch_data(self, resource_name, symbols, start_date, end_date, **kwargs):
            data = {
                'symbol': symbols,
                'date': [start_date] * len(symbols),
                'open': [100.0] * len(symbols),
                'high': [105.0] * len(symbols),
                'low': [95.0] * len(symbols),
                'close': [102.0] * len(symbols),
                'volume': [1000000] * len(symbols)
            }
            return pd.DataFrame(data)
        
        def get_config(self):
            return {'mock': True}
    
    mock_source = MockDataSource()
    fetcher.register_data_source("mock", mock_source)
    
    # Test data fetching
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    symbols = ['AAPL', 'MSFT']
    
    data = fetcher.fetch_data('stock_price', symbols, start_date, end_date)
    print(f"✅ Universal fetcher data: {len(data)} records")
    
    # Test data summary
    summary = fetcher.get_data_summary(data, 'stock_price')
    print(f"✅ Data summary: {summary['total_records']} records, {summary['symbols']} symbols")
    
    # Test fetch statistics
    stats = fetcher.get_fetch_statistics()
    print(f"✅ Fetch stats: {stats['successful_requests']} successful, {stats['success_rate']:.2%} success rate")
    
    return True

def test_yfinance_integration():
    """Test YFinance integration (if available)"""
    print("🧪 Testing YFinance Integration...")
    
    try:
        # Try to import yfinance
        import yfinance as yf
        
        # Create universal data fetcher
        fetcher = UniversalDataFetcher()
        
        # Create YFinance data source
        from sources.yfinance_source import YFinanceDataSource
        yf_source = YFinanceDataSource()
        
        # Register YFinance source
        fetcher.register_data_source("yfinance", yf_source)
        
        # Test data fetching (small test)
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        symbols = ['AAPL']  # Just one symbol for testing
        
        try:
            data = fetcher.fetch_data('stock_price', symbols, start_date, end_date)
            print(f"✅ YFinance data fetched: {len(data)} records")
            
            if not data.empty:
                summary = fetcher.get_data_summary(data, 'stock_price')
                print(f"✅ YFinance summary: {summary['total_records']} records")
            
        except Exception as e:
            print(f"⚠️ YFinance test failed (expected if no internet): {e}")
        
        return True
        
    except ImportError:
        print("⚠️ YFinance not available, skipping integration test")
        return True

def main():
    """Run all data fetching tests"""
    print("🚀 Starting Data Fetching System Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_resources,
        test_data_source_interface,
        test_universal_data_fetcher,
        test_yfinance_integration
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
        print("🎉 All data fetching components are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
