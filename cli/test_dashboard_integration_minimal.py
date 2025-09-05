#!/usr/bin/env python3
"""
Minimal test for dashboard integration
Tests the basic functionality without requiring full infrastructure
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_connection():
    """Test database connection"""
    try:
        from database.connection import get_db_connection
        db = get_db_connection()
        if db and db.test_connection():
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_pipeline_queries():
    """Test pipeline queries"""
    try:
        from database.pipeline_queries import PipelineQueries
        pq = PipelineQueries()
        
        # Test summary
        summary = pq.get_pipeline_summary()
        print(f"✅ Pipeline summary: {summary}")
        
        # Test status
        status = pq.get_pipeline_status()
        print(f"✅ Pipeline status: {status}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline queries error: {e}")
        return False

def test_signals_queries():
    """Test signals queries"""
    try:
        from database.signals_queries import SignalsQueries
        sq = SignalsQueries()
        
        # Test latest signals
        signals = sq.get_latest_signals()
        print(f"✅ Latest signals: {len(signals)} found")
        
        return True
    except Exception as e:
        print(f"❌ Signals queries error: {e}")
        return False

def test_api_handler():
    """Test API handler"""
    try:
        from handlers.api_handler import APIHandler
        api = APIHandler()
        
        # Test summary API
        summary_content, content_type, status_code = api.serve_summary_api()
        print(f"✅ Summary API: {status_code} - {content_type}")
        
        # Test pipeline status API
        status_content, content_type, status_code = api.serve_pipeline_status_api()
        print(f"✅ Pipeline status API: {status_code} - {content_type}")
        
        return True
    except Exception as e:
        print(f"❌ API handler error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Dashboard Integration...")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Pipeline Queries", test_pipeline_queries),
        ("Signals Queries", test_signals_queries),
        ("API Handler", test_api_handler),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
