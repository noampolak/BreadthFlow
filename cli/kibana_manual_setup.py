#!/usr/bin/env python3
"""
Manual Kibana Dashboard Setup Guide

Provides step-by-step instructions for creating a working dashboard manually in Kibana UI.
This avoids all the API configuration issues.
"""

import requests
import json
import time

def check_elasticsearch_data():
    """Check if we have data in Elasticsearch"""
    print("🔍 Checking Elasticsearch data...")
    
    try:
        response = requests.get("http://elasticsearch:9200/breadthflow-logs/_count", timeout=5)
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            print(f"✅ Found {count} log entries in Elasticsearch")
            return count > 0
        else:
            print(f"❌ Elasticsearch error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Elasticsearch: {e}")
        return False

def generate_sample_data():
    """Generate some sample data if needed"""
    print("📝 Generating sample pipeline data...")
    
    try:
        # Run a simple command to generate logs
        import subprocess
        result = subprocess.run([
            'docker', 'exec', 'spark-master', 
            'python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 
            'data', 'summary'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Generated fresh pipeline data")
            return True
        else:
            print(f"⚠️ Data generation had issues: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False

def print_manual_instructions():
    """Print step-by-step manual setup instructions"""
    
    print("\n" + "="*60)
    print("🎨 MANUAL KIBANA DASHBOARD SETUP")
    print("="*60)
    print("Since the automated dashboard creation has issues, let's create")
    print("a working dashboard manually through the Kibana UI.")
    print()
    
    print("📋 STEP-BY-STEP INSTRUCTIONS:")
    print()
    
    print("1️⃣  OPEN KIBANA")
    print("   • Go to: http://localhost:5601")
    print("   • Wait for Kibana to fully load")
    print()
    
    print("2️⃣  CREATE A NEW DASHBOARD")
    print("   • Click 'Dashboard' in the left sidebar")
    print("   • Click 'Create dashboard' button")
    print("   • You'll see an empty dashboard in edit mode")
    print()
    
    print("3️⃣  ADD A SEARCH PANEL")
    print("   • Click 'Add panel' button")
    print("   • Select 'Saved search'")
    print("   • If no saved searches exist, click 'Create new search'")
    print()
    
    print("4️⃣  CONFIGURE THE SEARCH")
    print("   • In the search interface:")
    print("     - Index pattern: Select 'breadthflow-logs'")
    print("     - Time range: Set to 'Last 24 hours'")
    print("     - Query: Leave as 'match_all' (or use: level:INFO)")
    print("     - Columns: Add @timestamp, level, message, run_id")
    print("     - Sort: @timestamp (descending)")
    print()
    
    print("5️⃣  SAVE THE SEARCH")
    print("   • Click 'Save search to Kibana'")
    print("   • Name it: 'BreadthFlow Pipeline Logs'")
    print("   • Click 'Save'")
    print()
    
    print("6️⃣  ADD TO DASHBOARD")
    print("   • The search will be added to your dashboard")
    print("   • Resize it to fill the available space")
    print()
    
    print("7️⃣  SAVE THE DASHBOARD")
    print("   • Click 'Save dashboard to Kibana'")
    print("   • Name it: 'BreadthFlow Pipeline Monitor'")
    print("   • Click 'Save'")
    print()
    
    print("8️⃣  TEST THE DASHBOARD")
    print("   • Switch to 'View mode'")
    print("   • You should see your pipeline logs")
    print("   • Set auto-refresh to 30 seconds if desired")
    print()
    
    print("🎯 ALTERNATIVE: USE DISCOVER")
    print("   • If dashboard creation is still problematic:")
    print("   • Go to 'Discover' in the left sidebar")
    print("   • Select 'breadthflow-logs' index")
    print("   • This gives you a simple log viewer")
    print()
    
    print("🔧 TROUBLESHOOTING:")
    print("   • Clear browser cache (Ctrl+Shift+R)")
    print("   • Try incognito/private browsing mode")
    print("   • Check browser console for JavaScript errors")
    print("   • Ensure Elasticsearch has data (see above)")
    print()

def main():
    """Main function"""
    print("🚀 BreadthFlow Kibana Manual Setup")
    print("="*50)
    
    # Check if we have data
    has_data = check_elasticsearch_data()
    
    if not has_data:
        print("⚠️  No data found in Elasticsearch")
        print("🔄 Generating sample data...")
        generate_sample_data()
        time.sleep(2)
        has_data = check_elasticsearch_data()
    
    if has_data:
        print("✅ Data is available for dashboard creation")
    else:
        print("❌ Still no data available")
    
    # Print manual instructions
    print_manual_instructions()
    
    print("🌐 Ready to create dashboard manually!")
    print("   Kibana URL: http://localhost:5601")

if __name__ == "__main__":
    main()
