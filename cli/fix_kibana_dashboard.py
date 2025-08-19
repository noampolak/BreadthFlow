#!/usr/bin/env python3
"""
Fix Kibana Dashboard Configuration

Creates a properly configured dashboard that won't have the searchSourceJSON error.
"""

import requests
import json
import time
from datetime import datetime

def wait_for_kibana():
    """Wait for Kibana to be ready"""
    kibana_url = "http://kibana:5601"
    max_attempts = 30
    
    print("⏳ Waiting for Kibana to be ready...")
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{kibana_url}/api/status", timeout=5)
            if response.status_code == 200:
                print("✅ Kibana is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print("❌ Kibana not ready after 60 seconds")
    return False

def create_working_dashboard():
    """Create a working dashboard without the searchSourceJSON error"""
    
    if not wait_for_kibana():
        return
    
    kibana_url = "http://kibana:5601"
    headers = {
        'Content-Type': 'application/json',
        'kbn-xsrf': 'true'
    }
    
    print("🎨 Creating Working BreadthFlow Dashboard...")
    print("=" * 50)
    
    # First, create a search object
    search_config = {
        "attributes": {
            "title": "BreadthFlow Pipeline Logs",
            "description": "Search for pipeline execution logs",
            "columns": ["@timestamp", "level", "message", "run_id"],
            "sort": [["@timestamp", "desc"]],
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "breadthflow-logs",
                    "query": {
                        "match_all": {}
                    },
                    "filter": []
                })
            }
        }
    }
    
    try:
        # Create the search
        response = requests.post(
            f"{kibana_url}/api/saved_objects/search/breadthflow-pipeline-logs",
            headers=headers,
            json=search_config,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: Pipeline Logs Search")
        else:
            print(f"⚠️ Search creation response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error creating search: {e}")
        return
    
    # Now create a dashboard that references this search
    dashboard_config = {
        "attributes": {
            "title": "🚀 BreadthFlow Working Dashboard",
            "description": "Real-time pipeline monitoring with proper configuration",
            "hits": 0,
            "description": "BreadthFlow pipeline monitoring dashboard",
            "panelsJSON": json.dumps([
                {
                    "version": "8.11.0",
                    "type": "search",
                    "gridData": {
                        "x": 0, "y": 0, "w": 48, "h": 20,
                        "i": "breadthflow-pipeline-logs"
                    },
                    "panelIndex": "breadthflow-pipeline-logs",
                    "embeddableConfig": {
                        "attributes": {
                            "title": "Pipeline Execution Logs",
                            "columns": ["@timestamp", "level", "message", "run_id"],
                            "sort": ["@timestamp", "desc"]
                        }
                    },
                    "savedObjectId": "breadthflow-pipeline-logs"
                }
            ]),
            "timeRestore": True,
            "timeTo": "now",
            "timeFrom": "now-24h",
            "refreshInterval": {
                "pause": False,
                "value": 30000
            },
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "breadthflow-logs",
                    "query": {
                        "match_all": {}
                    },
                    "filter": []
                })
            }
        }
    }
    
    try:
        # Create the dashboard
        response = requests.post(
            f"{kibana_url}/api/saved_objects/dashboard/breadthflow-working",
            headers=headers,
            json=dashboard_config,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: BreadthFlow Working Dashboard")
        else:
            print(f"⚠️ Dashboard creation response: {response.status_code}")
            print(response.text[:300])
            
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
    
    # Create a simple visualization dashboard
    viz_dashboard = {
        "attributes": {
            "title": "📊 BreadthFlow Analytics",
            "description": "Pipeline performance analytics",
            "panelsJSON": json.dumps([
                {
                    "version": "8.11.0",
                    "type": "visualization",
                    "gridData": {
                        "x": 0, "y": 0, "w": 24, "h": 10,
                        "i": "pipeline-status"
                    },
                    "panelIndex": "pipeline-status",
                    "embeddableConfig": {
                        "attributes": {
                            "title": "Pipeline Status Distribution"
                        }
                    }
                },
                {
                    "version": "8.11.0",
                    "type": "visualization", 
                    "gridData": {
                        "x": 24, "y": 0, "w": 24, "h": 10,
                        "i": "pipeline-timeline"
                    },
                    "panelIndex": "pipeline-timeline",
                    "embeddableConfig": {
                        "attributes": {
                            "title": "Pipeline Execution Timeline"
                        }
                    }
                }
            ]),
            "timeRestore": True,
            "timeTo": "now",
            "timeFrom": "now-24h",
            "refreshInterval": {
                "pause": False,
                "value": 30000
            },
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "breadthflow-logs",
                    "query": {
                        "match_all": {}
                    },
                    "filter": []
                })
            }
        }
    }
    
    try:
        response = requests.post(
            f"{kibana_url}/api/saved_objects/dashboard/breadthflow-analytics",
            headers=headers,
            json=viz_dashboard,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: BreadthFlow Analytics Dashboard")
        else:
            print(f"⚠️ Analytics dashboard response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error creating analytics dashboard: {e}")
    
    print("\n🎉 Dashboard setup completed!")
    print("🌐 Access Kibana: http://localhost:5601")
    print("📊 Available dashboards:")
    print("   • 🚀 BreadthFlow Working Dashboard")
    print("   • 📊 BreadthFlow Analytics")
    print("   • 🚀 BreadthFlow Pipeline Monitor (from previous setup)")
    print("\n💡 If you still see errors, try:")
    print("   1. Clear browser cache")
    print("   2. Go to Stack Management > Saved Objects")
    print("   3. Delete any broken dashboards")
    print("   4. Refresh the page")

if __name__ == "__main__":
    create_working_dashboard()
