#!/usr/bin/env python3
"""
Create Clean Kibana Dashboard

Creates a minimal, working dashboard without complex configurations that cause errors.
"""

import requests
import json
import time

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

def delete_existing_dashboards():
    """Delete existing dashboards to start fresh"""
    kibana_url = "http://kibana:5601"
    headers = {
        'Content-Type': 'application/json',
        'kbn-xsrf': 'true'
    }
    
    dashboards_to_delete = [
        'breadthflow-working',
        'breadthflow-simple', 
        'breadthflow-analytics',
        'breadthflow-pipeline-monitor'
    ]
    
    print("🗑️ Cleaning up existing dashboards...")
    for dashboard_id in dashboards_to_delete:
        try:
            response = requests.delete(
                f"{kibana_url}/api/saved_objects/dashboard/{dashboard_id}",
                headers=headers,
                timeout=10
            )
            if response.status_code in [200, 404]:
                print(f"   ✅ Deleted: {dashboard_id}")
            else:
                print(f"   ⚠️ Could not delete {dashboard_id}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error deleting {dashboard_id}: {e}")

def create_clean_dashboard():
    """Create a minimal, working dashboard"""
    
    if not wait_for_kibana():
        return
    
    delete_existing_dashboards()
    
    kibana_url = "http://kibana:5601"
    headers = {
        'Content-Type': 'application/json',
        'kbn-xsrf': 'true'
    }
    
    print("🎨 Creating Clean BreadthFlow Dashboard...")
    print("=" * 50)
    
    # Create a minimal dashboard with just a search panel
    clean_dashboard = {
        "attributes": {
            "title": "BreadthFlow Pipeline Monitor",
            "description": "Real-time pipeline monitoring",
            "hits": 0,
            "description": "BreadthFlow pipeline monitoring dashboard",
            "panelsJSON": json.dumps([
                {
                    "version": "8.11.0",
                    "type": "search",
                    "gridData": {
                        "x": 0, "y": 0, "w": 48, "h": 20,
                        "i": "search-panel"
                    },
                    "panelIndex": "search-panel",
                    "embeddableConfig": {
                        "attributes": {
                            "title": "Pipeline Logs",
                            "columns": ["@timestamp", "level", "message", "run_id"],
                            "sort": ["@timestamp", "desc"]
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
            f"{kibana_url}/api/saved_objects/dashboard/breadthflow-clean",
            headers=headers,
            json=clean_dashboard,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: BreadthFlow Pipeline Monitor (Clean)")
        else:
            print(f"⚠️ Dashboard creation response: {response.status_code}")
            print(response.text[:300])
            
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
    
    # Create an even simpler dashboard with no panels initially
    simple_dashboard = {
        "attributes": {
            "title": "BreadthFlow Simple",
            "description": "Simple pipeline monitoring",
            "hits": 0,
            "description": "Simple BreadthFlow dashboard",
            "panelsJSON": "[]",
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
            f"{kibana_url}/api/saved_objects/dashboard/breadthflow-simple-clean",
            headers=headers,
            json=simple_dashboard,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: BreadthFlow Simple (No Panels)")
        else:
            print(f"⚠️ Simple dashboard response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error creating simple dashboard: {e}")
    
    print("\n🎉 Clean dashboard setup completed!")
    print("🌐 Access Kibana: http://localhost:5601")
    print("📊 Available dashboards:")
    print("   • BreadthFlow Pipeline Monitor (Clean)")
    print("   • BreadthFlow Simple (No Panels)")
    print("\n💡 Instructions:")
    print("   1. Go to Dashboard → Browse dashboards")
    print("   2. Click on 'BreadthFlow Pipeline Monitor'")
    print("   3. If it works, you can add panels manually")
    print("   4. If it still has errors, try 'BreadthFlow Simple'")

if __name__ == "__main__":
    create_clean_dashboard()
