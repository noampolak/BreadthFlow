#!/usr/bin/env python3
"""
Simple Kibana Dashboard Setup

Creates basic dashboards using Kibana's saved objects API.
Focus on getting working dashboards quickly rather than complex configurations.
"""

import requests
import json
import click
from datetime import datetime

def create_simple_dashboards():
    """Create simple but functional dashboards"""
    
    kibana_url = "http://kibana:5601"
    headers = {
        'Content-Type': 'application/json',
        'kbn-xsrf': 'true'
    }
    
    print("🎨 Creating Simple BreadthFlow Dashboards...")
    print("=" * 50)
    
    # Create a basic dashboard with discover panels
    basic_dashboard = {
        "attributes": {
            "title": "🚀 BreadthFlow Pipeline Monitor",
            "description": "Real-time pipeline monitoring dashboard",
            "panelsJSON": json.dumps([
                {
                    "version": "8.11.0",
                    "type": "search",
                    "gridData": {
                        "x": 0, "y": 0, "w": 48, "h": 20,
                        "i": "pipeline-logs"
                    },
                    "panelIndex": "pipeline-logs",
                    "embeddableConfig": {
                        "attributes": {
                            "title": "Recent Pipeline Logs",
                            "columns": ["@timestamp", "level", "message", "run_id", "status"],
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
            f"{kibana_url}/api/saved_objects/dashboard/breadthflow-simple",
            headers=headers,
            json=basic_dashboard,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: BreadthFlow Pipeline Monitor")
        else:
            print(f"⚠️ Dashboard creation response: {response.status_code}")
            print(response.text[:300])
            
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
    
    # Also create a search for errors
    error_search = {
        "attributes": {
            "title": "🚨 BreadthFlow Errors",
            "description": "Pipeline errors and warnings",
            "columns": ["@timestamp", "level", "message", "run_id"],
            "sort": ["@timestamp", "desc"],
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "index": "breadthflow-logs",
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"level": "ERROR"}},
                                {"match": {"level": "WARN"}}
                            ]
                        }
                    },
                    "filter": []
                })
            }
        }
    }
    
    try:
        response = requests.post(
            f"{kibana_url}/api/saved_objects/search/breadthflow-errors",
            headers=headers,
            json=error_search,
            timeout=10
        )
        
        if response.status_code in [200, 409]:
            print("✅ Created: BreadthFlow Errors Search")
        else:
            print(f"⚠️ Error search response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error creating error search: {e}")
    
    print("\n🎉 Setup completed!")
    print(f"🌐 Access Kibana: http://localhost:5601")
    print("📊 Go to Dashboard > Browse dashboards")
    print("🔍 Look for: '🚀 BreadthFlow Pipeline Monitor'")
    
    return True

@click.command()
def setup():
    """Setup simple Kibana dashboards for BreadthFlow"""
    create_simple_dashboards()

if __name__ == '__main__':
    setup()
