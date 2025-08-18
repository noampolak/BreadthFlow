#!/usr/bin/env python3
"""
Kibana Dashboard Setup for BreadthFlow

Creates pre-configured dashboards in Kibana for long-term monitoring
and analysis of BreadthFlow pipeline runs and system metrics.
"""

import requests
import json
import time
import click
from datetime import datetime
from typing import Dict, List, Any

class KibanaSetup:
    """Setup Kibana dashboards for BreadthFlow monitoring"""
    
    def __init__(self, kibana_url: str = "http://localhost:5601"):
        self.kibana_url = kibana_url
        self.headers = {
            'Content-Type': 'application/json',
            'kbn-xsrf': 'true'
        }
    
    def wait_for_kibana(self, timeout: int = 300):
        """Wait for Kibana to be ready"""
        print("⏳ Waiting for Kibana to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.kibana_url}/api/status", timeout=5)
                if response.status_code == 200:
                    print("✅ Kibana is ready!")
                    return True
            except:
                pass
            time.sleep(10)
        
        print("❌ Kibana not ready after timeout")
        return False
    
    def create_index_pattern(self):
        """Create index pattern for BreadthFlow logs"""
        pattern_config = {
            "attributes": {
                "title": "breadthflow-*",
                "timeFieldName": "@timestamp"
            }
        }
        
        try:
            response = requests.post(
                f"{self.kibana_url}/api/saved_objects/index-pattern/breadthflow-logs",
                headers=self.headers,
                json=pattern_config
            )
            
            if response.status_code in [200, 409]:  # 409 = already exists
                print("✅ Created/Updated index pattern: breadthflow-*")
                return True
            else:
                print(f"❌ Failed to create index pattern: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error creating index pattern: {e}")
            return False
    
    def create_pipeline_dashboard(self):
        """Create main pipeline monitoring dashboard"""
        dashboard_config = {
            "attributes": {
                "title": "BreadthFlow Pipeline Monitoring",
                "type": "dashboard",
                "description": "Real-time monitoring of BreadthFlow data pipelines",
                "panelsJSON": json.dumps([
                    {
                        "version": "8.11.0",
                        "type": "lens",
                        "gridData": {
                            "x": 0, "y": 0, "w": 24, "h": 15,
                            "i": "pipeline-runs-over-time"
                        },
                        "panelIndex": "pipeline-runs-over-time",
                        "embeddableConfig": {
                            "attributes": {
                                "title": "Pipeline Runs Over Time",
                                "type": "lens",
                                "visualizationType": "lnsXY",
                                "state": {
                                    "datasourceStates": {
                                        "indexpattern": {
                                            "layers": {
                                                "layer1": {
                                                    "columns": {
                                                        "timestamp": {
                                                            "label": "Timestamp",
                                                            "dataType": "date",
                                                            "operationType": "date_histogram",
                                                            "sourceField": "@timestamp",
                                                            "isBucketed": True,
                                                            "params": {"interval": "auto"}
                                                        },
                                                        "count": {
                                                            "label": "Count of runs",
                                                            "dataType": "number", 
                                                            "operationType": "count",
                                                            "isBucketed": False
                                                        }
                                                    },
                                                    "columnOrder": ["timestamp", "count"],
                                                    "indexPatternId": "breadthflow-logs"
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    {
                        "version": "8.11.0",
                        "type": "lens", 
                        "gridData": {
                            "x": 24, "y": 0, "w": 24, "h": 15,
                            "i": "success-rate-gauge"
                        },
                        "panelIndex": "success-rate-gauge",
                        "embeddableConfig": {
                            "attributes": {
                                "title": "Pipeline Success Rate",
                                "type": "lens",
                                "visualizationType": "lnsGauge"
                            }
                        }
                    },
                    {
                        "version": "8.11.0",
                        "type": "lens",
                        "gridData": {
                            "x": 0, "y": 15, "w": 48, "h": 20,
                            "i": "recent-pipeline-logs"
                        },
                        "panelIndex": "recent-pipeline-logs",
                        "embeddableConfig": {
                            "attributes": {
                                "title": "Recent Pipeline Logs",
                                "type": "lens",
                                "visualizationType": "lnsDatatable"
                            }
                        }
                    }
                ]),
                "timeRestore": False,
                "timeTo": "now",
                "timeFrom": "now-24h",
                "refreshInterval": {
                    "pause": False,
                    "value": 30000  # 30 seconds
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.kibana_url}/api/saved_objects/dashboard/breadthflow-pipeline-monitoring",
                headers=self.headers,
                json=dashboard_config
            )
            
            if response.status_code in [200, 409]:
                print("✅ Created/Updated dashboard: BreadthFlow Pipeline Monitoring")
                return True
            else:
                print(f"❌ Failed to create dashboard: {response.status_code}")
                print(response.text)
                return False
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            return False
    
    def create_system_health_dashboard(self):
        """Create system health monitoring dashboard"""
        dashboard_config = {
            "attributes": {
                "title": "BreadthFlow System Health",
                "type": "dashboard", 
                "description": "Infrastructure monitoring and system health metrics",
                "panelsJSON": json.dumps([
                    {
                        "version": "8.11.0",
                        "type": "lens",
                        "gridData": {
                            "x": 0, "y": 0, "w": 24, "h": 15,
                            "i": "service-health-status"
                        },
                        "panelIndex": "service-health-status",
                        "embeddableConfig": {
                            "attributes": {
                                "title": "Service Health Status",
                                "type": "lens",
                                "visualizationType": "lnsMetric"
                            }
                        }
                    },
                    {
                        "version": "8.11.0", 
                        "type": "lens",
                        "gridData": {
                            "x": 24, "y": 0, "w": 24, "h": 15,
                            "i": "pipeline-duration-trends"
                        },
                        "panelIndex": "pipeline-duration-trends",
                        "embeddableConfig": {
                            "attributes": {
                                "title": "Pipeline Duration Trends",
                                "type": "lens",
                                "visualizationType": "lnsXY"
                            }
                        }
                    }
                ]),
                "timeRestore": False,
                "timeTo": "now", 
                "timeFrom": "now-7d"
            }
        }
        
        try:
            response = requests.post(
                f"{self.kibana_url}/api/saved_objects/dashboard/breadthflow-system-health",
                headers=self.headers,
                json=dashboard_config
            )
            
            if response.status_code in [200, 409]:
                print("✅ Created/Updated dashboard: BreadthFlow System Health")
                return True
            else:
                print(f"❌ Failed to create health dashboard: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error creating health dashboard: {e}")
            return False
    
    def setup_all_dashboards(self):
        """Setup all Kibana dashboards"""
        print("🔧 Setting up Kibana dashboards for BreadthFlow...")
        
        if not self.wait_for_kibana():
            return False
        
        success = True
        success &= self.create_index_pattern()
        success &= self.create_pipeline_dashboard()
        success &= self.create_system_health_dashboard()
        
        if success:
            print("🎉 All Kibana dashboards created successfully!")
            print(f"🌐 Access dashboards at: {self.kibana_url}/app/dashboards")
            print("📊 Available dashboards:")
            print("   • BreadthFlow Pipeline Monitoring")
            print("   • BreadthFlow System Health")
        else:
            print("⚠️ Some dashboards failed to create")
        
        return success

@click.command()
@click.option('--kibana-url', default='http://localhost:5601', help='Kibana URL')
@click.option('--wait-timeout', default=300, help='Timeout waiting for Kibana (seconds)')
def setup_kibana(kibana_url: str, wait_timeout: int):
    """Setup Kibana dashboards for BreadthFlow monitoring"""
    
    print("🚀 BreadthFlow Kibana Dashboard Setup")
    print("=" * 50)
    
    setup = KibanaSetup(kibana_url)
    success = setup.setup_all_dashboards()
    
    if success:
        print("\n✅ Setup completed successfully!")
        print(f"🌐 Open Kibana: {kibana_url}")
        print("📊 Navigate to Dashboards to view BreadthFlow monitoring")
    else:
        print("\n❌ Setup failed. Check Kibana connectivity and try again.")

if __name__ == '__main__':
    setup_kibana()
