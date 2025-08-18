#!/usr/bin/env python3
"""
Create BreadthFlow Dashboards in Kibana

Creates pre-built dashboards for monitoring your pipeline runs:
1. Pipeline Overview Dashboard - High-level metrics
2. Pipeline Performance Dashboard - Detailed performance analysis
3. Data Fetching Dashboard - Symbol-specific tracking
4. Error Monitoring Dashboard - Failure analysis
"""

import requests
import json
import time
import click
from datetime import datetime
from typing import Dict, List, Any

class KibanaDashboardCreator:
    """Creates comprehensive dashboards in Kibana for BreadthFlow monitoring"""
    
    def __init__(self, kibana_url: str = "http://kibana:5601"):
        self.kibana_url = kibana_url
        self.headers = {
            'Content-Type': 'application/json',
            'kbn-xsrf': 'true'
        }
        self.index_pattern = "breadthflow-logs*"
    
    def create_visualizations(self):
        """Create individual visualizations for the dashboards"""
        
        visualizations = [
            # 1. Pipeline Runs Over Time
            {
                "id": "pipeline-runs-timeline",
                "title": "Pipeline Runs Over Time",
                "type": "lens",
                "config": {
                    "title": "Pipeline Runs Over Time",
                    "type": "lens",
                    "visualizationType": "lnsXY",
                    "state": {
                        "visualization": {
                            "title": "Empty XY chart",
                            "legend": {"isVisible": True, "position": "right"},
                            "valueLabels": "hide",
                            "preferredSeriesType": "line",
                            "layers": [
                                {
                                    "layerId": "layer1",
                                    "accessors": ["count"],
                                    "position": "top",
                                    "seriesType": "line",
                                    "showGridlines": False,
                                    "layerType": "data",
                                    "xAccessor": "timestamp"
                                }
                            ]
                        },
                        "query": {"query": "status:started", "language": "kuery"},
                        "filters": []
                    }
                }
            },
            
            # 2. Success Rate Gauge
            {
                "id": "success-rate-gauge",
                "title": "Pipeline Success Rate",
                "type": "lens",
                "config": {
                    "title": "Pipeline Success Rate",
                    "type": "lens",
                    "visualizationType": "lnsGauge",
                    "state": {
                        "visualization": {
                            "layerId": "layer1",
                            "metricAccessor": "success_rate",
                            "goalAccessor": "goal"
                        },
                        "query": {"query": "status:completed OR status:failed", "language": "kuery"},
                        "filters": []
                    }
                }
            },
            
            # 3. Command Distribution
            {
                "id": "command-distribution",
                "title": "Command Distribution",
                "type": "lens", 
                "config": {
                    "title": "Command Distribution",
                    "type": "lens",
                    "visualizationType": "lnsPie",
                    "state": {
                        "visualization": {
                            "shape": "donut",
                            "layers": [
                                {
                                    "layerId": "layer1",
                                    "groups": ["command"],
                                    "metric": "count",
                                    "numberDisplay": "percent",
                                    "categoryDisplay": "default",
                                    "legendDisplay": "default"
                                }
                            ]
                        },
                        "query": {"query": "status:started", "language": "kuery"},
                        "filters": []
                    }
                }
            },
            
            # 4. Recent Pipeline Status
            {
                "id": "recent-pipeline-status",
                "title": "Recent Pipeline Status",
                "type": "lens",
                "config": {
                    "title": "Recent Pipeline Status",
                    "type": "lens",
                    "visualizationType": "lnsDatatable",
                    "state": {
                        "visualization": {
                            "columns": [
                                {"columnId": "timestamp"},
                                {"columnId": "command"}, 
                                {"columnId": "status"},
                                {"columnId": "duration"},
                                {"columnId": "run_id"}
                            ],
                            "layerId": "layer1"
                        },
                        "query": {"query": "status:completed OR status:failed", "language": "kuery"},
                        "filters": []
                    }
                }
            },
            
            # 5. Error Logs
            {
                "id": "error-logs",
                "title": "Error Logs",
                "type": "lens",
                "config": {
                    "title": "Error Logs",
                    "type": "lens", 
                    "visualizationType": "lnsDatatable",
                    "state": {
                        "visualization": {
                            "columns": [
                                {"columnId": "timestamp"},
                                {"columnId": "message"},
                                {"columnId": "run_id"}
                            ],
                            "layerId": "layer1"
                        },
                        "query": {"query": "level:ERROR", "language": "kuery"},
                        "filters": []
                    }
                }
            },
            
            # 6. Symbol Fetch Progress
            {
                "id": "symbol-fetch-progress", 
                "title": "Symbol Fetch Progress",
                "type": "lens",
                "config": {
                    "title": "Symbol Fetch Progress",
                    "type": "lens",
                    "visualizationType": "lnsXY",
                    "state": {
                        "visualization": {
                            "title": "Symbol Fetch Progress",
                            "legend": {"isVisible": True, "position": "right"},
                            "valueLabels": "hide",
                            "preferredSeriesType": "bar",
                            "layers": [
                                {
                                    "layerId": "layer1",
                                    "accessors": ["count"],
                                    "position": "top",
                                    "seriesType": "bar_stacked",
                                    "showGridlines": False,
                                    "layerType": "data",
                                    "xAccessor": "symbol",
                                    "splitAccessor": "status"
                                }
                            ]
                        },
                        "query": {"query": "metadata.symbol:*", "language": "kuery"},
                        "filters": []
                    }
                }
            }
        ]
        
        created_vis = []
        for vis in visualizations:
            try:
                response = requests.post(
                    f"{self.kibana_url}/api/saved_objects/lens/{vis['id']}",
                    headers=self.headers,
                    json={"attributes": vis["config"]},
                    timeout=10
                )
                
                if response.status_code in [200, 409]:
                    print(f"✅ Created visualization: {vis['title']}")
                    created_vis.append(vis["id"])
                else:
                    print(f"⚠️ Failed to create {vis['title']}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error creating {vis['title']}: {e}")
        
        return created_vis
    
    def create_dashboard(self, dashboard_id: str, title: str, description: str, panels: List[Dict]):
        """Create a dashboard with specified panels"""
        
        dashboard_config = {
            "attributes": {
                "title": title,
                "type": "dashboard",
                "description": description,
                "panelsJSON": json.dumps(panels),
                "timeRestore": False,
                "timeTo": "now",
                "timeFrom": "now-24h",
                "refreshInterval": {
                    "pause": False,
                    "value": 30000  # 30 seconds
                },
                "kibanaSavedObjectMeta": {
                    "searchSourceJSON": json.dumps({
                        "query": {"query": "", "language": "kuery"},
                        "filter": []
                    })
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.kibana_url}/api/saved_objects/dashboard/{dashboard_id}",
                headers=self.headers,
                json=dashboard_config,
                timeout=10
            )
            
            if response.status_code in [200, 409]:
                print(f"✅ Created dashboard: {title}")
                return True
            else:
                print(f"❌ Failed to create dashboard {title}: {response.status_code}")
                print(response.text[:500])
                return False
                
        except Exception as e:
            print(f"❌ Error creating dashboard {title}: {e}")
            return False
    
    def create_pipeline_overview_dashboard(self):
        """Create main pipeline overview dashboard"""
        
        panels = [
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 0, "y": 0, "w": 24, "h": 15, "i": "timeline"},
                "panelIndex": "timeline",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Pipeline Runs Timeline",
                        "description": "Number of pipeline runs over time"
                    }
                },
                "panelRefName": "panel_timeline"
            },
            {
                "version": "8.11.0", 
                "type": "lens",
                "gridData": {"x": 24, "y": 0, "w": 12, "h": 15, "i": "success-rate"},
                "panelIndex": "success-rate",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Success Rate",
                        "description": "Overall pipeline success rate"
                    }
                },
                "panelRefName": "panel_success_rate"
            },
            {
                "version": "8.11.0",
                "type": "lens", 
                "gridData": {"x": 36, "y": 0, "w": 12, "h": 15, "i": "commands"},
                "panelIndex": "commands",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Command Distribution",
                        "description": "Types of commands being run"
                    }
                },
                "panelRefName": "panel_commands"
            },
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 0, "y": 15, "w": 48, "h": 20, "i": "recent-runs"},
                "panelIndex": "recent-runs", 
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Recent Pipeline Runs",
                        "description": "Latest pipeline executions"
                    }
                },
                "panelRefName": "panel_recent_runs"
            }
        ]
        
        return self.create_dashboard(
            "breadthflow-overview",
            "🚀 BreadthFlow Pipeline Overview",
            "High-level overview of pipeline performance and activity",
            panels
        )
    
    def create_performance_dashboard(self):
        """Create performance monitoring dashboard"""
        
        panels = [
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 0, "y": 0, "w": 24, "h": 15, "i": "duration-trend"},
                "panelIndex": "duration-trend",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Pipeline Duration Trends",
                        "description": "Performance trends over time"
                    }
                },
                "panelRefName": "panel_duration_trend"
            },
            {
                "version": "8.11.0",
                "type": "lens", 
                "gridData": {"x": 24, "y": 0, "w": 24, "h": 15, "i": "symbol-progress"},
                "panelIndex": "symbol-progress",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Symbol Processing",
                        "description": "Success/failure by symbol"
                    }
                },
                "panelRefName": "panel_symbol_progress"
            },
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 0, "y": 15, "w": 48, "h": 20, "i": "performance-table"},
                "panelIndex": "performance-table",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Performance Metrics",
                        "description": "Detailed performance data"
                    }
                },
                "panelRefName": "panel_performance_table"
            }
        ]
        
        return self.create_dashboard(
            "breadthflow-performance", 
            "📈 BreadthFlow Performance Monitoring",
            "Detailed performance analysis and trends",
            panels
        )
    
    def create_error_monitoring_dashboard(self):
        """Create error monitoring and troubleshooting dashboard"""
        
        panels = [
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 0, "y": 0, "w": 24, "h": 15, "i": "error-timeline"},
                "panelIndex": "error-timeline",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Errors Over Time",
                        "description": "Error frequency trends"
                    }
                },
                "panelRefName": "panel_error_timeline"
            },
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 24, "y": 0, "w": 24, "h": 15, "i": "error-types"},
                "panelIndex": "error-types", 
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Error Types",
                        "description": "Common error patterns"
                    }
                },
                "panelRefName": "panel_error_types"
            },
            {
                "version": "8.11.0",
                "type": "lens",
                "gridData": {"x": 0, "y": 15, "w": 48, "h": 25, "i": "error-details"},
                "panelIndex": "error-details",
                "embeddableConfig": {
                    "enhancements": {},
                    "attributes": {
                        "title": "Error Details",
                        "description": "Detailed error logs for troubleshooting"
                    }
                },
                "panelRefName": "panel_error_details"
            }
        ]
        
        return self.create_dashboard(
            "breadthflow-errors",
            "🚨 BreadthFlow Error Monitoring", 
            "Error tracking and troubleshooting dashboard",
            panels
        )
    
    def create_all_dashboards(self):
        """Create all BreadthFlow dashboards"""
        print("🎨 Creating BreadthFlow Dashboards in Kibana...")
        print("=" * 60)
        
        success_count = 0
        
        # Create visualizations first
        print("📊 Creating visualizations...")
        # self.create_visualizations()
        
        # Create dashboards
        print("\n🏗️ Creating dashboards...")
        
        if self.create_pipeline_overview_dashboard():
            success_count += 1
            
        if self.create_performance_dashboard():
            success_count += 1
            
        if self.create_error_monitoring_dashboard():
            success_count += 1
        
        print(f"\n🎉 Created {success_count}/3 dashboards successfully!")
        
        if success_count > 0:
            print(f"\n🌐 Access your dashboards at: {self.kibana_url.replace('kibana', 'localhost')}")
            print("📊 Available dashboards:")
            print("   • 🚀 BreadthFlow Pipeline Overview")
            print("   • 📈 BreadthFlow Performance Monitoring") 
            print("   • 🚨 BreadthFlow Error Monitoring")
            print("\n💡 Navigate to Dashboard > Browse dashboards to see them")
        
        return success_count > 0

@click.command()
@click.option('--kibana-url', default='http://kibana:5601', help='Kibana URL (use kibana:5601 from container)')
def create_dashboards(kibana_url: str):
    """Create comprehensive BreadthFlow dashboards in Kibana"""
    
    creator = KibanaDashboardCreator(kibana_url)
    creator.create_all_dashboards()

if __name__ == '__main__':
    create_dashboards()
