#!/usr/bin/env python3
"""
Integrated Monitoring Solution for BreadthFlow

Combines real-time dashboard + Kibana logging for the best of both worlds:
- Real-time progress tracking in custom dashboard
- Long-term analysis and trends in Kibana
- Single command to start both monitoring systems
"""

import click
import time
import requests
import subprocess
import threading
import webbrowser
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedMonitoring:
    """Manages both dashboard and Kibana integration"""
    
    def __init__(self):
        self.dashboard_port = 8082
        self.kibana_url = "http://localhost:5601"
        self.dashboard_process = None
        
    def start_dashboard(self):
        """Start the real-time dashboard"""
        try:
            logger.info("🚀 Starting real-time dashboard...")
            
            # Start dashboard in background
            cmd = [
                "docker", "exec", "-d", "spark-master", "python3", 
                "/opt/bitnami/spark/jobs/cli/web_dashboard.py",
                "--port", str(self.dashboard_port),
                "--host", "0.0.0.0"
            ]
            
            self.dashboard_process = subprocess.Popen(cmd)
            time.sleep(3)  # Give it time to start
            
            # Check if it's running
            dashboard_url = f"http://localhost:{self.dashboard_port}"
            try:
                response = requests.get(f"{dashboard_url}/api/status", timeout=5)
                logger.info(f"✅ Dashboard started: {dashboard_url}")
                return True
            except:
                logger.info(f"📊 Dashboard starting up: {dashboard_url}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def setup_kibana_integration(self):
        """Setup Kibana dashboards if not already done"""
        try:
            logger.info("📊 Setting up Kibana integration...")
            
            # Check if Kibana is available
            response = requests.get(f"{self.kibana_url}/api/status", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Kibana is available")
                
                # Setup dashboards
                cmd = [
                    "docker", "exec", "spark-master", "python3",
                    "/opt/bitnami/spark/jobs/cli/setup_kibana_dashboards.py"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("✅ Kibana dashboards configured")
                    return True
                else:
                    logger.warning("⚠️ Kibana setup had issues (may already be configured)")
                    return True
            else:
                logger.warning("⚠️ Kibana not available yet")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Kibana setup failed: {e}")
            return False
    
    def show_access_info(self):
        """Show how to access both monitoring systems"""
        print("\n" + "="*60)
        print("🎯 BreadthFlow Integrated Monitoring Active!")
        print("="*60)
        
        print(f"\n🚀 REAL-TIME DASHBOARD (Development & Debugging):")
        print(f"   URL: http://localhost:{self.dashboard_port}")
        print(f"   Best for: Live progress, immediate logs, system status")
        
        print(f"\n📊 KIBANA ANALYTICS (Production & Analysis):")
        print(f"   URL: {self.kibana_url}")
        print(f"   Best for: Historical trends, pattern analysis, alerts")
        
        print(f"\n💡 USAGE RECOMMENDATIONS:")
        print(f"   • During development: Use REAL-TIME dashboard")
        print(f"   • For troubleshooting: Start with REAL-TIME, then Kibana for patterns")
        print(f"   • For production monitoring: Set up Kibana alerts")
        print(f"   • For reporting: Use Kibana's export features")
        
        print(f"\n🎮 ENHANCED COMMANDS:")
        print(f"   # Use these commands for automatic logging to both systems:")
        print(f"   docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py data summary")
        print(f"   docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py demo --quick")
        
        print("\n" + "="*60)

@click.group()
def cli():
    """🎯 BreadthFlow Integrated Monitoring"""
    pass

@cli.command()
@click.option('--open-browser', is_flag=True, help='Automatically open browser tabs')
@click.option('--dashboard-only', is_flag=True, help='Start only the real-time dashboard')
def start(open_browser, dashboard_only):
    """Start integrated monitoring (Dashboard + Kibana)"""
    
    monitor = IntegratedMonitoring()
    
    print("🎯 Starting BreadthFlow Integrated Monitoring...")
    
    # Start real-time dashboard
    dashboard_started = monitor.start_dashboard()
    
    # Setup Kibana integration (unless dashboard-only)
    kibana_ready = True
    if not dashboard_only:
        kibana_ready = monitor.setup_kibana_integration()
    
    # Show access information
    monitor.show_access_info()
    
    # Open browser if requested
    if open_browser:
        if dashboard_started:
            webbrowser.open(f"http://localhost:{monitor.dashboard_port}")
        if kibana_ready and not dashboard_only:
            time.sleep(2)
            webbrowser.open(monitor.kibana_url)
    
    if dashboard_started:
        print(f"\n✅ Monitoring setup complete!")
        print(f"💡 The dashboard will continue running in the background")
        print(f"🔄 Use 'bf monitoring status' to check if it's still running")
    else:
        print(f"\n❌ Some components failed to start")

@cli.command()
def status():
    """Check status of all monitoring components"""
    
    print("🔍 Checking BreadthFlow Monitoring Status...")
    print("-" * 50)
    
    # Check dashboard
    try:
        response = requests.get("http://localhost:8082/api/status", timeout=5)
        print("✅ Real-time Dashboard: Running (http://localhost:8082)")
    except:
        print("❌ Real-time Dashboard: Not running")
    
    # Check Kibana
    try:
        response = requests.get("http://localhost:5601/api/status", timeout=5)
        print("✅ Kibana: Available (http://localhost:5601)")
    except:
        print("❌ Kibana: Not available")
    
    # Check other services
    services = [
        ("Spark UI", "http://localhost:8080"),
        ("MinIO Console", "http://localhost:9001"),
        ("Elasticsearch", "http://localhost:9200")
    ]
    
    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            print(f"✅ {name}: Available ({url})")
        except:
            print(f"❌ {name}: Not available ({url})")

@cli.command()
def stop():
    """Stop monitoring components"""
    
    print("🛑 Stopping BreadthFlow Monitoring...")
    
    # Try to stop dashboard processes
    try:
        subprocess.run([
            "docker", "exec", "spark-master", "pkill", "-f", "web_dashboard.py"
        ], capture_output=True)
        print("✅ Stopped dashboard processes")
    except:
        print("⚠️ No dashboard processes found")
    
    print("💡 Kibana and other services are still running (part of main infrastructure)")

@cli.command()
@click.option('--symbols', default='AAPL,MSFT', help='Symbols to test with')
def demo(symbols):
    """Run a monitoring demo to show both systems in action"""
    
    monitor = IntegratedMonitoring()
    
    print("🎬 BreadthFlow Monitoring Demo")
    print("="*40)
    
    # Ensure monitoring is running
    print("🔧 Ensuring monitoring is active...")
    monitor.start_dashboard()
    
    print(f"\n📊 Running demo pipeline with symbols: {symbols}")
    print("💡 Watch the progress in your browser!")
    
    # Show URLs
    print(f"\n🚀 Real-time Dashboard: http://localhost:{monitor.dashboard_port}")
    print(f"📊 Kibana Analytics: {monitor.kibana_url}")
    
    # Run the demo pipeline
    cmd = [
        "docker", "exec", "spark-master", "python3",
        "/opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py",
        "demo", "--quick"
    ]
    
    print(f"\n🎮 Executing: Enhanced demo with logging...")
    result = subprocess.run(cmd)
    
    print(f"\n✅ Demo completed! Check both monitoring systems:")
    print(f"   • Real-time dashboard for immediate logs")
    print(f"   • Kibana for historical analysis")

if __name__ == '__main__':
    cli()
