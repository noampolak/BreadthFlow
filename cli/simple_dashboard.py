#!/usr/bin/env python3
"""
Simple BreadthFlow Dashboard
A basic web dashboard for pipeline monitoring
"""

import click
import json
import sqlite3
import os
import requests
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any, Optional
import urllib.parse

# Database configuration
DB_PATH = os.getenv('DB_PATH', '/app/data/pipeline.db')

def get_db_connection():
    """Get SQLite database connection"""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return sqlite3.connect(DB_PATH)
    except Exception:
        return sqlite3.connect('dashboard.db')

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.serve_dashboard()
        elif self.path == '/api/summary':
            self.serve_summary()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>BreadthFlow Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .stat-card { background: #f0f0f0; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #333; }
        .stat-label { color: #666; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 BreadthFlow Pipeline Dashboard</h1>
        <p>Real-time pipeline monitoring</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value" id="total-runs">-</div>
            <div class="stat-label">Total Runs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="success-rate">-</div>
            <div class="stat-label">Success Rate</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="recent-runs">-</div>
            <div class="stat-label">Recent Runs</div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/summary');
                const data = await response.json();
                
                document.getElementById('total-runs').textContent = data.total_runs || 0;
                document.getElementById('success-rate').textContent = (data.success_rate || 0) + '%';
                document.getElementById('recent-runs').textContent = data.recent_runs || 0;
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        // Load data on page load
        loadData();
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_summary(self):
        try:
            data = self.get_summary()
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def get_summary(self):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Total runs
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs')
            total_runs = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            # Success rate
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE status = "completed"')
            successful = cursor.fetchone()[0] if cursor.fetchone() else 0
            success_rate = (successful / total_runs * 100) if total_runs > 0 else 0
            
            # Recent runs (last 24h)
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE start_time > ?', (yesterday,))
            recent_runs = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            conn.close()
            
            return {
                'total_runs': total_runs,
                'success_rate': round(success_rate, 1),
                'recent_runs': recent_runs,
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

@click.command()
@click.option('--port', default=8080, help='Port to run dashboard on')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
def start_dashboard(port: int, host: str):
    """Start the simple web dashboard"""
    
    print(f"🚀 Starting BreadthFlow Dashboard...")
    print(f"📊 Dashboard URL: http://localhost:{port}")
    
    httpd = HTTPServer((host, port), DashboardHandler)
    
    print(f"✅ Dashboard ready at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        httpd.shutdown()

if __name__ == '__main__':
    start_dashboard()
