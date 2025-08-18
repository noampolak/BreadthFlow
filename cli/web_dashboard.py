#!/usr/bin/env python3
"""
BreadthFlow Web Dashboard - Real-time Pipeline Monitoring

Provides a modern web UI for monitoring data pipelines, showing:
- Real-time progress of data fetching and processing
- Historical run logs and statistics
- System health and status
- Interactive charts and visualizations
"""

import click
import json
import time
import sqlite3
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineRun:
    """Represents a pipeline execution run"""
    run_id: str
    command: str
    status: str  # 'running', 'completed', 'failed'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    logs: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.metadata is None:
            self.metadata = {}

class DashboardDB:
    """SQLite database for storing pipeline runs and logs"""
    
    def __init__(self, db_path: str = "dashboard.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Pipeline runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration REAL,
                metadata TEXT
            )
        ''')
        
        # Logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
            )
        ''')
        
        # System status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_status (
                timestamp TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_run(self, run: PipelineRun):
        """Add a new pipeline run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO pipeline_runs 
            (run_id, command, status, start_time, end_time, duration, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            run.run_id,
            run.command,
            run.status,
            run.start_time.isoformat(),
            run.end_time.isoformat() if run.end_time else None,
            run.duration,
            json.dumps(run.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def add_log(self, run_id: str, level: str, message: str, timestamp: Optional[datetime] = None):
        """Add a log entry"""
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO logs (run_id, timestamp, level, message)
            VALUES (?, ?, ?, ?)
        ''', (run_id, timestamp.isoformat(), level, message))
        
        conn.commit()
        conn.close()
    
    def get_recent_runs(self, limit: int = 50) -> List[Dict]:
        """Get recent pipeline runs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT run_id, command, status, start_time, end_time, duration, metadata
            FROM pipeline_runs
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        runs = []
        for row in cursor.fetchall():
            run_data = {
                'run_id': row[0],
                'command': row[1],
                'status': row[2],
                'start_time': row[3],
                'end_time': row[4],
                'duration': row[5],
                'metadata': json.loads(row[6]) if row[6] else {}
            }
            runs.append(run_data)
        
        conn.close()
        return runs
    
    def get_run_logs(self, run_id: str) -> List[Dict]:
        """Get logs for a specific run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, level, message
            FROM logs
            WHERE run_id = ?
            ORDER BY timestamp ASC
        ''', (run_id,))
        
        logs = []
        for row in cursor.fetchall():
            logs.append({
                'timestamp': row[0],
                'level': row[1],
                'message': row[2]
            })
        
        conn.close()
        return logs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total runs
        cursor.execute('SELECT COUNT(*) FROM pipeline_runs')
        total_runs = cursor.fetchone()[0]
        
        # Success rate
        cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE status = "completed"')
        successful_runs = cursor.fetchone()[0]
        
        # Recent activity (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE start_time > ?', (yesterday,))
        recent_runs = cursor.fetchone()[0]
        
        # Average duration
        cursor.execute('SELECT AVG(duration) FROM pipeline_runs WHERE duration IS NOT NULL')
        avg_duration = cursor.fetchone()[0] or 0
        
        conn.close()
        
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        return {
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'success_rate': round(success_rate, 1),
            'recent_runs': recent_runs,
            'avg_duration': round(avg_duration, 2) if avg_duration else 0
        }

class DashboardServer(BaseHTTPRequestHandler):
    """HTTP server for the web dashboard"""
    
    def __init__(self, *args, db: DashboardDB, **kwargs):
        self.db = db
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/dashboard':
            self.serve_dashboard()
        elif path == '/api/runs':
            self.serve_runs_api()
        elif path.startswith('/api/logs/'):
            run_id = path.split('/')[-1]
            self.serve_logs_api(run_id)
        elif path == '/api/stats':
            self.serve_stats_api()
        elif path == '/api/status':
            self.serve_status_api()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html = self.get_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_runs_api(self):
        """Serve recent runs as JSON"""
        runs = self.db.get_recent_runs()
        self.send_json_response(runs)
    
    def serve_logs_api(self, run_id: str):
        """Serve logs for a specific run"""
        logs = self.db.get_run_logs(run_id)
        self.send_json_response(logs)
    
    def serve_stats_api(self):
        """Serve dashboard statistics"""
        stats = self.db.get_stats()
        self.send_json_response(stats)
    
    def serve_status_api(self):
        """Serve system status"""
        # Check if services are running
        status = {
            'timestamp': datetime.now().isoformat(),
            'services': {
                'spark': self.check_service_health('localhost', 8080),
                'minio': self.check_service_health('localhost', 9001),
                'kafka': self.check_service_health('localhost', 9092),
                'kibana': self.check_service_health('localhost', 5601),
                'elasticsearch': self.check_service_health('localhost', 9200)
            }
        }
        self.send_json_response(status)
    
    def check_service_health(self, host: str, port: int) -> bool:
        """Check if a service is responding"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except:
            return False
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #666; font-size: 1.1em; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label { color: #666; font-size: 0.9em; margin-top: 5px; }
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .panel h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        .runs-list {
            max-height: 500px;
            overflow-y: auto;
        }
        .run-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #eee;
            transition: background 0.3s ease;
        }
        .run-item:hover { background: #f8f9fa; }
        .run-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-completed { background: #d4edda; color: #155724; }
        .status-running { background: #fff3cd; color: #856404; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .services-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .service-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .service-status {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .status-healthy { background: #28a745; }
        .status-unhealthy { background: #dc3545; }
        .refresh-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.3s ease;
        }
        .refresh-btn:hover { transform: scale(1.05); }
        .logs-panel {
            grid-column: 1 / -1;
        }
        .logs-container {
            background: #1e1e1e;
            color: #00ff00;
            border-radius: 8px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .log-entry {
            margin-bottom: 5px;
            word-wrap: break-word;
        }
        .log-timestamp { color: #888; }
        .log-level-INFO { color: #00ff00; }
        .log-level-WARN { color: #ffaa00; }
        .log-level-ERROR { color: #ff4444; }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 BreadthFlow Dashboard</h1>
            <p class="subtitle">Real-time Pipeline Monitoring & Analytics</p>
        </div>
        
        <div class="stats-grid">
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
                <div class="stat-label">Last 24h</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-duration">-</div>
                <div class="stat-label">Avg Duration (s)</div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="panel">
                <h2>📊 Recent Pipeline Runs</h2>
                <button class="refresh-btn" onclick="loadData()">🔄 Refresh</button>
                <div class="runs-list" id="runs-list">
                    Loading...
                </div>
            </div>
            
            <div class="panel">
                <h2>🏥 System Status</h2>
                <div class="services-grid" id="services-status">
                    Loading...
                </div>
            </div>
            
            <div class="panel logs-panel">
                <h2>📝 Latest Logs</h2>
                <div class="logs-container" id="logs-container">
                    Click on a run to view logs...
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentRunId = null;
        
        async function loadData() {
            try {
                // Load stats
                const statsResponse = await fetch('/api/stats');
                const stats = await statsResponse.json();
                document.getElementById('total-runs').textContent = stats.total_runs;
                document.getElementById('success-rate').textContent = stats.success_rate + '%';
                document.getElementById('recent-runs').textContent = stats.recent_runs;
                document.getElementById('avg-duration').textContent = stats.avg_duration;
                
                // Load runs
                const runsResponse = await fetch('/api/runs');
                const runs = await runsResponse.json();
                displayRuns(runs);
                
                // Load system status
                const statusResponse = await fetch('/api/status');
                const status = await statusResponse.json();
                displaySystemStatus(status);
                
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        function displayRuns(runs) {
            const runsList = document.getElementById('runs-list');
            runsList.innerHTML = '';
            
            runs.forEach(run => {
                const runItem = document.createElement('div');
                runItem.className = 'run-item';
                runItem.style.cursor = 'pointer';
                runItem.onclick = () => loadLogs(run.run_id);
                
                const startTime = new Date(run.start_time).toLocaleString();
                const duration = run.duration ? `${run.duration.toFixed(1)}s` : 'N/A';
                
                runItem.innerHTML = `
                    <div>
                        <strong>${run.command}</strong><br>
                        <small>${startTime} • ${duration}</small>
                    </div>
                    <span class="run-status status-${run.status}">${run.status}</span>
                `;
                
                runsList.appendChild(runItem);
            });
        }
        
        function displaySystemStatus(status) {
            const servicesStatus = document.getElementById('services-status');
            servicesStatus.innerHTML = '';
            
            Object.entries(status.services).forEach(([service, healthy]) => {
                const serviceItem = document.createElement('div');
                serviceItem.className = 'service-item';
                serviceItem.innerHTML = `
                    <span>${service.charAt(0).toUpperCase() + service.slice(1)}</span>
                    <span class="service-status ${healthy ? 'status-healthy' : 'status-unhealthy'}"></span>
                `;
                servicesStatus.appendChild(serviceItem);
            });
        }
        
        async function loadLogs(runId) {
            currentRunId = runId;
            try {
                const response = await fetch(`/api/logs/${runId}`);
                const logs = await response.json();
                displayLogs(logs);
            } catch (error) {
                console.error('Error loading logs:', error);
            }
        }
        
        function displayLogs(logs) {
            const logsContainer = document.getElementById('logs-container');
            logsContainer.innerHTML = '';
            
            logs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                
                const timestamp = new Date(log.timestamp).toLocaleTimeString();
                logEntry.innerHTML = `
                    <span class="log-timestamp">[${timestamp}]</span>
                    <span class="log-level-${log.level}">[${log.level}]</span>
                    ${log.message}
                `;
                
                logsContainer.appendChild(logEntry);
            });
            
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
        
        // Initial load
        loadData();
    </script>
</body>
</html>
        '''

# Global dashboard instance
dashboard_db = DashboardDB()

def create_server_class(db: DashboardDB):
    """Create server class with database dependency injection"""
    class CustomHandler(DashboardServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, db=db, **kwargs)
    return CustomHandler

@click.command()
@click.option('--port', default=8081, help='Port to run dashboard on')
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--auto-open', is_flag=True, help='Automatically open browser')
def start_dashboard(port: int, host: str, auto_open: bool):
    """Start the web dashboard for pipeline monitoring"""
    
    print(f"🚀 Starting BreadthFlow Dashboard...")
    print(f"📊 Dashboard URL: http://{host}:{port}")
    print(f"🔄 Auto-refresh: Every 30 seconds")
    print(f"💾 Database: {dashboard_db.db_path}")
    
    server_class = create_server_class(dashboard_db)
    httpd = HTTPServer((host, port), server_class)
    
    if auto_open:
        def open_browser():
            time.sleep(1)  # Wait for server to start
            webbrowser.open(f'http://{host}:{port}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    print(f"✅ Dashboard ready at http://{host}:{port}")
    print("💡 Available endpoints:")
    print(f"   • Main Dashboard: http://{host}:{port}/dashboard")
    print(f"   • API Stats: http://{host}:{port}/api/stats")
    print(f"   • API Status: http://{host}:{port}/api/status")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        httpd.shutdown()

if __name__ == '__main__':
    start_dashboard()
