#!/usr/bin/env python3
"""
BreadthFlow Comprehensive Web Dashboard

A modern web interface providing complete visibility into:
- Pipeline run status and performance
- Symbol processing analytics  
- Financial data insights
- System health monitoring
- Historical trends and patterns

Features:
- Real-time updates
- Interactive charts and tables
- Export capabilities
- Mobile responsive design
"""

import click
import json
import sqlite3
import pandas as pd
import requests
import threading
import webbrowser
import os
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any, Optional
import urllib.parse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - use environment variable or default
DB_PATH = os.getenv('DB_PATH', '/app/data/pipeline.db')

def get_db_connection():
    """Get SQLite database connection with proper error handling"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return sqlite3.connect(DB_PATH)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        # Fallback to current directory
        return sqlite3.connect('dashboard.db')

class PipelineDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the comprehensive pipeline dashboard"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/dashboard':
            self.serve_dashboard()
        elif path == '/api/summary':
            self.serve_summary_api()
        elif path == '/api/runs':
            self.serve_runs_api()
        elif path == '/api/symbols':
            self.serve_symbols_api()
        elif path == '/api/performance':
            self.serve_performance_api()
        elif path == '/api/storage':
            self.serve_storage_api()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML"""
        html = self.get_dashboard_html()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_summary_api(self):
        """Serve overall system summary"""
        try:
            data = self.get_system_summary()
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({"error": str(e)})
    
    def serve_runs_api(self):
        """Serve recent pipeline runs"""
        try:
            data = self.get_recent_runs()
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({"error": str(e)})
    
    def serve_symbols_api(self):
        """Serve symbol processing statistics"""
        try:
            data = self.get_symbol_stats()
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({"error": str(e)})
    
    def serve_performance_api(self):
        """Serve performance metrics"""
        try:
            data = self.get_performance_metrics()
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({"error": str(e)})
    
    def serve_storage_api(self):
        """Serve storage statistics"""
        try:
            data = self.get_storage_stats()
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({"error": str(e)}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall system summary"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Total runs
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs')
            total_runs = cursor.fetchone()[0]
            
            # Success metrics
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE status = "completed"')
            successful_runs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE status = "failed"')
            failed_runs = cursor.fetchone()[0]
            
            # Recent activity
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute('SELECT COUNT(*) FROM pipeline_runs WHERE start_time > ?', (yesterday,))
            recent_runs = cursor.fetchone()[0]
            
            # Performance metrics
            cursor.execute('SELECT AVG(duration), MAX(duration), MIN(duration) FROM pipeline_runs WHERE duration IS NOT NULL')
            avg_duration, max_duration, min_duration = cursor.fetchone()
            
            # Symbol processing
            cursor.execute('''
                SELECT SUM(CAST(json_extract(metadata, "$.symbols_count") AS INTEGER))
                FROM pipeline_runs 
                WHERE metadata IS NOT NULL AND json_extract(metadata, "$.symbols_count") IS NOT NULL
            ''')
            total_symbols = cursor.fetchone()[0] or 0
            
            conn.close()
            
            success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
            
            return {
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': round(success_rate, 1),
                'recent_runs': recent_runs,
                'avg_duration': round(avg_duration or 0, 2),
                'max_duration': round(max_duration or 0, 2),
                'min_duration': round(min_duration or 0, 2),
                'total_symbols_processed': total_symbols,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system summary: {e}")
            return {'error': str(e)}
    
    def get_recent_runs(self) -> List[Dict[str, Any]]:
        """Get recent pipeline runs with details"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT run_id, command, status, start_time, end_time, duration, metadata
                FROM pipeline_runs
                ORDER BY start_time DESC
                LIMIT 50
            ''')
            
            runs = []
            for row in cursor.fetchall():
                run_id, command, status, start_time, end_time, duration, metadata_json = row
                
                metadata = {}
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except:
                        pass
                
                runs.append({
                    'run_id': run_id,
                    'command': command,
                    'status': status,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'symbols_count': metadata.get('symbols_count', 0),
                    'successful_symbols': metadata.get('successful_symbols', 0),
                    'failed_symbols': metadata.get('failed_symbols', 0),
                    'progress': metadata.get('progress', 0),
                    'metadata': metadata
                })
            
            conn.close()
            return runs
            
        except Exception as e:
            logger.error(f"Error getting recent runs: {e}")
            return []
    
    def get_symbol_stats(self) -> Dict[str, Any]:
        """Get symbol processing statistics"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get all runs with symbol data
            cursor.execute('''
                SELECT metadata FROM pipeline_runs 
                WHERE metadata IS NOT NULL 
                AND json_extract(metadata, "$.symbols") IS NOT NULL
            ''')
            
            symbol_counts = {}
            symbol_success = {}
            
            for (metadata_json,) in cursor.fetchall():
                try:
                    metadata = json.loads(metadata_json)
                    symbols = metadata.get('symbols', [])
                    successful = metadata.get('successful_symbols', 0)
                    failed = metadata.get('failed_symbols', 0)
                    
                    if isinstance(symbols, list):
                        for symbol in symbols:
                            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                            
                            if symbol not in symbol_success:
                                symbol_success[symbol] = {'successful': 0, 'failed': 0}
                            
                            # This is simplified - in real implementation you'd track per-symbol success
                            if successful > failed:
                                symbol_success[symbol]['successful'] += 1
                            else:
                                symbol_success[symbol]['failed'] += 1
                                
                except:
                    continue
            
            # Top symbols by processing count
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Symbol success rates
            symbol_stats = []
            for symbol, count in top_symbols:
                success_data = symbol_success.get(symbol, {'successful': 0, 'failed': 0})
                total = success_data['successful'] + success_data['failed']
                success_rate = (success_data['successful'] / total * 100) if total > 0 else 0
                
                symbol_stats.append({
                    'symbol': symbol,
                    'total_processed': count,
                    'success_rate': round(success_rate, 1),
                    'successful': success_data['successful'],
                    'failed': success_data['failed']
                })
            
            conn.close()
            
            return {
                'total_unique_symbols': len(symbol_counts),
                'most_processed': symbol_stats,
                'symbol_counts': dict(top_symbols)
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol stats: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance trends and metrics"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Daily performance trends (last 30 days)
            thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('''
                SELECT DATE(start_time) as date, 
                       COUNT(*) as runs,
                       AVG(duration) as avg_duration,
                       SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) as successful
                FROM pipeline_runs
                WHERE start_time > ?
                GROUP BY DATE(start_time)
                ORDER BY date
            ''', (thirty_days_ago,))
            
            daily_trends = []
            for date, runs, avg_duration, successful in cursor.fetchall():
                success_rate = (successful / runs * 100) if runs > 0 else 0
                daily_trends.append({
                    'date': date,
                    'runs': runs,
                    'avg_duration': round(avg_duration or 0, 2),
                    'success_rate': round(success_rate, 1)
                })
            
            # Command performance
            cursor.execute('''
                SELECT command, 
                       COUNT(*) as total_runs,
                       AVG(duration) as avg_duration,
                       SUM(CASE WHEN status = "completed" THEN 1 ELSE 0 END) as successful
                FROM pipeline_runs
                WHERE start_time > ?
                GROUP BY command
                ORDER BY total_runs DESC
            ''', (thirty_days_ago,))
            
            command_stats = []
            for command, total, avg_duration, successful in cursor.fetchall():
                success_rate = (successful / total * 100) if total > 0 else 0
                command_stats.append({
                    'command': command,
                    'total_runs': total,
                    'avg_duration': round(avg_duration or 0, 2),
                    'success_rate': round(success_rate, 1)
                })
            
            conn.close()
            
            return {
                'daily_trends': daily_trends,
                'command_performance': command_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from MinIO"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                endpoint_url='http://minio:9000',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
                region_name='us-east-1'
            )
            
            # Get OHLCV data
            response = s3_client.list_objects_v2(Bucket='breadthflow', Prefix='ohlcv/')
            
            storage_by_symbol = {}
            total_size = 0
            total_files = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    modified = obj['LastModified']
                    
                    path_parts = key.split('/')
                    if len(path_parts) >= 2 and path_parts[1]:
                        symbol = path_parts[1]
                        if symbol not in storage_by_symbol:
                            storage_by_symbol[symbol] = {
                                'files': 0,
                                'size_mb': 0,
                                'latest_update': modified.isoformat()
                            }
                        
                        storage_by_symbol[symbol]['files'] += 1
                        storage_by_symbol[symbol]['size_mb'] += size / (1024*1024)
                        if modified.isoformat() > storage_by_symbol[symbol]['latest_update']:
                            storage_by_symbol[symbol]['latest_update'] = modified.isoformat()
                    
                    total_size += size
                    total_files += 1
            
            # Top symbols by storage size
            top_by_size = sorted(
                [(symbol, data['size_mb']) for symbol, data in storage_by_symbol.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            return {
                'total_symbols': len(storage_by_symbol),
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024*1024), 2),
                'storage_by_symbol': storage_by_symbol,
                'top_by_size': [{'symbol': symbol, 'size_mb': round(size, 2)} for symbol, size in top_by_size]
            }
            
        except Exception as e:
            logger.warning(f"Could not get MinIO stats: {e}")
            return {'error': 'MinIO not accessible', 'details': str(e)}
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())
    
    def get_dashboard_html(self) -> str:
        """Generate comprehensive dashboard HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Comprehensive Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .header h1 {
            font-size: 2.8em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .subtitle { color: #666; font-size: 1.2em; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            grid-template-columns: 1fr 1fr;
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
        .full-width { grid-column: 1 / -1; }
        .table-container {
            max-height: 400px;
            overflow-y: auto;
            border-radius: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .status-completed { color: #28a745; font-weight: bold; }
        .status-failed { color: #dc3545; font-weight: bold; }
        .status-running { color: #ffc107; font-weight: bold; }
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            margin-bottom: 20px;
            transition: transform 0.3s ease;
        }
        .refresh-btn:hover { transform: scale(1.05); }
        .loading { text-align: center; color: #666; font-style: italic; }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 BreadthFlow Comprehensive Dashboard</h1>
            <p class="subtitle">Complete Pipeline Visibility & Analytics</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-runs">-</div>
                <div class="stat-label">Total Pipeline Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="success-rate">-</div>
                <div class="stat-label">Overall Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="symbols-processed">-</div>
                <div class="stat-label">Symbols Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-duration">-</div>
                <div class="stat-label">Avg Duration (s)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="recent-activity">-</div>
                <div class="stat-label">Last 24h Activity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="data-storage">-</div>
                <div class="stat-label">Data Storage (MB)</div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="panel">
                <h2>📊 Recent Pipeline Runs</h2>
                <button class="refresh-btn" onclick="loadAllData()">🔄 Refresh Data</button>
                <div class="table-container">
                    <table id="runs-table">
                        <thead>
                            <tr>
                                <th>Command</th>
                                <th>Status</th>
                                <th>Duration</th>
                                <th>Symbols</th>
                                <th>Success%</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody id="runs-tbody">
                            <tr><td colspan="6" class="loading">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="panel">
                <h2>📈 Top Processed Symbols</h2>
                <div class="table-container">
                    <table id="symbols-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Times Processed</th>
                                <th>Success Rate</th>
                                <th>Storage (MB)</th>
                            </tr>
                        </thead>
                        <tbody id="symbols-tbody">
                            <tr><td colspan="4" class="loading">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="panel full-width">
                <h2>📈 Performance Trends (Last 30 Days)</h2>
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
            </div>
            
            <div class="panel">
                <h2>⚡ Command Performance</h2>
                <div class="table-container">
                    <table id="commands-table">
                        <thead>
                            <tr>
                                <th>Command</th>
                                <th>Runs</th>
                                <th>Avg Duration</th>
                                <th>Success Rate</th>
                            </tr>
                        </thead>
                        <tbody id="commands-tbody">
                            <tr><td colspan="4" class="loading">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="panel">
                <h2>💾 Storage Analytics</h2>
                <div id="storage-stats">
                    <div class="loading">Loading storage data...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let performanceChart = null;
        
        async function loadAllData() {
            try {
                await Promise.all([
                    loadSummary(),
                    loadRuns(),
                    loadSymbols(),
                    loadPerformance(),
                    loadStorage()
                ]);
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        async function loadSummary() {
            try {
                const response = await fetch('/api/summary');
                const data = await response.json();
                
                document.getElementById('total-runs').textContent = data.total_runs || 0;
                document.getElementById('success-rate').textContent = (data.success_rate || 0) + '%';
                document.getElementById('symbols-processed').textContent = data.total_symbols_processed || 0;
                document.getElementById('avg-duration').textContent = (data.avg_duration || 0).toFixed(1);
                document.getElementById('recent-activity').textContent = data.recent_runs || 0;
                
            } catch (error) {
                console.error('Error loading summary:', error);
            }
        }
        
        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const runs = await response.json();
                
                const tbody = document.getElementById('runs-tbody');
                tbody.innerHTML = '';
                
                runs.slice(0, 20).forEach(run => {
                    const row = document.createElement('tr');
                    const startTime = new Date(run.start_time).toLocaleString();
                    const command = run.command.length > 25 ? run.command.substring(0, 25) + '...' : run.command;
                    
                    row.innerHTML = `
                        <td>${command}</td>
                        <td><span class="status-${run.status}">${run.status}</span></td>
                        <td>${(run.duration || 0).toFixed(1)}s</td>
                        <td>${run.symbols_count || 0}</td>
                        <td>${run.symbols_count > 0 ? ((run.successful_symbols || 0) / run.symbols_count * 100).toFixed(1) : 0}%</td>
                        <td>${startTime}</td>
                    `;
                    tbody.appendChild(row);
                });
                
            } catch (error) {
                console.error('Error loading runs:', error);
            }
        }
        
        async function loadSymbols() {
            try {
                const [symbolsResponse, storageResponse] = await Promise.all([
                    fetch('/api/symbols'),
                    fetch('/api/storage')
                ]);
                
                const symbolsData = await symbolsResponse.json();
                const storageData = await storageResponse.json();
                
                const tbody = document.getElementById('symbols-tbody');
                tbody.innerHTML = '';
                
                if (symbolsData.most_processed) {
                    symbolsData.most_processed.slice(0, 15).forEach(symbol => {
                        const row = document.createElement('tr');
                        const storageInfo = storageData.storage_by_symbol?.[symbol.symbol];
                        const storageMB = storageInfo ? storageInfo.size_mb.toFixed(2) : '0';
                        
                        row.innerHTML = `
                            <td><strong>${symbol.symbol}</strong></td>
                            <td>${symbol.total_processed}</td>
                            <td>${symbol.success_rate}%</td>
                            <td>${storageMB}</td>
                        `;
                        tbody.appendChild(row);
                    });
                }
                
            } catch (error) {
                console.error('Error loading symbols:', error);
            }
        }
        
        async function loadPerformance() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                
                // Update commands table
                const tbody = document.getElementById('commands-tbody');
                tbody.innerHTML = '';
                
                if (data.command_performance) {
                    data.command_performance.forEach(cmd => {
                        const row = document.createElement('tr');
                        const command = cmd.command.length > 20 ? cmd.command.substring(0, 20) + '...' : cmd.command;
                        
                        row.innerHTML = `
                            <td>${command}</td>
                            <td>${cmd.total_runs}</td>
                            <td>${cmd.avg_duration.toFixed(1)}s</td>
                            <td>${cmd.success_rate}%</td>
                        `;
                        tbody.appendChild(row);
                    });
                }
                
                // Create performance chart
                if (data.daily_trends && data.daily_trends.length > 0) {
                    createPerformanceChart(data.daily_trends);
                }
                
            } catch (error) {
                console.error('Error loading performance:', error);
            }
        }
        
        async function loadStorage() {
            try {
                const response = await fetch('/api/storage');
                const data = await response.json();
                
                const container = document.getElementById('storage-stats');
                
                if (data.error) {
                    container.innerHTML = `<div style="color: #dc3545;">⚠️ ${data.error}</div>`;
                    document.getElementById('data-storage').textContent = 'N/A';
                    return;
                }
                
                document.getElementById('data-storage').textContent = data.total_size_mb || 0;
                
                container.innerHTML = `
                    <div style="margin-bottom: 15px;">
                        <strong>📊 Storage Overview:</strong><br>
                        • ${data.total_symbols || 0} symbols<br>
                        • ${data.total_files || 0} files<br>
                        • ${(data.total_size_mb || 0).toFixed(2)} MB total
                    </div>
                    <div>
                        <strong>🏆 Top by Storage:</strong><br>
                        ${(data.top_by_size || []).slice(0, 5).map(item => 
                            `• ${item.symbol}: ${item.size_mb} MB`
                        ).join('<br>')}
                    </div>
                `;
                
            } catch (error) {
                console.error('Error loading storage:', error);
                document.getElementById('data-storage').textContent = 'Error';
            }
        }
        
        function createPerformanceChart(trends) {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: trends.map(t => t.date),
                    datasets: [
                        {
                            label: 'Success Rate (%)',
                            data: trends.map(t => t.success_rate),
                            borderColor: '#28a745',
                            backgroundColor: 'rgba(40, 167, 69, 0.1)',
                            yAxisID: 'y'
                        },
                        {
                            label: 'Avg Duration (s)',
                            data: trends.map(t => t.avg_duration),
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    scales: {
                        x: {
                            display: true,
                            title: { display: true, text: 'Date' }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Success Rate (%)' }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Duration (s)' },
                            grid: { drawOnChartArea: false }
                        }
                    }
                }
            });
        }
        
        // Auto-refresh every 60 seconds
        setInterval(loadAllData, 60000);
        
        // Initial load
        loadAllData();
    </script>
</body>
</html>
        '''

@click.command()
@click.option('--port', default=8083, help='Port to run dashboard on')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--auto-open', is_flag=True, help='Automatically open browser')
def start_dashboard(port: int, host: str, auto_open: bool):
    """Start the comprehensive web dashboard for complete pipeline visibility"""
    
    print(f"🚀 Starting BreadthFlow Comprehensive Dashboard...")
    print(f"📊 Dashboard URL: http://localhost:{port}")
    print(f"🔄 Auto-refresh: Every 60 seconds")
    print(f"💾 Data sources: SQLite + Elasticsearch + MinIO")
    
    httpd = HTTPServer((host, port), PipelineDashboardHandler)
    
    if auto_open:
        def open_browser():
            import time
            time.sleep(2)
            webbrowser.open(f'http://localhost:{port}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    print(f"✅ Dashboard ready at http://localhost:{port}")
    print("💡 Features:")
    print("   • Complete pipeline run history and status")
    print("   • Symbol processing analytics and success rates")
    print("   • Performance trends and duration tracking")
    print("   • Storage analytics and data insights")
    print("   • Real-time updates and interactive charts")
    print("   • Mobile responsive design")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down dashboard...")
        httpd.shutdown()

if __name__ == '__main__':
    start_dashboard()
