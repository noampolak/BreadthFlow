#!/usr/bin/env python3
"""
BreadthFlow PostgreSQL Dashboard
A web dashboard using PostgreSQL as the backend database
"""

import click
import json
import os
import requests
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Any, Optional
import urllib.parse
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://pipeline:pipeline123@postgres:5432/breadthflow')

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        engine = create_engine(DATABASE_URL)
        return engine.connect()
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables if they don't exist"""
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Create pipeline_runs table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id VARCHAR(255) PRIMARY KEY,
                    command TEXT,
                    status VARCHAR(50),
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration FLOAT,
                    error_message TEXT,
                    metadata JSONB
                )
            """))
            conn.commit()
            print("✅ Database tables initialized")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/dashboard':
            self.serve_dashboard()
        elif self.path == '/infrastructure':
            self.serve_infrastructure()
        elif self.path == '/trading':
            self.serve_trading()
        elif self.path == '/commands':
            self.serve_commands()
        elif self.path == '/api/summary':
            self.serve_summary()
        elif self.path == '/api/runs':
            self.serve_runs()
        elif self.path.startswith('/api/run/'):
            run_id = self.path.split('/')[-1]
            self.serve_run_details(run_id)
        elif self.path == '/api/signals/latest':
            self.serve_latest_signals()
        elif self.path == '/api/signals/export':
            self.serve_signals_export()
        else:
            self.send_error(404)
    
    def do_POST(self):
        if self.path == '/api/execute-command':
            self.serve_execute_command()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Dashboard</title>
    <style>
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .stats { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .stat-card { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px; 
            border-radius: 15px; 
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
        .stat-label { color: #666; margin-top: 5px; }
        .nav-buttons {
            margin-top: 20px;
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }
        .nav-btn {
            background: rgba(255, 255, 255, 0.8);
            color: #333;
            border: 2px solid transparent;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .nav-btn:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
        }
        .nav-btn.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.3s ease;
        }
                            .refresh-btn:hover { transform: scale(1.05); }
                    
                    .details-btn {
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        color: white;
                        border: none;
                        padding: 5px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 0.8em;
                        transition: all 0.3s ease;
                    }
                    .details-btn:hover { transform: scale(1.05); background: linear-gradient(135deg, #764ba2, #667eea); }
                    
                    .export-btn {
                        background: linear-gradient(135deg, #28a745, #20c997);
                        color: white;
                        border: none;
                        padding: 5px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 0.8em;
                        margin-left: 5px;
                        transition: all 0.3s ease;
                    }
                    .export-btn:hover { transform: scale(1.05); }
                    
                    .modal {
                        display: none;
                        position: fixed;
                        z-index: 1000;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0,0,0,0.5);
                    }
                    
                    .modal-content {
                        background: rgba(255, 255, 255, 0.95);
                        backdrop-filter: blur(10px);
                        margin: 5% auto;
                        padding: 20px;
                        border-radius: 15px;
                        width: 90%;
                        max-width: 800px;
                        max-height: 80vh;
                        overflow-y: auto;
                        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                    }
                    
                    .close {
                        color: #aaa;
                        float: right;
                        font-size: 28px;
                        font-weight: bold;
                        cursor: pointer;
                    }
                    .close:hover { color: #000; }
        .runs-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .runs-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .runs-table th, .runs-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        .runs-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .status-completed { color: #28a745; font-weight: bold; }
        .status-failed { color: #dc3545; font-weight: bold; }
        .status-running { color: #ffc107; font-weight: bold; }
        .last-updated { color: #666; font-size: 0.9em; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BreadthFlow Pipeline Dashboard</h1>
            <p>Real-time pipeline monitoring with PostgreSQL backend</p>
                                                    <div class="nav-buttons">
                    <button class="nav-btn active" onclick="window.location.href='/'">Dashboard</button>
                    <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                    <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                    <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                    <button class="refresh-btn" onclick="loadData()">Refresh Now</button>
                </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="total-runs">-</div>
                <div class="stat-label">Total Pipeline Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="success-rate">-</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="recent-runs">-</div>
                <div class="stat-label">Last 24h Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-duration">-</div>
                <div class="stat-label">Avg Duration (s)</div>
            </div>
        </div>
        
        <div class="runs-section">
            <h2>Recent Pipeline Runs</h2>
            <table class="runs-table">
                <thead>
                    <tr>
                        <th>Command</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Start Time</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="runs-tbody">
                    <tr><td colspan="5">Loading...</td></tr>
                </tbody>
            </table>
            <div class="last-updated" id="last-updated">Last updated: Never</div>
        </div>
    </div>
    
    <!-- Run Details Modal -->
    <div id="runModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalContent">Loading...</div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                // Load summary data
                const summaryResponse = await fetch('/api/summary');
                const summary = await summaryResponse.json();
                
                if (summary.error) {
                    console.error('Summary error:', summary.error);
                    document.getElementById('total-runs').textContent = 'Error';
                    return;
                }
                
                document.getElementById('total-runs').textContent = summary.total_runs || 0;
                document.getElementById('success-rate').textContent = (summary.success_rate || 0) + '%';
                document.getElementById('recent-runs').textContent = summary.recent_runs || 0;
                document.getElementById('avg-duration').textContent = (summary.avg_duration || 0).toFixed(1);
                
                // Load runs data
                const runsResponse = await fetch('/api/runs');
                const runs = await runsResponse.json();
                
                const tbody = document.getElementById('runs-tbody');
                tbody.innerHTML = '';
                
                if (runs.error) {
                    tbody.innerHTML = '<tr><td colspan="5">Error loading runs: ' + runs.error + '</td></tr>';
                    return;
                }
                
                if (!runs.length) {
                    tbody.innerHTML = '<tr><td colspan="5">No pipeline runs found. Run a demo to see data!</td></tr>';
                    return;
                }
                
                runs.slice(0, 10).forEach(run => {
                    const row = document.createElement('tr');
                    const startTime = new Date(run.start_time).toLocaleString();
                    const command = run.command.length > 30 ? run.command.substring(0, 30) + '...' : run.command;
                    const runId = run.run_id.substring(0, 8);
                    
                    row.innerHTML = `
                        <td>${command}</td>
                        <td><span class="status-${run.status}">${run.status}</span></td>
                        <td>${(run.duration || 0).toFixed(1)}s</td>
                        <td>${startTime}</td>
                        <td>
                            <button class="details-btn" onclick="showRunDetails('${run.run_id}')">Details</button>
                            ${run.command.includes('signals') || run.command.includes('backtest') ? 
                              `<button class="export-btn" onclick="exportRunData('${run.run_id}')">Export</button>` : ''}
                        </td>
                    `;
                    tbody.appendChild(row);
                });
                
                document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleString();
                
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('total-runs').textContent = 'Error';
            }
        }
        
        // Load data on page load
        loadData();
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
        
        // Modal functions
        function showRunDetails(runId) {
            document.getElementById('runModal').style.display = 'block';
            document.getElementById('modalContent').innerHTML = 'Loading run details...';
            
            fetch(`/api/run/${runId}`)
                .then(response => response.json())
                .then(data => {
                    document.getElementById('modalContent').innerHTML = formatRunDetails(data);
                })
                .catch(error => {
                    document.getElementById('modalContent').innerHTML = `<p>Error loading details: ${error}</p>`;
                });
        }
        
        function closeModal() {
            document.getElementById('runModal').style.display = 'none';
        }
        
        function formatRunDetails(data) {
            if (data.error) {
                return `<p>Error: ${data.error}</p>`;
            }
            
            let html = `
                <h2>Run Details</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <div>
                        <h3>Basic Information</h3>
                        <p><strong>Run ID:</strong> ${data.run_id}</p>
                        <p><strong>Command:</strong> ${data.command}</p>
                        <p><strong>Status:</strong> <span class="status-${data.status}">${data.status}</span></p>
                        <p><strong>Duration:</strong> ${data.duration ? data.duration.toFixed(2) + 's' : 'N/A'}</p>
                    </div>
                    <div>
                        <h3>Timing</h3>
                        <p><strong>Started:</strong> ${new Date(data.start_time).toLocaleString()}</p>
                        <p><strong>Ended:</strong> ${data.end_time ? new Date(data.end_time).toLocaleString() : 'N/A'}</p>
                        ${data.error_message ? `<p><strong>Error:</strong> ${data.error_message}</p>` : ''}
                    </div>
                </div>
            `;
            
            // Add signal-specific information if available
            if (data.command.includes('signals') && data.signals_data) {
                html += formatSignalsData(data.signals_data);
            }
            
            // Add backtest-specific information if available
            if (data.command.includes('backtest') && data.backtest_data) {
                html += formatBacktestData(data.backtest_data);
            }
            
            return html;
        }
        
        function formatSignalsData(signals) {
            return `
                <div style="margin-top: 20px;">
                    <h3>Signal Summary</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${signals.total_signals || 0}</div>
                            <div class="stat-label">Total Signals</div>
                        </div>
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${signals.buy_signals || 0}</div>
                            <div class="stat-label">Buy Signals</div>
                        </div>
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${signals.sell_signals || 0}</div>
                            <div class="stat-label">Sell Signals</div>
                        </div>
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${signals.avg_confidence || 0}%</div>
                            <div class="stat-label">Avg Confidence</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function formatBacktestData(backtest) {
            return `
                <div style="margin-top: 20px;">
                    <h3>Backtest Results</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${(backtest.total_return * 100).toFixed(1)}%</div>
                            <div class="stat-label">Total Return</div>
                        </div>
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${backtest.sharpe_ratio || 0}</div>
                            <div class="stat-label">Sharpe Ratio</div>
                        </div>
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${(backtest.max_drawdown * 100).toFixed(1)}%</div>
                            <div class="stat-label">Max Drawdown</div>
                        </div>
                        <div class="stat-card" style="margin: 0;">
                            <div class="stat-value">${backtest.total_trades || 0}</div>
                            <div class="stat-label">Total Trades</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function exportRunData(runId) {
            window.open(`/api/signals/export?run_id=${runId}`, '_blank');
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('runModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_infrastructure(self):
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Infrastructure</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .nav-buttons {
            margin-top: 20px;
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }
        .nav-btn {
            background: rgba(255, 255, 255, 0.8);
            color: #333;
            border: 2px solid transparent;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .nav-btn:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
        }
        .nav-btn.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .content-grid {
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
        .full-width { grid-column: 1 / -1; }
        .panel h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        .architecture-diagram {
            text-align: center;
            margin: 20px 0;
        }
        .node {
            stroke: #fff;
            stroke-width: 3px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .node:hover {
            stroke-width: 4px;
            r: 45;
        }
        .link {
            stroke: #999;
            stroke-width: 2px;
            fill: none;
            marker-end: url(#arrowhead);
        }
        .node-label {
            fill: white;
            font-family: 'Segoe UI', sans-serif;
            font-size: 11px;
            font-weight: bold;
            text-anchor: middle;
            dominant-baseline: middle;
            pointer-events: none;
        }
        .service-info {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        .service-info h4 {
            margin: 0 0 8px 0;
            color: #333;
        }
        .service-info p {
            margin: 0;
            color: #666;
            font-size: 0.9em;
        }
        .tech-badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin: 2px;
        }
        @media (max-width: 968px) {
            .content-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BreadthFlow Infrastructure</h1>
            <p>System Architecture &amp; Component Overview</p>
                                    <div class="nav-buttons">
                            <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                            <button class="nav-btn active" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                            <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                            <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                        </div>
        </div>
        
        <div class="content-grid">
            <div class="panel full-width">
                <h2>Project Overview</h2>
                <p><strong>BreadthFlow</strong> is a comprehensive financial data pipeline designed for real-time market analysis and backtesting. The system processes market data, generates trading signals, and provides portfolio backtesting capabilities using modern big data technologies.</p>
                
                <div style="margin: 20px 0;">
                    <h3>Key Features:</h3>
                    <ul style="list-style: none; padding: 0;">
                        <li><strong>Real-time Data Processing:</strong> Apache Spark for distributed analytics</li>
                        <li><strong>Modern Storage:</strong> MinIO (S3-compatible) + PostgreSQL</li>
                        <li><strong>Advanced Monitoring:</strong> Elasticsearch + Kibana + Custom Dashboard</li>
                        <li><strong>Streaming Analytics:</strong> Apache Kafka for real-time data streams</li>
                        <li><strong>Financial Analysis:</strong> Technical indicators, signal generation, backtesting</li>
                        <li><strong>Containerized:</strong> Full Docker-based microservices architecture</li>
                    </ul>
                </div>
            </div>
            
            <div class="panel">
                <h2>Architecture Diagram</h2>
                <div class="architecture-diagram">
                    <svg width="100%" height="450" id="architecture-svg" style="border: 1px solid #eee; border-radius: 8px; background: white;"></svg>
                </div>
            </div>
            
            <div class="panel">
                <h2>Technologies Used</h2>
                <div style="margin: 15px 0;">
                    <div class="tech-badge">Apache Spark</div>
                    <div class="tech-badge">PostgreSQL</div>
                    <div class="tech-badge">MinIO S3</div>
                    <div class="tech-badge">Apache Kafka</div>
                    <div class="tech-badge">Elasticsearch</div>
                    <div class="tech-badge">Kibana</div>
                    <div class="tech-badge">Docker</div>
                    <div class="tech-badge">Python</div>
                    <div class="tech-badge">PySpark</div>
                    <div class="tech-badge">Flask</div>
                    <div class="tech-badge">yfinance</div>
                    <div class="tech-badge">boto3</div>
                </div>
                
                <div class="service-info">
                    <h4>Use Cases</h4>
                    <p>Algorithmic trading strategy development<br>
                    Market data analysis and research<br>
                    Portfolio performance backtesting<br>
                    Real-time trading signal generation<br>
                    Risk management and monitoring</p>
                </div>
            </div>
            
            <div class="panel full-width">
                <h2>Service Components</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <div class="service-info">
                        <h4>Spark Master + Workers</h4>
                        <p><strong>Purpose:</strong> Distributed data processing and analytics<br>
                        <strong>Ports:</strong> 8080 (UI), 7077 (Master), 18080 (History)<br>
                        <strong>Handles:</strong> OHLCV data processing, technical indicators, backtesting</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>PostgreSQL Database</h4>
                        <p><strong>Purpose:</strong> Pipeline metadata and run tracking<br>
                        <strong>Port:</strong> 5432<br>
                        <strong>Stores:</strong> Pipeline runs, execution logs, performance metrics</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>MinIO Object Storage</h4>
                        <p><strong>Purpose:</strong> Financial data storage (S3-compatible)<br>
                        <strong>Ports:</strong> 9000 (API), 9001 (Console)<br>
                        <strong>Stores:</strong> OHLCV data, analytics results, backtest results</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>Apache Kafka</h4>
                        <p><strong>Purpose:</strong> Real-time data streaming and replay<br>
                        <strong>Port:</strong> 9092<br>
                        <strong>Handles:</strong> Market data streams, historical data replay</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>Elasticsearch</h4>
                        <p><strong>Purpose:</strong> Log storage and search analytics<br>
                        <strong>Ports:</strong> 9200 (HTTP), 9300 (Transport)<br>
                        <strong>Indexes:</strong> Pipeline logs, performance metrics, error tracking</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>Kibana</h4>
                        <p><strong>Purpose:</strong> Data visualization and monitoring<br>
                        <strong>Port:</strong> 5601<br>
                        <strong>Provides:</strong> Log analysis, performance dashboards, alerting</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>Web Dashboard</h4>
                        <p><strong>Purpose:</strong> Real-time pipeline monitoring<br>
                        <strong>Port:</strong> 8083<br>
                        <strong>Features:</strong> Live metrics, run history, infrastructure overview</p>
                    </div>
                    
                    <div class="service-info">
                        <h4>Zookeeper</h4>
                        <p><strong>Purpose:</strong> Kafka cluster coordination<br>
                        <strong>Port:</strong> 2181<br>
                        <strong>Manages:</strong> Kafka broker discovery, topic management</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // D3.js Architecture Diagram
        const svg = d3.select("#architecture-svg");
        const width = 600;
        const height = 450;
        
        svg.attr("viewBox", `0 0 ${width} ${height}`);
        
        // Define arrow marker
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 25)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#666");
        
        // Node data with better positioning
        const nodes = [
            {id: "dashboard", name: "Web\\nDashboard", x: 300, y: 60, color: "#28a745"},
            {id: "postgres", name: "PostgreSQL", x: 150, y: 160, color: "#336791"},
            {id: "spark", name: "Spark\\nCluster", x: 300, y: 160, color: "#e25a1c"},
            {id: "minio", name: "MinIO\\nStorage", x: 450, y: 160, color: "#c72e29"},
            {id: "kafka", name: "Apache\\nKafka", x: 200, y: 280, color: "#231f20"},
            {id: "elastic", name: "Elasticsearch", x: 400, y: 280, color: "#005571"},
            {id: "kibana", name: "Kibana", x: 300, y: 380, color: "#e8478b"},
            {id: "zookeeper", name: "Zookeeper", x: 100, y: 380, color: "#d4af37"}
        ];
        
        // Link data
        const links = [
            {source: "dashboard", target: "postgres"},
            {source: "spark", target: "postgres"},
            {source: "spark", target: "minio"},
            {source: "kafka", target: "spark"},
            {source: "spark", target: "elastic"},
            {source: "elastic", target: "kibana"},
            {source: "kafka", target: "zookeeper"},
            {source: "dashboard", target: "spark"}
        ];
        
        // Create links
        svg.selectAll(".link")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("x1", d => nodes.find(n => n.id === d.source).x)
            .attr("y1", d => nodes.find(n => n.id === d.source).y)
            .attr("x2", d => nodes.find(n => n.id === d.target).x)
            .attr("y2", d => nodes.find(n => n.id === d.target).y);
        
        // Create nodes
        const nodeGroups = svg.selectAll(".node-group")
            .data(nodes)
            .enter().append("g")
            .attr("class", "node-group");
        
        nodeGroups.append("circle")
            .attr("class", "node")
            .attr("r", 40)
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("fill", d => d.color);
        
        nodeGroups.each(function(d) {
            const group = d3.select(this);
            const lines = d.name.split("\\n");
            
            lines.forEach((line, i) => {
                group.append("text")
                    .attr("class", "node-label")
                    .attr("x", d.x)
                    .attr("y", d.y + (i - (lines.length - 1) / 2) * 14)
                    .text(line);
            });
        });
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_summary(self):
        try:
            data = self.get_summary()
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def serve_runs(self):
        try:
            data = self.get_recent_runs()
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def get_summary(self):
        conn = get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}
        
        try:
            # Total runs
            result = conn.execute(text('SELECT COUNT(*) FROM pipeline_runs'))
            total_runs = result.fetchone()[0]
            
            # Success rate
            result = conn.execute(text('SELECT COUNT(*) FROM pipeline_runs WHERE status = :status'), {'status': 'completed'})
            successful = result.fetchone()[0]
            success_rate = (successful / total_runs * 100) if total_runs > 0 else 0
            
            # Recent runs (last 24h)
            yesterday = datetime.now() - timedelta(days=1)
            result = conn.execute(text('SELECT COUNT(*) FROM pipeline_runs WHERE start_time > :yesterday'), {'yesterday': yesterday})
            recent_runs = result.fetchone()[0]
            
            # Average duration
            result = conn.execute(text('SELECT AVG(duration) FROM pipeline_runs WHERE duration IS NOT NULL'))
            avg_duration = result.fetchone()[0] or 0
            
            return {
                'total_runs': total_runs,
                'success_rate': round(success_rate, 1),
                'recent_runs': recent_runs,
                'avg_duration': round(float(avg_duration), 2),
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()
    
    def get_recent_runs(self):
        conn = get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}
        
        try:
            result = conn.execute(text('''
                SELECT run_id, command, status, start_time, end_time, duration, metadata
                FROM pipeline_runs
                ORDER BY start_time DESC
                LIMIT 20
            '''))
            
            runs = []
            for row in result:
                runs.append({
                    'run_id': row[0],
                    'command': row[1],
                    'status': row[2],
                    'start_time': row[3].isoformat() if row[3] else None,
                    'end_time': row[4].isoformat() if row[4] else None,
                    'duration': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {}
                })
            
            return runs
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()
    
    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2, default=str).encode())
    
    def serve_run_details(self, run_id):
        try:
            data = self.get_run_details(run_id)
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def serve_latest_signals(self):
        try:
            data = self.get_latest_signals()
            self.send_json(data)
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def serve_signals_export(self):
        try:
            run_id = self.get_query_param('run_id')
            if run_id:
                data = self.export_run_signals(run_id)
                self.send_csv_response(data, f"signals_{run_id[:8]}.csv")
            else:
                data = self.export_latest_signals()
                self.send_csv_response(data, "latest_signals.csv")
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def serve_trading(self):
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Trading Signals</title>
    <style>
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        .nav-buttons {
            margin-top: 20px;
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }
        .nav-btn {
            background: rgba(255, 255, 255, 0.8);
            color: #333;
            border: 2px solid transparent;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .nav-btn:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
        }
        .nav-btn.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .signal-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .signal-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .signal-card.buy { border-left: 5px solid #28a745; }
        .signal-card.sell { border-left: 5px solid #dc3545; }
        .signal-card.hold { border-left: 5px solid #ffc107; }
        .export-section {
            text-align: center;
            margin-top: 30px;
        }
        .export-btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin: 0 10px;
            transition: all 0.3s ease;
        }
        .export-btn:hover { transform: scale(1.05); }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Trading Signals Dashboard</h1>
            <p>Real-time trading signals and market analysis</p>
                            <div class="nav-buttons">
                    <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                    <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                    <button class="nav-btn active" onclick="window.location.href='/trading'">Trading Signals</button>
                    <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                </div>
        </div>
        
        <div class="panel">
            <h2>Latest Trading Signals</h2>
            <div id="signals-container">Loading signals...</div>
        </div>
        
        <div class="panel">
            <h2>Signal Export</h2>
            <div class="export-section">
                <button class="export-btn" onclick="exportSignals('csv')">Export as CSV</button>
                <button class="export-btn" onclick="exportSignals('json')">Export as JSON</button>
            </div>
        </div>
    </div>
    
    <script>
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals/latest');
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('signals-container').innerHTML = `<p>Error: ${data.error}</p>`;
                    return;
                }
                
                const container = document.getElementById('signals-container');
                if (data.signals && data.signals.length > 0) {
                    container.innerHTML = data.signals.map(signal => formatSignalCard(signal)).join('');
                } else {
                    container.innerHTML = '<p>No trading signals available. Run signal generation first.</p>';
                }
            } catch (error) {
                document.getElementById('signals-container').innerHTML = `<p>Error loading signals: ${error}</p>`;
            }
        }
        
        function formatSignalCard(signal) {
            const signalClass = signal.signal_type || 'hold';
            const confidence = signal.confidence || 0;
            const strength = signal.strength || 'medium';
            
            return `
                <div class="signal-card ${signalClass}">
                    <h3>${signal.symbol || 'UNKNOWN'}</h3>
                    <p><strong>Signal:</strong> ${signal.signal_type?.toUpperCase() || 'HOLD'}</p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <p><strong>Strength:</strong> ${strength}</p>
                    <p><strong>Date:</strong> ${signal.date || 'N/A'}</p>
                </div>
            `;
        }
        
        function exportSignals(format) {
            window.open(`/api/signals/export?format=${format}`, '_blank');
        }
        
        // Load signals on page load
        loadSignals();
        
        // Auto-refresh every 60 seconds
        setInterval(loadSignals, 60000);
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def get_run_details(self, run_id):
        conn = get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}
        
        try:
            # Get basic run information
            result = conn.execute(text('''
                SELECT run_id, command, status, start_time, end_time, duration, error_message, metadata
                FROM pipeline_runs 
                WHERE run_id = :run_id
            '''), {'run_id': run_id})
            
            row = result.fetchone()
            if not row:
                return {"error": "Run not found"}
            
            run_data = {
                'run_id': row[0],
                'command': row[1],
                'status': row[2],
                'start_time': row[3],
                'end_time': row[4],
                'duration': row[5],
                'error_message': row[6],
                'metadata': row[7]
            }
            
            # Add mock signal/backtest data based on command type
            if 'signals' in run_data['command']:
                run_data['signals_data'] = {
                    'total_signals': 45,
                    'buy_signals': 18,
                    'sell_signals': 12,
                    'hold_signals': 15,
                    'avg_confidence': 73.5
                }
            
            if 'backtest' in run_data['command']:
                run_data['backtest_data'] = {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.08,
                    'total_trades': 45,
                    'win_rate': 0.65
                }
            
            return run_data
            
        except Exception as e:
            return {"error": f"Database query failed: {str(e)}"}
        finally:
            conn.close()
    
    def get_latest_signals(self):
        # Mock data for now - would integrate with MinIO signal storage
        return {
            "signals": [
                {
                    "symbol": "AAPL",
                    "signal_type": "buy",
                    "confidence": 85,
                    "strength": "strong",
                    "date": "2025-08-19"
                },
                {
                    "symbol": "MSFT", 
                    "signal_type": "buy",
                    "confidence": 78,
                    "strength": "medium",
                    "date": "2025-08-19"
                },
                {
                    "symbol": "GOOGL",
                    "signal_type": "hold",
                    "confidence": 65,
                    "strength": "weak",
                    "date": "2025-08-19"
                }
            ]
        }
    
    def export_run_signals(self, run_id):
        # Mock CSV data for signals export
        return [
            ["Symbol", "Signal", "Confidence", "Strength", "Date"],
            ["AAPL", "BUY", "85%", "Strong", "2025-08-19"],
            ["MSFT", "BUY", "78%", "Medium", "2025-08-19"],
            ["GOOGL", "HOLD", "65%", "Weak", "2025-08-19"]
        ]
    
    def export_latest_signals(self):
        # Mock CSV data for latest signals export
        return [
            ["Symbol", "Signal", "Confidence", "Strength", "Date"],
            ["AAPL", "BUY", "85%", "Strong", "2025-08-19"],
            ["MSFT", "BUY", "78%", "Medium", "2025-08-19"],
            ["GOOGL", "HOLD", "65%", "Weak", "2025-08-19"]
        ]
    
    def get_query_param(self, param_name):
        from urllib.parse import urlparse, parse_qs
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        return params.get(param_name, [None])[0]
    
    def send_csv_response(self, data, filename):
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(data)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/csv')
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(output.getvalue().encode('utf-8'))
    
    def serve_commands(self):
        """Serve the Commands page"""
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Commands</title>
    <style>
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        .nav-buttons {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .nav-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }
        .nav-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .nav-btn.active { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
        .content-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
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
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        .command-section {
            margin-bottom: 25px;
        }
        .command-section h3 {
            color: #555;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .command-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .command-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        .command-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .command-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .command-description {
            color: #666;
            margin-bottom: 15px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .command-params {
            margin-bottom: 15px;
        }
        .param-group {
            margin-bottom: 10px;
        }
        .param-label {
            display: block;
            font-weight: 500;
            color: #555;
            margin-bottom: 5px;
            font-size: 0.9em;
        }
        .param-input {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 0.9em;
            box-sizing: border-box;
        }
        .param-select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 0.9em;
            box-sizing: border-box;
        }
        .execute-btn {
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            width: 100%;
            transition: all 0.3s ease;
        }
        .execute-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }
        .execute-btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }
        .flow-section {
            background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
        }
        .flow-section h2 {
            color: white;
            margin-top: 0;
        }
        .flow-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .flow-option {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .flow-option:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        .flow-option.selected {
            background: rgba(255, 255, 255, 0.3);
            border: 2px solid rgba(255, 255, 255, 0.5);
        }
        .flow-option h3 {
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }
        .flow-option p {
            margin: 0;
            opacity: 0.9;
            font-size: 0.9em;
        }
        .status-area {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            min-height: 100px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            overflow-y: auto;
            max-height: 300px;
        }
        .status-success { color: #28a745; }
        .status-error { color: #dc3545; }
        .status-info { color: #007bff; }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BreadthFlow Commands</h1>
            <p>Execute pipeline commands directly from the web interface</p>
            <div class="nav-buttons">
                <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                <button class="nav-btn active" onclick="window.location.href='/commands'">Commands</button>
            </div>
        </div>
        
        <div class="flow-section">
            <h2>🚀 Quick Flows</h2>
            <p>Execute complete pipeline flows with predefined configurations</p>
            <div class="flow-options">
                <div class="flow-option" onclick="selectFlow('demo')">
                    <h3>🎬 Demo Flow</h3>
                    <p>Quick demo with 2-3 symbols</p>
                </div>
                <div class="flow-option" onclick="selectFlow('small')">
                    <h3>📊 Small Flow</h3>
                    <p>5-10 symbols for testing</p>
                </div>
                <div class="flow-option" onclick="selectFlow('medium')">
                    <h3>📈 Medium Flow</h3>
                    <p>10-25 symbols for analysis</p>
                </div>
                <div class="flow-option" onclick="selectFlow('full')">
                    <h3>🏭 Full Flow</h3>
                    <p>All symbols for production</p>
                </div>
            </div>
        </div>
        
        <div class="content-grid">
            <div class="panel">
                <h2>📊 Data Commands</h2>
                
                <div class="command-section">
                    <h3>Data Fetching</h3>
                    <div class="command-grid">
                        <div class="command-card">
                            <div class="command-title">Data Summary</div>
                            <div class="command-description">Show current data status and statistics</div>
                            <button class="execute-btn" onclick="executeCommand('data_summary')">Execute</button>
                        </div>
                        
                        <div class="command-card">
                            <div class="command-title">Fetch Market Data</div>
                            <div class="command-description">Fetch real market data from Yahoo Finance</div>
                            <div class="command-params">
                                <div class="param-group">
                                    <label class="param-label">Symbols:</label>
                                    <input type="text" class="param-input" id="fetch_symbols" placeholder="AAPL,MSFT,GOOGL" value="AAPL,MSFT">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">Start Date:</label>
                                    <input type="date" class="param-input" id="fetch_start_date" value="2024-08-15">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">End Date:</label>
                                    <input type="date" class="param-input" id="fetch_end_date" value="2024-08-16">
                                </div>
                            </div>
                            <button class="execute-btn" onclick="executeCommand('data_fetch')">Execute</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>🎯 Signal Commands</h2>
                
                <div class="command-section">
                    <h3>Signal Generation</h3>
                    <div class="command-grid">
                        <div class="command-card">
                            <div class="command-title">Generate Signals</div>
                            <div class="command-description">Generate trading signals using technical analysis</div>
                            <div class="command-params">
                                <div class="param-group">
                                    <label class="param-label">Symbols:</label>
                                    <input type="text" class="param-input" id="signal_symbols" placeholder="AAPL,MSFT,GOOGL" value="AAPL,MSFT">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">Start Date:</label>
                                    <input type="date" class="param-input" id="signal_start_date" value="2024-08-15">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">End Date:</label>
                                    <input type="date" class="param-input" id="signal_end_date" value="2024-08-16">
                                </div>
                            </div>
                            <button class="execute-btn" onclick="executeCommand('signal_generate')">Execute</button>
                        </div>
                        
                        <div class="command-card">
                            <div class="command-title">Signal Summary</div>
                            <div class="command-description">Show summary of generated signals</div>
                            <button class="execute-btn" onclick="executeCommand('signal_summary')">Execute</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>🔄 Backtesting Commands</h2>
                
                <div class="command-section">
                    <h3>Backtesting</h3>
                    <div class="command-grid">
                        <div class="command-card">
                            <div class="command-title">Run Backtest</div>
                            <div class="command-description">Run backtesting simulation with historical data</div>
                            <div class="command-params">
                                <div class="param-group">
                                    <label class="param-label">Symbols:</label>
                                    <input type="text" class="param-input" id="backtest_symbols" placeholder="AAPL,MSFT,GOOGL" value="AAPL,MSFT">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">From Date:</label>
                                    <input type="date" class="param-input" id="backtest_from_date" value="2024-08-15">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">To Date:</label>
                                    <input type="date" class="param-input" id="backtest_to_date" value="2024-08-16">
                                </div>
                                <div class="param-group">
                                    <label class="param-label">Initial Capital ($):</label>
                                    <input type="number" class="param-input" id="backtest_capital" value="100000">
                                </div>
                            </div>
                            <button class="execute-btn" onclick="executeCommand('backtest_run')">Execute</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="panel">
                <h2>🎨 Kafka Commands</h2>
                
                <div class="command-section">
                    <h3>Streaming & Kafka</h3>
                    <div class="command-grid">
                        <div class="command-card">
                            <div class="command-title">Kafka Demo</div>
                            <div class="command-description">Demonstrate Kafka streaming capabilities</div>
                            <button class="execute-btn" onclick="executeCommand('kafka_demo')">Execute</button>
                        </div>
                        
                        <div class="command-card">
                            <div class="command-title">Real Kafka Test</div>
                            <div class="command-description">Test real Kafka integration with Spark</div>
                            <button class="execute-btn" onclick="executeCommand('kafka_real_test')">Execute</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>📋 Command Status</h2>
            <div id="status-area" class="status-area">
                Ready to execute commands. Select a command above to get started.
            </div>
        </div>
    </div>
    
    <script>
        let selectedFlow = null;
        
        function selectFlow(flow) {
            selectedFlow = flow;
            
            // Update UI
            document.querySelectorAll('.flow-option').forEach(opt => opt.classList.remove('selected'));
            event.target.closest('.flow-option').classList.add('selected');
            
            // Update status
            updateStatus('info', `Selected flow: ${flow.toUpperCase()}`);
            
            // Auto-fill parameters based on flow
            updateParametersForFlow(flow);
        }
        
        function updateParametersForFlow(flow) {
            const flows = {
                'demo': { symbols: 'AAPL,MSFT', startDate: '2024-08-15', endDate: '2024-08-16' },
                'small': { symbols: 'AAPL,MSFT,GOOGL,NVDA,TSLA', startDate: '2024-08-15', endDate: '2024-08-16' },
                'medium': { symbols: 'AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN,META,BRK-B,JPM,V', startDate: '2024-08-15', endDate: '2024-08-16' },
                'full': { symbols: 'AAPL,MSFT,GOOGL,NVDA,TSLA,AMZN,META,BRK-B,JPM,V,UNH,PG,JNJ,HD,MA', startDate: '2024-08-15', endDate: '2024-08-16' }
            };
            
            const config = flows[flow];
            if (config) {
                // Update all symbol inputs
                document.getElementById('fetch_symbols').value = config.symbols;
                document.getElementById('signal_symbols').value = config.symbols;
                document.getElementById('backtest_symbols').value = config.symbols;
                
                // Update all date inputs
                document.getElementById('fetch_start_date').value = config.startDate;
                document.getElementById('fetch_end_date').value = config.endDate;
                document.getElementById('signal_start_date').value = config.startDate;
                document.getElementById('signal_end_date').value = config.endDate;
                document.getElementById('backtest_from_date').value = config.startDate;
                document.getElementById('backtest_to_date').value = config.endDate;
            }
        }
        
        function executeCommand(command) {
            updateStatus('info', `Executing ${command}...`);
            
            // Disable all buttons
            document.querySelectorAll('.execute-btn').forEach(btn => btn.disabled = true);
            
            // Prepare parameters
            let params = {};
            
            switch(command) {
                case 'data_fetch':
                    params = {
                        symbols: document.getElementById('fetch_symbols').value,
                        start_date: document.getElementById('fetch_start_date').value,
                        end_date: document.getElementById('fetch_end_date').value
                    };
                    break;
                case 'signal_generate':
                    params = {
                        symbols: document.getElementById('signal_symbols').value,
                        start_date: document.getElementById('signal_start_date').value,
                        end_date: document.getElementById('signal_end_date').value
                    };
                    break;
                case 'backtest_run':
                    params = {
                        symbols: document.getElementById('backtest_symbols').value,
                        from_date: document.getElementById('backtest_from_date').value,
                        to_date: document.getElementById('backtest_to_date').value,
                        initial_capital: document.getElementById('backtest_capital').value
                    };
                    break;
            }
            
            // Execute command
            fetch('/api/execute-command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    command: command,
                    parameters: params
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus('success', data.output);
                } else {
                    updateStatus('error', data.error);
                }
            })
            .catch(error => {
                updateStatus('error', `Error: ${error.message}`);
            })
            .finally(() => {
                // Re-enable all buttons
                document.querySelectorAll('.execute-btn').forEach(btn => btn.disabled = false);
            });
        }
        
        function updateStatus(type, message) {
            const statusArea = document.getElementById('status-area');
            const timestamp = new Date().toLocaleTimeString();
            
            let className = '';
            switch(type) {
                case 'success': className = 'status-success'; break;
                case 'error': className = 'status-error'; break;
                case 'info': className = 'status-info'; break;
            }
            
            statusArea.innerHTML += `[${timestamp}] <span class="${className}">${message}</span>\\n`;
            statusArea.scrollTop = statusArea.scrollHeight;
        }
    </script>
</body>
</html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def serve_execute_command(self):
        """Handle command execution requests"""
        if self.command == 'POST':
            import json
            
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            command = request_data.get('command')
            parameters = request_data.get('parameters', {})
            
            try:
                # Execute the command
                result = self.execute_command(command, parameters)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': True,
                    'output': result
                }).encode('utf-8'))
                
            except Exception as e:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': False,
                    'error': str(e)
                }).encode('utf-8'))
        else:
            self.send_error(405)
    
    def execute_command(self, command, parameters):
        """Execute a pipeline command"""
        import requests
        import json
        import subprocess
        import sys
        import os
        
        # Commands that work in dashboard container
        dashboard_commands = {
            'kafka_demo': ['python3', '/app/cli/kafka_demo.py'],
            'kafka_real_test': ['python3', '/app/cli/real_kafka_integration_test.py']
        }
        
        # Commands that need Spark container (via HTTP API)
        spark_commands = ['data_summary', 'data_fetch', 'signal_generate', 'signal_summary', 'backtest_run']
        
        if command in dashboard_commands:
            # Execute command in dashboard container
            env = os.environ.copy()
            env.update({
                'MINIO_ENDPOINT': 'http://minio:9000',
                'MINIO_ACCESS_KEY': 'minioadmin',
                'MINIO_SECRET_KEY': 'minioadmin',
                'DATABASE_URL': 'postgresql://pipeline:pipeline123@postgres:5432/breadthflow',
                'ELASTICSEARCH_URL': 'http://elasticsearch:9200'
            })
            
            cmd = dashboard_commands[command]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd='/app/cli')
            
            if result.returncode == 0:
                return result.stdout
            else:
                raise Exception(f"Command failed: {result.stderr}")
                
        elif command in spark_commands:
            # Execute command via HTTP API to Spark container
            try:
                url = "http://spark-master:8081/execute"
                payload = {
                    "command": command,
                    "parameters": parameters
                }
                
                response = requests.post(url, json=payload, timeout=300)
                response.raise_for_status()
                
                result = response.json()
                if result.get("success"):
                    return result.get("output", "")
                else:
                    raise Exception(result.get("error", "Unknown error"))
                    
            except requests.exceptions.RequestException as e:
                raise Exception(f"Failed to connect to Spark command server: {e}")
            except Exception as e:
                raise Exception(f"Command execution failed: {e}")
        else:
            raise ValueError(f"Unknown command: {command}")

@click.command()
@click.option('--port', default=8080, help='Port to run dashboard on')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
def start_dashboard(port: int, host: str):
    """Start the PostgreSQL-backed web dashboard"""
    
    print(f"🚀 Starting BreadthFlow PostgreSQL Dashboard...")
    print(f"📊 Dashboard URL: http://localhost:{port}")
    print(f"🐘 Database: {DATABASE_URL}")
    
    # Initialize database
    init_database()
    
    httpd = HTTPServer((host, port), DashboardHandler)
    
    print(f"✅ Dashboard ready at http://localhost:{port}")
    print("💡 Features:")
    print("   • PostgreSQL backend (standalone database)")
    print("   • Real-time pipeline monitoring")
    print("   • Auto-refresh every 30 seconds")
    print("   • Manual refresh button")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        httpd.shutdown()

if __name__ == '__main__':
    start_dashboard()
