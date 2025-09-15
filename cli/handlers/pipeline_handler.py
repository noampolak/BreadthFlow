"""
Pipeline handler for the pipeline management page
"""


class PipelineHandler:
    def __init__(self):
        pass

    def serve_pipeline_management(self):
        """Serve the comprehensive pipeline management page"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Pipeline Management</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <style>
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }

        .header h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 2.5em;
            font-weight: 700;
        }

        .header p {
            color: #666;
            margin: 0;
            font-size: 1.1em;
        }

        .nav-buttons {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .nav-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
        }

        .nav-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }

        .nav-btn.active {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }

        .refresh-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
        }

        .control-panel {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .control-panel h2 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.8em;
        }

        .control-buttons {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .control-btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
        }

        .start-btn {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
        }

        .start-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        }

        .start-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .stop-btn {
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
        }

        .stop-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(244, 67, 54, 0.3);
        }

        .stop-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .refresh-status-btn {
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
        }

        .refresh-status-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
        }

        .config-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .config-section h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.4em;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .config-group {
            display: flex;
            flex-direction: column;
        }

        .config-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }

        .config-group select,
        .config-group input {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        .config-group select:focus,
        .config-group input:focus {
            outline: none;
            border-color: #667eea;
        }

        .metrics-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .metrics-section h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.4em;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .metric-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .metric-value {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .status-running {
            color: #4CAF50 !important;
        }

        .status-stopped {
            color: #f44336 !important;
        }

        .runs-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .runs-section h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.4em;
        }

        .runs-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .runs-table th,
        .runs-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        .runs-table th {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: 600;
        }

        .runs-table tr:hover {
            background: rgba(102, 126, 234, 0.05);
        }

        .status-running {
            color: #4CAF50;
            font-weight: 600;
        }

        .status-stopped {
            color: #f44336;
            font-weight: 600;
        }

        .status-completed {
            color: #2196F3;
            font-weight: 600;
        }

        .status-failed {
            color: #ff9800;
            font-weight: 600;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }

        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #f44336;
        }

        .success {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }

        .warning {
            background: #fff3e0;
            color: #ef6c00;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #ff9800;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Pipeline Management</h1>
            <p>Automated batch processing with continuous execution</p>

            <div class="nav-buttons">
                <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                <button class="nav-btn active" onclick="window.location.href='/pipeline'">Pipeline</button>
                <button class="nav-btn" onclick="window.location.href='/parameters'">Parameters</button>
                <button class="nav-btn" onclick="window.location.href='/training'">Training</button>
                <button class="refresh-btn" onclick="loadPipelineData()">Refresh Now</button>
            </div>
        </div>

        <div class="control-panel">
            <h2>🎛️ Pipeline Control</h2>
            <div class="control-buttons">
                <button id="start-btn" class="control-btn start-btn" onclick="startPipeline()">🚀 Start Pipeline</button>
                <button id="stop-btn" class="control-btn stop-btn" onclick="stopPipeline()">🛑 Stop Pipeline</button>
                <button class="control-btn refresh-status-btn" onclick="loadPipelineData()">🔄 Refresh Status</button>
            </div>
            <div id="control-message"></div>
        </div>

        <div class="config-section">
            <h3>⚙️ Pipeline Configuration</h3>
            <div class="config-grid">
                <div class="config-group">
                    <label for="pipeline-mode">Mode:</label>
                    <select id="pipeline-mode">
                        <option value="demo">Demo (AAPL, MSFT)</option>
                        <option value="all">All Symbols</option>
                        <option value="custom">Custom Symbols</option>
                    </select>
                </div>
                <div class="config-group">
                    <label for="pipeline-interval">Interval:</label>
                    <select id="pipeline-interval">
                        <option value="1m">1 minute</option>
                        <option value="5m">5 minutes</option>
                        <option value="15m">15 minutes</option>
                        <option value="30m">30 minutes</option>
                        <option value="1h">1 hour</option>
                        <option value="6h">6 hours</option>
                        <option value="12h">12 hours</option>
                        <option value="1d">1 day</option>
                    </select>
                </div>
                <div class="config-group">
                    <label for="pipeline-timeframe">Timeframe:</label>
                    <select id="pipeline-timeframe">
                        <option value="1day">Daily</option>
                        <option value="1hour">Hourly</option>
                        <option value="1min">Minute</option>
                    </select>
                </div>
                <div class="config-group">
                    <label for="pipeline-symbols">Symbols (comma-separated):</label>
                    <input type="text" id="pipeline-symbols" placeholder="AAPL,MSFT,GOOGL" value="AAPL,MSFT">
                </div>
                <div class="config-group">
                    <label for="pipeline-data-source">Data Source:</label>
                    <select id="pipeline-data-source">
                        <option value="yfinance">Yahoo Finance</option>
                        <option value="alpha_vantage">Alpha Vantage</option>
                        <option value="polygon">Polygon.io</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="metrics-section">
            <h3>📊 Pipeline Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="status-value">Loading...</div>
                    <div class="metric-label">Status</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="total-runs">0</div>
                    <div class="metric-label">Total Runs</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="successful-runs">0</div>
                    <div class="metric-label">Successful</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="failed-runs">0</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="uptime">0s</div>
                    <div class="metric-label">Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="last-run">Never</div>
                    <div class="metric-label">Last Run</div>
                </div>
            </div>
        </div>

        <div class="runs-section">
            <h3>📋 Recent Pipeline Runs (Last 2 Days)</h3>
            <div id="runs-content">
                <div class="loading">Loading recent runs...</div>
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setInterval(loadPipelineData, 30000);

        // Load data on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadPipelineData();
        });

        function loadPipelineData() {
            // Load pipeline status
            fetch('/api/pipeline/status')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateMetrics(data.status);
                        updateButtonStates(data.status.state);
                    } else {
                        console.error('Failed to load pipeline status:', data.error);
                    }
                })
                .catch(error => {
                    console.error('Error loading pipeline status:', error);
                });

            // Load recent runs
            fetch('/api/pipeline/runs')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateRunsTable(data.runs);
                    } else {
                        console.error('Failed to load pipeline runs:', data.error);
                    }
                })
                .catch(error => {
                    console.error('Error loading pipeline runs:', error);
                });
        }

        function updateMetrics(status) {
            document.getElementById('status-value').textContent = status.state || 'Unknown';
            document.getElementById('total-runs').textContent = status.total_runs || 0;
            document.getElementById('successful-runs').textContent = status.successful_runs || 0;
            document.getElementById('failed-runs').textContent = status.failed_runs || 0;
            document.getElementById('uptime').textContent = status.uptime_seconds ? status.uptime_seconds + 's' : '0s';
            document.getElementById('last-run').textContent = status.last_run_time || 'Never';

            // Update status color
            const statusElement = document.getElementById('status-value');
            statusElement.className = 'metric-value';
            if (status.state === 'running') {
                statusElement.classList.add('status-running');
            } else if (status.state === 'stopped') {
                statusElement.classList.add('status-stopped');
            }
        }

        function updateButtonStates(pipelineState) {
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');

            if (pipelineState === 'running') {
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }

        function updateRunsTable(runs) {
            const content = document.getElementById('runs-content');

            if (!runs || runs.length === 0) {
                content.innerHTML = '<div class="loading">No recent pipeline runs found</div>';
                return;
            }

            let table = '<table class="runs-table">';
            table += '<thead><tr><th>Run ID</th><th>Command</th><th>Status</th><th>Start Time</th><th>End Time</th><th>Duration</th><th>Error</th></tr></thead>';
            table += '<tbody>';

            runs.forEach(run => {
                const statusClass = run.status === 'completed' ? 'status-completed' :
                                  run.status === 'failed' ? 'status-failed' :
                                  run.status === 'running' ? 'status-running' : 'status-stopped';

                const duration = run.duration_seconds ? run.duration_seconds + 's' : '-';
                const endTime = run.end_time || '-';

                table += `<tr>
                    <td>${run.run_id.substring(0, 12)}...</td>
                    <td>${run.command}</td>
                    <td class="${statusClass}">${run.status}</td>
                    <td>${run.start_time}</td>
                    <td>${endTime}</td>
                    <td>${duration}</td>
                    <td>${run.error_message || '-'}</td>
                </tr>`;
            });

            table += '</tbody></table>';
            content.innerHTML = table;
        }

        function startPipeline() {
            const mode = document.getElementById('pipeline-mode').value;
            const interval = document.getElementById('pipeline-interval').value;
            const timeframe = document.getElementById('pipeline-timeframe').value;
            const symbols = document.getElementById('pipeline-symbols').value;
            const dataSource = document.getElementById('pipeline-data-source').value;

            const messageDiv = document.getElementById('control-message');
            messageDiv.innerHTML = '<div class="loading">Starting pipeline...</div>';

            fetch('/api/pipeline/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mode: mode,
                    interval: interval,
                    timeframe: timeframe,
                    symbols: symbols,
                    data_source: dataSource
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    messageDiv.innerHTML = `<div class="success">${data.message}</div>`;
                    setTimeout(loadPipelineData, 2000);
                } else {
                    messageDiv.innerHTML = `<div class="error">${data.error}</div>`;
                }
            })
            .catch(error => {
                messageDiv.innerHTML = `<div class="error">Failed to start pipeline: ${error}</div>`;
            });
        }

        function stopPipeline() {
            const messageDiv = document.getElementById('control-message');
            messageDiv.innerHTML = '<div class="loading">Stopping pipeline...</button>';

            fetch('/api/pipeline/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    messageDiv.innerHTML = `<div class="success">${data.message}</div>`;
                    setTimeout(loadPipelineData, 2000);
                } else {
                    messageDiv.innerHTML = `<div class="error">${data.error}</div>`;
                }
            })
            .catch(error => {
                messageDiv.innerHTML = `<div class="error">Failed to stop pipeline: ${error}</div>`;
            });
        }
    </script>
</body>
</html>"""

        return html, "text/html; charset=utf-8", 200
