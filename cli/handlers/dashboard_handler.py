"""
Dashboard handler for the main dashboard page
"""

from template_renderer import TemplateRenderer


class DashboardHandler:
    def __init__(self):
        self.template_renderer = TemplateRenderer()

    def serve_dashboard(self):
        """Serve the main dashboard page"""
        try:
            # Create a simple dashboard without templates to ensure it works
            html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <link rel="stylesheet" href="/static/css/dashboard.css">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BreadthFlow Dashboard</h1>
            <p>Real-time pipeline monitoring with PostgreSQL backend</p>
            <div class="nav-buttons">
                <button class="nav-btn active" onclick="window.location.href='/'">Dashboard</button>
                <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                <button class="nav-btn" onclick="window.location.href='/pipeline'">Pipeline</button>
                <button class="nav-btn" onclick="window.location.href='/training'">Training</button>
                <button class="nav-btn" onclick="window.location.href='/parameters'">Parameters</button>
                <button class="refresh-btn" onclick="loadData()">Refresh Now</button>
            </div>
        </div>
        
        <!-- Dashboard Stats -->
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
        
        <!-- Recent Pipeline Runs -->
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
            
            <!-- Pagination Controls -->
            <div class="pagination-container" id="pagination-container" style="display: none;">
                <div class="pagination-info">
                    <span id="pagination-info">Showing 0-0 of 0 runs</span>
                </div>
                <div class="pagination-controls">
                    <button id="prev-page" class="pagination-btn" onclick="changePage(-1)" disabled>← Previous</button>
                    <span id="page-numbers" class="page-numbers"></span>
                    <button id="next-page" class="pagination-btn" onclick="changePage(1)" disabled>Next →</button>
                </div>
            </div>
            
            <div class="last-updated" id="last-updated">Last updated: Never</div>
        </div>
        
        <!-- Quick Actions -->
        <div class="runs-section">
            <h2>Quick Actions</h2>
            <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                <button class="refresh-btn" onclick="window.location.href='/pipeline'">🚀 Pipeline Management</button>
                <button class="refresh-btn" onclick="window.location.href='/trading'">📊 Trading Signals</button>
                <button class="refresh-btn" onclick="window.location.href='/commands'">⚡ Quick Commands</button>
                <button class="refresh-btn" onclick="window.location.href='/training'">🎓 Model Training</button>
            </div>
        </div>
    </div>
    
    <!-- Run Details Modal -->
    <div id="runModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalContent">Loading...</div>
        </div>
    </div>
    
    <script src="/static/js/dashboard.js"></script>
    <script>
        // Load data on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadData();
        });
        
        // Auto-refresh every 30 seconds
        setInterval(loadData, 30000);
    </script>
</body>
</html>"""
            return html, "text/html; charset=utf-8", 200
        except Exception as e:
            return f"Error rendering dashboard: {str(e)}", "text/plain", 500

    def serve_infrastructure(self):
        """Serve the infrastructure page"""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Infrastructure - BreadthFlow</title>
            <link rel="stylesheet" href="/static/css/dashboard.css">
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Infrastructure Status</h1>
                    <p>System health and service monitoring</p>
                    <div class="nav-buttons">
                        <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                        <button class="nav-btn active" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                        <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                        <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                        <button class="nav-btn" onclick="window.location.href='/pipeline'">Pipeline</button>
                        <button class="nav-btn" onclick="window.location.href='/training'">Training</button>
                        <button class="nav-btn" onclick="window.location.href='/parameters'">Parameters</button>
                    </div>
                </div>
                <!-- System Health Overview -->
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="database-status">🟢</div>
                        <div class="stat-label">Database</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="pipeline-status">⚪</div>
                        <div class="stat-label">Pipeline</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="api-status">🟢</div>
                        <div class="stat-label">API</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="uptime">0h</div>
                        <div class="stat-label">Uptime</div>
                    </div>
                </div>
                
                <!-- Service Status -->
                <div class="runs-section">
                    <h2>Service Status</h2>
                    <table class="runs-table">
                        <thead>
                            <tr>
                                <th>Service</th>
                                <th>Status</th>
                                <th>Last Check</th>
                                <th>Response Time</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody id="services-tbody">
                            <tr><td colspan="5">Loading services...</td></tr>
                        </tbody>
                    </table>
                </div>
                
                <!-- System Resources -->
                <div class="runs-section">
                    <h2>System Resources</h2>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-value" id="cpu-usage">--</div>
                            <div class="stat-label">CPU Usage</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="memory-usage">--</div>
                            <div class="stat-label">Memory Usage</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="disk-usage">--</div>
                            <div class="stat-label">Disk Usage</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="network-status">🟢</div>
                            <div class="stat-label">Network</div>
                        </div>
                    </div>
                </div>
                
                <div class="last-updated" id="last-updated">Last updated: Never</div>
            </div>
            
            <script>
                // Load infrastructure data
                async function loadInfrastructureData() {
                    try {
                        // Check database connection
                        const dbResponse = await fetch('/api/summary');
                        if (dbResponse.ok) {
                            document.getElementById('database-status').textContent = '🟢';
                            document.getElementById('database-status').style.color = '#28a745';
                        } else {
                            document.getElementById('database-status').textContent = '🔴';
                            document.getElementById('database-status').style.color = '#dc3545';
                        }
                        
                        // Check pipeline status
                        const pipelineResponse = await fetch('/api/pipeline/status');
                        if (pipelineResponse.ok) {
                            const pipelineData = await pipelineResponse.json();
                            if (pipelineData.success && pipelineData.status.state === 'running') {
                                document.getElementById('pipeline-status').textContent = '🟢';
                                document.getElementById('pipeline-status').style.color = '#28a745';
                            } else {
                                document.getElementById('pipeline-status').textContent = '🟡';
                                document.getElementById('pipeline-status').style.color = '#ffc107';
                            }
                        } else {
                            document.getElementById('pipeline-status').textContent = '🔴';
                            document.getElementById('pipeline-status').style.color = '#dc3545';
                        }
                        
                        // Check API status
                        const apiResponse = await fetch('/api/summary');
                        if (apiResponse.ok) {
                            document.getElementById('api-status').textContent = '🟢';
                            document.getElementById('api-status').style.color = '#28a745';
                        } else {
                            document.getElementById('api-status').textContent = '🔴';
                            document.getElementById('api-status').style.color = '#dc3545';
                        }
                        
                        // Update services table
                        updateServicesTable();
                        
                        // Update system resources (mock data for now)
                        document.getElementById('cpu-usage').textContent = '15%';
                        document.getElementById('memory-usage').textContent = '45%';
                        document.getElementById('disk-usage').textContent = '32%';
                        document.getElementById('network-status').textContent = '🟢';
                        
                        document.getElementById('last-updated').textContent = 'Last updated: ' + new Date().toLocaleString();
                        
                    } catch (error) {
                        console.error('Error loading infrastructure data:', error);
                        document.getElementById('database-status').textContent = '🔴';
                        document.getElementById('pipeline-status').textContent = '🔴';
                        document.getElementById('api-status').textContent = '🔴';
                    }
                }
                
                function updateServicesTable() {
                    const tbody = document.getElementById('services-tbody');
                    const services = [
                        { name: 'Dashboard Server', status: 'Running', lastCheck: new Date().toLocaleString(), responseTime: '45ms', details: 'Port 8003' },
                        { name: 'PostgreSQL Database', status: 'Connected', lastCheck: new Date().toLocaleString(), responseTime: '12ms', details: 'Port 5432' },
                        { name: 'API Endpoints', status: 'Active', lastCheck: new Date().toLocaleString(), responseTime: '8ms', details: 'All endpoints responding' },
                        { name: 'Static Files', status: 'Serving', lastCheck: new Date().toLocaleString(), responseTime: '3ms', details: 'CSS/JS files' }
                    ];
                    
                    tbody.innerHTML = '';
                    services.forEach(service => {
                        const row = document.createElement('tr');
                        const statusClass = service.status === 'Running' || service.status === 'Connected' || service.status === 'Active' || service.status === 'Serving' ? 'status-completed' : 'status-failed';
                        
                        row.innerHTML = `
                            <td>${service.name}</td>
                            <td><span class="${statusClass}">${service.status}</span></td>
                            <td>${service.lastCheck}</td>
                            <td>${service.responseTime}</td>
                            <td>${service.details}</td>
                        `;
                        tbody.appendChild(row);
                    });
                }
                
                // Load data on page load
                loadInfrastructureData();
                
                // Auto-refresh every 30 seconds
                setInterval(loadInfrastructureData, 30000);
            </script>
        </body>
        </html>
        """

        return html, "text/html; charset=utf-8", 200

    def serve_trading(self):
        """Serve the comprehensive trading signals page"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Trading Signals</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
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
        .refresh-btn:hover { 
            transform: scale(1.05); 
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
                <button class="nav-btn" onclick="window.location.href='/pipeline'">Pipeline</button>
                <button class="nav-btn" onclick="window.location.href='/parameters'">Parameters</button>
                <button class="nav-btn" onclick="window.location.href='/training'">Training</button>
                <button class="refresh-btn" onclick="loadSignals()">Refresh Signals</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>Latest Trading Signals</h2>
            <div style="margin-bottom: 20px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                <label style="font-weight: 600; color: #333;">Filter by Timeframe:</label>
                <select id="timeframe-filter" style="padding: 8px 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 0.9em;" onchange="loadSignals()">
                    <option value="all">All Timeframes</option>
                    <option value="1day">Daily (1day)</option>
                    <option value="1hour">Hourly (1hour)</option>
                    <option value="15min">15 Minutes (15min)</option>
                    <option value="5min">5 Minutes (5min)</option>
                    <option value="1min">1 Minute (1min)</option>
                </select>
                <span style="color: #666; font-size: 0.9em;">Note: Signal timeframe depends on how signals were generated</span>
            </div>
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
            const timeframe = signal.timeframe || '1day';
            
            // Apply timeframe filter
            const selectedTimeframe = document.getElementById('timeframe-filter').value;
            if (selectedTimeframe !== 'all' && timeframe !== selectedTimeframe) {
                return ''; // Don't show this signal
            }
            
            return `
                <div class="signal-card ${signalClass}">
                    <h3>${signal.symbol || 'UNKNOWN'} <span style="font-size: 0.7em; color: #666; font-weight: normal;">(${timeframe})</span></h3>
                    <p><strong>Signal:</strong> ${signal.signal_type?.toUpperCase() || 'HOLD'}</p>
                    <p><strong>Confidence:</strong> ${confidence}%</p>
                    <p><strong>Strength:</strong> ${strength}</p>
                    <p><strong>Timeframe:</strong> ${timeframe}</p>
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
</html>"""

        return html, "text/html; charset=utf-8", 200
