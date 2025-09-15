"""
Commands handler for the commands page
"""


class CommandsHandler:
    def __init__(self):
        pass

    def serve_commands(self):
        """Serve the commands page"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BreadthFlow Commands</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
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
            border-radius: 8px;
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
                <button class="nav-btn" onclick="window.location.href='/pipeline'">Pipeline</button>
                <button class="nav-btn" onclick="window.location.href='/training'">Training</button>
                <button class="nav-btn" onclick="window.location.href='/parameters'">Parameters</button>
                <button class="refresh-btn" onclick="location.reload()">Refresh Page</button>
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
                                <div class="param-group">
                                    <label class="param-label">Timeframe:</label>
                                    <select class="param-select" id="fetch_timeframe">
                                        <option value="1day">Daily (1day)</option>
                                        <option value="1hour">Hourly (1hour)</option>
                                        <option value="15min">15 Minutes (15min)</option>
                                        <option value="5min">5 Minutes (5min)</option>
                                        <option value="1min">1 Minute (1min)</option>
                                    </select>
                                </div>
                                <div class="param-group">
                                    <label class="param-label">Data Source:</label>
                                    <select class="param-select" id="fetch_data_source">
                                        <option value="yfinance">Yahoo Finance (yfinance)</option>
                                        <option value="alpha_vantage">Alpha Vantage</option>
                                        <option value="polygon">Polygon.io</option>
                                    </select>
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
                    <div class="warning-box" style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px;">
                        <strong>⚠️ Important:</strong> Signal generation requires data files with specific naming pattern:<br>
                        <code>ohlcv/{SYMBOL}/{SYMBOL}_{START_DATE}_{END_DATE}.parquet</code><br>
                        Example: <code>ohlcv/AAPL/AAPL_2024-01-01_2024-12-31.parquet</code><br>
                        Make sure to run data fetch with matching date ranges first.
                </div>
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
                                <div class="param-group">
                                    <label class="param-label">Timeframe:</label>
                                    <select class="param-select" id="signal_timeframe">
                                        <option value="1day">Daily (1day)</option>
                                        <option value="1hour">Hourly (1hour)</option>
                                        <option value="15min">15 Minutes (15min)</option>
                                        <option value="5min">5 Minutes (5min)</option>
                                        <option value="1min">1 Minute (1min)</option>
                                    </select>
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
                                <div class="param-group">
                                    <label class="param-label">Timeframe:</label>
                                    <select class="param-select" id="backtest_timeframe">
                                        <option value="1day">Daily (1day)</option>
                                        <option value="1hour">Hourly (1hour)</option>
                                        <option value="15min">15 Minutes (15min)</option>
                                        <option value="5min">5 Minutes (5min)</option>
                                        <option value="1min">1 Minute (1min)</option>
                                    </select>
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
                        end_date: document.getElementById('fetch_end_date').value,
                        timeframe: document.getElementById('fetch_timeframe').value,
                        data_source: document.getElementById('fetch_data_source').value
                    };
                    break;
                case 'signal_generate':
                    params = {
                        symbols: document.getElementById('signal_symbols').value,
                        start_date: document.getElementById('signal_start_date').value,
                        end_date: document.getElementById('signal_end_date').value,
                        timeframe: document.getElementById('signal_timeframe').value
                    };
                    break;
                case 'backtest_run':
                    params = {
                        symbols: document.getElementById('backtest_symbols').value,
                        from_date: document.getElementById('backtest_from_date').value,
                        to_date: document.getElementById('backtest_to_date').value,
                        initial_capital: document.getElementById('backtest_capital').value,
                        timeframe: document.getElementById('backtest_timeframe').value
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
</html>"""

        return html, "text/html; charset=utf-8", 200
