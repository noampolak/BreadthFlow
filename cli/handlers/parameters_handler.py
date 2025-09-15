"""
Parameters handler for the parameters configuration page
"""


class ParametersHandler:
    def __init__(self):
        pass

    def serve_parameters(self):
        """Serve the comprehensive parameters configuration page"""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Parameters Configuration</title>
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
        
        .timeframe-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .timeframe-tab {
            padding: 12px 24px;
            background: rgba(255, 255, 255, 0.8);
            color: #333;
            border: 2px solid transparent;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .timeframe-tab:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
        }
        
        .timeframe-tab.active {
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border-color: #4CAF50;
        }
        
        .parameters-section {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        
        .parameters-section h3 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 20px;
            font-size: 1.6em;
        }
        
        .parameters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .parameter-group {
            display: flex;
            flex-direction: column;
        }
        
        .parameter-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 14px;
        }
        
        .parameter-group input,
        .parameter-group select {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s ease;
            background: white;
        }
        
        .parameter-group input:focus,
        .parameter-group select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .parameter-group input[type="number"] {
            width: 100%;
        }
        
        .parameter-group input[type="text"] {
            width: 100%;
        }
        
        .parameter-group select {
            width: 100%;
        }
        
        .save-section {
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }
        
        .save-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 0 10px;
        }
        
        .save-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.3);
        }
        
        .reset-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #ff9800, #f57c00);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 0 10px;
        }
        
        .reset-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 152, 0, 0.3);
        }
        
        .load-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            margin: 0 10px;
        }
        
        .load-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.3);
        }
        
        .message {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-weight: 600;
        }
        
        .message.success {
            background: #e8f5e8;
            color: #2e7d32;
            border-left: 4px solid #4CAF50;
        }
        
        .message.error {
            background: #ffebee;
            color: #c62828;
            border-left: 4px solid #f44336;
        }
        
        .message.info {
            background: #e3f2fd;
            color: #1565c0;
            border-left: 4px solid #2196F3;
        }
        
        .hidden {
            display: none;
        }
        
        .parameter-description {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            font-style: italic;
        }
        
        .advanced-toggle {
            text-align: center;
            margin: 20px 0;
        }
        
        .advanced-toggle button {
            padding: 10px 20px;
            background: rgba(102, 126, 234, 0.1);
            color: #667eea;
            border: 2px solid #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .advanced-toggle button:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚙️ Parameters Configuration</h1>
            <p>Configure trading strategy parameters and system settings</p>
            
            <div class="nav-buttons">
                <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                <button class="nav-btn" onclick="window.location.href='/pipeline'">Pipeline</button>
                <button class="nav-btn active" onclick="window.location.href='/parameters'">Parameters</button>
                <button class="nav-btn" onclick="window.location.href='/training'">Training</button>
                <button class="refresh-btn" onclick="loadParameters()">Refresh Now</button>
            </div>
        </div>
        
        <div class="timeframe-tabs">
            <button class="timeframe-tab active" onclick="showTimeframe('daily')">📅 Daily (1day)</button>
            <button class="timeframe-tab" onclick="showTimeframe('hourly')">⏰ Hourly (1hour)</button>
            <button class="timeframe-tab" onclick="showTimeframe('minute')">⚡ Minute (1min)</button>
            <button class="timeframe-tab" onclick="showTimeframe('system')">🔧 System</button>
        </div>
        
        <!-- Daily Parameters -->
        <div id="daily-params" class="parameters-section">
            <h3>📅 Daily Trading Parameters</h3>
            <div class="parameters-grid">
                <div class="parameter-group">
                    <label for="daily-lookback">Lookback Period (days)</label>
                    <input type="number" id="daily-lookback" value="30" min="1" max="365">
                    <div class="parameter-description">Number of days to look back for analysis</div>
                </div>
                <div class="parameter-group">
                    <label for="daily-threshold">Signal Threshold</label>
                    <input type="number" id="daily-threshold" value="0.6" min="0" max="1" step="0.1">
                    <div class="parameter-description">Minimum confidence for signal generation</div>
                </div>
                <div class="parameter-group">
                    <label for="daily-strategy">Strategy</label>
                    <select id="daily-strategy">
                        <option value="momentum">Momentum</option>
                        <option value="mean_reversion">Mean Reversion</option>
                        <option value="breakout">Breakout</option>
                        <option value="hybrid">Hybrid</option>
                    </select>
                    <div class="parameter-description">Trading strategy to use</div>
                </div>
                <div class="parameter-group">
                    <label for="daily-capital">Capital Allocation (%)</label>
                    <input type="number" id="daily-capital" value="10" min="1" max="100">
                    <div class="parameter-description">Percentage of capital to allocate per trade</div>
                </div>
                <div class="parameter-group">
                    <label for="daily-stop-loss">Stop Loss (%)</label>
                    <input type="number" id="daily-stop-loss" value="5" min="0.1" max="50" step="0.1">
                    <div class="parameter-description">Stop loss percentage</div>
                </div>
                <div class="parameter-group">
                    <label for="daily-take-profit">Take Profit (%)</label>
                    <input type="number" id="daily-take-profit" value="15" min="0.1" max="100" step="0.1">
                    <div class="parameter-description">Take profit percentage</div>
                </div>
            </div>
        </div>
        
        <!-- Hourly Parameters -->
        <div id="hourly-params" class="parameters-section hidden">
            <h3>⏰ Hourly Trading Parameters</h3>
            <div class="parameters-grid">
                <div class="parameter-group">
                    <label for="hourly-lookback">Lookback Period (hours)</label>
                    <input type="number" id="hourly-lookback" value="168" min="1" max="8760">
                    <div class="parameter-description">Number of hours to look back for analysis</div>
                </div>
                <div class="parameter-group">
                    <label for="hourly-threshold">Signal Threshold</label>
                    <input type="number" id="hourly-threshold" value="0.65" min="0" max="1" step="0.1">
                    <div class="parameter-description">Minimum confidence for signal generation</div>
                </div>
                <div class="parameter-group">
                    <label for="hourly-strategy">Strategy</label>
                    <select id="hourly-strategy">
                        <option value="scalping">Scalping</option>
                        <option value="swing">Swing Trading</option>
                        <option value="intraday">Intraday</option>
                        <option value="hybrid">Hybrid</option>
                    </select>
                    <div class="parameter-description">Trading strategy to use</div>
                </div>
                <div class="parameter-group">
                    <label for="hourly-capital">Capital Allocation (%)</label>
                    <input type="number" id="hourly-capital" value="8" min="1" max="100">
                    <div class="parameter-description">Percentage of capital to allocate per trade</div>
                </div>
                <div class="parameter-group">
                    <label for="hourly-stop-loss">Stop Loss (%)</label>
                    <input type="number" id="hourly-stop-loss" value="3" min="0.1" max="50" step="0.1">
                    <div class="parameter-description">Stop loss percentage</div>
                </div>
                <div class="parameter-group">
                    <label for="hourly-take-profit">Take Profit (%)</label>
                    <input type="number" id="hourly-take-profit" value="8" min="0.1" max="100" step="0.1">
                    <div class="parameter-description">Take profit percentage</div>
                </div>
            </div>
        </div>
        
        <!-- Minute Parameters -->
        <div id="minute-params" class="parameters-section hidden">
            <h3>⚡ Minute Trading Parameters</h3>
            <div class="parameters-grid">
                <div class="parameter-group">
                    <label for="minute-lookback">Lookback Period (minutes)</label>
                    <input type="number" id="minute-lookback" value="1440" min="1" max="10080">
                    <div class="parameter-description">Number of minutes to look back for analysis</div>
                </div>
                <div class="parameter-group">
                    <label for="minute-threshold">Signal Threshold</label>
                    <input type="number" id="minute-threshold" value="0.7" min="0" max="1" step="0.1">
                    <div class="parameter-description">Minimum confidence for signal generation</div>
                </div>
                <div class="parameter-group">
                    <label for="minute-strategy">Strategy</label>
                    <select id="minute-strategy">
                        <option value="high_frequency">High Frequency</option>
                        <option value="arbitrage">Arbitrage</option>
                        <option value="market_making">Market Making</option>
                        <option value="hybrid">Hybrid</option>
                    </select>
                    <div class="parameter-description">Trading strategy to use</div>
                </div>
                <div class="parameter-group">
                    <label for="minute-capital">Capital Allocation (%)</label>
                    <input type="number" id="minute-capital" value="5" min="1" max="100">
                    <div class="parameter-description">Percentage of capital to allocate per trade</div>
                </div>
                <div class="parameter-group">
                    <label for="minute-stop-loss">Stop Loss (%)</label>
                    <input type="number" id="minute-stop-loss" value="2" min="0.1" max="50" step="0.1">
                    <div class="parameter-description">Stop loss percentage</div>
                </div>
                <div class="minute-take-profit">Take Profit (%)</label>
                    <input type="number" id="minute-take-profit" value="4" min="0.1" max="100" step="0.1">
                    <div class="parameter-description">Take profit percentage</div>
                </div>
            </div>
        </div>
        
        <!-- System Parameters -->
        <div id="system-params" class="parameters-section hidden">
            <h3>🔧 System Configuration</h3>
            <div class="parameters-grid">
                <div class="parameter-group">
                    <label for="system-max-positions">Max Concurrent Positions</label>
                    <input type="number" id="system-max-positions" value="10" min="1" max="100">
                    <div class="parameter-description">Maximum number of open positions</div>
                </div>
                <div class="parameter-group">
                    <label for="system-risk-per-trade">Risk Per Trade (%)</label>
                    <input type="number" id="system-risk-per-trade" value="2" min="0.1" max="10" step="0.1">
                    <div class="parameter-description">Maximum risk per individual trade</div>
                </div>
                <div class="parameter-group">
                    <label for="system-max-drawdown">Max Drawdown (%)</label>
                    <input type="number" id="system-max-drawdown" value="20" min="1" max="50">
                    <div class="parameter-description">Maximum allowed drawdown</div>
                </div>
                <div class="parameter-group">
                    <label for="system-rebalance-frequency">Rebalance Frequency</label>
                    <select id="system-rebalance-frequency">
                        <option value="daily">Daily</option>
                        <option value="weekly">Weekly</option>
                        <option value="monthly">Monthly</option>
                        <option value="never">Never</option>
                    </select>
                    <div class="parameter-description">Portfolio rebalancing frequency</div>
                </div>
                <div class="parameter-group">
                    <label for="system-data-source">Primary Data Source</label>
                    <select id="system-data-source">
                        <option value="yfinance">Yahoo Finance</option>
                        <option value="alpha_vantage">Alpha Vantage</option>
                        <option value="polygon">Polygon.io</option>
                        <option value="iex">IEX Cloud</option>
                    </select>
                    <div class="parameter-description">Primary data source for market data</div>
                </div>
                <div class="parameter-group">
                    <label for="system-backup-source">Backup Data Source</label>
                    <select id="system-backup-source">
                        <option value="alpha_vantage">Alpha Vantage</option>
                        <option value="yfinance">Yahoo Finance</option>
                        <option value="polygon">Polygon.io</option>
                        <option value="iex">IEX Cloud</option>
                    </select>
                    <div class="parameter-description">Backup data source if primary fails</div>
                </div>
            </div>
        </div>
        
        <div class="save-section">
            <button class="save-btn" onclick="saveParameters()">💾 Save Parameters</button>
            <button class="reset-btn" onclick="resetParameters()">🔄 Reset to Defaults</button>
            <button class="load-btn" onclick="loadParameters()">📥 Load from Database</button>
        </div>
        
        <div id="message-container"></div>
    </div>
    
    <script>
        let currentTimeframe = 'daily';
        
        function showTimeframe(timeframe) {
            // Hide all parameter sections
            document.querySelectorAll('.parameters-section').forEach(section => {
                section.classList.add('hidden');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.timeframe-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected timeframe section
            document.getElementById(timeframe + '-params').classList.remove('hidden');
            
            // Add active class to selected tab
            event.target.classList.add('active');
            
            currentTimeframe = timeframe;
        }
        
        function saveParameters() {
            const params = {
                timeframe: currentTimeframe,
                parameters: {}
            };
            
            // Collect parameters based on current timeframe
            if (currentTimeframe === 'daily') {
                params.parameters = {
                    lookback: document.getElementById('daily-lookback').value,
                    threshold: document.getElementById('daily-threshold').value,
                    strategy: document.getElementById('daily-strategy').value,
                    capital: document.getElementById('daily-capital').value,
                    stop_loss: document.getElementById('daily-stop-loss').value,
                    take_profit: document.getElementById('daily-take-profit').value
                };
            } else if (currentTimeframe === 'hourly') {
                params.parameters = {
                    lookback: document.getElementById('hourly-lookback').value,
                    threshold: document.getElementById('hourly-threshold').value,
                    strategy: document.getElementById('hourly-strategy').value,
                    capital: document.getElementById('hourly-capital').value,
                    stop_loss: document.getElementById('hourly-stop-loss').value,
                    take_profit: document.getElementById('hourly-take-profit').value
                };
            } else if (currentTimeframe === 'minute') {
                params.parameters = {
                    lookback: document.getElementById('minute-lookback').value,
                    threshold: document.getElementById('minute-threshold').value,
                    strategy: document.getElementById('minute-strategy').value,
                    capital: document.getElementById('minute-capital').value,
                    stop_loss: document.getElementById('minute-stop-loss').value,
                    take_profit: document.getElementById('minute-take-profit').value
                };
            } else if (currentTimeframe === 'system') {
                params.parameters = {
                    max_positions: document.getElementById('system-max-positions').value,
                    risk_per_trade: document.getElementById('system-risk-per-trade').value,
                    max_drawdown: document.getElementById('system-max-drawdown').value,
                    rebalance_frequency: document.getElementById('system-rebalance-frequency').value,
                    data_source: document.getElementById('system-data-source').value,
                    backup_source: document.getElementById('system-backup-source').value
                };
            }
            
            // Save to database
            fetch('/api/parameters/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage('Parameters saved successfully!', 'success');
                } else {
                    showMessage('Failed to save parameters: ' + data.error, 'error');
                }
            })
            .catch(error => {
                showMessage('Error saving parameters: ' + error, 'error');
            });
        }
        
        function resetParameters() {
            if (confirm('Are you sure you want to reset all parameters to defaults?')) {
                // Reset daily parameters
                document.getElementById('daily-lookback').value = '30';
                document.getElementById('daily-threshold').value = '0.6';
                document.getElementById('daily-strategy').value = 'momentum';
                document.getElementById('daily-capital').value = '10';
                document.getElementById('daily-stop-loss').value = '5';
                document.getElementById('daily-take-profit').value = '15';
                
                // Reset hourly parameters
                document.getElementById('hourly-lookback').value = '168';
                document.getElementById('hourly-threshold').value = '0.65';
                document.getElementById('hourly-strategy').value = 'scalping';
                document.getElementById('hourly-capital').value = '8';
                document.getElementById('hourly-stop-loss').value = '3';
                document.getElementById('hourly-take-profit').value = '8';
                
                // Reset minute parameters
                document.getElementById('minute-lookback').value = '1440';
                document.getElementById('minute-threshold').value = '0.7';
                document.getElementById('minute-strategy').value = 'high_frequency';
                document.getElementById('minute-capital').value = '5';
                document.getElementById('minute-stop-loss').value = '2';
                document.getElementById('minute-take-profit').value = '4';
                
                // Reset system parameters
                document.getElementById('system-max-positions').value = '10';
                document.getElementById('system-risk-per-trade').value = '2';
                document.getElementById('system-max-drawdown').value = '20';
                document.getElementById('system-rebalance-frequency').value = 'daily';
                document.getElementById('system-data-source').value = 'yfinance';
                document.getElementById('system-backup-source').value = 'alpha_vantage';
                
                showMessage('Parameters reset to defaults!', 'success');
            }
        }
        
        function loadParameters() {
            // Load parameters from database
            fetch('/api/parameters/load?timeframe=' + currentTimeframe)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        populateParameters(data.parameters);
                        showMessage('Parameters loaded successfully!', 'success');
                    } else {
                        showMessage('Failed to load parameters: ' + data.error, 'error');
                    }
                })
                .catch(error => {
                    showMessage('Error loading parameters: ' + error, 'error');
                });
        }
        
        function populateParameters(params) {
            if (currentTimeframe === 'daily') {
                if (params.lookback) document.getElementById('daily-lookback').value = params.lookback;
                if (params.threshold) document.getElementById('daily-threshold').value = params.threshold;
                if (params.strategy) document.getElementById('daily-strategy').value = params.strategy;
                if (params.capital) document.getElementById('daily-capital').value = params.capital;
                if (params.stop_loss) document.getElementById('daily-stop-loss').value = params.stop_loss;
                if (params.take_profit) document.getElementById('daily-take-profit').value = params.take_profit;
            } else if (currentTimeframe === 'hourly') {
                if (params.lookback) document.getElementById('hourly-lookback').value = params.lookback;
                if (params.threshold) document.getElementById('hourly-threshold').value = params.threshold;
                if (params.strategy) document.getElementById('hourly-strategy').value = params.strategy;
                if (params.capital) document.getElementById('hourly-capital').value = params.capital;
                if (params.stop_loss) document.getElementById('hourly-stop-loss').value = params.stop_loss;
                if (params.take_profit) document.getElementById('hourly-take-profit').value = params.take_profit;
            } else if (currentTimeframe === 'minute') {
                if (params.lookback) document.getElementById('minute-lookback').value = params.lookback;
                if (params.threshold) document.getElementById('minute-threshold').value = params.threshold;
                if (params.strategy) document.getElementById('minute-strategy').value = params.strategy;
                if (params.capital) document.getElementById('minute-capital').value = params.capital;
                if (params.stop_loss) document.getElementById('minute-stop-loss').value = params.stop_loss;
                if (params.take_profit) document.getElementById('minute-take-profit').value = params.take_profit;
            } else if (currentTimeframe === 'system') {
                if (params.max_positions) document.getElementById('system-max-positions').value = params.max_positions;
                if (params.risk_per_trade) document.getElementById('system-risk-per-trade').value = params.risk_per_trade;
                if (params.max_drawdown) document.getElementById('system-max-drawdown').value = params.max_drawdown;
                if (params.rebalance_frequency) document.getElementById('system-rebalance-frequency').value = params.rebalance_frequency;
                if (params.data_source) document.getElementById('system-data-source').value = params.data_source;
                if (params.backup_source) document.getElementById('system-backup-source').value = params.backup_source;
            }
        }
        
        function showMessage(message, type) {
            const container = document.getElementById('message-container');
            container.innerHTML = `<div class="message ${type}">${message}</div>`;
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                container.innerHTML = '';
            }, 5000);
        }
        
        // Load parameters on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadParameters();
        });
    </script>
</body>
</html>"""

        return html, "text/html; charset=utf-8", 200
