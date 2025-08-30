"""
Training Dashboard for BreadthFlow

This module provides a comprehensive training interface for the BreadthFlow
abstraction system, including model training, management, and analytics.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from http.server import BaseHTTPRequestHandler
import urllib.parse

logger = logging.getLogger(__name__)

class TrainingDashboard:
    """Training dashboard HTML generator"""
    
    def generate_html(self) -> str:
        """Generate the complete training management HTML page"""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BreadthFlow Training Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg">
    <style>
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
        }}
        .header {{ 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .header h1 {{ 
            color: #667eea; 
            margin: 0 0 10px 0; 
            font-size: 2.5em;
        }}
        .header p {{ 
            color: #666; 
            margin: 0 0 20px 0; 
            font-size: 1.1em;
        }}
        .nav-buttons {{ 
            display: flex; 
            gap: 10px; 
            justify-content: center; 
            flex-wrap: wrap; 
            margin-bottom: 20px;
        }}
        .nav-btn {{ 
            background: rgba(102, 126, 234, 0.1); 
            color: #667eea; 
            border: 2px solid #667eea; 
            padding: 10px 20px; 
            border-radius: 25px; 
            cursor: pointer; 
            transition: all 0.3s ease; 
            text-decoration: none; 
            font-weight: bold;
        }}
        .nav-btn:hover {{ 
            background: #667eea; 
            color: white; 
            transform: translateY(-2px);
        }}
        .nav-btn.active {{ 
            background: #667eea; 
            color: white;
        }}
        .refresh-btn {{ 
            background: #28a745; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 25px; 
            cursor: pointer; 
            transition: all 0.3s ease;
        }}
        .refresh-btn:hover {{ 
            background: #218838; 
            transform: translateY(-2px);
        }}
        .quick-actions {{ 
            margin-top: 20px; 
            display: flex; 
            gap: 15px; 
            justify-content: center; 
            flex-wrap: wrap;
        }}
        .action-btn {{ 
            background: linear-gradient(135deg, #667eea, #764ba2); 
            color: white; 
            border: none; 
            padding: 12px 20px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-weight: bold; 
            transition: all 0.3s ease;
        }}
        .action-btn:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        .section {{ 
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .section h2 {{ 
            color: #667eea; 
            margin-bottom: 25px; 
            font-size: 1.8em;
        }}
        .config-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
            gap: 20px; 
            margin-bottom: 25px;
        }}
        .config-group {{ 
            display: flex; 
            flex-direction: column;
        }}
        .config-group label {{ 
            font-weight: bold; 
            margin-bottom: 8px; 
            color: #555;
        }}
        .config-group input, .config-group select, .config-group textarea {{ 
            padding: 12px; 
            border: 2px solid #e1e5e9; 
            border-radius: 8px; 
            font-size: 14px; 
            transition: border-color 0.3s ease;
        }}
        .config-group input:focus, .config-group select:focus, .config-group textarea:focus {{ 
            outline: none; 
            border-color: #667eea;
        }}
        .training-controls {{ 
            display: flex; 
            gap: 15px; 
            justify-content: center; 
            flex-wrap: wrap; 
            margin-top: 25px;
        }}
        .btn-primary {{ 
            background: linear-gradient(135deg, #667eea, #764ba2); 
            color: white; 
            border: none; 
            padding: 15px 30px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-weight: bold; 
            font-size: 16px; 
            transition: all 0.3s ease;
        }}
        .btn-primary:hover {{ 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        .btn-secondary {{ 
            background: #6c757d; 
            color: white; 
            border: none; 
            padding: 15px 30px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-weight: bold; 
            font-size: 16px; 
            transition: all 0.3s ease;
        }}
        .btn-outline {{ 
            background: transparent; 
            color: #667eea; 
            border: 2px solid #667eea; 
            padding: 15px 30px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-weight: bold; 
            font-size: 16px; 
            transition: all 0.3s ease;
        }}
        .btn-outline:hover {{ 
            background: #667eea; 
            color: white;
        }}
        .progress-section {{ 
            margin-top: 25px; 
            padding: 20px; 
            background: rgba(102, 126, 234, 0.1); 
            border-radius: 10px;
        }}
        .progress-bar {{ 
            width: 100%; 
            height: 20px; 
            background: #e1e5e9; 
            border-radius: 10px; 
            overflow: hidden; 
            margin: 15px 0;
        }}
        .progress-fill {{ 
            height: 100%; 
            background: linear-gradient(90deg, #667eea, #764ba2); 
            width: 0%; 
            transition: width 0.3s ease;
        }}
        .progress-details {{ 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 15px;
        }}
        .progress-details span {{ 
            font-weight: bold; 
            color: #667eea;
        }}
        #training-log {{ 
            max-height: 200px; 
            overflow-y: auto; 
            background: rgba(0,0,0,0.05); 
            padding: 15px; 
            border-radius: 8px; 
            font-family: monospace; 
            font-size: 12px;
        }}
        .model-card {{ 
            background: white; 
            border-radius: 15px; 
            padding: 20px; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
            border-left: 5px solid #667eea;
        }}
        .model-header {{ 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            margin-bottom: 15px;
        }}
        .model-name {{ 
            font-size: 1.2em; 
            font-weight: bold; 
            color: #333;
        }}
        .model-actions {{ 
            display: flex; 
            gap: 10px;
        }}
        .btn-small {{ 
            padding: 8px 15px; 
            border-radius: 5px; 
            border: none; 
            cursor: pointer; 
            font-size: 12px; 
            font-weight: bold; 
            transition: all 0.3s ease;
        }}
        .btn-deploy {{ 
            background: #28a745; 
            color: white;
        }}
        .btn-view {{ 
            background: #17a2b8; 
            color: white;
        }}
        .btn-delete {{ 
            background: #dc3545; 
            color: white;
        }}
        .model-metrics {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
            gap: 15px; 
            margin-bottom: 15px;
        }}
        .metric {{ 
            text-align: center; 
            padding: 10px; 
            background: rgba(102, 126, 234, 0.1); 
            border-radius: 8px;
        }}
        .metric-value {{ 
            font-size: 1.5em; 
            font-weight: bold; 
            color: #667eea;
        }}
        .metric-label {{ 
            font-size: 0.9em; 
            color: #666;
        }}
        .analytics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 20px;
        }}
        .analytics-card {{ 
            background: white; 
            border-radius: 15px; 
            padding: 20px; 
            text-align: center; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .analytics-value {{ 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #667eea; 
            margin-bottom: 10px;
        }}
        .analytics-label {{ 
            color: #666; 
            font-size: 1.1em;
        }}
        .message {{ 
            padding: 15px; 
            border-radius: 8px; 
            margin: 15px 0; 
            font-weight: bold;
        }}
        .message.success {{ 
            background: #d4edda; 
            color: #155724; 
            border: 1px solid #c3e6cb;
        }}
        .message.error {{ 
            background: #f8d7da; 
            color: #721c24; 
            border: 1px solid #f5c6cb;
        }}
        .message.info {{ 
            background: #d1ecf1; 
            color: #0c5460; 
            border: 1px solid #bee5eb;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Model Training & Management</h1>
            <p>Train, manage, and deploy machine learning models for trading signals</p>
            
            <div class="nav-buttons">
                <button class="nav-btn" onclick="window.location.href='/'">Dashboard</button>
                <button class="nav-btn" onclick="window.location.href='/infrastructure'">Infrastructure</button>
                <button class="nav-btn" onclick="window.location.href='/trading'">Trading Signals</button>
                <button class="nav-btn" onclick="window.location.href='/commands'">Commands</button>
                <button class="nav-btn" onclick="window.location.href='/pipeline'">Pipeline</button>
                <button class="nav-btn active" onclick="window.location.href='/training'">Training</button>
                <button class="refresh-btn" onclick="loadTrainingData()">Refresh</button>
            </div>
            
            <!-- Training Quick Actions -->
            <div class="quick-actions">
                <button class="action-btn" onclick="document.getElementById('training-section').scrollIntoView({{behavior: 'smooth'}})">
                    🚀 Start New Training
                </button>
                <button class="action-btn" onclick="document.getElementById('models-section').scrollIntoView({{behavior: 'smooth'}})">
                    📊 View Models
                </button>
                <button class="action-btn" onclick="document.getElementById('analytics-section').scrollIntoView({{behavior: 'smooth'}})">
                    📈 Analytics
                </button>
                <button class="action-btn" onclick="alert('🎓 Training Dashboard Help\\n\\n🚀 Getting Started:\\n1. Configure Data: Select symbols, timeframe, and date range\\n2. Choose Model: Select from Random Forest, XGBoost, etc.\\n3. Set Strategy: Choose classification or regression\\n4. Start Training: Click \\"Start Training\\" to begin\\n\\n📊 Model Management:\\n- View all trained models with performance metrics\\n- Deploy best-performing models for production\\n- Delete underperforming models\\n\\n📈 Analytics:\\n- Track accuracy, precision, recall, and F1-score\\n- Monitor training sessions and model evolution\\n- Compare different models and strategies')">
                    ❓ Help
                </button>
            </div>
        </div>
        
        <!-- Model Training Section -->
        <div class="section" id="training-section">
            <h2>🎯 Model Training</h2>
            
            <div class="config-grid">
                <div class="config-group">
                    <label>Symbols:</label>
                    <input type="text" id="symbols" placeholder="AAPL,MSFT,GOOGL" value="AAPL,MSFT,GOOGL">
                </div>
                <div class="config-group">
                    <label>Timeframe:</label>
                    <select id="timeframe">
                        <option value="1min">1 Minute</option>
                        <option value="5min">5 Minutes</option>
                        <option value="15min">15 Minutes</option>
                        <option value="1hour">1 Hour</option>
                        <option value="1day" selected>1 Day</option>
                    </select>
                </div>
                <div class="config-group">
                    <label>Start Date:</label>
                    <input type="date" id="start-date" value="2024-01-01">
                </div>
                <div class="config-group">
                    <label>End Date:</label>
                    <input type="date" id="end-date" value="2024-12-31">
                </div>
                <div class="config-group">
                    <label>Data Sources:</label>
                    <select id="data-sources" multiple>
                        <option value="yfinance" selected>YFinance</option>
                        <option value="alpha_vantage">Alpha Vantage</option>
                        <option value="polygon">Polygon</option>
                    </select>
                </div>
                <div class="config-group">
                    <label>Model Type:</label>
                    <select id="model-type">
                        <option value="random_forest">Random Forest</option>
                        <option value="xgboost">XGBoost</option>
                        <option value="lightgbm">LightGBM</option>
                        <option value="neural_network">Neural Network</option>
                        <option value="svm">SVM</option>
                        <option value="logistic_regression">Logistic Regression</option>
                    </select>
                </div>
                <div class="config-group">
                    <label>Strategy:</label>
                    <select id="strategy">
                        <option value="classification">Classification (Buy/Sell/Hold)</option>
                        <option value="regression">Regression (Price Prediction)</option>
                        <option value="ensemble">Ensemble</option>
                    </select>
                </div>
                <div class="config-group">
                    <label>Cross-Validation Folds:</label>
                    <input type="number" id="cv-folds" value="5" min="2" max="10">
                </div>
                <div class="config-group">
                    <label>Test Split:</label>
                    <input type="number" id="test-split" value="0.2" min="0.1" max="0.5" step="0.1">
                </div>
            </div>
            
            <div class="config-group">
                <label>Hyperparameters (JSON):</label>
                <textarea id="hyperparameters" placeholder='{{"n_estimators": 100, "max_depth": 10, "random_state": 42}}' rows="4">{{"n_estimators": 100, "max_depth": 10, "random_state": 42}}</textarea>
            </div>
            
            <div class="training-controls">
                <button id="start-training" class="btn-primary" onclick="startTraining()">🚀 Start Training</button>
                <button id="stop-training" class="btn-secondary" onclick="stopTraining()" disabled>⏹️ Stop Training</button>
                <button id="save-config" class="btn-outline" onclick="saveConfiguration()">💾 Save Configuration</button>
                <button id="load-config" class="btn-outline" onclick="loadConfiguration()">📂 Load Configuration</button>
            </div>
            
            <div id="progress-section" class="progress-section" style="display: none;">
                <h4>📈 Training Progress</h4>
                <div class="progress-bar">
                    <div class="progress-fill" id="training-progress"></div>
                </div>
                <div class="progress-details">
                    <span id="current-epoch">Epoch: 0/100</span>
                    <span id="current-loss">Loss: 0.000</span>
                    <span id="current-accuracy">Accuracy: 0.00%</span>
                </div>
                <div id="training-log"></div>
            </div>
        </div>
        
        <!-- Model Management Section -->
        <div class="section" id="models-section">
            <h2>📊 Model Management</h2>
            <div id="models-container">Loading models...</div>
        </div>
        
        <!-- Training Analytics Section -->
        <div class="section" id="analytics-section">
            <h2>📈 Training Analytics</h2>
            <div id="analytics-container">Loading analytics...</div>
        </div>
    </div>
    
    <script>
        // Load training data on page load
        document.addEventListener('DOMContentLoaded', function() {{
            loadTrainingData();
        }});
        
        function loadTrainingData() {{
            loadModels();
            loadAnalytics();
        }}
        
        function loadModels() {{
            fetch('/api/training/models')
                .then(response => response.json())
                .then(data => {{
                    const container = document.getElementById('models-container');
                    if (data.models && data.models.length > 0) {{
                        container.innerHTML = data.models.map(model => formatModelCard(model)).join('');
                    }} else {{
                        container.innerHTML = '<div class="message info">No trained models found. Start training to create your first model!</div>';
                    }}
                }})
                .catch(error => {{
                    console.error('Error loading models:', error);
                    document.getElementById('models-container').innerHTML = '<div class="message error">Error loading models</div>';
                }});
        }}
        
        function formatModelCard(model) {{
            return `
                <div class="model-card">
                    <div class="model-header">
                        <div class="model-name">${{model.name}}</div>
                        <div class="model-actions">
                            <button class="btn-small btn-view" onclick="viewModel('${{model.id}}')">View</button>
                            <button class="btn-small btn-deploy" onclick="deployModel('${{model.id}}')">Deploy</button>
                            <button class="btn-small btn-delete" onclick="deleteModel('${{model.id}}')">Delete</button>
                        </div>
                    </div>
                    <div class="model-metrics">
                        <div class="metric">
                            <div class="metric-value">${{model.accuracy}}%</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${{model.precision}}</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${{model.recall}}</div>
                            <div class="metric-label">Recall</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${{model.f1_score}}</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                    </div>
                    <div style="color: #666; font-size: 0.9em;">
                        <strong>Type:</strong> ${{model.type}} | <strong>Strategy:</strong> ${{model.strategy}} | <strong>Created:</strong> ${{model.created_at}}
                    </div>
                </div>
            `;
        }}
        
        function loadAnalytics() {{
            fetch('/api/training/analytics')
                .then(response => response.json())
                .then(data => {{
                    const container = document.getElementById('analytics-container');
                    if (data.analytics) {{
                        const analytics = data.analytics;
                        container.innerHTML = `
                            <div class="analytics-grid">
                                <div class="analytics-card">
                                    <div class="analytics-value">${{analytics.total_models}}</div>
                                    <div class="analytics-label">Total Models</div>
                                </div>
                                <div class="analytics-card">
                                    <div class="analytics-value">${{analytics.avg_accuracy}}%</div>
                                    <div class="analytics-label">Avg Accuracy</div>
                                </div>
                                <div class="analytics-card">
                                    <div class="analytics-value">${{analytics.training_sessions}}</div>
                                    <div class="analytics-label">Training Sessions</div>
                                </div>
                                <div class="analytics-card">
                                    <div class="analytics-value">${{analytics.deployed_models}}</div>
                                    <div class="analytics-label">Deployed Models</div>
                                </div>
                            </div>
                        `;
                    }} else {{
                        container.innerHTML = '<div class="message info">No analytics data available</div>';
                    }}
                }})
                .catch(error => {{
                    console.error('Error loading analytics:', error);
                    document.getElementById('analytics-container').innerHTML = '<div class="message error">Error loading analytics</div>';
                }});
        }}
        
        function startTraining() {{
            const config = {{
                symbols: document.getElementById('symbols').value,
                timeframe: document.getElementById('timeframe').value,
                start_date: document.getElementById('start-date').value,
                end_date: document.getElementById('end-date').value,
                data_sources: Array.from(document.getElementById('data-sources').selectedOptions).map(opt => opt.value),
                model_type: document.getElementById('model-type').value,
                strategy: document.getElementById('strategy').value,
                cv_folds: parseInt(document.getElementById('cv-folds').value),
                test_split: parseFloat(document.getElementById('test-split').value),
                hyperparameters: JSON.parse(document.getElementById('hyperparameters').value)
            }};
            
            fetch('/api/training/start', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify(config)
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    showMessage('Training started successfully!', 'success');
                    document.getElementById('start-training').disabled = true;
                    document.getElementById('stop-training').disabled = false;
                    startProgressMonitoring();
                }} else {{
                    showMessage('Failed to start training: ' + data.error, 'error');
                }}
            }})
            .catch(error => {{
                console.error('Error starting training:', error);
                showMessage('Error starting training', 'error');
            }});
        }}
        
        function stopTraining() {{
            fetch('/api/training/stop', {{method: 'POST'}})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        showMessage('Training stopped successfully!', 'success');
                        document.getElementById('start-training').disabled = false;
                        document.getElementById('stop-training').disabled = true;
                        document.getElementById('progress-section').style.display = 'none';
                    }} else {{
                        showMessage('Failed to stop training: ' + data.error, 'error');
                    }}
                }})
                .catch(error => {{
                    console.error('Error stopping training:', error);
                    showMessage('Error stopping training', 'error');
                }});
        }}
        
        function startProgressMonitoring() {{
            document.getElementById('progress-section').style.display = 'block';
            const progressInterval = setInterval(() => {{
                fetch('/api/training/progress')
                    .then(response => response.json())
                    .then(data => {{
                        if (data.progress) {{
                            const progress = data.progress;
                            document.getElementById('training-progress').style.width = progress.percentage + '%';
                            document.getElementById('current-epoch').textContent = `Epoch: ${{progress.current_epoch}}/${{progress.total_epochs}}`;
                            document.getElementById('current-loss').textContent = `Loss: ${{progress.current_loss.toFixed(3)}}`;
                            document.getElementById('current-accuracy').textContent = `Accuracy: ${{progress.current_accuracy.toFixed(2)}}%`;
                            
                            if (progress.log) {{
                                const logElement = document.getElementById('training-log');
                                logElement.innerHTML += progress.log + '<br>';
                                logElement.scrollTop = logElement.scrollHeight;
                            }}
                            
                            if (progress.percentage >= 100) {{
                                clearInterval(progressInterval);
                                showMessage('Training completed successfully!', 'success');
                                loadTrainingData();
                            }}
                        }}
                    }})
                    .catch(error => {{
                        console.error('Error monitoring progress:', error);
                    }});
            }}, 1000);
        }}
        
        function deployModel(modelId) {{
            fetch('/api/training/deploy', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{model_id: modelId}})
            }})
            .then(response => response.json())
            .then(data => {{
                if (data.success) {{
                    showMessage('Model deployed successfully!', 'success');
                    loadTrainingData();
                }} else {{
                    showMessage('Failed to deploy model: ' + data.error, 'error');
                }}
            }})
            .catch(error => {{
                console.error('Error deploying model:', error);
                showMessage('Error deploying model', 'error');
            }});
        }}
        
        function deleteModel(modelId) {{
            if (confirm('Are you sure you want to delete this model?')) {{
                fetch('/api/training/delete', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{model_id: modelId}})
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        showMessage('Model deleted successfully!', 'success');
                        loadTrainingData();
                    }} else {{
                        showMessage('Failed to delete model: ' + data.error, 'error');
                    }}
                }})
                .catch(error => {{
                    console.error('Error deleting model:', error);
                    showMessage('Error deleting model', 'error');
                }});
            }}
        }}
        
        function viewModel(modelId) {{
            alert('View model details for: ' + modelId);
        }}
        
        function saveConfiguration() {{
            const config = {{
                symbols: document.getElementById('symbols').value,
                timeframe: document.getElementById('timeframe').value,
                start_date: document.getElementById('start-date').value,
                end_date: document.getElementById('end-date').value,
                model_type: document.getElementById('model-type').value,
                strategy: document.getElementById('strategy').value,
                hyperparameters: document.getElementById('hyperparameters').value
            }};
            
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(config, null, 2));
            const downloadAnchorNode = document.createElement('a');
            downloadAnchorNode.setAttribute("href", dataStr);
            downloadAnchorNode.setAttribute("download", "training_config.json");
            document.body.appendChild(downloadAnchorNode);
            downloadAnchorNode.click();
            downloadAnchorNode.remove();
            
            showMessage('Configuration saved successfully!', 'success');
        }}
        
        function loadConfiguration() {{
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = function(e) {{
                const file = e.target.files[0];
                const reader = new FileReader();
                reader.onload = function(e) {{
                    try {{
                        const config = JSON.parse(e.target.result);
                        document.getElementById('symbols').value = config.symbols || '';
                        document.getElementById('timeframe').value = config.timeframe || '1day';
                        document.getElementById('start-date').value = config.start_date || '';
                        document.getElementById('end-date').value = config.end_date || '';
                        document.getElementById('model-type').value = config.model_type || 'random_forest';
                        document.getElementById('strategy').value = config.strategy || 'classification';
                        document.getElementById('hyperparameters').value = config.hyperparameters || '';
                        showMessage('Configuration loaded successfully!', 'success');
                    }} catch (error) {{
                        showMessage('Error loading configuration file', 'error');
                    }}
                }};
                reader.readAsText(file);
            }};
            input.click();
        }}
        
        function showMessage(message, type) {{
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{type}}`;
            messageDiv.textContent = message;
            document.querySelector('.container').insertBefore(messageDiv, document.querySelector('.section'));
            
            setTimeout(() => {{
                messageDiv.remove();
            }}, 5000);
        }}
        
        // Auto-refresh every 30 seconds
        setInterval(loadTrainingData, 30000);
    </script>
</body>
</html>
        """
