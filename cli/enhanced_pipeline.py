#!/usr/bin/env python3
"""
Enhanced Pipeline with Airflow-like Features

This provides some of the benefits of Airflow without the complexity:
- Task history and logging
- Retry logic
- Basic monitoring
- Web dashboard (optional)
"""

import click
import subprocess
import time
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, asdict
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    status: str  # 'success', 'failed', 'running'
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    error_message: Optional[str]
    retry_count: int = 0

@dataclass
class PipelineRun:
    """Result of a pipeline execution"""
    run_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    status: str  # 'success', 'failed', 'running'
    tasks: List[TaskResult]
    symbol_list: str
    interval: int

class PipelineManager:
    """Manages pipeline execution with Airflow-like features"""
    
    def __init__(self, db_path: str = "pipeline.db"):
        self.db_path = db_path
        self.init_database()
        self.runs: List[PipelineRun] = []
        self.current_run: Optional[PipelineRun] = None
        
    def init_database(self):
        """Initialize SQLite database for storing pipeline history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                duration REAL,
                status TEXT,
                symbol_list TEXT,
                interval INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                task_id TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                duration REAL,
                error_message TEXT,
                retry_count INTEGER,
                FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_run(self, symbol_list: str, interval: int) -> PipelineRun:
        """Start a new pipeline run"""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = PipelineRun(
            run_id=run_id,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            status='running',
            tasks=[],
            symbol_list=symbol_list,
            interval=interval
        )
        self.runs.append(self.current_run)
        logger.info(f"Started pipeline run: {run_id}")
        return self.current_run
    
    def end_run(self, status: str = 'success'):
        """End the current pipeline run"""
        if self.current_run:
            self.current_run.end_time = datetime.now()
            self.current_run.duration = (self.current_run.end_time - self.current_run.start_time).total_seconds()
            self.current_run.status = status
            self.save_run_to_db(self.current_run)
            logger.info(f"Ended pipeline run: {self.current_run.run_id} - {status}")
    
    def execute_task(self, task_id: str, command: List[str], max_retries: int = 3) -> TaskResult:
        """Execute a task with retry logic"""
        task = TaskResult(
            task_id=task_id,
            status='running',
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            error_message=None,
            retry_count=0
        )
        
        if self.current_run:
            self.current_run.tasks.append(task)
        
        logger.info(f"Starting task: {task_id}")
        
        for attempt in range(max_retries + 1):
            try:
                task.retry_count = attempt
                task.start_time = datetime.now()
                
                # Execute command
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300  # 5 minute timeout
                )
                
                task.end_time = datetime.now()
                task.duration = (task.end_time - task.start_time).total_seconds()
                task.status = 'success'
                
                logger.info(f"Task {task_id} completed successfully in {task.duration:.2f}s")
                return task
                
            except subprocess.CalledProcessError as e:
                task.error_message = f"Command failed: {e.stderr}"
                logger.error(f"Task {task_id} failed (attempt {attempt + 1}): {e.stderr}")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying task {task_id} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    task.end_time = datetime.now()
                    task.duration = (task.end_time - task.start_time).total_seconds()
                    task.status = 'failed'
                    logger.error(f"Task {task_id} failed after {max_retries + 1} attempts")
                    return task
                    
            except subprocess.TimeoutExpired:
                task.error_message = "Task timed out after 5 minutes"
                logger.error(f"Task {task_id} timed out (attempt {attempt + 1})")
                
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying task {task_id} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    task.end_time = datetime.now()
                    task.duration = (task.end_time - task.start_time).total_seconds()
                    task.status = 'failed'
                    logger.error(f"Task {task_id} failed after {max_retries + 1} attempts")
                    return task
    
    def save_run_to_db(self, run: PipelineRun):
        """Save pipeline run to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save pipeline run
        cursor.execute('''
            INSERT OR REPLACE INTO pipeline_runs 
            (run_id, start_time, end_time, duration, status, symbol_list, interval)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            run.run_id,
            run.start_time.isoformat(),
            run.end_time.isoformat() if run.end_time else None,
            run.duration,
            run.status,
            run.symbol_list,
            run.interval
        ))
        
        # Save task results
        for task in run.tasks:
            cursor.execute('''
                INSERT INTO task_results 
                (run_id, task_id, status, start_time, end_time, duration, error_message, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run.run_id,
                task.task_id,
                task.status,
                task.start_time.isoformat(),
                task.end_time.isoformat() if task.end_time else None,
                task.duration,
                task.error_message,
                task.retry_count
            ))
        
        conn.commit()
        conn.close()
    
    def get_run_history(self, limit: int = 10) -> List[Dict]:
        """Get recent pipeline run history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT run_id, start_time, end_time, duration, status, symbol_list, interval
            FROM pipeline_runs
            ORDER BY start_time DESC
            LIMIT ?
        ''', (limit,))
        
        runs = []
        for row in cursor.fetchall():
            runs.append({
                'run_id': row[0],
                'start_time': row[1],
                'end_time': row[2],
                'duration': row[3],
                'status': row[4],
                'symbol_list': row[5],
                'interval': row[6]
            })
        
        conn.close()
        return runs
    
    def get_task_history(self, run_id: str) -> List[Dict]:
        """Get task history for a specific run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT task_id, status, start_time, end_time, duration, error_message, retry_count
            FROM task_results
            WHERE run_id = ?
            ORDER BY start_time
        ''', (run_id,))
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                'task_id': row[0],
                'status': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'duration': row[4],
                'error_message': row[5],
                'retry_count': row[6]
            })
        
        conn.close()
        return tasks

class DashboardHandler(BaseHTTPRequestHandler):
    """Simple HTTP server for dashboard"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Get pipeline history
            manager = PipelineManager()
            runs = manager.get_run_history(20)
            
            html = self.generate_dashboard(runs)
            self.wfile.write(html.encode())
            
        elif self.path == '/api/runs':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            manager = PipelineManager()
            runs = manager.get_run_history(50)
            self.wfile.write(json.dumps(runs).encode())
            
        elif self.path.startswith('/api/tasks/'):
            run_id = self.path.split('/')[-1]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            manager = PipelineManager()
            tasks = manager.get_task_history(run_id)
            self.wfile.write(json.dumps(tasks).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def generate_dashboard(self, runs):
        """Generate HTML dashboard"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Breadth/Thrust Pipeline Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .run { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .success { border-left: 5px solid #4CAF50; }
                .failed { border-left: 5px solid #f44336; }
                .running { border-left: 5px solid #2196F3; }
                .task { margin: 5px 0; padding: 5px; background: #f9f9f9; }
                .task.success { background: #e8f5e8; }
                .task.failed { background: #ffe8e8; }
                .refresh { margin: 20px 0; }
                button { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 3px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Breadth/Thrust Pipeline Dashboard</h1>
                <p>Enhanced CLI Pipeline with Airflow-like Features</p>
                <div class="refresh">
                    <button onclick="location.reload()">🔄 Refresh</button>
                </div>
            </div>
        """
        
        for run in runs:
            status_class = run['status']
            duration = f"{run['duration']:.2f}s" if run['duration'] else "N/A"
            
            html += f"""
            <div class="run {status_class}">
                <h3>🎯 Run: {run['run_id']}</h3>
                <p><strong>Status:</strong> {run['status'].upper()}</p>
                <p><strong>Symbol List:</strong> {run['symbol_list']}</p>
                <p><strong>Duration:</strong> {duration}</p>
                <p><strong>Start:</strong> {run['start_time']}</p>
                <p><strong>End:</strong> {run['end_time'] or 'Running...'}</p>
                <button onclick="showTasks('{run['run_id']}')">📊 View Tasks</button>
                <div id="tasks-{run['run_id']}" style="display:none;"></div>
            </div>
            """
        
        html += """
        <script>
        function showTasks(runId) {
            const taskDiv = document.getElementById('tasks-' + runId);
            if (taskDiv.style.display === 'none') {
                fetch('/api/tasks/' + runId)
                    .then(response => response.json())
                    .then(tasks => {
                        let taskHtml = '<h4>Tasks:</h4>';
                        tasks.forEach(task => {
                            const duration = task.duration ? task.duration.toFixed(2) + 's' : 'N/A';
                            taskHtml += `
                                <div class="task ${task.status}">
                                    <strong>${task.task_id}</strong> - ${task.status.toUpperCase()}
                                    <br>Duration: ${duration} | Retries: ${task.retry_count}
                                    ${task.error_message ? '<br>Error: ' + task.error_message : ''}
                                </div>
                            `;
                        });
                        taskDiv.innerHTML = taskHtml;
                        taskDiv.style.display = 'block';
                    });
            } else {
                taskDiv.style.display = 'none';
            }
        }
        </script>
        </body>
        </html>
        """
        
        return html

def start_dashboard_server(host='localhost', port=8081):
    """Start the dashboard server standalone"""
    server = HTTPServer((host, port), DashboardHandler)
    print(f"🌐 Dashboard started at http://{host}:{port}")
    print("📊 Features:")
    print("  • Pipeline run history")
    print("  • Task success/failure tracking")
    print("  • Duration metrics")
    print("  • Error messages and retry counts")
    print()
    print("🌐 Dashboard will run until stopped")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
        server.shutdown()


@click.command()
@click.option('--symbol-list', default='demo_small', help='Symbol list to use')
@click.option('--interval', default=300, help='Interval between runs in seconds')
def enhanced_pipeline(symbol_list, interval):
    """Enhanced pipeline with Airflow-like features"""
    
    manager = PipelineManager()
    
    print("🔄 Enhanced Pipeline Mode")
    print("=" * 50)
    print(f"📊 Symbol List: {symbol_list}")
    print(f"⏰ Interval: {interval} seconds")
    print()
    print("🔄 Pipeline will run continuously until stopped")
    print("⏹️  Press Ctrl+C to stop the pipeline")
    print()
    
    run_count = 0
    
    try:
        while True:
            run_count += 1
            
            # Start pipeline run
            run = manager.start_run(symbol_list, interval)
            print(f"\n🔄 Pipeline Run #{run_count}")
            print(f"⏰ Started at: {run.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)
            
            # Execute tasks with retry logic (using simple fetcher to avoid Java requirement)
            tasks = [
                ("fetch_data", ["poetry", "run", "bf", "data", "fetch", "--symbol-list", symbol_list]),
                ("generate_signals", ["poetry", "run", "bf", "signals", "generate", "--symbol-list", symbol_list]),
                ("run_backtest", ["poetry", "run", "bf", "backtest", "run", "--symbol-list", symbol_list, "--from-date", "2024-01-01", "--to-date", "2024-12-31"])
            ]
            
            all_success = True
            
            for task_id, command in tasks:
                result = manager.execute_task(task_id, command, max_retries=2)
                if result.status == 'failed':
                    all_success = False
                    print(f"❌ Task {task_id} failed: {result.error_message}")
                else:
                    print(f"✅ Task {task_id} completed in {result.duration:.2f}s")
            
            # End pipeline run
            status = 'success' if all_success else 'failed'
            manager.end_run(status)
            
            print(f"✅ Pipeline run #{run_count} completed - {status.upper()}")
            
            # Wait for next run
            if interval > 0:
                print(f"⏳ Waiting {interval} seconds until next run...")
                time.sleep(interval)
            else:
                print("🔄 Running immediately (no interval)")
                
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline stopped by user")
        if manager.current_run:
            manager.end_run('stopped')
        print(f"📊 Total runs completed: {run_count}")

if __name__ == '__main__':
    enhanced_pipeline()
