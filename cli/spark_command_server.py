#!/usr/bin/env python3
"""
Spark Command Server
Simple HTTP server to receive and execute commands from the dashboard
"""

import os
import sys
import json
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests - health check"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"status": "healthy", "server": "spark-command-server"}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests - execute commands"""
        if self.path == '/execute':
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                command = data.get('command')
                parameters = data.get('parameters', {})
                
                if not command:
                    raise ValueError("No command specified")
                
                # Execute the command
                result = self.execute_command(command, parameters)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "success": True,
                    "command": command,
                    "output": result,
                    "timestamp": time.time()
                }
                response_data = json.dumps(response).encode('utf-8')
                self.send_header('Content-Length', str(len(response_data)))
                self.wfile.write(response_data)
                
            except Exception as e:
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
                response_data = json.dumps(response).encode('utf-8')
                self.send_header('Content-Length', str(len(response_data)))
                self.wfile.write(response_data)
        else:
            self.send_response(404)
            self.end_headers()
    
    def execute_command(self, command, parameters):
        """Execute a command with parameters"""
        # Map commands to actual CLI commands with timeframe support
        command_map = {
            'data_summary': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'data', 'summary'],
            'data_fetch': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'data', 'fetch', 
                          '--symbols', parameters.get("symbols", "AAPL,MSFT"),
                          '--start-date', parameters.get("start_date", "2024-08-15"),
                          '--end-date', parameters.get("end_date", "2024-08-16"),
                          '--timeframe', parameters.get("timeframe", "1day"),
                          '--data-source', parameters.get("data_source", "yfinance")],
            'signal_generate': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'signals', 'generate',
                               '--symbols', parameters.get("symbols", "AAPL,MSFT"),
                               '--start-date', parameters.get("start_date", "2024-08-15"),
                               '--end-date', parameters.get("end_date", "2024-08-16"),
                               '--timeframe', parameters.get("timeframe", "1day")],
            'signal_summary': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'signals', 'summary'],
            'backtest_run': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'backtest', 'run',
                            '--symbols', parameters.get("symbols", "AAPL,MSFT"),
                            '--from-date', parameters.get("from_date", "2024-08-15"),
                            '--to-date', parameters.get("to_date", "2024-08-16"),
                            '--timeframe', parameters.get("timeframe", "1day"),
                            '--initial-capital', parameters.get("initial_capital", "100000")],
            # New pipeline commands with timeframe support
            'pipeline_start': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'pipeline', 'start',
                              '--mode', parameters.get("mode", "demo"),
                              '--interval', parameters.get("interval", "300"),
                              '--timeframe', parameters.get("timeframe", "1day"),
                              '--symbols', parameters.get("symbols", ""),
                              '--data-source', parameters.get("data_source", "yfinance")],
            'pipeline_stop': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'pipeline', 'stop'],
            'pipeline_status': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'pipeline', 'status'],
            'pipeline_logs': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'pipeline', 'logs',
                             '--lines', parameters.get("lines", "20")],
            'pipeline_run': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'pipeline', 'run',
                            '--mode', parameters.get("mode", "demo"),
                            '--interval', parameters.get("interval", "300"),
                            '--timeframe', parameters.get("timeframe", "1day"),
                            '--cycles', parameters.get("cycles", "3")]
        }
        
        # Special commands that use methods
        special_commands = {
            'simple_pipeline_run': self._execute_simple_pipeline,
            'cron_pipeline_start': self._start_continuous_pipeline_cron,
            'cron_pipeline_stop': self._stop_continuous_pipeline_cron,
            'test_continuous': self._test_continuous,
            'spark_streaming_start': self._start_spark_streaming_pipeline,
            'spark_streaming_stop': self._stop_spark_streaming_pipeline,
            'spark_streaming_status': self._get_spark_streaming_status
        }
        
        # Check if it's a special command
        if command in special_commands:
            return special_commands[command](parameters)
        
        # Check if it's a regular command
        if command not in command_map:
            raise ValueError(f"Unknown command: {command}")
        
        # Execute regular command
        cmd = command_map[command]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(f"Command failed: {result.stderr}")
    
    def _execute_simple_pipeline(self, parameters):
        """Execute a simple pipeline: data fetch -> signals -> backtest"""
        import uuid
        import sys
        sys.path.insert(0, '/opt/bitnami/spark/jobs')
        from cli.kibana_enhanced_bf import DualLogger
        
        symbols = parameters.get("symbols", "AAPL,MSFT")
        timeframe = parameters.get("timeframe", "1day")
        start_date = parameters.get("start_date", "2025-08-20")
        end_date = parameters.get("end_date", "2025-08-23")
        data_source = parameters.get("data_source", "yfinance")
        
        # Create logger for the simple pipeline
        run_id = str(uuid.uuid4())
        dual_logger = DualLogger(run_id, f"simple_pipeline_run --symbols {symbols} --timeframe {timeframe}")
        
        output = []
        output.append("🚀 Starting Simple Pipeline")
        output.append("=" * 50)
        
        # Step 1: Data Fetch
        output.append("📥 Step 1: Data Fetch")
        dual_logger.log("INFO", "📥 Step 1: Data Fetch")
        try:
            fetch_cmd = ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'data', 'fetch',
                        '--symbols', symbols,
                        '--start-date', start_date,
                        '--end-date', end_date,
                        '--timeframe', timeframe,
                        '--data-source', data_source]
            
            result = subprocess.run(fetch_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                output.append("✅ Data Fetch completed successfully")
                dual_logger.log("INFO", "✅ Data Fetch completed successfully")
            else:
                output.append(f"❌ Data Fetch failed: {result.stderr}")
                dual_logger.log("ERROR", f"❌ Data Fetch failed: {result.stderr}")
                dual_logger.complete('failed')
                return "\n".join(output)
        except Exception as e:
            output.append(f"❌ Data Fetch error: {str(e)}")
            dual_logger.log("ERROR", f"❌ Data Fetch error: {str(e)}")
            dual_logger.complete('failed')
            return "\n".join(output)
        
        # Step 2: Signal Generation
        output.append("📊 Step 2: Signal Generation")
        dual_logger.log("INFO", "📊 Step 2: Signal Generation")
        try:
            signals_cmd = ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'signals', 'generate',
                          '--symbols', symbols,
                          '--start-date', start_date,
                          '--end-date', end_date,
                          '--timeframe', timeframe]
            
            result = subprocess.run(signals_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                output.append("✅ Signal Generation completed successfully")
                dual_logger.log("INFO", "✅ Signal Generation completed successfully")
            else:
                output.append(f"❌ Signal Generation failed: {result.stderr}")
                dual_logger.log("ERROR", f"❌ Signal Generation failed: {result.stderr}")
                dual_logger.complete('failed')
                return "\n".join(output)
        except Exception as e:
            output.append(f"❌ Signal Generation error: {str(e)}")
            dual_logger.log("ERROR", f"❌ Signal Generation error: {str(e)}")
            dual_logger.complete('failed')
            return "\n".join(output)
        
        # Step 3: Backtesting
        output.append("📈 Step 3: Backtesting")
        dual_logger.log("INFO", "📈 Step 3: Backtesting")
        try:
            backtest_cmd = ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'backtest', 'run',
                           '--symbols', symbols,
                           '--from-date', start_date,
                           '--to-date', end_date,
                           '--timeframe', timeframe,
                           '--initial-capital', '100000']
            
            result = subprocess.run(backtest_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                output.append("✅ Backtesting completed successfully")
                dual_logger.log("INFO", "✅ Backtesting completed successfully")
            else:
                output.append(f"❌ Backtesting failed: {result.stderr}")
                dual_logger.log("ERROR", f"❌ Backtesting failed: {result.stderr}")
                dual_logger.complete('failed')
                return "\n".join(output)
        except Exception as e:
            output.append(f"❌ Backtesting error: {str(e)}")
            dual_logger.log("ERROR", f"❌ Backtesting error: {str(e)}")
            dual_logger.complete('failed')
            return "\n".join(output)
        
        output.append("=" * 50)
        output.append("🎉 Simple Pipeline completed successfully!")
        dual_logger.log("INFO", "🎉 Simple Pipeline completed successfully!")
        dual_logger.complete('completed')
        
        return "\n".join(output)
    
    def _start_continuous_pipeline_cron(self, parameters):
        """Start a continuous pipeline using cron scheduling"""
        import subprocess
        import os
        
        symbols = parameters.get("symbols", "AAPL,MSFT")
        timeframe = parameters.get("timeframe", "1day")
        interval_minutes = int(parameters.get("interval_seconds", 60)) // 60  # Convert to minutes
        data_source = parameters.get("data_source", "yfinance")
        
        # Create a unique job ID
        job_id = f"breadthflow_pipeline_{symbols.replace(',', '_')}_{timeframe}"
        
        # Create the cron job command
        cron_command = f"python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols {symbols} --timeframe {timeframe} --data-source {data_source} --start-date 2025-08-20 --end-date 2025-08-23"
        
        # Create cron entry (every X minutes)
        cron_entry = f"*/{interval_minutes} * * * * {cron_command} >> /tmp/{job_id}.log 2>&1"
        
        try:
            # Add to crontab
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            current_crontab = result.stdout if result.returncode == 0 else ""
            
            # Check if job already exists
            if job_id not in current_crontab:
                # Add new cron job
                new_crontab = current_crontab + f"\n{cron_entry}\n"
                
                # Write to temporary file
                with open('/tmp/new_crontab', 'w') as f:
                    f.write(new_crontab)
                
                # Install new crontab
                subprocess.run(['crontab', '/tmp/new_crontab'], check=True)
                
                # Store job info
                self._active_cron_jobs = getattr(self, '_active_cron_jobs', {})
                self._active_cron_jobs[job_id] = {
                    'symbols': symbols,
                    'timeframe': timeframe,
                    'interval_minutes': interval_minutes,
                    'cron_entry': cron_entry
                }
                
                return f"🚀 Continuous pipeline started with cron scheduling\n📊 Symbols: {symbols}\n⏰ Timeframe: {timeframe}\n⏱️ Interval: {interval_minutes} minutes\n🔄 Will run automatically until stopped\n📝 Job ID: {job_id}"
            else:
                return f"❌ Pipeline job '{job_id}' is already running"
                
        except Exception as e:
            return f"❌ Failed to start continuous pipeline: {str(e)}"
    
    def _stop_continuous_pipeline_cron(self, parameters):
        """Stop the continuous pipeline by removing cron job"""
        import subprocess
        
        job_id = parameters.get("job_id", "")
        
        if not job_id:
            return "❌ Please specify job_id to stop"
        
        try:
            # Get current crontab
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode != 0:
                return "❌ No cron jobs found"
            
            current_crontab = result.stdout
            
            # Remove the specific job
            lines = current_crontab.split('\n')
            new_lines = [line for line in lines if job_id not in line and line.strip()]
            
            # Write new crontab
            new_crontab = '\n'.join(new_lines) + '\n'
            
            with open('/tmp/new_crontab', 'w') as f:
                f.write(new_crontab)
            
            subprocess.run(['crontab', '/tmp/new_crontab'], check=True)
            
            # Remove from active jobs
            if hasattr(self, '_active_cron_jobs') and job_id in self._active_cron_jobs:
                del self._active_cron_jobs[job_id]
            
            return f"🛑 Continuous pipeline '{job_id}' stopped successfully"
            
        except Exception as e:
            return f"❌ Failed to stop pipeline: {str(e)}"
    
    def _test_continuous(self, parameters):
        """Simple test method for continuous pipeline"""
        return "✅ Test continuous method works!"
    
    def _start_spark_streaming_pipeline(self, parameters):
        """Start Spark Structured Streaming pipeline"""
        import sys
        sys.path.insert(0, '/opt/bitnami/spark/jobs')
        from cli.spark_streaming_pipeline import SparkStreamingPipeline
        
        symbols = parameters.get("symbols", "AAPL,MSFT").split(",")
        timeframe = parameters.get("timeframe", "1day")
        interval_seconds = int(parameters.get("interval_seconds", 300))  # Default 5 minutes for safety
        data_source = parameters.get("data_source", "yfinance")
        
        # Use existing Spark session from the command server
        # The Spark session is already available in the container
        try:
            # Create and start pipeline using existing Spark context
            pipeline = SparkStreamingPipeline(None)  # Will use existing Spark session
            result = pipeline.start_continuous_pipeline(
                symbols=symbols,
                timeframe=timeframe,
                interval_seconds=interval_seconds,
                data_source=data_source
            )
            
            # Store pipeline instance
            self._spark_streaming_pipeline = pipeline
            
            return result
            
        except Exception as e:
            return f"❌ Failed to start Spark streaming pipeline: {str(e)}"
    
    def _stop_spark_streaming_pipeline(self, parameters):
        """Stop Spark Structured Streaming pipeline"""
        if hasattr(self, '_spark_streaming_pipeline'):
            result = self._spark_streaming_pipeline.stop_pipeline()
            if hasattr(self, '_spark_session'):
                self._spark_session.stop()
            return result
        else:
            return "❌ No Spark streaming pipeline is running"
    
    def _get_spark_streaming_status(self, parameters):
        """Get Spark streaming pipeline status"""
        if hasattr(self, '_spark_streaming_pipeline'):
            status = self._spark_streaming_pipeline.get_status()
            return json.dumps(status, indent=2)
        else:
            return json.dumps({"state": "stopped", "is_running": False}, indent=2)
    
    def _stop_continuous_pipeline(self, parameters):
        """Stop the continuous pipeline"""
        if hasattr(self, '_continuous_pipeline_running') and self._continuous_pipeline_running:
            self._pipeline_stop_event.set()
            cycles_completed = getattr(self, '_pipeline_cycle_count', 0)
            return f"🛑 Stopping continuous pipeline...\n📊 Cycles completed: {cycles_completed}"
        else:
            return "❌ No continuous pipeline is running"
    
    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        pass

def main():
    port = int(os.environ.get('COMMAND_SERVER_PORT', 8081))
    
    server = HTTPServer(('0.0.0.0', port), CommandHandler)
    print(f"🚀 Spark Command Server starting on port {port}")
    print(f"📡 Ready to receive commands from dashboard")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Spark Command Server")
        server.shutdown()

if __name__ == "__main__":
    main()
