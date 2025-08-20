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
                self.wfile.write(json.dumps(response).encode())
                
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
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def execute_command(self, command, parameters):
        """Execute a command with parameters"""
        # Map commands to actual CLI commands
        command_map = {
            'data_summary': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'data', 'summary'],
            'data_fetch': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'data', 'fetch', 
                          '--symbols', parameters.get("symbols", "AAPL,MSFT"),
                          '--start-date', parameters.get("start_date", "2024-08-15"),
                          '--end-date', parameters.get("end_date", "2024-08-16")],
            'signal_generate': ['python3', '/opt/bitnami/spark/jobs/cli/bf_minio.py', 'signals', 'generate',
                               '--symbols', parameters.get("symbols", "AAPL,MSFT"),
                               '--start-date', parameters.get("start_date", "2024-08-15"),
                               '--end-date', parameters.get("end_date", "2024-08-16")],
            'signal_summary': ['python3', '/opt/bitnami/spark/jobs/cli/bf_minio.py', 'signals', 'summary'],
            'backtest_run': ['python3', '/opt/bitnami/spark/jobs/cli/bf_minio.py', 'backtest', 'run',
                            '--symbols', parameters.get("symbols", "AAPL,MSFT"),
                            '--from-date', parameters.get("from_date", "2024-08-15"),
                            '--to-date', parameters.get("to_date", "2024-08-16"),
                            '--initial-capital', parameters.get("initial_capital", "100000")]
        }
        
        if command not in command_map:
            raise ValueError(f"Unknown command: {command}")
        
        # Execute the command
        cmd = command_map[command]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            return result.stdout
        else:
            raise Exception(f"Command failed: {result.stderr}")
    
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
