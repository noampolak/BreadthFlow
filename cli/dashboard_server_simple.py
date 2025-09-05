#!/usr/bin/env python3
"""
Simplified BreadthFlow Dashboard Server
Uses extracted templates and static files
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# Add the cli directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from template_renderer import TemplateRenderer
from static_server import StaticFileServer
from handlers.dashboard_handler import DashboardHandler as DashboardPageHandler
from handlers.commands_handler import CommandsHandler
from handlers.pipeline_handler import PipelineHandler
from handlers.training_handler import TrainingHandler
from handlers.parameters_handler import ParametersHandler
from handlers.signals_handler import SignalsHandler
from handlers.api_handler import APIHandler

class DashboardServer(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.template_renderer = TemplateRenderer()
        self.static_server = StaticFileServer()
        self.dashboard_handler = DashboardPageHandler()
        self.commands_handler = CommandsHandler()
        self.pipeline_handler = PipelineHandler()
        self.training_handler = TrainingHandler()
        self.parameters_handler = ParametersHandler()
        self.signals_handler = SignalsHandler()
        self.api_handler = APIHandler()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            # Handle static files first
            if self.static_server.is_static_request(self.path):
                self._serve_static_file()
                return
            
            # Route to appropriate handler
            if self.path == '/':
                self._serve_dashboard()
            elif self.path == '/infrastructure':
                self._serve_infrastructure()
            elif self.path == '/trading':
                self._serve_trading()
            elif self.path == '/commands':
                self._serve_commands()
            elif self.path == '/pipeline':
                self._serve_pipeline_management()
            elif self.path == '/training':
                self._serve_training()
            elif self.path == '/parameters':
                self._serve_parameters()
            elif self.path.startswith('/api/'):
                self._serve_api(self.path)
            else:
                self.send_error(404, "Page not found")
                
        except Exception as e:
            print(f"Error handling GET request: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            if self.path.startswith('/api/'):
                if self.path == '/api/pipeline/start':
                    # Read request body for pipeline start
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length)
                    content, content_type, status_code = self.api_handler.serve_pipeline_start_api(post_data.decode('utf-8'))
                elif self.path == '/api/pipeline/stop':
                    content, content_type, status_code = self.api_handler.serve_pipeline_stop_api()
                else:
                    # For other API endpoints, use the standard handler
                    content, content_type, status_code = self.api_handler.serve_training_api(self.path)
                
                self.send_response(status_code)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            else:
                self.send_error(404, "Endpoint not found")
                
        except Exception as e:
            print(f"Error handling POST request: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def _serve_static_file(self):
        """Serve static files (CSS, JS)"""
        content, content_type, status_code = self.static_server.serve_file(self.path)
        
        if status_code == 200:
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(status_code, "File not found")
    
    def _serve_dashboard(self):
        """Serve the main dashboard page"""
        content, content_type, status_code = self.dashboard_handler.serve_dashboard()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_infrastructure(self):
        """Serve the infrastructure page"""
        content, content_type, status_code = self.dashboard_handler.serve_infrastructure()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_trading(self):
        """Serve the trading signals page"""
        content, content_type, status_code = self.dashboard_handler.serve_trading()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_commands(self):
        """Serve the commands page"""
        content, content_type, status_code = self.commands_handler.serve_commands()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_pipeline_management(self):
        """Serve the pipeline management page"""
        content, content_type, status_code = self.pipeline_handler.serve_pipeline_management()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_training(self):
        """Serve the training page"""
        content, content_type, status_code = self.training_handler.serve_training()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_parameters(self):
        """Serve the parameters page"""
        content, content_type, status_code = self.parameters_handler.serve_parameters()
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _serve_api(self, path):
        """Serve API endpoints"""
        if path == '/api/summary':
            content, content_type, status_code = self.api_handler.serve_summary_api()
        elif path.startswith('/api/runs'):
            content, content_type, status_code = self.api_handler.serve_runs_api(path)
        elif path.startswith('/api/run/'):
            content, content_type, status_code = self.api_handler.serve_run_details_api(path)
        elif path.startswith('/api/training'):
            content, content_type, status_code = self.api_handler.serve_training_api(path)
        elif path == '/api/pipeline/status':
            content, content_type, status_code = self.api_handler.serve_pipeline_status_api()
        elif path == '/api/pipeline/runs':
            content, content_type, status_code = self.api_handler.serve_pipeline_runs_api()
        elif path == '/api/signals/latest':
            content, content_type, status_code = self.api_handler.serve_signals_latest_api()
        elif path.startswith('/api/signals/export'):
            content, content_type, status_code = self.api_handler.serve_signals_export_api(path)
        else:
            self.send_error(404, "API endpoint not found")
            return
        
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))

def run_server(port=8003):
    """Run the dashboard server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardServer)
    print(f"Dashboard server running on port {port}")
    print(f"Open http://localhost:{port} in your browser")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
