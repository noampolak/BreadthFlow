#!/usr/bin/env python3
"""
Simple Spark Command Server for BreadthFlow
Provides a basic HTTP interface for Spark commands
"""

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkCommandHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Spark commands"""

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/health":
            self._handle_health()
        elif parsed_path.path == "/status":
            self._handle_status()
        else:
            self._handle_not_found()

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/execute":
            self._handle_execute()
        else:
            self._handle_not_found()

    def _handle_health(self):
        """Handle health check requests"""
        try:
            response = {"status": "healthy", "service": "spark-command-server", "version": "1.0.0", "timestamp": time.time()}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            logger.info("Health check requested - returning healthy status")

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            self.send_response(500)
            self.end_headers()

    def _handle_status(self):
        """Handle status requests"""
        try:
            response = {
                "status": "running",
                "service": "spark-command-server",
                "uptime": time.time() - start_time,
                "timestamp": time.time(),
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            logger.info("Status requested - returning running status")

        except Exception as e:
            logger.error(f"Error in status check: {e}")
            self.send_response(500)
            self.end_headers()

    def _handle_execute(self):
        """Handle command execution requests"""
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            command_data = json.loads(post_data.decode("utf-8"))

            # Simple command processing
            response = {
                "status": "accepted",
                "command_id": f"cmd_{int(time.time())}",
                "message": "Command accepted for processing",
                "timestamp": time.time(),
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            logger.info(f"Command execution requested: {command_data}")

        except Exception as e:
            logger.error(f"Error in command execution: {e}")
            self.send_response(500)
            self.end_headers()

    def _handle_not_found(self):
        """Handle 404 requests"""
        response = {"error": "Not Found", "message": "The requested endpoint was not found", "timestamp": time.time()}

        self.send_response(404)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"{self.address_string()} - {format % args}")


def start_server(host="0.0.0.0", port=8081):
    """Start the Spark Command Server"""
    global start_time
    start_time = time.time()

    server = HTTPServer((host, port), SparkCommandHandler)
    logger.info(f"🚀 Starting Spark Command Server on {host}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Spark Command Server...")
        server.shutdown()


if __name__ == "__main__":
    start_server()
