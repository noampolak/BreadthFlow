#!/usr/bin/env python3
"""
Pipeline Controller Module
Handles pipeline start, stop, and status management
"""

import json
import os
import psycopg2
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://pipeline:pipeline123@postgres:5432/breadthflow')

class PipelineController:
    """Handles pipeline operations and state management"""
    
    def __init__(self):
        self.spark_server_url = "http://spark-master:8081/execute"
    
    def get_db_connection(self):
        """Get PostgreSQL database connection"""
        try:
            return psycopg2.connect(DATABASE_URL)
        except Exception as e:
            print(f"Database connection error: {e}")
            return None
    
    def is_pipeline_running(self) -> bool:
        """Check if any pipeline is currently running"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM pipeline_runs 
                WHERE status = 'running' AND (command LIKE '%spark_streaming_pipeline%' OR command LIKE '%pipeline%')
            ''')
            
            running_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return running_count > 0
            
        except Exception as e:
            print(f"Error checking pipeline status: {e}")
            return False
    
    def get_running_pipeline_id(self) -> Optional[str]:
        """Get the ID of the currently running pipeline"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return None
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT run_id FROM pipeline_runs 
                WHERE status = 'running' AND (command LIKE '%spark_streaming_pipeline%' OR command LIKE '%pipeline%')
                ORDER BY start_time DESC
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error getting running pipeline ID: {e}")
            return None
    
    def start_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new pipeline"""
        try:
            # Check if pipeline is already running
            if self.is_pipeline_running():
                return {
                    "success": False,
                    "error": "A pipeline is already running. Please stop the current pipeline before starting a new one."
                }
            
            # Generate unique pipeline ID
            pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Note: Database record will be created by the Spark command server
            # when it executes the pipeline, to avoid duplicate records
            
            # Call Spark command server
            try:
                command_data = {
                    "command": "spark_streaming_start",
                    "parameters": {
                        "pipeline_id": pipeline_id,
                        "symbols": config.get('symbols', 'AAPL,MSFT'),
                        "timeframe": config.get('timeframe', '1day'),
                        "interval_seconds": 60,
                        "data_source": config.get('data_source', 'yfinance')
                    }
                }
                
                req = urllib.request.Request(
                    self.spark_server_url,
                    data=json.dumps(command_data).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    
                    if result.get("success"):
                        return {
                            "success": True,
                            "message": f"Pipeline started successfully with ID: {pipeline_id}",
                            "pipeline_id": pipeline_id,
                            "config": config
                        }
                    else:
                        # Update pipeline status to failed
                        self._update_pipeline_status(pipeline_id, 'failed', result.get('error', 'Unknown error'))
                        return {
                            "success": False,
                            "error": f"Pipeline start failed: {result.get('error')}"
                        }
                        
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                # Update pipeline status to failed
                self._update_pipeline_status(pipeline_id, 'failed', str(e))
                return {
                    "success": False,
                    "error": f"Cannot connect to pipeline server: {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start pipeline: {str(e)}"
            }
    
    def stop_pipeline(self) -> Dict[str, Any]:
        """Stop the currently running pipeline"""
        try:
            # Get running pipeline ID
            running_pipeline_id = self.get_running_pipeline_id()
            if not running_pipeline_id:
                return {
                    "success": False,
                    "error": "No pipeline is currently running"
                }
            
            # Call Spark command server to stop
            try:
                command_data = {
                    "command": "pipeline_stop",
                    "parameters": {
                        "pipeline_id": running_pipeline_id
                    }
                }
                
                req = urllib.request.Request(
                    self.spark_server_url,
                    data=json.dumps(command_data).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                
                with urllib.request.urlopen(req, timeout=15) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    
                    if result.get("success"):
                        # Update pipeline status to stopped
                        self._update_pipeline_status(running_pipeline_id, 'stopped')
                        return {
                            "success": True,
                            "message": f"Pipeline {running_pipeline_id} stopped successfully",
                            "pipeline_id": running_pipeline_id
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Pipeline stop failed: {result.get('error')}"
                        }
                        
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                return {
                    "success": False,
                    "error": f"Cannot connect to pipeline server: {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to stop pipeline: {str(e)}"
            }
    
    def _update_pipeline_status(self, pipeline_id: str, status: str, error_message: str = None):
        """Update pipeline status in database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return
            
            cursor = conn.cursor()
            
            if status == 'stopped':
                cursor.execute('''
                    UPDATE pipeline_runs 
                    SET status = %s, end_time = %s, duration = EXTRACT(EPOCH FROM (end_time - start_time))
                    WHERE run_id = %s
                ''', (status, datetime.now(), pipeline_id))
            else:
                cursor.execute('''
                    UPDATE pipeline_runs 
                    SET status = %s, error_message = %s
                    WHERE run_id = %s
                ''', (status, error_message, pipeline_id))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error updating pipeline status: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return {
                    "state": "unknown",
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "uptime_seconds": 0,
                    "last_run_time": None
                }
            
            cursor = conn.cursor()
            
            # Check if pipeline is running
            is_running = self.is_pipeline_running()
            
            # Get pipeline metrics from last 2 days
            two_days_ago = datetime.now() - timedelta(days=2)
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_runs,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_runs,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_runs,
                    COUNT(CASE WHEN status = 'stopped' THEN 1 END) as stopped_runs,
                    MAX(start_time) as last_run_time
                FROM pipeline_runs 
                WHERE (command LIKE '%spark_streaming%' OR command LIKE '%pipeline%')
                AND start_time >= NOW() - INTERVAL '2 days'
            ''')
            
            row = cursor.fetchone()
            total_runs, successful_runs, failed_runs, stopped_runs, last_run_time = row
            
            # Calculate uptime for running pipeline
            uptime_seconds = 0
            if is_running:
                running_pipeline_id = self.get_running_pipeline_id()
                if running_pipeline_id:
                    cursor.execute('''
                        SELECT start_time FROM pipeline_runs 
                        WHERE run_id = %s
                    ''', (running_pipeline_id,))
                    
                    start_time = cursor.fetchone()
                    if start_time and start_time[0]:
                        uptime_seconds = int((datetime.now() - start_time[0]).total_seconds())
            
            cursor.close()
            conn.close()
            
            return {
                "state": "running" if is_running else "stopped",
                "total_runs": total_runs or 0,
                "successful_runs": successful_runs or 0,
                "failed_runs": failed_runs or 0,
                "stopped_runs": stopped_runs or 0,
                "uptime_seconds": uptime_seconds,
                "last_run_time": last_run_time.strftime("%Y-%m-%d %H:%M:%S") if last_run_time else None
            }
            
        except Exception as e:
            print(f"Error getting pipeline status: {e}")
            return {
                "state": "unknown",
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "uptime_seconds": 0,
                "last_run_time": None
            }
    
    def get_recent_pipeline_runs(self, days: int = 2) -> List[Dict[str, Any]]:
        """Get recent pipeline runs from the last N days"""
        try:
            conn = self.get_db_connection()
            if not conn:
                return []
            
            cursor = conn.cursor()
            
            # Get pipeline runs from last N days
            start_date = datetime.now() - timedelta(days=days)
            
            # Get pipeline runs from last N days
            cursor.execute('''
                SELECT 
                    run_id,
                    command,
                    status,
                    start_time,
                    end_time,
                    duration,
                    error_message,
                    metadata
                FROM pipeline_runs 
                WHERE (command LIKE '%spark_streaming%' OR command LIKE '%pipeline%')
                ORDER BY start_time DESC
            ''')
            
            rows = cursor.fetchall()
            runs = []
            
            for row in rows:
                run_id, command, status, start_time, end_time, duration, error_message, metadata = row
                
                # Calculate duration if not set
                if not duration and start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
                elif not duration and start_time and status == 'running':
                    duration = (datetime.now() - start_time).total_seconds()
                else:
                    duration = duration or 0
                
                # Parse metadata
                config = {}
                if metadata:
                    try:
                        config = json.loads(metadata)
                    except:
                        pass
                
                runs.append({
                    "run_id": run_id,
                    "command": command,
                    "status": status,
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None,
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None,
                    "duration_seconds": int(duration) if duration else 0,
                    "error_message": error_message,
                    "config": config
                })
            
            cursor.close()
            conn.close()
            
            return runs
            
        except Exception as e:
            print(f"Error getting recent pipeline runs: {e}")
            return []
