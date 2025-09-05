"""
Pipeline-related database queries for BreadthFlow dashboard
"""

from datetime import datetime, timedelta
from .connection import get_db_connection

class PipelineQueries:
    def __init__(self):
        self.db = get_db_connection()
    
    def get_pipeline_summary(self):
        """Get pipeline summary statistics"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return {
                    'total_runs': 0,
                    'success_rate': 0,
                    'recent_runs': 0,
                    'avg_duration': 0
                }
            
            # Get total runs
            total_runs_query = "SELECT COUNT(*) FROM pipeline_runs"
            total_runs_result = self.db.execute_query(total_runs_query)
            total_runs = total_runs_result.fetchone()[0] if total_runs_result else 0
            
            # Get success rate
            success_rate_query = """
                SELECT 
                    ROUND(
                        (COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*)), 2
                    ) as success_rate
                FROM pipeline_runs
            """
            success_rate_result = self.db.execute_query(success_rate_query)
            success_rate = success_rate_result.fetchone()[0] if success_rate_result else 0
            
            # Get recent runs (last 24h)
            recent_runs_query = """
                SELECT COUNT(*) FROM pipeline_runs 
                WHERE start_time >= NOW() - INTERVAL '24 hours'
            """
            recent_runs_result = self.db.execute_query(recent_runs_query)
            recent_runs = recent_runs_result.fetchone()[0] if recent_runs_result else 0
            
            # Get average duration
            avg_duration_query = """
                SELECT ROUND(AVG(EXTRACT(EPOCH FROM (end_time - start_time))), 2)
                FROM pipeline_runs 
                WHERE status = 'completed' AND end_time IS NOT NULL
            """
            avg_duration_result = self.db.execute_query(avg_duration_query)
            avg_duration = avg_duration_result.fetchone()[0] if avg_duration_result else 0
            
            return {
                'total_runs': total_runs,
                'success_rate': success_rate,
                'recent_runs': recent_runs,
                'avg_duration': avg_duration
            }
            
        except Exception as e:
            print(f"Error getting pipeline summary: {e}")
            return {
                'total_runs': 0,
                'success_rate': 0,
                'recent_runs': 0,
                'avg_duration': 0
            }
    
    def get_pipeline_runs(self, page=1, per_page=10):
        """Get paginated pipeline runs"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return {
                    'runs': [],
                    'pagination': {
                        'page': page,
                        'per_page': per_page,
                        'total_count': 0,
                        'total_pages': 0,
                        'has_prev': False,
                        'has_next': False
                    }
                }
            
            offset = (page - 1) * per_page
            
            # Get total count
            count_query = "SELECT COUNT(*) FROM pipeline_runs"
            count_result = self.db.execute_query(count_query)
            total_count = count_result.fetchone()[0] if count_result else 0
            
            # Get runs for current page
            runs_query = """
                SELECT 
                    run_id, command, status, start_time, end_time,
                    EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) as duration
                FROM pipeline_runs 
                ORDER BY start_time DESC 
                LIMIT :per_page OFFSET :offset
            """
            runs_result = self.db.execute_query(runs_query, {
                'per_page': per_page,
                'offset': offset
            })
            
            runs = []
            if runs_result:
                for row in runs_result:
                    runs.append({
                        'run_id': row[0],
                        'command': row[1],
                        'status': row[2],
                        'start_time': row[3].isoformat() if row[3] else None,
                        'end_time': row[4].isoformat() if row[4] else None,
                        'duration': float(row[5]) if row[5] else 0
                    })
            
            # Calculate pagination info
            total_pages = (total_count + per_page - 1) // per_page
            has_prev = page > 1
            has_next = page < total_pages
            
            return {
                'runs': runs,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'has_prev': has_prev,
                    'has_next': has_next
                }
            }
            
        except Exception as e:
            print(f"Error getting pipeline runs: {e}")
            return {
                'runs': [],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_count': 0,
                    'total_pages': 0,
                    'has_prev': False,
                    'has_next': False
                }
            }
    
    def get_run_details(self, run_id):
        """Get detailed information about a specific run"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return None
            
            run_query = """
                SELECT 
                    run_id, command, status, start_time, end_time,
                    EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) as duration,
                    error_message, metadata
                FROM pipeline_runs 
                WHERE run_id = :run_id
            """
            run_result = self.db.execute_query(run_query, {'run_id': run_id})
            
            if run_result:
                row = run_result.fetchone()
                return {
                    'run_id': row[0],
                    'command': row[1],
                    'status': row[2],
                    'start_time': row[3].isoformat() if row[3] else None,
                    'end_time': row[4].isoformat() if row[4] else None,
                    'duration': float(row[5]) if row[5] else 0,
                    'error_message': row[6],
                    'metadata': row[7] if row[7] else {}
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error getting run details: {e}")
            return None
    
    def get_pipeline_status(self):
        """Get current pipeline status"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return {
                    "success": False,
                    "error": "No database connection"
                }
            
            # Check if pipeline is currently running
            running_query = """
                SELECT 
                    run_id, command, start_time,
                    EXTRACT(EPOCH FROM (NOW() - start_time)) as uptime_seconds
                FROM pipeline_runs 
                WHERE status = 'running' 
                ORDER BY start_time DESC 
                LIMIT 1
            """
            running_result = self.db.execute_query(running_query)
            
            if running_result:
                row = running_result.fetchone()
                return {
                    "success": True,
                    "status": {
                        "state": "running",
                        "run_id": row[0],
                        "command": row[1],
                        "start_time": row[2].isoformat() if row[2] else None,
                        "uptime_seconds": int(row[3]) if row[3] else 0
                    }
                }
            else:
                # Get last run info
                last_run_query = """
                    SELECT 
                        run_id, command, status, start_time, end_time
                    FROM pipeline_runs 
                    ORDER BY start_time DESC 
                    LIMIT 1
                """
                last_run_result = self.db.execute_query(last_run_query)
                
                if last_run_result:
                    row = last_run_result.fetchone()
                    return {
                        "success": True,
                        "status": {
                            "state": "stopped",
                            "last_run_id": row[0],
                            "last_command": row[1],
                            "last_status": row[2],
                            "last_run_time": row[3].isoformat() if row[3] else None
                        }
                    }
                else:
                    return {
                        "success": True,
                        "status": {
                            "state": "never_run",
                            "last_run_time": None
                        }
                    }
                    
        except Exception as e:
            print(f"Error getting pipeline status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_recent_pipeline_runs(self):
        """Get recent pipeline runs for the last 2 days"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                return []
            
            runs_query = """
                SELECT 
                    run_id, command, status, start_time, end_time,
                    EXTRACT(EPOCH FROM (COALESCE(end_time, NOW()) - start_time)) as duration_seconds,
                    error_message
                FROM pipeline_runs 
                WHERE start_time >= NOW() - INTERVAL '2 days'
                ORDER BY start_time DESC 
                LIMIT 20
            """
            runs_result = self.db.execute_query(runs_query)
            
            runs = []
            if runs_result:
                for row in runs_result:
                    runs.append({
                        'run_id': row[0],
                        'command': row[1],
                        'status': row[2],
                        'start_time': row[3].isoformat() if row[3] else None,
                        'end_time': row[4].isoformat() if row[4] else None,
                        'duration_seconds': float(row[5]) if row[5] else 0,
                        'error_message': row[6]
                    })
            
            return runs
            
        except Exception as e:
            print(f"Error getting recent pipeline runs: {e}")
            return []
    
    def start_pipeline(self, config):
        """Start a new pipeline with the given configuration"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                raise Exception("No database connection available")
            
            # Create a new pipeline run record
            insert_query = """
                INSERT INTO pipeline_runs (run_id, command, status, start_time, metadata)
                VALUES (:run_id, :command, 'running', NOW(), :metadata)
                RETURNING run_id
            """
            
            import uuid
            run_id = str(uuid.uuid4())
            command = f"pipeline_{config.get('mode', 'demo')}_{config.get('interval', '1m')}"
            metadata = {
                'mode': config.get('mode'),
                'interval': config.get('interval'),
                'timeframe': config.get('timeframe'),
                'symbols': config.get('symbols'),
                'data_source': config.get('data_source')
            }
            
            result = self.db.execute_query(insert_query, {
                'run_id': run_id,
                'command': command,
                'metadata': metadata
            })
            
            if result:
                return {'pipeline_id': run_id}
            else:
                raise Exception("Failed to create pipeline run record")
                
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            raise e
    
    def stop_pipeline(self):
        """Stop the currently running pipeline"""
        try:
            if not self.db or not self.db.connection:
                print("No database connection available")
                raise Exception("No database connection available")
            
            # Find and stop any running pipeline
            update_query = """
                UPDATE pipeline_runs 
                SET status = 'stopped', end_time = NOW()
                WHERE status = 'running'
            """
            
            result = self.db.execute_query(update_query)
            
            if result:
                return {'stopped': True}
            else:
                return {'stopped': False, 'message': 'No running pipeline found'}
                
        except Exception as e:
            print(f"Error stopping pipeline: {e}")
            raise e
