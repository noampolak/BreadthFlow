#!/usr/bin/env python3
"""
Elasticsearch Logger for BreadthFlow

Sends pipeline logs to Elasticsearch for Kibana visualization
while maintaining SQLite logging for the web dashboard.
"""

import json
import requests
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ElasticsearchLogger:
    """Sends logs to Elasticsearch for Kibana dashboards"""
    
    def __init__(self, es_url: str = "http://elasticsearch:9200"):
        self.es_url = es_url
        self.index_name = "breadthflow-logs"
        self.session = requests.Session()
        
    def ensure_index_exists(self):
        """Create index with proper mapping if it doesn't exist"""
        index_url = f"{self.es_url}/{self.index_name}"
        
        # Check if index exists
        response = self.session.head(index_url)
        if response.status_code == 200:
            return True
            
        # Create index with mapping
        mapping = {
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "run_id": {"type": "keyword"},
                    "command": {"type": "text"},
                    "status": {"type": "keyword"},
                    "level": {"type": "keyword"},
                    "message": {"type": "text"},
                    "duration": {"type": "float"},
                    "symbols_count": {"type": "integer"},
                    "symbols": {"type": "keyword"},
                    "success_count": {"type": "integer"},
                    "failed_count": {"type": "integer"},
                    "progress": {"type": "float"},
                    "metadata": {"type": "object"}
                }
            }
        }
        
        try:
            response = self.session.put(index_url, json=mapping)
            if response.status_code in [200, 201]:
                logger.info(f"✅ Created Elasticsearch index: {self.index_name}")
                return True
            else:
                logger.error(f"❌ Failed to create index: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return False
    
    def log_pipeline_start(self, run_id: str, command: str, metadata: Dict[str, Any] = None):
        """Log pipeline start event"""
        doc = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "command": command,
            "status": "started",
            "level": "INFO",
            "message": f"Pipeline started: {command}",
            "metadata": metadata or {}
        }
        
        self._send_to_elasticsearch(doc)
    
    def log_pipeline_progress(self, run_id: str, message: str, level: str = "INFO", 
                            progress: Optional[float] = None, metadata: Dict[str, Any] = None):
        """Log pipeline progress update"""
        doc = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "level": level,
            "message": message,
            "metadata": metadata or {}
        }
        
        if progress is not None:
            doc["progress"] = progress
            
        self._send_to_elasticsearch(doc)
    
    def log_pipeline_complete(self, run_id: str, command: str, status: str, 
                            duration: float, metadata: Dict[str, Any] = None):
        """Log pipeline completion"""
        doc = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "command": command,
            "status": status,
            "level": "INFO" if status == "completed" else "ERROR",
            "message": f"Pipeline {status}: {command} (Duration: {duration:.1f}s)",
            "duration": duration,
            "metadata": metadata or {}
        }
        
        self._send_to_elasticsearch(doc)
    
    def log_data_fetch_progress(self, run_id: str, symbol: str, current: int, total: int, 
                               success: bool, records: int = 0):
        """Log data fetching progress for specific symbols"""
        try:
            # Ensure current and total are integers
            current_int = int(current) if current is not None else 0
            total_int = int(total) if total is not None else 0
            progress = (current_int / total_int) * 100 if total_int > 0 else 0
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating progress: current={current}, total={total}, error={e}")
            progress = 0
        status = "success" if success else "failed"
        
        doc = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": run_id,
            "level": "INFO" if success else "WARN",
            "message": f"Symbol {symbol} ({current}/{total}): {status}",
            "progress": progress,
            "metadata": {
                "symbol": symbol,
                "current_symbol": current,
                "total_symbols": total,
                "records_fetched": records,
                "fetch_status": status
            }
        }
        
        self._send_to_elasticsearch(doc)
    
    def _send_to_elasticsearch(self, doc: Dict[str, Any]):
        """Send document to Elasticsearch"""
        try:
            url = f"{self.es_url}/{self.index_name}/_doc"
            response = self.session.post(url, json=doc)
            
            if response.status_code not in [200, 201]:
                logger.error(f"Failed to send to Elasticsearch: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending to Elasticsearch: {e}")

# Global instance
es_logger = ElasticsearchLogger()
