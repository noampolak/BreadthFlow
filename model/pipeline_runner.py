#!/usr/bin/env python3
"""
BreadthFlow Pipeline Runner

This module provides automated pipeline execution capabilities for continuous 
batch processing of financial data. It triggers existing commands (fetch, signals, 
backtest) in sequence with configurable intervals and modes.

Key Features:
- Continuous execution until stopped
- Multiple execution modes (demo, all_symbols, custom)
- Comprehensive logging and metadata tracking
- Error handling and recovery
- Integration with existing Spark infrastructure
"""

import time
import threading
import signal
import sys
import subprocess
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineState:
    """Thread-safe pipeline state management"""
    def __init__(self):
        self._state = "stopped"
        self._lock = threading.Lock()
        self._start_time = None
        self._current_run_id = None
        self._stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "last_run_time": None,
            "last_error": None
        }
    
    def set_state(self, state: str):
        with self._lock:
            self._state = state
            if state == "running" and self._start_time is None:
                self._start_time = datetime.now()
            elif state == "stopped":
                self._start_time = None
    
    def get_state(self) -> str:
        with self._lock:
            return self._state
    
    def set_current_run(self, run_id: str):
        with self._lock:
            self._current_run_id = run_id
    
    def get_current_run(self) -> Optional[str]:
        with self._lock:
            return self._current_run_id
    
    def update_stats(self, success: bool, error: str = None):
        with self._lock:
            self._stats["total_runs"] += 1
            if success:
                self._stats["successful_runs"] += 1
            else:
                self._stats["failed_runs"] += 1
                self._stats["last_error"] = error
            self._stats["last_run_time"] = datetime.now().isoformat()
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = self._stats.copy()
            stats["state"] = self._state
            stats["start_time"] = self._start_time.isoformat() if self._start_time else None
            stats["current_run_id"] = self._current_run_id
            stats["uptime_seconds"] = (datetime.now() - self._start_time).total_seconds() if self._start_time else 0
            return stats

class PipelineConfig:
    """Pipeline configuration management"""
    def __init__(self, mode: str = "demo", interval: str = "5m", symbols: List[str] = None, 
                 timeframe: str = "1day", data_source: str = "yfinance"):
        self.mode = mode
        self.interval = interval
        self.symbols = symbols or self._get_default_symbols(mode)
        self.timeframe = timeframe
        self.data_source = data_source
        self.validate()
    
    def _get_default_symbols(self, mode: str) -> List[str]:
        """Get default symbols based on mode"""
        if mode == "demo":
            return ["AAPL", "MSFT"]
        elif mode == "demo_small":
            return ["AAPL", "MSFT", "GOOGL"]
        elif mode == "tech_leaders":
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        else:
            return ["AAPL", "MSFT", "GOOGL"]
    
    def get_interval_seconds(self) -> int:
        """Convert interval string to seconds"""
        if self.interval.endswith('s'):
            return int(self.interval[:-1])
        elif self.interval.endswith('m'):
            return int(self.interval[:-1]) * 60
        elif self.interval.endswith('h'):
            return int(self.interval[:-1]) * 3600
        else:
            # Default to minutes if no unit specified
            return int(self.interval) * 60
    
    def validate(self):
        """Validate configuration parameters"""
        valid_modes = ["demo", "demo_small", "tech_leaders", "all_symbols", "custom"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Valid modes: {valid_modes}")
        
        valid_timeframes = ["1min", "5min", "15min", "1hour", "1day"]
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe: {self.timeframe}. Valid timeframes: {valid_timeframes}")
        
        if self.get_interval_seconds() < 60:  # Minimum 1 minute
            raise ValueError("Interval must be at least 1 minute")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "interval": self.interval,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "data_source": self.data_source,
            "interval_seconds": self.get_interval_seconds()
        }

class PipelineRunner:
    """Main pipeline runner class"""
    
    def __init__(self, config: PipelineConfig, cli_path: str = "/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py"):
        self.config = config
        self.cli_path = cli_path
        self.state = PipelineState()
        self._stop_event = threading.Event()
        self._thread = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, stopping pipeline...")
        self.stop()
    
    def start(self) -> Dict[str, Any]:
        """Start the pipeline runner"""
        if self.state.get_state() == "running":
            return {"success": False, "error": "Pipeline is already running"}
        
        logger.info(f"Starting pipeline with config: {self.config.to_dict()}")
        self.state.set_state("running")
        self._stop_event.clear()
        
        # Start pipeline thread
        self._thread = threading.Thread(target=self._run_pipeline_loop, daemon=True)
        self._thread.start()
        
        return {
            "success": True,
            "message": "Pipeline started successfully",
            "config": self.config.to_dict(),
            "state": self.state.get_stats()
        }
    
    def stop(self) -> Dict[str, Any]:
        """Stop the pipeline runner"""
        if self.state.get_state() == "stopped":
            return {"success": False, "error": "Pipeline is not running"}
        
        logger.info("Stopping pipeline...")
        self._stop_event.set()
        self.state.set_state("stopping")
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)  # 30 second timeout
        
        self.state.set_state("stopped")
        
        return {
            "success": True,
            "message": "Pipeline stopped successfully",
            "final_stats": self.state.get_stats()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "success": True,
            "state": self.state.get_stats(),
            "config": self.config.to_dict()
        }
    
    def _run_pipeline_loop(self):
        """Main pipeline execution loop"""
        logger.info("Pipeline loop started")
        
        while not self._stop_event.is_set():
            try:
                # Execute one pipeline run
                run_result = self._execute_pipeline_run()
                
                # Update statistics
                self.state.update_stats(
                    success=run_result.get("success", False),
                    error=run_result.get("error")
                )
                
                # Wait for next interval or stop signal
                if not self._stop_event.wait(timeout=self.config.get_interval_seconds()):
                    continue  # Timeout reached, run next iteration
                else:
                    break  # Stop event was set
                    
            except Exception as e:
                logger.error(f"Error in pipeline loop: {e}")
                self.state.update_stats(success=False, error=str(e))
                
                # Wait before retrying
                if not self._stop_event.wait(timeout=60):  # 1 minute retry delay
                    continue
                else:
                    break
        
        logger.info("Pipeline loop ended")
    
    def _execute_pipeline_run(self) -> Dict[str, Any]:
        """Execute a single pipeline run"""
        run_id = str(uuid.uuid4())
        self.state.set_current_run(run_id)
        
        logger.info(f"Starting pipeline run {run_id}")
        start_time = datetime.now()
        
        try:
            # Execute commands in sequence
            results = {}
            
            # 1. Data Fetch
            logger.info("Executing data fetch...")
            fetch_result = self._execute_command(
                "data", "fetch",
                symbols=",".join(self.config.symbols),
                start_date=self._get_start_date(),
                end_date=self._get_end_date(),
                timeframe=self.config.timeframe,
                data_source=self.config.data_source
            )
            results["fetch"] = fetch_result
            
            if not fetch_result.get("success", False):
                logger.error(f"Data fetch failed: {fetch_result.get('error')}")
                return {"success": False, "error": "Data fetch failed", "results": results}
            
            # 2. Signal Generation
            logger.info("Executing signal generation...")
            signals_result = self._execute_command(
                "signals", "generate",
                symbols=",".join(self.config.symbols),
                start_date=self._get_start_date(),
                end_date=self._get_end_date(),
                timeframe=self.config.timeframe
            )
            results["signals"] = signals_result
            
            if not signals_result.get("success", False):
                logger.error(f"Signal generation failed: {signals_result.get('error')}")
                return {"success": False, "error": "Signal generation failed", "results": results}
            
            # 3. Backtesting (optional)
            logger.info("Executing backtesting...")
            backtest_result = self._execute_command(
                "backtest", "run",
                symbols=",".join(self.config.symbols),
                start_date=self._get_start_date(),
                end_date=self._get_end_date(),
                timeframe=self.config.timeframe
            )
            results["backtest"] = backtest_result
            
            # Calculate run duration
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Pipeline run {run_id} completed successfully in {duration:.1f}s")
            
            return {
                "success": True,
                "run_id": run_id,
                "duration": duration,
                "results": results,
                "symbols_processed": len(self.config.symbols),
                "config": self.config.to_dict()
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Pipeline run {run_id} failed after {duration:.1f}s: {e}")
            
            return {
                "success": False,
                "run_id": run_id,
                "duration": duration,
                "error": str(e)
            }
    
    def _execute_command(self, group: str, command: str, **kwargs) -> Dict[str, Any]:
        """Execute a CLI command and return result"""
        try:
            # Build command
            cmd = ["python3", self.cli_path, group, command]
            
            # Add parameters
            for key, value in kwargs.items():
                if value is not None:
                    cmd.extend([f"--{key.replace('_', '-')}", str(value)])
            
            logger.debug(f"Executing command: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd="/opt/bitnami/spark/jobs"
            )
            
            if result.returncode == 0:
                return {"success": True, "output": result.stdout}
            else:
                return {"success": False, "error": result.stderr or result.stdout}
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_start_date(self) -> str:
        """Get start date for data fetching (default: 30 days ago)"""
        if self.config.timeframe == "1day":
            days_back = 30
        elif self.config.timeframe == "1hour":
            days_back = 7  # 1 week for hourly
        else:
            days_back = 3  # 3 days for minute data
        
        start_date = datetime.now() - timedelta(days=days_back)
        return start_date.strftime("%Y-%m-%d")
    
    def _get_end_date(self) -> str:
        """Get end date for data fetching (default: today)"""
        return datetime.now().strftime("%Y-%m-%d")

# Global pipeline runner instance
_pipeline_runner: Optional[PipelineRunner] = None

def get_pipeline_runner() -> Optional[PipelineRunner]:
    """Get the global pipeline runner instance"""
    return _pipeline_runner

def create_pipeline_runner(config: PipelineConfig) -> PipelineRunner:
    """Create and set the global pipeline runner instance"""
    global _pipeline_runner
    _pipeline_runner = PipelineRunner(config)
    return _pipeline_runner

def start_pipeline(mode: str = "demo", interval: str = "5m", symbols: List[str] = None,
                  timeframe: str = "1day", data_source: str = "yfinance") -> Dict[str, Any]:
    """Start the pipeline with the given configuration"""
    try:
        config = PipelineConfig(mode, interval, symbols, timeframe, data_source)
        runner = create_pipeline_runner(config)
        return runner.start()
    except Exception as e:
        return {"success": False, "error": str(e)}

def stop_pipeline() -> Dict[str, Any]:
    """Stop the running pipeline"""
    runner = get_pipeline_runner()
    if runner:
        return runner.stop()
    else:
        return {"success": False, "error": "No pipeline is running"}

def get_pipeline_status() -> Dict[str, Any]:
    """Get the current pipeline status"""
    runner = get_pipeline_runner()
    if runner:
        return runner.get_status()
    else:
        return {
            "success": True,
            "state": {
                "state": "stopped",
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "last_run_time": None,
                "uptime_seconds": 0
            },
            "config": None
        }

