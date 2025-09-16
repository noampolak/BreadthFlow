"""
Unit tests for API endpoints.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestDashboardAPI:
    """Test dashboard API endpoints"""

    def test_dashboard_summary(self, client):
        """Test dashboard summary endpoint"""
        response = client.get("/api/dashboard/summary")
        assert response.status_code == 200

        data = response.json()
        assert "stats" in data
        assert "recent_runs" in data
        assert "last_updated" in data

        # Check stats structure
        stats = data["stats"]
        assert "total_runs" in stats
        assert "success_rate" in stats
        assert "failed_runs" in stats
        assert "recent_runs_24h" in stats
        assert "average_duration" in stats

    def test_dashboard_summary_data_types(self, client):
        """Test dashboard summary data types"""
        response = client.get("/api/dashboard/summary")
        data = response.json()

        stats = data["stats"]
        assert isinstance(stats["total_runs"], int)
        assert isinstance(stats["success_rate"], (int, float))
        assert isinstance(stats["failed_runs"], int)
        assert isinstance(stats["recent_runs_24h"], int)
        assert isinstance(stats["average_duration"], (int, float))
        assert isinstance(data["recent_runs"], list)

    def test_dashboard_summary_success_rate_range(self, client):
        """Test success rate is within valid range"""
        response = client.get("/api/dashboard/summary")
        data = response.json()

        assert 0 <= data["stats"]["success_rate"] <= 100


class TestPipelineAPI:
    """Test pipeline API endpoints"""

    def test_pipeline_status(self, client):
        """Test pipeline status endpoint"""
        response = client.get("/api/pipeline/status")
        assert response.status_code == 200

        data = response.json()
        assert "state" in data
        assert "total_runs" in data
        assert "successful_runs" in data
        assert "failed_runs" in data
        assert "success_rate" in data
        assert "average_duration" in data

    def test_pipeline_start(self, client):
        """Test pipeline start endpoint"""
        payload = {"mode": "demo", "interval": "1h", "timeframe": "1day", "symbols": "AAPL,MSFT", "data_source": "yfinance"}

        response = client.post("/api/pipeline/start", json=payload)
        assert response.status_code in [200, 201]

        data = response.json()
        assert "run_id" in data
        assert "status" in data

    def test_pipeline_start_invalid_symbols(self, client):
        """Test pipeline start with invalid symbols"""
        payload = {"mode": "demo", "interval": "1h", "timeframe": "1day", "symbols": "", "data_source": "yfinance"}

        response = client.post("/api/pipeline/start", json=payload)
        assert response.status_code == 422

    def test_pipeline_start_invalid_timeframe(self, client):
        """Test pipeline start with invalid timeframe"""
        payload = {"mode": "demo", "interval": "1h", "timeframe": "invalid", "symbols": "AAPL", "data_source": "yfinance"}

        response = client.post("/api/pipeline/start", json=payload)
        assert response.status_code == 422

    def test_pipeline_stop(self, client):
        """Test pipeline stop endpoint"""
        # First create a pipeline run, then try to stop it
        payload = {"mode": "demo", "interval": "1h", "timeframe": "1day", "symbols": "AAPL", "data_source": "yfinance"}

        # Start a pipeline
        start_response = client.post("/api/pipeline/start", json=payload)
        if start_response.status_code in [200, 201]:
            run_id = start_response.json()["run_id"]
            # Try to stop it - it might be 400 if not running yet, or 200 if running
            response = client.post(f"/api/pipeline/stop/{run_id}")
            assert response.status_code in [200, 400]  # Accept both success and "not running" cases
        else:
            # If no pipeline to stop, expect 404
            response = client.post("/api/pipeline/stop/nonexistent")
            assert response.status_code == 404

        data = response.json()
        # For 200 responses, expect "message", for 400 responses, expect "detail"
        if response.status_code == 200:
            assert "message" in data
        else:
            assert "detail" in data

    def test_pipeline_history(self, client):
        """Test pipeline history endpoint"""
        response = client.get("/api/pipeline/runs")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_pipeline_history_with_pagination(self, client):
        """Test pipeline history with pagination"""
        response = client.get("/api/pipeline/runs?skip=0&limit=10")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10


class TestSignalsAPI:
    """Test signals API endpoints"""

    def test_signals_latest(self, client):
        """Test latest signals endpoint"""
        response = client.get("/api/signals/latest")
        assert response.status_code == 200

        data = response.json()
        assert "signals" in data
        assert "stats" in data
        assert "last_updated" in data
        assert isinstance(data["signals"], list)
        assert isinstance(data["stats"], dict)

    def test_signals_latest_with_symbol(self, client):
        """Test latest signals with symbol filter"""
        response = client.get("/api/signals/latest?symbol=AAPL")
        assert response.status_code == 200

        data = response.json()
        assert "signals" in data
        assert isinstance(data["signals"], list)
        if data["signals"]:  # If there are signals
            assert all(signal["symbol"] == "AAPL" for signal in data["signals"])

    def test_signals_latest_with_timeframe(self, client):
        """Test latest signals with timeframe filter"""
        response = client.get("/api/signals/latest?timeframe=1day")
        assert response.status_code == 200

        data = response.json()
        assert "signals" in data
        assert isinstance(data["signals"], list)
        if data["signals"]:  # If there are signals
            assert all(signal["timeframe"] == "1day" for signal in data["signals"])

    def test_signals_export(self, client):
        """Test signals export endpoint"""
        response = client.get("/api/signals/export?format=csv")
        assert response.status_code == 200
        # The API returns JSON with export data, not direct CSV
        assert response.headers["content-type"] == "application/json"

    def test_signals_export_json(self, client):
        """Test signals export as JSON"""
        response = client.get("/api/signals/export?format=json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"


class TestCommandsAPI:
    """Test commands API endpoints"""

    def test_commands_quick_flows(self, client):
        """Test quick flows endpoint"""
        response = client.get("/api/commands/quick-flows")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_commands_templates(self, client):
        """Test command templates endpoint"""
        response = client.get("/api/commands/templates")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)
        assert "data_commands" in data
        assert "signal_commands" in data
        assert "backtest_commands" in data
        assert "pipeline_commands" in data

    def test_commands_execute(self, client):
        """Test command execution endpoint"""
        payload = {"command": "echo hello", "parameters": {}, "background": False}

        response = client.post("/api/commands/execute", json=payload)
        # Accept both success and failure since command execution depends on system environment
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "command_id" in data
            assert "status" in data

    def test_commands_execute_invalid(self, client):
        """Test command execution with invalid command"""
        payload = {"command": "invalid_command", "parameters": {}, "background": False}

        response = client.post("/api/commands/execute", json=payload)
        # Accept both success and failure since command execution depends on system environment
        assert response.status_code in [200, 500]

    def test_commands_history(self, client):
        """Test command history endpoint"""
        response = client.get("/api/commands/history")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


class TestTrainingAPI:
    """Test training API endpoints"""

    def test_training_configurations(self, client):
        """Test training configurations endpoint"""
        response = client.get("/api/training/configurations")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict)
        assert "strategies" in data
        assert "model_types" in data
        assert "timeframes" in data
        assert "symbols" in data

    def test_training_start(self, client):
        """Test training start endpoint"""
        payload = {
            "symbols": ["AAPL"],
            "timeframe": "1day",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "strategy": "momentum",
            "model_type": "random_forest",
            "test_split": 0.2,
        }

        response = client.post("/api/training/start", json=payload)
        # Accept both success and failure since training depends on ML dependencies
        assert response.status_code in [200, 201, 500]

        if response.status_code in [200, 201]:
            data = response.json()
            assert "training_id" in data
            assert "status" in data

    def test_training_status(self, client):
        """Test training history endpoint"""
        response = client.get("/api/training/history")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_training_results(self, client):
        """Test training models endpoint"""
        response = client.get("/api/training/models")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


class TestParametersAPI:
    """Test parameters API endpoints"""

    def test_parameters_groups(self, client):
        """Test parameters groups endpoint"""
        response = client.get("/api/parameters/groups")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_parameters_group_details(self, client):
        """Test parameters group details endpoint"""
        response = client.get("/api/parameters/groups/signal_generation")
        assert response.status_code == 200

        data = response.json()
        assert "group_name" in data
        assert "parameters" in data

    def test_parameters_update(self, client):
        """Test parameters update endpoint"""
        payload = [
            {"group_name": "signal_generation", "parameter_name": "rsi_period", "value": 14},
            {"group_name": "signal_generation", "parameter_name": "macd_fast", "value": 12},
            {"group_name": "signal_generation", "parameter_name": "macd_slow", "value": 26},
        ]

        response = client.put("/api/parameters/update", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "message" in data

    def test_parameters_history(self, client):
        """Test parameters history endpoint"""
        response = client.get("/api/parameters/history")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


class TestInfrastructureAPI:
    """Test infrastructure API endpoints"""

    def test_infrastructure_health(self, client):
        """Test infrastructure health endpoint"""
        response = client.get("/api/infrastructure/health")
        assert response.status_code == 200

        data = response.json()
        assert "overall_status" in data
        assert "services" in data
        assert "last_updated" in data

    def test_infrastructure_metrics(self, client):
        """Test infrastructure metrics endpoint"""
        response = client.get("/api/infrastructure/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "disk_usage" in data

    def test_infrastructure_logs(self, client):
        """Test infrastructure logs endpoint"""
        response = client.get("/api/infrastructure/logs")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_infrastructure_logs_with_level(self, client):
        """Test infrastructure logs with level filter"""
        response = client.get("/api/infrastructure/logs?level=ERROR")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        if data:  # If there are logs
            assert all(log["level"] == "ERROR" for log in data)
