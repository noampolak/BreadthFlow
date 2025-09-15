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
        assert "total_runs" in data
        assert "success_rate" in data
        assert "active_pipelines" in data
        assert "recent_runs" in data

    def test_dashboard_summary_data_types(self, client):
        """Test dashboard summary data types"""
        response = client.get("/api/dashboard/summary")
        data = response.json()

        assert isinstance(data["total_runs"], int)
        assert isinstance(data["success_rate"], (int, float))
        assert isinstance(data["active_pipelines"], int)
        assert isinstance(data["recent_runs"], list)

    def test_dashboard_summary_success_rate_range(self, client):
        """Test success rate is within valid range"""
        response = client.get("/api/dashboard/summary")
        data = response.json()

        assert 0 <= data["success_rate"] <= 100


class TestPipelineAPI:
    """Test pipeline API endpoints"""

    def test_pipeline_status(self, client):
        """Test pipeline status endpoint"""
        response = client.get("/api/pipeline/status")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "active_runs" in data
        assert "queue_size" in data

    def test_pipeline_start(self, client):
        """Test pipeline start endpoint"""
        payload = {"symbols": ["AAPL", "MSFT"], "timeframe": "1day", "strategy": "technical"}

        response = client.post("/api/pipeline/start", json=payload)
        assert response.status_code in [200, 201]

        data = response.json()
        assert "run_id" in data
        assert "status" in data

    def test_pipeline_start_invalid_symbols(self, client):
        """Test pipeline start with invalid symbols"""
        payload = {"symbols": [], "timeframe": "1day"}

        response = client.post("/api/pipeline/start", json=payload)
        assert response.status_code == 400

    def test_pipeline_start_invalid_timeframe(self, client):
        """Test pipeline start with invalid timeframe"""
        payload = {"symbols": ["AAPL"], "timeframe": "invalid"}

        response = client.post("/api/pipeline/start", json=payload)
        assert response.status_code == 400

    def test_pipeline_stop(self, client):
        """Test pipeline stop endpoint"""
        response = client.post("/api/pipeline/stop")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data

    def test_pipeline_history(self, client):
        """Test pipeline history endpoint"""
        response = client.get("/api/pipeline/history")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_pipeline_history_with_pagination(self, client):
        """Test pipeline history with pagination"""
        response = client.get("/api/pipeline/history?page=1&limit=10")
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
        assert isinstance(data, list)

    def test_signals_latest_with_symbol(self, client):
        """Test latest signals with symbol filter"""
        response = client.get("/api/signals/latest?symbol=AAPL")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        if data:  # If there are signals
            assert all(signal["symbol"] == "AAPL" for signal in data)

    def test_signals_latest_with_timeframe(self, client):
        """Test latest signals with timeframe filter"""
        response = client.get("/api/signals/latest?timeframe=1day")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_signals_export(self, client):
        """Test signals export endpoint"""
        response = client.get("/api/signals/export?format=csv")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"

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
        assert isinstance(data, list)

    def test_commands_execute(self, client):
        """Test command execution endpoint"""
        payload = {"command": "health", "parameters": {}}

        response = client.post("/api/commands/execute", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "output" in data
        assert "status" in data

    def test_commands_execute_invalid(self, client):
        """Test command execution with invalid command"""
        payload = {"command": "invalid_command", "parameters": {}}

        response = client.post("/api/commands/execute", json=payload)
        assert response.status_code == 400

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
        assert isinstance(data, list)

    def test_training_start(self, client):
        """Test training start endpoint"""
        payload = {"model_type": "random_forest", "strategy": "technical", "symbols": ["AAPL"], "timeframe": "1day"}

        response = client.post("/api/training/start", json=payload)
        assert response.status_code in [200, 201]

        data = response.json()
        assert "training_id" in data
        assert "status" in data

    def test_training_status(self, client):
        """Test training status endpoint"""
        response = client.get("/api/training/status")
        assert response.status_code == 200

        data = response.json()
        assert "active_trainings" in data
        assert "queue_size" in data

    def test_training_results(self, client):
        """Test training results endpoint"""
        response = client.get("/api/training/results")
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
        payload = {"group_name": "signal_generation", "parameters": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26}}

        response = client.put("/api/parameters/update", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "updated_parameters" in data

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
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data

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
