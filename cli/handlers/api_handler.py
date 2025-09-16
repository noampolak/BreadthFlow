"""
API handler for all API endpoints
"""

import json

from database.pipeline_queries import PipelineQueries
from database.signals_queries import SignalsQueries


class APIHandler:
    def __init__(self):
        self.pipeline_queries = PipelineQueries()
        self.signals_queries = SignalsQueries()

    def serve_summary_api(self):
        """Serve summary API endpoint"""
        try:
            summary_data = self.pipeline_queries.get_pipeline_summary()
            return json.dumps(summary_data), "application/json", 200
        except Exception as e:
            response_data = {"error": "Failed to get summary data", "message": str(e)}
            return json.dumps(response_data), "application/json", 500

    def serve_runs_api(self, path):
        """Serve runs API endpoint"""
        try:
            # Parse query parameters
            from urllib.parse import parse_qs, urlparse

            parsed_url = urlparse(path)
            query_params = parse_qs(parsed_url.query)

            page = int(query_params.get("page", [1])[0])
            per_page = int(query_params.get("per_page", [10])[0])

            runs_data = self.pipeline_queries.get_pipeline_runs(page, per_page)
            return json.dumps(runs_data), "application/json", 200
        except Exception as e:
            response_data = {"error": "Failed to get runs data", "message": str(e)}
            return json.dumps(response_data), "application/json", 500

    def serve_run_details_api(self, path):
        """Serve run details API endpoint"""
        try:
            # Extract run_id from path
            run_id = path.split("/")[-1]

            run_details = self.pipeline_queries.get_run_details(run_id)
            if run_details:
                # Add signals and backtest data if available
                if "signals" in run_details["command"].lower():
                    run_details["signals_data"] = self.signals_queries.get_signals_data(run_id)

                if "backtest" in run_details["command"].lower():
                    run_details["backtest_data"] = self.signals_queries.get_backtest_data(run_id)

                return json.dumps(run_details), "application/json", 200
            else:
                response_data = {"error": "Run not found", "message": f"Run with ID {run_id} was not found"}
                return json.dumps(response_data), "application/json", 404
        except Exception as e:
            response_data = {"error": "Failed to get run details", "message": str(e)}
            return json.dumps(response_data), "application/json", 500

    def serve_training_api(self, path):
        """Serve training API endpoints"""
        # TODO: Implement actual training API logic
        response_data = {
            "error": "Training API not yet implemented",
            "message": "This endpoint will handle training operations when implemented",
        }

        return json.dumps(response_data), "application/json", 501

    def serve_pipeline_status_api(self):
        """Serve pipeline status API endpoint"""
        try:
            # Get pipeline status from database
            status_data = self.pipeline_queries.get_pipeline_status()
            return json.dumps(status_data), "application/json", 200
        except Exception as e:
            response_data = {"success": False, "error": f"Failed to get pipeline status: {str(e)}"}
            return json.dumps(response_data), "application/json", 500

    def serve_pipeline_runs_api(self):
        """Serve pipeline runs API endpoint"""
        try:
            # Get recent pipeline runs from database
            runs_data = self.pipeline_queries.get_recent_pipeline_runs()
            response_data = {"success": True, "runs": runs_data}
            return json.dumps(response_data), "application/json", 200
        except Exception as e:
            response_data = {"success": False, "error": f"Failed to get pipeline runs: {str(e)}"}
            return json.dumps(response_data), "application/json", 500

    def serve_pipeline_start_api(self, request_body):
        """Serve pipeline start API endpoint"""
        try:
            # Parse request body
            config = json.loads(request_body) if isinstance(request_body, str) else request_body

            # Start pipeline with configuration
            result = self.pipeline_queries.start_pipeline(config)
            response_data = {
                "success": True,
                "message": "Pipeline started successfully",
                "pipeline_id": result.get("pipeline_id"),
            }
            return json.dumps(response_data), "application/json", 200
        except Exception as e:
            response_data = {"success": False, "error": f"Failed to start pipeline: {str(e)}"}
            return json.dumps(response_data), "application/json", 500

    def serve_pipeline_stop_api(self):
        """Serve pipeline stop API endpoint"""
        try:
            # Stop running pipeline
            result = self.pipeline_queries.stop_pipeline()
            response_data = {"success": True, "message": "Pipeline stopped successfully"}
            return json.dumps(response_data), "application/json", 200
        except Exception as e:
            response_data = {"success": False, "error": f"Failed to stop pipeline: {str(e)}"}
            return json.dumps(response_data), "application/json", 500

    def serve_signals_latest_api(self):
        """Serve latest signals API endpoint"""
        try:
            # Get latest trading signals from database
            signals_data = self.signals_queries.get_latest_signals()
            response_data = {"success": True, "signals": signals_data}
            return json.dumps(response_data), "application/json", 200
        except Exception as e:
            response_data = {"success": False, "error": f"Failed to get latest signals: {str(e)}"}
            return json.dumps(response_data), "application/json", 500

    def serve_signals_export_api(self, path):
        """Serve signals export API endpoint"""
        try:
            # Parse query parameters
            from urllib.parse import parse_qs, urlparse

            parsed_url = urlparse(path)
            query_params = parse_qs(parsed_url.query)

            export_format = query_params.get("format", ["csv"])[0]
            run_id = query_params.get("run_id", [None])[0]

            # Export signals in requested format
            export_data = self.signals_queries.export_signals(export_format, run_id)

            if export_format == "csv":
                content_type = "text/csv"
            else:
                content_type = "application/json"

            return export_data, content_type, 200
        except Exception as e:
            response_data = {"success": False, "error": f"Failed to export signals: {str(e)}"}
            return json.dumps(response_data), "application/json", 500
