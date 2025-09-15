from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Dict, Any
import json
import uuid
from datetime import datetime

from .models import ParameterConfig, ParameterHistory
from .schemas import ParameterGroup, ParameterValue, ParameterUpdate, ParameterType


class ParametersService:
    def __init__(self, db: Session):
        self.db = db
        self._initialize_default_parameters()

    def _initialize_default_parameters(self):
        """Initialize default parameters if they don't exist"""
        if self.db.query(ParameterConfig).count() == 0:
            default_params = self._get_default_parameters()
            for group_name, params in default_params.items():
                for param_name, param_data in params.items():
                    param_config = ParameterConfig(
                        group_name=group_name,
                        parameter_name=param_name,
                        value=json.dumps(param_data["value"]),
                        default_value=json.dumps(param_data["default_value"]),
                        description=param_data["description"],
                        parameter_type=param_data["type"],
                        options=json.dumps(param_data.get("options", [])),
                        min_value=param_data.get("min_value"),
                        max_value=param_data.get("max_value"),
                        required=param_data.get("required", True),
                    )
                    self.db.add(param_config)
            self.db.commit()

    def _get_default_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get default parameter configurations"""
        return {
            "data_fetching": {
                "default_symbols": {
                    "value": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                    "default_value": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                    "description": "Default symbols for data fetching",
                    "type": ParameterType.MULTISELECT,
                    "options": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"],
                    "required": True,
                },
                "default_timeframe": {
                    "value": "1day",
                    "default_value": "1day",
                    "description": "Default timeframe for data fetching",
                    "type": ParameterType.SELECT,
                    "options": ["1min", "5min", "15min", "1hour", "1day"],
                    "required": True,
                },
                "data_source": {
                    "value": "yfinance",
                    "default_value": "yfinance",
                    "description": "Data source for fetching market data",
                    "type": ParameterType.SELECT,
                    "options": ["yfinance", "alpha_vantage", "quandl"],
                    "required": True,
                },
                "cache_duration": {
                    "value": 3600,
                    "default_value": 3600,
                    "description": "Cache duration in seconds",
                    "type": ParameterType.INTEGER,
                    "min_value": 60,
                    "max_value": 86400,
                    "required": True,
                },
            },
            "signal_generation": {
                "default_strategy": {
                    "value": "momentum",
                    "default_value": "momentum",
                    "description": "Default trading strategy",
                    "type": ParameterType.SELECT,
                    "options": ["momentum", "mean_reversion", "breakout", "rsi", "macd"],
                    "required": True,
                },
                "confidence_threshold": {
                    "value": 0.7,
                    "default_value": 0.7,
                    "description": "Minimum confidence threshold for signals",
                    "type": ParameterType.FLOAT,
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "required": True,
                },
                "signal_strength_threshold": {
                    "value": "medium",
                    "default_value": "medium",
                    "description": "Minimum signal strength",
                    "type": ParameterType.SELECT,
                    "options": ["weak", "medium", "strong"],
                    "required": True,
                },
                "max_signals_per_day": {
                    "value": 10,
                    "default_value": 10,
                    "description": "Maximum number of signals per day",
                    "type": ParameterType.INTEGER,
                    "min_value": 1,
                    "max_value": 100,
                    "required": True,
                },
            },
            "backtesting": {
                "default_capital": {
                    "value": 100000,
                    "default_value": 100000,
                    "description": "Default starting capital for backtesting",
                    "type": ParameterType.INTEGER,
                    "min_value": 1000,
                    "max_value": 10000000,
                    "required": True,
                },
                "commission_rate": {
                    "value": 0.001,
                    "default_value": 0.001,
                    "description": "Commission rate per trade",
                    "type": ParameterType.FLOAT,
                    "min_value": 0.0,
                    "max_value": 0.01,
                    "required": True,
                },
                "slippage": {
                    "value": 0.0005,
                    "default_value": 0.0005,
                    "description": "Slippage rate per trade",
                    "type": ParameterType.FLOAT,
                    "min_value": 0.0,
                    "max_value": 0.01,
                    "required": True,
                },
                "risk_free_rate": {
                    "value": 0.02,
                    "default_value": 0.02,
                    "description": "Risk-free interest rate",
                    "type": ParameterType.FLOAT,
                    "min_value": 0.0,
                    "max_value": 0.1,
                    "required": True,
                },
            },
            "pipeline": {
                "default_mode": {
                    "value": "demo",
                    "default_value": "demo",
                    "description": "Default pipeline mode",
                    "type": ParameterType.SELECT,
                    "options": ["demo", "small", "medium", "full"],
                    "required": True,
                },
                "execution_interval": {
                    "value": 300,
                    "default_value": 300,
                    "description": "Pipeline execution interval in seconds",
                    "type": ParameterType.INTEGER,
                    "min_value": 60,
                    "max_value": 3600,
                    "required": True,
                },
                "max_concurrent_runs": {
                    "value": 3,
                    "default_value": 3,
                    "description": "Maximum concurrent pipeline runs",
                    "type": ParameterType.INTEGER,
                    "min_value": 1,
                    "max_value": 10,
                    "required": True,
                },
                "auto_restart": {
                    "value": True,
                    "default_value": True,
                    "description": "Automatically restart failed pipelines",
                    "type": ParameterType.BOOLEAN,
                    "required": True,
                },
            },
            "notifications": {
                "email_enabled": {
                    "value": False,
                    "default_value": False,
                    "description": "Enable email notifications",
                    "type": ParameterType.BOOLEAN,
                    "required": True,
                },
                "email_recipients": {
                    "value": [],
                    "default_value": [],
                    "description": "Email recipients for notifications",
                    "type": ParameterType.MULTISELECT,
                    "options": [],
                    "required": False,
                },
                "slack_enabled": {
                    "value": False,
                    "default_value": False,
                    "description": "Enable Slack notifications",
                    "type": ParameterType.BOOLEAN,
                    "required": True,
                },
                "notification_level": {
                    "value": "info",
                    "default_value": "info",
                    "description": "Minimum notification level",
                    "type": ParameterType.SELECT,
                    "options": ["debug", "info", "warning", "error"],
                    "required": True,
                },
            },
        }

    def get_parameter_groups(self) -> List[ParameterGroup]:
        """Get all parameter groups"""
        groups = {}

        # Get all parameters from database
        parameters = self.db.query(ParameterConfig).all()

        for param in parameters:
            if param.group_name not in groups:
                groups[param.group_name] = {
                    "group_name": param.group_name,
                    "display_name": param.group_name.replace("_", " ").title(),
                    "description": f"Configuration parameters for {param.group_name.replace('_', ' ')}",
                    "parameters": [],
                    "last_modified": None,
                }

            # Parse parameter data
            value = json.loads(param.value)
            default_value = json.loads(param.default_value)
            options = json.loads(param.options) if param.options else None

            param_value = ParameterValue(
                name=param.parameter_name,
                value=value,
                default_value=default_value,
                description=param.description,
                parameter_type=param.parameter_type,
                options=options,
                min_value=param.min_value,
                max_value=param.max_value,
                required=param.required,
                last_modified=param.last_modified,
            )

            groups[param.group_name]["parameters"].append(param_value)

            # Track last modified time
            if param.last_modified and (
                not groups[param.group_name]["last_modified"]
                or param.last_modified > groups[param.group_name]["last_modified"]
            ):
                groups[param.group_name]["last_modified"] = param.last_modified

        return [ParameterGroup(**group_data) for group_data in groups.values()]

    def update_parameters(self, updates: List[ParameterUpdate]):
        """Update parameter values"""
        for update in updates:
            param = (
                self.db.query(ParameterConfig)
                .filter(
                    ParameterConfig.group_name == update.group_name, ParameterConfig.parameter_name == update.parameter_name
                )
                .first()
            )

            if param:
                # Store old value for history
                old_value = param.value

                # Update parameter
                param.value = json.dumps(update.value)
                param.last_modified = datetime.utcnow()

                # Create history record
                history = ParameterHistory(
                    history_id=str(uuid.uuid4()),
                    group_name=update.group_name,
                    parameter_name=update.parameter_name,
                    old_value=old_value,
                    new_value=json.dumps(update.value),
                    changed_by="admin",  # In a real app, this would be the current user
                    change_time=datetime.utcnow(),
                    change_reason="Manual update",
                )

                self.db.add(history)

        self.db.commit()

    def reset_parameter_group(self, group_name: str):
        """Reset parameter group to default values"""
        parameters = self.db.query(ParameterConfig).filter(ParameterConfig.group_name == group_name).all()

        for param in parameters:
            # Store old value for history
            old_value = param.value

            # Reset to default
            param.value = param.default_value
            param.last_modified = datetime.utcnow()

            # Create history record
            history = ParameterHistory(
                history_id=str(uuid.uuid4()),
                group_name=group_name,
                parameter_name=param.parameter_name,
                old_value=old_value,
                new_value=param.default_value,
                changed_by="admin",
                change_time=datetime.utcnow(),
                change_reason="Reset to default",
            )

            self.db.add(history)

        self.db.commit()

    def export_parameters(self, format: str = "json") -> Dict[str, Any]:
        """Export parameters in specified format"""
        groups = self.get_parameter_groups()

        export_data = {}
        for group in groups:
            export_data[group.group_name] = {}
            for param in group.parameters:
                export_data[group.group_name][param.name] = param.value

        return {"format": format, "data": export_data, "exported_at": datetime.utcnow().isoformat()}

    def import_parameters(self, parameters_data: Dict[str, Any]):
        """Import parameters from data"""
        for group_name, params in parameters_data.items():
            for param_name, value in params.items():
                param = (
                    self.db.query(ParameterConfig)
                    .filter(ParameterConfig.group_name == group_name, ParameterConfig.parameter_name == param_name)
                    .first()
                )

                if param:
                    # Store old value for history
                    old_value = param.value

                    # Update parameter
                    param.value = json.dumps(value)
                    param.last_modified = datetime.utcnow()

                    # Create history record
                    history = ParameterHistory(
                        history_id=str(uuid.uuid4()),
                        group_name=group_name,
                        parameter_name=param_name,
                        old_value=old_value,
                        new_value=json.dumps(value),
                        changed_by="admin",
                        change_time=datetime.utcnow(),
                        change_reason="Import",
                    )

                    self.db.add(history)

        self.db.commit()

    def get_parameter_history(self, limit: int = 50) -> List[ParameterHistory]:
        """Get parameter change history"""
        history = self.db.query(ParameterHistory).order_by(desc(ParameterHistory.change_time)).limit(limit).all()

        return [
            ParameterHistory(
                history_id=h.history_id,
                group_name=h.group_name,
                parameter_name=h.parameter_name,
                old_value=h.old_value,
                new_value=h.new_value,
                changed_by=h.changed_by,
                change_time=h.change_time,
                change_reason=h.change_reason,
            )
            for h in history
        ]
