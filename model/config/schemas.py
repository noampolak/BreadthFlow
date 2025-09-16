"""
Configuration Schemas

Predefined schemas for configuration validation across different
component types in the BreadthFlow system.
"""

from .configuration_manager import ConfigSchema

# Global system configuration schema
GLOBAL_CONFIG_SCHEMA = {
    "database": ConfigSchema(name="database", type="dict", required=True, description="Database configuration"),
    "logging": ConfigSchema(
        name="logging", type="dict", required=False, default={"level": "INFO"}, description="Logging configuration"
    ),
    "components": ConfigSchema(
        name="components", type="dict", required=False, default={}, description="Component-specific configurations"
    ),
}

# Data source configuration schema
DATA_SOURCE_CONFIG_SCHEMA = {
    "api_key": ConfigSchema(name="api_key", type="string", required=False, description="API key for data source"),
    "rate_limits": ConfigSchema(
        name="rate_limits",
        type="dict",
        required=False,
        default={"requests_per_minute": 60},
        description="Rate limiting configuration",
    ),
    "timeout": ConfigSchema(
        name="timeout",
        type="integer",
        required=False,
        default=30,
        validation_rules={"min": 1, "max": 300},
        description="Request timeout in seconds",
    ),
}

# Signal generator configuration schema
SIGNAL_GENERATOR_CONFIG_SCHEMA = {
    "parameters": ConfigSchema(name="parameters", type="dict", required=True, description="Signal generation parameters"),
    "required_data": ConfigSchema(name="required_data", type="list", required=True, description="Required data resources"),
    "output_format": ConfigSchema(
        name="output_format", type="string", required=False, default="standard", description="Output format for signals"
    ),
}

# Backtest engine configuration schema
BACKTEST_ENGINE_CONFIG_SCHEMA = {
    "initial_capital": ConfigSchema(
        name="initial_capital",
        type="float",
        required=True,
        validation_rules={"min": 1000},
        description="Initial capital for backtesting",
    ),
    "commission_rate": ConfigSchema(
        name="commission_rate",
        type="float",
        required=False,
        default=0.001,
        validation_rules={"min": 0, "max": 0.1},
        description="Commission rate for trades",
    ),
    "slippage_rate": ConfigSchema(
        name="slippage_rate",
        type="float",
        required=False,
        default=0.0005,
        validation_rules={"min": 0, "max": 0.01},
        description="Slippage rate for trade execution",
    ),
}

# Training strategy configuration schema
TRAINING_STRATEGY_CONFIG_SCHEMA = {
    "model_type": ConfigSchema(name="model_type", type="string", required=True, description="Type of model to train"),
    "parameters": ConfigSchema(name="parameters", type="dict", required=False, default={}, description="Training parameters"),
    "validation_split": ConfigSchema(
        name="validation_split",
        type="float",
        required=False,
        default=0.2,
        validation_rules={"min": 0.1, "max": 0.5},
        description="Validation data split ratio",
    ),
}
