"""
Configuration Manager Implementation

Centralized configuration management for BreadthFlow system
with validation, schema support, and component-specific settings.
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation"""

    name: str
    type: str
    required: bool = False
    default: Any = None
    description: str = ""
    validation_rules: Dict[str, Any] = None


class ConfigurationManager:
    """Manages system configuration and component settings"""

    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.schemas: Dict[str, Dict[str, ConfigSchema]] = {}
        self._load_configurations()

    def load_config(self, config_name: str, config_type: str = "yaml") -> Dict[str, Any]:
        """Load configuration from file"""

        config_file = self.config_path / f"{config_name}.{config_type}"

        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}

        try:
            with open(config_file, "r") as f:
                if config_type == "yaml":
                    config_data = yaml.safe_load(f)
                elif config_type == "json":
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")

            # Validate configuration
            if config_name in self.schemas:
                self._validate_config(config_data, self.schemas[config_name])

            self.configs[config_name] = config_data
            logger.info(f"✅ Loaded configuration: {config_name}")
            return config_data

        except Exception as e:
            logger.error(f"❌ Failed to load configuration {config_name}: {e}")
            return {}

    def save_config(self, config_name: str, config_data: Dict[str, Any], config_type: str = "yaml"):
        """Save configuration to file"""

        # Ensure config directory exists
        self.config_path.mkdir(parents=True, exist_ok=True)

        config_file = self.config_path / f"{config_name}.{config_type}"

        try:
            with open(config_file, "w") as f:
                if config_type == "yaml":
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif config_type == "json":
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")

            self.configs[config_name] = config_data
            logger.info(f"✅ Saved configuration: {config_name}")

        except Exception as e:
            logger.error(f"❌ Failed to save configuration {config_name}: {e}")

    def get_config(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""

        if config_name not in self.configs:
            self.load_config(config_name)

        if config_name not in self.configs:
            return default

        config_data = self.configs[config_name]

        if key is None:
            return config_data

        # Support nested keys (e.g., "database.host")
        keys = key.split(".")
        value = config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set_config(self, config_name: str, key: str, value: Any):
        """Set configuration value"""

        if config_name not in self.configs:
            self.configs[config_name] = {}

        # Support nested keys
        keys = key.split(".")
        config_data = self.configs[config_name]

        for k in keys[:-1]:
            if k not in config_data:
                config_data[k] = {}
            config_data = config_data[k]

        config_data[keys[-1]] = value

    def validate_config(self, config_data: Dict[str, Any], schema: Dict[str, ConfigSchema]) -> bool:
        """Validate configuration against schema"""

        for field_name, field_schema in schema.items():
            if field_schema.required and field_name not in config_data:
                logger.error(f"❌ Required field missing: {field_name}")
                return False

            if field_name in config_data:
                value = config_data[field_name]

                # Type validation
                if not self._validate_type(value, field_schema.type):
                    logger.error(f"❌ Invalid type for {field_name}: expected {field_schema.type}")
                    return False

                # Custom validation rules
                if field_schema.validation_rules:
                    if not self._validate_rules(value, field_schema.validation_rules):
                        logger.error(f"❌ Validation failed for {field_name}")
                        return False

        return True

    def get_component_config(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """Get configuration for specific component"""

        # Try component-specific config first
        component_config = self.get_config(f"{component_type}_{component_name}")
        if component_config:
            return component_config

        # Fall back to component type config
        type_config = self.get_config(f"{component_type}_default")
        if type_config:
            return type_config

        # Fall back to global config
        return self.get_config("global", f"components.{component_type}", {})

    def register_schema(self, config_name: str, schema: Dict[str, ConfigSchema]):
        """Register configuration schema for validation"""
        self.schemas[config_name] = schema

    def _validate_config(self, config_data: Dict[str, Any], schema: Dict[str, ConfigSchema]):
        """Internal validation method"""
        if not self.validate_config(config_data, schema):
            raise ValueError(f"Configuration validation failed for {config_name}")

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""

        type_mapping = {"string": str, "integer": int, "float": (int, float), "boolean": bool, "list": list, "dict": dict}

        expected_class = type_mapping.get(expected_type)
        if expected_class is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected_class)

    def _validate_rules(self, value: Any, rules: Dict[str, Any]) -> bool:
        """Validate against custom rules"""

        for rule, rule_value in rules.items():
            if rule == "min" and value < rule_value:
                return False
            elif rule == "max" and value > rule_value:
                return False
            elif rule == "min_length" and len(value) < rule_value:
                return False
            elif rule == "max_length" and len(value) > rule_value:
                return False
            elif rule == "pattern" and not re.match(rule_value, str(value)):
                return False

        return True

    def _load_configurations(self):
        """Load all configuration files on startup"""

        if not self.config_path.exists():
            return

        # Load all yaml and json files
        for config_file in self.config_path.glob("*.yaml"):
            config_name = config_file.stem
            self.load_config(config_name, "yaml")

        for config_file in self.config_path.glob("*.json"):
            config_name = config_file.stem
            self.load_config(config_name, "json")
