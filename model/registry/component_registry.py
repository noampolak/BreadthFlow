"""
Component Registry Implementation

Central registry for managing all BreadthFlow system components
with dynamic discovery, validation, and lifecycle management.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class ComponentMetadata:
    """Metadata for registered components"""

    name: str
    type: str
    version: str
    description: str
    author: str
    created_date: datetime
    last_updated: datetime
    dependencies: List[str]
    configuration_schema: Dict[str, Any]
    is_active: bool = True


class ComponentRegistry:
    """Central registry for all system components"""

    def __init__(self, registry_path: str = "config/component_registry.json"):
        self.registry_path = registry_path
        self.components: Dict[str, Dict[str, Any]] = {}
        self.component_metadata: Dict[str, ComponentMetadata] = {}
        self._load_registry()

    def register_component(self, component_type: str, name: str, component_class: Type, metadata: ComponentMetadata):
        """Register a new component"""

        if component_type not in self.components:
            self.components[component_type] = {}

        # Validate component
        if not self._validate_component(component_class, metadata):
            raise ValueError(f"Invalid component: {name}")

        # Register component
        self.components[component_type][name] = {"class": component_class, "metadata": metadata, "instance": None}

        self.component_metadata[f"{component_type}:{name}"] = metadata
        self._save_registry()

        logger.info(f"✅ Registered {component_type}:{name}")

    def get_component(self, component_type: str, name: str, create_instance: bool = True, **kwargs):
        """Get a component by type and name"""

        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")

        if name not in self.components[component_type]:
            raise ValueError(f"Unknown component: {component_type}:{name}")

        component_info = self.components[component_type][name]

        # Create instance if requested and not exists
        if create_instance and component_info["instance"] is None:
            component_info["instance"] = component_info["class"](**kwargs)

        return component_info["instance"] if create_instance else component_info["class"]

    def list_components(self, component_type: str = None) -> List[Dict[str, Any]]:
        """List all components or components of a specific type"""

        if component_type:
            if component_type not in self.components:
                return []

            return [
                {"name": name, "type": component_type, "metadata": asdict(info["metadata"])}
                for name, info in self.components[component_type].items()
            ]
        else:
            all_components = []
            for comp_type, components in self.components.items():
                for name, info in components.items():
                    all_components.append({"name": name, "type": comp_type, "metadata": asdict(info["metadata"])})
            return all_components

    def validate_component(self, component_type: str, name: str) -> bool:
        """Validate component configuration"""

        if component_type not in self.components:
            return False

        if name not in self.components[component_type]:
            return False

        component_info = self.components[component_type][name]
        metadata = component_info["metadata"]

        # Check if component is active
        if not metadata.is_active:
            return False

        # Check dependencies
        for dependency in metadata.dependencies:
            if not self._check_dependency(dependency):
                return False

        return True

    def update_component(self, component_type: str, name: str, updates: Dict[str, Any]):
        """Update component metadata"""

        if component_type not in self.components or name not in self.components[component_type]:
            raise ValueError(f"Component not found: {component_type}:{name}")

        component_info = self.components[component_type][name]
        metadata = component_info["metadata"]

        # Update metadata fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.last_updated = datetime.now()
        self._save_registry()

    def deactivate_component(self, component_type: str, name: str):
        """Deactivate a component"""
        self.update_component(component_type, name, {"is_active": False})

    def activate_component(self, component_type: str, name: str):
        """Activate a component"""
        self.update_component(component_type, name, {"is_active": True})

    def remove_component(self, component_type: str, name: str):
        """Remove a component from registry"""

        if component_type in self.components and name in self.components[component_type]:
            del self.components[component_type][name]

            metadata_key = f"{component_type}:{name}"
            if metadata_key in self.component_metadata:
                del self.component_metadata[metadata_key]

            self._save_registry()

    def _validate_component(self, component_class: Type, metadata: ComponentMetadata) -> bool:
        """Validate component class and metadata"""

        # Check if class has required methods (basic validation)
        required_methods = self._get_required_methods(metadata.type)

        for method in required_methods:
            if not hasattr(component_class, method):
                logger.error(f"❌ Component {metadata.name} missing required method: {method}")
                return False

        return True

    def _get_required_methods(self, component_type: str) -> List[str]:
        """Get required methods for component type"""

        method_requirements = {
            "data_source": ["get_name", "get_supported_resources", "fetch_data"],
            "signal_generator": ["get_name", "get_config", "generate_signals"],
            "backtest_engine": ["get_name", "get_config", "run_backtest"],
            "training_strategy": ["get_name", "get_config", "train_model"],
            "execution_engine": ["execute_trade", "calculate_execution_price"],
            "risk_manager": ["validate_trade", "calculate_position_size"],
            "performance_analyzer": ["calculate_returns", "analyze_trades"],
        }

        return method_requirements.get(component_type, [])

    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is available"""

        # Parse dependency (format: "type:name")
        if ":" in dependency:
            dep_type, dep_name = dependency.split(":", 1)
            return self.validate_component(dep_type, dep_name)

        return True  # Assume satisfied if no specific component required

    def _load_registry(self):
        """Load component registry from disk"""

        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    registry_data = json.load(f)

                    # Load components
                    self.components = registry_data.get("components", {})

                    # Load metadata
                    metadata_data = registry_data.get("metadata", {})
                    for key, meta_dict in metadata_data.items():
                        # Convert datetime strings back to datetime objects
                        meta_dict["created_date"] = datetime.fromisoformat(meta_dict["created_date"])
                        meta_dict["last_updated"] = datetime.fromisoformat(meta_dict["last_updated"])
                        self.component_metadata[key] = ComponentMetadata(**meta_dict)

            except Exception as e:
                logger.warning(f"⚠️ Warning: Could not load component registry: {e}")
                self.components = {}
                self.component_metadata = {}

    def _save_registry(self):
        """Save component registry to disk"""

        # Prepare data for serialization
        registry_data = {
            "components": self.components,
            "metadata": {key: asdict(metadata) for key, metadata in self.component_metadata.items()},
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)

        # Save to file
        with open(self.registry_path, "w") as f:
            json.dump(registry_data, f, indent=2, default=str)
