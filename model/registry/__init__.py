"""
Component Registry System

This module provides the central component registry for BreadthFlow,
enabling dynamic component discovery, registration, and management.
"""

from .component_registry import ComponentRegistry, ComponentMetadata
from .register_components import register_default_components

__all__ = ['ComponentRegistry', 'ComponentMetadata', 'register_default_components']
