"""
Orchestration System
===================

This package provides the main orchestration layer that integrates all components:
- Data fetching
- Signal generation
- Backtesting
- Training
- Component registry
- Configuration management
- Error handling and logging

The orchestration system provides a unified interface for running complete workflows
and managing the entire BreadthFlow system.
"""

from .pipeline_orchestrator import PipelineOrchestrator
from .system_monitor import SystemMonitor
from .workflow_manager import WorkflowManager

__all__ = ["PipelineOrchestrator", "WorkflowManager", "SystemMonitor"]
