"""
Pipeline Orchestrator
====================

Main orchestration class that coordinates all BreadthFlow components:
- Data fetching
- Signal generation
- Backtesting
- Training
- Component management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

# Optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Core system imports
from model.registry.component_registry import ComponentRegistry
from model.config.configuration_manager import ConfigurationManager
from model.logging.error_handler import ErrorHandler
from model.logging.enhanced_logger import EnhancedLogger
from model.logging.error_recovery import ErrorRecovery

# Data layer imports
from model.data.universal_data_fetcher import UniversalDataFetcher
from model.data.resources.data_resources import DataResource

# Signal layer imports
from model.signals.composite_signal_generator import CompositeSignalGenerator
from model.signals.signal_config import SignalConfig

# Backtesting layer imports
from model.backtesting.backtest_config import BacktestConfig
from model.backtesting.engines.standard_backtest_engine import StandardBacktestEngine

# Training layer imports (when available)
try:
    from model.training.training_manager import TrainingManager

    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestration"""

    pipeline_id: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    data_resources: List[DataResource]
    signal_config: SignalConfig
    backtest_config: BacktestConfig
    enable_training: bool = False
    training_config: Optional[Dict[str, Any]] = None
    max_workers: int = 4
    timeout_seconds: int = 3600


@dataclass
class PipelineResult:
    """Result of a pipeline execution"""

    pipeline_id: str
    success: bool
    start_time: datetime
    end_time: datetime
    data_fetch_result: Optional[Dict[str, Any]] = None
    signal_result: Optional[Dict[str, Any]] = None
    backtest_result: Optional[Dict[str, Any]] = None
    training_result: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class PipelineOrchestrator:
    """
    Main orchestration class for BreadthFlow system

    Coordinates all components and provides unified interface for:
    - Complete pipeline execution
    - Component management
    - Error handling and recovery
    - Performance monitoring
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline orchestrator

        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = EnhancedLogger("PipelineOrchestrator")
        self.error_handler = ErrorHandler()

        # Initialize core systems
        self.component_registry = ComponentRegistry()
        self.config_manager = ConfigurationManager()

        # Load configuration
        if config_path:
            self.config_manager.load_config(config_path)

        # Initialize components
        self._initialize_components()

        # Performance tracking
        self.performance_metrics = {}

    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Register default components
            try:
                from model.registry.register_components import register_default_components

                register_default_components(self.component_registry)
            except ImportError:
                # If register_components doesn't exist, skip component registration
                self.logger.warning("register_components module not found, skipping component registration")

            # Initialize data fetcher
            self.data_fetcher = UniversalDataFetcher(config={"component_registry": self.component_registry})

            # Initialize signal generator
            self.signal_generator = CompositeSignalGenerator(config={"component_registry": self.component_registry})

            # Initialize backtest engine
            self.backtest_engine = StandardBacktestEngine(
                name="standard_backtest",
                config=BacktestConfig(name="default_backtest", symbols=[], start_date=datetime.now(), end_date=datetime.now()),
            )

            # Initialize training manager (if available)
            if TRAINING_AVAILABLE:
                self.training_manager = TrainingManager(config={"component_registry": self.component_registry})
            else:
                self.training_manager = None

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.error_handler.record_error(
                error=e,
                context={"component": "PipelineOrchestrator"},
                component="PipelineOrchestrator",
                operation="initialize_components",
            )
            raise

    @ErrorRecovery.retry(max_attempts=3, backoff_factor=2)
    async def run_pipeline(self, config: PipelineConfig) -> PipelineResult:
        """
        Run a complete pipeline from data fetching to backtesting

        Args:
            config: Pipeline configuration

        Returns:
            PipelineResult with execution results
        """
        start_time = datetime.now()
        result = PipelineResult(
            pipeline_id=config.pipeline_id, success=False, start_time=start_time, end_time=start_time, errors=[]
        )

        self.logger.info(f"Starting pipeline {config.pipeline_id}")

        try:
            # Step 1: Data Fetching
            self.logger.info("Step 1: Fetching data")
            data_result = await self._fetch_data(config)
            result.data_fetch_result = data_result

            if not data_result.get("success", False):
                result.errors.append("Data fetching failed")
                return result

            # Step 2: Signal Generation
            self.logger.info("Step 2: Generating signals")
            signal_result = await self._generate_signals(config, data_result["data"])
            result.signal_result = signal_result

            if not signal_result.get("success", False):
                result.errors.append("Signal generation failed")
                return result

            # Step 3: Training (if enabled)
            if config.enable_training and self.training_manager:
                self.logger.info("Step 3: Training models")
                training_result = await self._train_models(config, data_result["data"])
                result.training_result = training_result

                if not training_result.get("success", False):
                    result.errors.append("Training failed")
                    return result

            # Step 4: Backtesting
            self.logger.info("Step 4: Running backtest")
            backtest_result = await self._run_backtest(config, signal_result["signals"])
            result.backtest_result = backtest_result

            if not backtest_result.get("success", False):
                result.errors.append("Backtesting failed")
                return result

            # Success
            result.success = True
            result.end_time = datetime.now()
            result.performance_metrics = self._calculate_performance_metrics(result)

            self.logger.info(f"Pipeline {config.pipeline_id} completed successfully")

        except Exception as e:
            result.errors.append(f"Pipeline execution failed: {str(e)}")
            self.logger.error(f"Pipeline {config.pipeline_id} failed: {e}")
            self.error_handler.record_error(
                error=e,
                context={"pipeline_id": config.pipeline_id},
                component="PipelineOrchestrator",
                operation="run_pipeline",
            )

        return result

    async def _fetch_data(self, config: PipelineConfig) -> Dict[str, Any]:
        """Fetch data for all symbols and resources"""
        try:
            data = {}
            for symbol in config.symbols:
                symbol_data = {}
                for resource in config.data_resources:
                    resource_data = self.data_fetcher.fetch_data(
                        resource_name=resource.name, symbols=[symbol], start_date=config.start_date, end_date=config.end_date
                    )
                    symbol_data[resource.name] = resource_data
                data[symbol] = symbol_data

            return {
                "success": True,
                "data": data,
                "symbols": config.symbols,
                "resources": [r.name for r in config.data_resources],
            }

        except Exception as e:
            self.logger.error(f"Data fetching failed: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_signals(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate signals using the composite signal generator"""
        try:
            signals = await self.signal_generator.generate_signals(config=config.signal_config, data=data)

            return {"success": True, "signals": signals, "strategy": config.signal_config.strategy_name}

        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _train_models(self, config: PipelineConfig, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train models if training is enabled"""
        try:
            if not self.training_manager:
                return {"success": False, "error": "Training manager not available"}

            training_result = await self.training_manager.train_models(data=data, config=config.training_config or {})

            return {
                "success": True,
                "models_trained": training_result.get("models_trained", []),
                "performance": training_result.get("performance", {}),
            }

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    async def _run_backtest(self, config: PipelineConfig, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest using the configured engine"""
        try:
            backtest_result = await self.backtest_engine.run_backtest(signals=signals, config=config.backtest_config)

            return {"success": True, "results": backtest_result, "metrics": backtest_result.get("performance_metrics", {})}

        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_performance_metrics(self, result: PipelineResult) -> Dict[str, Any]:
        """Calculate performance metrics for the pipeline"""
        duration = (result.end_time - result.start_time).total_seconds()

        return {
            "total_duration_seconds": duration,
            "steps_completed": sum(
                [
                    result.data_fetch_result is not None,
                    result.signal_result is not None,
                    result.backtest_result is not None,
                    result.training_result is not None,
                ]
            ),
            "success_rate": 1.0 if result.success else 0.0,
            "error_count": len(result.errors),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health"""
        return {
            "component_registry": {
                "total_components": len(self.component_registry.list_components()),
                "data_sources": len(self.component_registry.list_components("data_source")),
                "signal_generators": len(self.component_registry.list_components("signal_generator")),
                "backtest_engines": len(self.component_registry.list_components("backtest_engine")),
            },
            "error_handler": {
                "total_errors": len(self.error_handler.get_errors()),
                "critical_errors": len([e for e in self.error_handler.get_errors() if e.severity == "CRITICAL"]),
                "recent_errors": len(
                    [e for e in self.error_handler.get_errors() if e.timestamp > datetime.now() - timedelta(hours=1)]
                ),
            },
            "performance": self.performance_metrics,
            "training_available": TRAINING_AVAILABLE,
        }

    def list_available_components(self) -> Dict[str, List[str]]:
        """List all available components by type"""
        components = self.component_registry.list_components()

        categorized = {"data_sources": [], "signal_generators": [], "backtest_engines": [], "training_strategies": []}

        for component in components:
            component_type = component.get("type", "unknown")
            if component_type in categorized:
                categorized[component_type].append(component["name"])

        return categorized
