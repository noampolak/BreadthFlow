"""
Base Signal Strategy

Abstract base class for signal generation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
from datetime import datetime
import logging
from ..signal_config import SignalConfig

logger = logging.getLogger(__name__)


class BaseSignalStrategy(ABC):
    """Abstract base class for signal generation strategies"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.required_data = []
        self.supported_timeframes = []

        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0.0,
        }

    @abstractmethod
    def generate_signals(self, data: Dict[str, Any], config: SignalConfig):
        """Generate signals based on data and configuration"""
        pass

    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate that required data is available"""
        if not PANDAS_AVAILABLE:
            return False
        pass

    def get_name(self) -> str:
        """Get strategy name"""
        return self.name

    def get_config(self) -> Dict[str, Any]:
        """Get strategy configuration"""
        return self.config

    def get_required_data(self) -> List[str]:
        """Get list of required data resources"""
        return self.required_data

    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes"""
        return self.supported_timeframes

    def validate_config(self, config: SignalConfig) -> bool:
        """Validate strategy configuration"""
        # Check if required data is available
        for required_resource in self.required_data:
            if required_resource not in config.required_resources:
                logger.error(f"Strategy {self.name} requires {required_resource} but it's not in config")
                return False

        # Check if timeframe is supported
        for timeframe in config.required_timeframes:
            if timeframe not in self.supported_timeframes:
                logger.warning(f"Strategy {self.name} may not support timeframe {timeframe}")

        return True

    def preprocess_data(self, data: Dict[str, Any], config: SignalConfig):
        """Preprocess data before signal generation"""
        if not PANDAS_AVAILABLE:
            return {}

        processed_data = {}

        for resource_name, resource_data in data.items():
            if resource_data.empty:
                logger.warning(f"Empty data for resource: {resource_name}")
                continue

            # Ensure data has minimum required points
            if len(resource_data) < config.min_data_points:
                logger.warning(
                    f"Insufficient data points for {resource_name}: {len(resource_data)} < {config.min_data_points}"
                )
                continue

            # Sort by date if date column exists
            if "date" in resource_data.columns:
                resource_data = resource_data.sort_values("date").reset_index(drop=True)

            # Remove duplicates
            resource_data = resource_data.drop_duplicates()

            processed_data[resource_name] = resource_data

        return processed_data

    def postprocess_signals(self, signals, config: SignalConfig):
        """Postprocess generated signals"""
        if not PANDAS_AVAILABLE:
            return {}

        if signals.empty:
            return signals

        # Apply signal threshold
        if "signal_strength" in signals.columns:
            signals = signals[signals["signal_strength"].abs() >= config.signal_threshold]

        # Apply confidence threshold
        if "confidence" in signals.columns:
            signals = signals[signals["confidence"] >= config.confidence_threshold]

        # Apply signal smoothing if enabled
        if config.signal_smoothing and "signal_strength" in signals.columns:
            signals["signal_strength"] = signals["signal_strength"].rolling(window=3, center=True).mean()

        # Add metadata if requested
        if config.include_metadata:
            signals["strategy_name"] = self.name
            signals["generation_timestamp"] = datetime.now()
            signals["config_hash"] = hash(str(config.to_dict()))

        return signals

    def calculate_signal_confidence(self, signals, confidence_factors: List[str]):
        """Calculate signal confidence based on multiple factors"""
        if not PANDAS_AVAILABLE:
            return None

        if signals.empty or not confidence_factors:
            return pd.Series(0.5, index=signals.index)

        confidence_scores = []

        for factor in confidence_factors:
            if factor in signals.columns:
                # Normalize factor to 0-1 range
                factor_data = signals[factor]
                if factor_data.dtype in ["float64", "int64"]:
                    normalized = (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min())
                    confidence_scores.append(normalized)

        if not confidence_scores:
            return pd.Series(0.5, index=signals.index)

        # Calculate average confidence
        confidence = pd.concat(confidence_scores, axis=1).mean(axis=1)

        return confidence

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        metrics = self.generation_stats.copy()

        if metrics["total_generations"] > 0:
            metrics["success_rate"] = metrics["successful_generations"] / metrics["total_generations"]
        else:
            metrics["success_rate"] = 0.0

        return metrics

    def update_performance_stats(self, success: bool, generation_time: float):
        """Update performance statistics"""
        self.generation_stats["total_generations"] += 1

        if success:
            self.generation_stats["successful_generations"] += 1
        else:
            self.generation_stats["failed_generations"] += 1

        # Update average generation time
        current_avg = self.generation_stats["average_generation_time"]
        total_generations = self.generation_stats["total_generations"]

        self.generation_stats["average_generation_time"] = (
            current_avg * (total_generations - 1) + generation_time
        ) / total_generations

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "required_data": self.required_data,
            "supported_timeframes": self.supported_timeframes,
            "performance_metrics": self.get_performance_metrics(),
        }
