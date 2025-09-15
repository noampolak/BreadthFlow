"""
Signal Configuration

Defines configuration structures for signal generation in the BreadthFlow system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SignalConfig:
    """Configuration for signal generation"""

    # Basic configuration
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime

    # Strategy-specific parameters
    parameters: Dict[str, Any] = None

    # Data requirements
    required_resources: List[str] = None
    required_timeframes: List[str] = None

    # Signal generation settings
    signal_threshold: float = 0.5
    confidence_threshold: float = 0.7
    min_data_points: int = 30

    # Output configuration
    output_format: str = "dataframe"  # "dataframe", "json", "dict"
    include_metadata: bool = True
    include_confidence: bool = True

    # Performance settings
    enable_caching: bool = True
    cache_duration: int = 3600  # seconds
    parallel_processing: bool = False
    max_workers: int = 4

    # Validation settings
    validate_signals: bool = True
    outlier_detection: bool = True
    signal_smoothing: bool = False

    def __post_init__(self):
        """Initialize default values"""
        if self.parameters is None:
            self.parameters = {}

        if self.required_resources is None:
            self.required_resources = ["stock_price"]

        if self.required_timeframes is None:
            self.required_timeframes = ["1day"]

    def validate(self) -> bool:
        """Validate signal configuration"""
        if not self.strategy_name:
            return False

        if not self.symbols:
            return False

        if self.start_date >= self.end_date:
            return False

        if self.signal_threshold < 0 or self.signal_threshold > 1:
            return False

        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "parameters": self.parameters,
            "required_resources": self.required_resources,
            "required_timeframes": self.required_timeframes,
            "signal_threshold": self.signal_threshold,
            "confidence_threshold": self.confidence_threshold,
            "min_data_points": self.min_data_points,
            "output_format": self.output_format,
            "include_metadata": self.include_metadata,
            "include_confidence": self.include_confidence,
            "enable_caching": self.enable_caching,
            "cache_duration": self.cache_duration,
            "parallel_processing": self.parallel_processing,
            "max_workers": self.max_workers,
            "validate_signals": self.validate_signals,
            "outlier_detection": self.outlier_detection,
            "signal_smoothing": self.signal_smoothing,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SignalConfig":
        """Create configuration from dictionary"""
        # Convert date strings back to datetime
        if "start_date" in config_dict and isinstance(config_dict["start_date"], str):
            config_dict["start_date"] = datetime.fromisoformat(config_dict["start_date"])

        if "end_date" in config_dict and isinstance(config_dict["end_date"], str):
            config_dict["end_date"] = datetime.fromisoformat(config_dict["end_date"])

        return cls(**config_dict)
