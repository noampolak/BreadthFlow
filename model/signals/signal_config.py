"""
Signal Configuration

Defines configuration structures for signal generation in the BreadthFlow system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SignalConfig:
    """Configuration for signal generation"""

    # Basic configuration
    strategy_name: str = "default"
    symbols: List[str] = None
    start_date: datetime = None
    end_date: datetime = None

    # Strategy-specific parameters
    parameters: Dict[str, Any] = None

    # Test compatibility parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    pe_threshold: float = 15.0
    pb_threshold: float = 1.5
    roe_threshold: float = 0.15

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

        if self.symbols is None:
            self.symbols = ["AAPL"]

        if self.start_date is None:
            from datetime import datetime, timedelta
            self.start_date = datetime.now() - timedelta(days=365)

        if self.end_date is None:
            from datetime import datetime
            self.end_date = datetime.now()

    def validate(self) -> bool:
        """Validate signal configuration"""
        if not self.strategy_name:
            raise ValueError("Strategy name is required")

        if not self.symbols:
            raise ValueError("Symbols list cannot be empty")

        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")

        if self.signal_threshold < 0 or self.signal_threshold > 1:
            raise ValueError("Signal threshold must be between 0 and 1")

        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")

        # Validate test parameters
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")

        if self.macd_fast <= 0 or self.macd_slow <= 0 or self.macd_signal <= 0:
            raise ValueError("MACD parameters must be positive")

        if self.bb_period <= 0 or self.bb_std <= 0:
            raise ValueError("Bollinger Bands parameters must be positive")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "strategy_name": self.strategy_name,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
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
            # Test compatibility parameters
            "rsi_period": self.rsi_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "pe_threshold": self.pe_threshold,
            "pb_threshold": self.pb_threshold,
            "roe_threshold": self.roe_threshold,
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
