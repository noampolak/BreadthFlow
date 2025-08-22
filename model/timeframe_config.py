#!/usr/bin/env python3
"""
Timeframe Configuration Management

This module provides centralized configuration management for timeframe-specific
settings, parameters, and database schema enhancements.

Key Features:
- Centralized timeframe configuration
- Database schema for timeframe metadata
- Parameter optimization by timeframe
- Validation and error handling
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe."""
    timeframe: str
    display_name: str
    description: str
    
    # Data fetching parameters
    default_lookback_days: int
    max_lookback_days: int
    min_data_points: int
    
    # Technical analysis parameters
    ma_short_period: int
    ma_long_period: int
    rsi_period: int
    rsi_oversold: float
    rsi_overbought: float
    bb_period: int
    bb_std_dev: float
    volume_ma_period: int
    
    # Signal generation parameters
    min_confidence: float
    price_change_threshold: float
    volume_ratio_threshold: float
    
    # Backtesting parameters
    commission_rate: float
    slippage_rate: float
    market_impact: float
    max_position_size: float
    execution_delay_bars: int
    bid_ask_spread: float
    liquidity_requirement: int
    
    # Data storage parameters
    storage_folder: str
    file_suffix: str
    compression: str
    
    # Performance parameters
    batch_size: int
    parallel_workers: int
    memory_allocation: str
    
    # Market hours and availability
    intraday: bool
    market_hours_only: bool
    weekend_data: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeframeConfig':
        """Create from dictionary."""
        return cls(**data)

class TimeframeConfigManager:
    """
    Centralized timeframe configuration manager.
    
    Provides access to all timeframe-specific configurations and parameters.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or "/opt/bitnami/spark/jobs/config/timeframe_config.json"
        self.configs: Dict[str, TimeframeConfig] = {}
        self.supported_timeframes = ['1min', '5min', '15min', '1hour', '1day']
        
        # Initialize with default configurations
        self._initialize_default_configs()
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            self.load_from_file()
        else:
            logger.info(f"No config file found at {self.config_file}, using defaults")
    
    def _initialize_default_configs(self):
        """Initialize with optimal default configurations for each timeframe."""
        
        # 1-minute timeframe configuration
        self.configs['1min'] = TimeframeConfig(
            timeframe='1min',
            display_name='1 Minute',
            description='Ultra-high frequency trading with 1-minute bars',
            
            # Data fetching
            default_lookback_days=7,
            max_lookback_days=30,
            min_data_points=20,
            
            # Technical analysis (fast-reacting)
            ma_short_period=5,
            ma_long_period=10,
            rsi_period=8,
            rsi_oversold=10.0,
            rsi_overbought=90.0,
            bb_period=10,
            bb_std_dev=1.2,
            volume_ma_period=10,
            
            # Signal generation (sensitive)
            min_confidence=0.2,
            price_change_threshold=0.005,  # 0.5%
            volume_ratio_threshold=1.05,
            
            # Backtesting (high costs)
            commission_rate=0.003,      # 0.3%
            slippage_rate=0.0025,       # 0.25%
            market_impact=0.002,        # 0.2%
            max_position_size=0.03,     # 3%
            execution_delay_bars=0,
            bid_ask_spread=0.001,       # 0.1%
            liquidity_requirement=50000,
            
            # Storage
            storage_folder='minute',
            file_suffix='_1M',
            compression='snappy',
            
            # Performance
            batch_size=1000,
            parallel_workers=4,
            memory_allocation='2g',
            
            # Market hours
            intraday=True,
            market_hours_only=True,
            weekend_data=False
        )
        
        # 5-minute timeframe configuration
        self.configs['5min'] = TimeframeConfig(
            timeframe='5min',
            display_name='5 Minutes',
            description='High frequency trading with 5-minute bars',
            
            # Data fetching
            default_lookback_days=14,
            max_lookback_days=60,
            min_data_points=24,
            
            # Technical analysis
            ma_short_period=6,
            ma_long_period=12,
            rsi_period=10,
            rsi_oversold=15.0,
            rsi_overbought=85.0,
            bb_period=12,
            bb_std_dev=1.4,
            volume_ma_period=15,
            
            # Signal generation
            min_confidence=0.3,
            price_change_threshold=0.008,  # 0.8%
            volume_ratio_threshold=1.1,
            
            # Backtesting
            commission_rate=0.0025,     # 0.25%
            slippage_rate=0.002,        # 0.2%
            market_impact=0.0015,       # 0.15%
            max_position_size=0.05,     # 5%
            execution_delay_bars=0,
            bid_ask_spread=0.0005,      # 0.05%
            liquidity_requirement=100000,
            
            # Storage
            storage_folder='minute',
            file_suffix='_5M',
            compression='snappy',
            
            # Performance
            batch_size=2000,
            parallel_workers=3,
            memory_allocation='2g',
            
            # Market hours
            intraday=True,
            market_hours_only=True,
            weekend_data=False
        )
        
        # 15-minute timeframe configuration
        self.configs['15min'] = TimeframeConfig(
            timeframe='15min',
            display_name='15 Minutes',
            description='Medium frequency trading with 15-minute bars',
            
            # Data fetching
            default_lookback_days=30,
            max_lookback_days=120,
            min_data_points=32,
            
            # Technical analysis
            ma_short_period=8,
            ma_long_period=16,
            rsi_period=14,
            rsi_oversold=20.0,
            rsi_overbought=80.0,
            bb_period=16,
            bb_std_dev=1.6,
            volume_ma_period=20,
            
            # Signal generation
            min_confidence=0.4,
            price_change_threshold=0.01,  # 1%
            volume_ratio_threshold=1.2,
            
            # Backtesting
            commission_rate=0.002,      # 0.2%
            slippage_rate=0.0015,       # 0.15%
            market_impact=0.001,        # 0.1%
            max_position_size=0.06,     # 6%
            execution_delay_bars=0,
            bid_ask_spread=0.0003,      # 0.03%
            liquidity_requirement=250000,
            
            # Storage
            storage_folder='minute',
            file_suffix='_15M',
            compression='snappy',
            
            # Performance
            batch_size=5000,
            parallel_workers=2,
            memory_allocation='3g',
            
            # Market hours
            intraday=True,
            market_hours_only=True,
            weekend_data=False
        )
        
        # 1-hour timeframe configuration
        self.configs['1hour'] = TimeframeConfig(
            timeframe='1hour',
            display_name='1 Hour',
            description='Intraday trading with hourly bars',
            
            # Data fetching
            default_lookback_days=60,
            max_lookback_days=365,
            min_data_points=48,
            
            # Technical analysis
            ma_short_period=12,
            ma_long_period=24,
            rsi_period=14,
            rsi_oversold=25.0,
            rsi_overbought=75.0,
            bb_period=20,
            bb_std_dev=1.8,
            volume_ma_period=24,
            
            # Signal generation
            min_confidence=0.5,
            price_change_threshold=0.015,  # 1.5%
            volume_ratio_threshold=1.3,
            
            # Backtesting
            commission_rate=0.0015,     # 0.15%
            slippage_rate=0.001,        # 0.1%
            market_impact=0.0005,       # 0.05%
            max_position_size=0.08,     # 8%
            execution_delay_bars=0,
            bid_ask_spread=0.0002,      # 0.02%
            liquidity_requirement=500000,
            
            # Storage
            storage_folder='hourly',
            file_suffix='_1H',
            compression='snappy',
            
            # Performance
            batch_size=10000,
            parallel_workers=2,
            memory_allocation='4g',
            
            # Market hours
            intraday=True,
            market_hours_only=False,
            weekend_data=False
        )
        
        # Daily timeframe configuration (backward compatible)
        self.configs['1day'] = TimeframeConfig(
            timeframe='1day',
            display_name='Daily',
            description='Traditional daily trading with end-of-day bars',
            
            # Data fetching
            default_lookback_days=365,
            max_lookback_days=3650,  # 10 years
            min_data_points=50,
            
            # Technical analysis (traditional)
            ma_short_period=20,
            ma_long_period=50,
            rsi_period=14,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            bb_period=20,
            bb_std_dev=2.0,
            volume_ma_period=20,
            
            # Signal generation
            min_confidence=0.6,
            price_change_threshold=0.02,  # 2%
            volume_ratio_threshold=1.5,
            
            # Backtesting (lower costs)
            commission_rate=0.001,      # 0.1%
            slippage_rate=0.0005,       # 0.05%
            market_impact=0.0002,       # 0.02%
            max_position_size=0.1,      # 10%
            execution_delay_bars=1,     # Next day
            bid_ask_spread=0.0001,      # 0.01%
            liquidity_requirement=1000000,
            
            # Storage
            storage_folder='daily',
            file_suffix='',  # No suffix for backward compatibility
            compression='snappy',
            
            # Performance
            batch_size=50000,
            parallel_workers=2,
            memory_allocation='4g',
            
            # Market hours
            intraday=False,
            market_hours_only=False,
            weekend_data=False
        )
        
        logger.info(f"Initialized default configurations for {len(self.configs)} timeframes")
    
    def get_config(self, timeframe: str) -> TimeframeConfig:
        """
        Get configuration for a specific timeframe.
        
        Args:
            timeframe: Timeframe identifier
            
        Returns:
            TimeframeConfig object
            
        Raises:
            ValueError: If timeframe is not supported
        """
        if timeframe not in self.configs:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(self.configs.keys())}")
        
        return self.configs[timeframe]
    
    def get_all_configs(self) -> Dict[str, TimeframeConfig]:
        """Get all timeframe configurations."""
        return self.configs.copy()
    
    def get_supported_timeframes(self) -> List[str]:
        """Get list of supported timeframes."""
        return list(self.configs.keys())
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Validate if a timeframe is supported.
        
        Args:
            timeframe: Timeframe to validate
            
        Returns:
            True if supported, False otherwise
        """
        return timeframe in self.configs
    
    def get_storage_path(self, timeframe: str, data_type: str = 'ohlcv') -> str:
        """
        Get storage path for timeframe and data type.
        
        Args:
            timeframe: Timeframe identifier
            data_type: Type of data ('ohlcv', 'signals', 'analytics')
            
        Returns:
            Storage path string
        """
        config = self.get_config(timeframe)
        
        # Backward compatibility for daily data
        if timeframe == '1day':
            return f"{data_type}/"
        else:
            return f"{data_type}/{config.storage_folder}/"
    
    def get_file_name(self, symbol: str, start_date: str, end_date: str, 
                     timeframe: str, file_format: str = 'parquet') -> str:
        """
        Generate timeframe-aware file name.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            timeframe: Timeframe identifier
            file_format: File format
            
        Returns:
            Generated file name
        """
        config = self.get_config(timeframe)
        
        base_name = f"{symbol}_{start_date}_{end_date}{config.file_suffix}.{file_format}"
        return base_name
    
    def save_to_file(self) -> bool:
        """
        Save configurations to file.
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Convert configs to dictionary
            data = {
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'timeframes': {tf: config.to_dict() for tf, config in self.configs.items()}
            }
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved timeframe configurations to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configurations: {str(e)}")
            return False
    
    def load_from_file(self) -> bool:
        """
        Load configurations from file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            # Load timeframe configurations
            if 'timeframes' in data:
                for tf, config_data in data['timeframes'].items():
                    self.configs[tf] = TimeframeConfig.from_dict(config_data)
            
            logger.info(f"Loaded timeframe configurations from {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            return False
    
    def update_config(self, timeframe: str, **kwargs):
        """
        Update configuration for a specific timeframe.
        
        Args:
            timeframe: Timeframe to update
            **kwargs: Configuration parameters to update
        """
        if timeframe not in self.configs:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        config = self.configs[timeframe]
        
        # Update only provided parameters
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        logger.info(f"Updated configuration for {timeframe}")
    
    def get_comparison_table(self) -> Dict[str, Any]:
        """
        Get a comparison table of all timeframe configurations.
        
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            'timeframes': list(self.configs.keys()),
            'parameters': {}
        }
        
        # Get all unique parameter names
        all_params = set()
        for config in self.configs.values():
            all_params.update(config.to_dict().keys())
        
        # Build comparison table
        for param in sorted(all_params):
            comparison['parameters'][param] = {}
            for timeframe, config in self.configs.items():
                config_dict = config.to_dict()
                comparison['parameters'][param][timeframe] = config_dict.get(param, 'N/A')
        
        return comparison

# Global instance
_config_manager = None

def get_timeframe_config_manager() -> TimeframeConfigManager:
    """Get the global timeframe configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = TimeframeConfigManager()
    return _config_manager

# Convenience functions
def get_timeframe_config(timeframe: str) -> TimeframeConfig:
    """Get configuration for a specific timeframe."""
    return get_timeframe_config_manager().get_config(timeframe)

def validate_timeframe(timeframe: str) -> bool:
    """Validate if a timeframe is supported."""
    return get_timeframe_config_manager().validate_timeframe(timeframe)

def get_supported_timeframes() -> List[str]:
    """Get list of supported timeframes."""
    return get_timeframe_config_manager().get_supported_timeframes()

# Example usage and testing
if __name__ == "__main__":
    # Test the configuration manager
    print("=== Testing Timeframe Configuration Manager ===")
    
    manager = TimeframeConfigManager()
    
    print(f"Supported timeframes: {manager.get_supported_timeframes()}")
    
    # Test configuration access
    for timeframe in ['1day', '1hour', '5min']:
        config = manager.get_config(timeframe)
        print(f"\n{timeframe} configuration:")
        print(f"  - MA periods: {config.ma_short_period}/{config.ma_long_period}")
        print(f"  - RSI levels: {config.rsi_oversold}/{config.rsi_overbought}")
        print(f"  - Commission: {config.commission_rate:.3f}")
        print(f"  - Storage: {manager.get_storage_path(timeframe)}")
    
    # Test file operations
    print("\n=== Testing File Operations ===")
    success = manager.save_to_file()
    print(f"Save to file: {'Success' if success else 'Failed'}")
    
    # Test comparison table
    print("\n=== Testing Comparison Table ===")
    comparison = manager.get_comparison_table()
    print(f"Comparison table has {len(comparison['parameters'])} parameters")
    print(f"Commission rates: {comparison['parameters']['commission_rate']}")
