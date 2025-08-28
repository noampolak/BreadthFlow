"""
Component Registration Module

Registers default system components with the Component Registry.
This module provides a centralized way to register all built-in components.
"""

from datetime import datetime
from .component_registry import ComponentRegistry, ComponentMetadata

def register_default_components(registry: ComponentRegistry):
    """Register default system components"""
    
    # Register YFinance Data Source
    yfinance_metadata = ComponentMetadata(
        name="yfinance",
        type="data_source",
        version="1.0.0",
        description="Yahoo Finance data source with multi-resource support",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "api_key": {"type": "string", "required": False},
            "rate_limits": {"type": "object", "required": False}
        }
    )
    
    # Note: We'll register the actual class when it's implemented
    # registry.register_component("data_source", "yfinance", YFinanceDataSource, yfinance_metadata)
    
    # Register Technical Analysis Strategy
    technical_metadata = ComponentMetadata(
        name="technical_analysis",
        type="signal_generator",
        version="1.0.0",
        description="Technical analysis-based signal generation",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "rsi_period": {"type": "integer", "default": 14},
            "ma_short": {"type": "integer", "default": 20},
            "ma_long": {"type": "integer", "default": 50}
        }
    )
    
    # Note: We'll register the actual class when it's implemented
    # registry.register_component("signal_generator", "technical_analysis", TechnicalAnalysisStrategy, technical_metadata)
    
    # Register Standard Backtest Engine
    backtest_metadata = ComponentMetadata(
        name="standard_engine",
        type="backtest_engine",
        version="1.0.0",
        description="Standard backtesting engine with modular components",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=["execution_engine", "risk_manager"],
        configuration_schema={
            "initial_capital": {"type": "float", "default": 100000},
            "commission_rate": {"type": "float", "default": 0.001}
        }
    )
    
    # Note: We'll register the actual class when it's implemented
    # registry.register_component("backtest_engine", "standard_engine", BaseBacktestEngine, backtest_metadata)
    
    print("📋 Default component metadata prepared (actual registration will happen when components are implemented)")

def register_legacy_adapters(registry: ComponentRegistry):
    """Register legacy system adapters for backward compatibility"""
    
    # Legacy TimeframeAgnosticFetcher adapter
    legacy_fetcher_metadata = ComponentMetadata(
        name="legacy_fetcher",
        type="data_source",
        version="1.0.0",
        description="Legacy TimeframeAgnosticFetcher adapter for backward compatibility",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "timeframe": {"type": "string", "default": "1day"},
            "symbols": {"type": "list", "required": True}
        }
    )
    
    # Legacy TimeframeAgnosticSignalGenerator adapter
    legacy_signal_metadata = ComponentMetadata(
        name="legacy_signal_generator",
        type="signal_generator",
        version="1.0.0",
        description="Legacy TimeframeAgnosticSignalGenerator adapter for backward compatibility",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "timeframe": {"type": "string", "default": "1day"},
            "symbols": {"type": "list", "required": True}
        }
    )
    
    # Legacy TimeframeAgnosticBacktestEngine adapter
    legacy_backtest_metadata = ComponentMetadata(
        name="legacy_backtest_engine",
        type="backtest_engine",
        version="1.0.0",
        description="Legacy TimeframeAgnosticBacktestEngine adapter for backward compatibility",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "timeframe": {"type": "string", "default": "1day"},
            "symbols": {"type": "list", "required": True},
            "initial_capital": {"type": "float", "default": 100000}
        }
    )
    
    print("🔄 Legacy adapter metadata prepared for backward compatibility")
