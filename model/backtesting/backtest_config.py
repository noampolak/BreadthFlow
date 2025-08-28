"""
Backtest Configuration

Defines configuration structures for backtesting in the BreadthFlow system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class ExecutionType(Enum):
    """Types of trade execution"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class RiskModel(Enum):
    """Types of risk management models"""
    STANDARD = "standard"
    VAR = "var"
    CUSTOM = "custom"

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    
    # Basic configuration
    name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    
    # Signal configuration
    signal_data: Optional[Dict[str, Any]] = None
    signal_threshold: float = 0.5
    confidence_threshold: float = 0.7
    
    # Execution configuration
    execution_type: ExecutionType = ExecutionType.MARKET
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    min_trade_size: float = 100.0
    max_trade_size: float = 10000.0
    
    # Risk management configuration
    risk_model: RiskModel = RiskModel.STANDARD
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    stop_loss_rate: float = 0.05  # 5% stop loss
    take_profit_rate: float = 0.1  # 10% take profit
    max_drawdown: float = 0.2  # 20% max drawdown
    
    # Position sizing configuration
    position_sizing_method: str = "fixed"  # "fixed", "kelly", "risk_parity"
    fixed_position_size: float = 0.05  # 5% per trade
    kelly_fraction: float = 0.25  # 25% of Kelly criterion
    risk_per_trade: float = 0.01  # 1% risk per trade
    
    # Performance configuration
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02  # 2% risk-free rate
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"
    
    # Advanced configuration
    enable_short_selling: bool = True
    enable_leverage: bool = False
    max_leverage: float = 1.0
    enable_dividends: bool = True
    enable_splits: bool = True
    
    # Output configuration
    output_format: str = "dataframe"  # "dataframe", "json", "dict"
    include_trades: bool = True
    include_positions: bool = True
    include_metrics: bool = True
    include_charts: bool = False
    
    # Performance settings
    enable_caching: bool = True
    cache_duration: int = 3600  # seconds
    parallel_processing: bool = False
    max_workers: int = 4
    
    def __post_init__(self):
        """Initialize default values"""
        if self.signal_data is None:
            self.signal_data = {}
    
    def validate(self) -> bool:
        """Validate backtest configuration"""
        if not self.name:
            return False
        
        if not self.symbols:
            return False
        
        if self.start_date >= self.end_date:
            return False
        
        if self.initial_capital <= 0:
            return False
        
        if self.signal_threshold < 0 or self.signal_threshold > 1:
            return False
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            return False
        
        if self.commission_rate < 0:
            return False
        
        if self.slippage_rate < 0:
            return False
        
        if self.max_position_size <= 0 or self.max_position_size > 1:
            return False
        
        if self.max_portfolio_risk <= 0 or self.max_portfolio_risk > 1:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'name': self.name,
            'symbols': self.symbols,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'signal_data': self.signal_data,
            'signal_threshold': self.signal_threshold,
            'confidence_threshold': self.confidence_threshold,
            'execution_type': self.execution_type.value,
            'commission_rate': self.commission_rate,
            'slippage_rate': self.slippage_rate,
            'min_trade_size': self.min_trade_size,
            'max_trade_size': self.max_trade_size,
            'risk_model': self.risk_model.value,
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'stop_loss_rate': self.stop_loss_rate,
            'take_profit_rate': self.take_profit_rate,
            'max_drawdown': self.max_drawdown,
            'position_sizing_method': self.position_sizing_method,
            'fixed_position_size': self.fixed_position_size,
            'kelly_fraction': self.kelly_fraction,
            'risk_per_trade': self.risk_per_trade,
            'benchmark_symbol': self.benchmark_symbol,
            'risk_free_rate': self.risk_free_rate,
            'rebalance_frequency': self.rebalance_frequency,
            'enable_short_selling': self.enable_short_selling,
            'enable_leverage': self.enable_leverage,
            'max_leverage': self.max_leverage,
            'enable_dividends': self.enable_dividends,
            'enable_splits': self.enable_splits,
            'output_format': self.output_format,
            'include_trades': self.include_trades,
            'include_positions': self.include_positions,
            'include_metrics': self.include_metrics,
            'include_charts': self.include_charts,
            'enable_caching': self.enable_caching,
            'cache_duration': self.cache_duration,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """Create configuration from dictionary"""
        # Convert date strings back to datetime
        if 'start_date' in config_dict and isinstance(config_dict['start_date'], str):
            config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        
        if 'end_date' in config_dict and isinstance(config_dict['end_date'], str):
            config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
        
        # Convert enums back
        if 'execution_type' in config_dict:
            config_dict['execution_type'] = ExecutionType(config_dict['execution_type'])
        
        if 'risk_model' in config_dict:
            config_dict['risk_model'] = RiskModel(config_dict['risk_model'])
        
        return cls(**config_dict)
