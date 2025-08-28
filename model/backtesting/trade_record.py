"""
Trade and Position Records

Defines data structures for tracking trades and positions in backtesting.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class TradeType(Enum):
    """Types of trades"""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"

class TradeStatus(Enum):
    """Status of trades"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class TradeRecord:
    """Record of a single trade"""
    
    # Trade identification
    trade_id: str
    symbol: str
    trade_type: TradeType
    timestamp: datetime
    
    # Trade details
    quantity: float
    price: float
    commission: float = 0.0
    slippage: float = 0.0
    
    # Trade status
    status: TradeStatus = TradeStatus.EXECUTED
    
    # Signal information
    signal_strength: Optional[float] = None
    signal_confidence: Optional[float] = None
    signal_type: Optional[str] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: Optional[float] = None
    
    # Metadata
    strategy_name: Optional[str] = None
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.total_cost = self.quantity * self.price
        self.net_amount = self.total_cost + self.commission + self.slippage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade record to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'trade_type': self.trade_type.value,
            'timestamp': self.timestamp.isoformat(),
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'status': self.status.value,
            'signal_strength': self.signal_strength,
            'signal_confidence': self.signal_confidence,
            'signal_type': self.signal_type,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_amount': self.risk_amount,
            'strategy_name': self.strategy_name,
            'notes': self.notes,
            'metadata': self.metadata,
            'total_cost': self.total_cost,
            'net_amount': self.net_amount
        }
    
    @classmethod
    def from_dict(cls, trade_dict: Dict[str, Any]) -> 'TradeRecord':
        """Create trade record from dictionary"""
        # Convert enums back
        if 'trade_type' in trade_dict:
            trade_dict['trade_type'] = TradeType(trade_dict['trade_type'])
        
        if 'status' in trade_dict:
            trade_dict['status'] = TradeStatus(trade_dict['status'])
        
        # Convert timestamp back
        if 'timestamp' in trade_dict and isinstance(trade_dict['timestamp'], str):
            trade_dict['timestamp'] = datetime.fromisoformat(trade_dict['timestamp'])
        
        return cls(**trade_dict)

@dataclass
class PositionRecord:
    """Record of a position at a specific point in time"""
    
    # Position identification
    symbol: str
    timestamp: datetime
    
    # Position details
    quantity: float
    average_price: float
    current_price: float
    
    # Position metrics
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Risk metrics
    position_size: float = 0.0  # Percentage of portfolio
    risk_amount: float = 0.0
    var_95: Optional[float] = None
    
    # Metadata
    entry_date: Optional[datetime] = None
    exit_date: Optional[datetime] = None
    strategy_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity
        self.total_pnl = self.unrealized_pnl + self.realized_pnl
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.average_price) * self.quantity
        self.total_pnl = self.unrealized_pnl + self.realized_pnl
    
    def add_trade(self, trade: TradeRecord):
        """Add a trade to the position"""
        if trade.symbol != self.symbol:
            raise ValueError(f"Trade symbol {trade.symbol} doesn't match position symbol {self.symbol}")
        
        # Update position based on trade type
        if trade.trade_type in [TradeType.BUY, TradeType.COVER]:
            # Adding to position
            total_quantity = self.quantity + trade.quantity
            total_cost = (self.quantity * self.average_price) + trade.total_cost
            self.average_price = total_cost / total_quantity if total_quantity > 0 else 0
            self.quantity = total_quantity
        
        elif trade.trade_type in [TradeType.SELL, TradeType.SHORT]:
            # Reducing position
            if trade.quantity <= self.quantity:
                # Partial or full exit
                exit_quantity = trade.quantity
                remaining_quantity = self.quantity - exit_quantity
                
                # Calculate realized P&L
                exit_value = exit_quantity * trade.price
                cost_basis = exit_quantity * self.average_price
                trade_pnl = exit_value - cost_basis - trade.commission - trade.slippage
                self.realized_pnl += trade_pnl
                
                # Update position
                self.quantity = remaining_quantity
                if remaining_quantity == 0:
                    self.average_price = 0
            else:
                raise ValueError(f"Trade quantity {trade.quantity} exceeds position quantity {self.quantity}")
        
        # Update current metrics
        self.update_price(self.current_price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position record to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'quantity': self.quantity,
            'average_price': self.average_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'position_size': self.position_size,
            'risk_amount': self.risk_amount,
            'var_95': self.var_95,
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'strategy_name': self.strategy_name,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, position_dict: Dict[str, Any]) -> 'PositionRecord':
        """Create position record from dictionary"""
        # Convert timestamps back
        for field in ['timestamp', 'entry_date', 'exit_date']:
            if field in position_dict and position_dict[field] and isinstance(position_dict[field], str):
                position_dict[field] = datetime.fromisoformat(position_dict[field])
        
        return cls(**position_dict)

@dataclass
class PortfolioRecord:
    """Record of portfolio state at a specific point in time"""
    
    # Portfolio identification
    timestamp: datetime
    
    # Portfolio values
    total_value: float
    cash_value: float
    positions_value: float
    
    # Performance metrics
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_return: float = 0.0
    daily_return: float = 0.0
    
    # Risk metrics
    volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_95: Optional[float] = None
    
    # Position information
    positions: List[PositionRecord] = field(default_factory=list)
    num_positions: int = 0
    
    # Metadata
    benchmark_value: Optional[float] = None
    benchmark_return: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.num_positions = len(self.positions)
        self.positions_value = sum(pos.market_value for pos in self.positions)
        self.total_value = self.cash_value + self.positions_value
    
    def add_position(self, position: PositionRecord):
        """Add a position to the portfolio"""
        self.positions.append(position)
        self.num_positions = len(self.positions)
        self.positions_value = sum(pos.market_value for pos in self.positions)
        self.total_value = self.cash_value + self.positions_value
    
    def update_positions(self, new_prices: Dict[str, float]):
        """Update all positions with new prices"""
        for position in self.positions:
            if position.symbol in new_prices:
                position.update_price(new_prices[position.symbol])
        
        # Recalculate portfolio values
        self.positions_value = sum(pos.market_value for pos in self.positions)
        self.total_value = self.cash_value + self.positions_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio record to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'cash_value': self.cash_value,
            'positions_value': self.positions_value,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'total_return': self.total_return,
            'daily_return': self.daily_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'positions': [pos.to_dict() for pos in self.positions],
            'num_positions': self.num_positions,
            'benchmark_value': self.benchmark_value,
            'benchmark_return': self.benchmark_return,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, portfolio_dict: Dict[str, Any]) -> 'PortfolioRecord':
        """Create portfolio record from dictionary"""
        # Convert timestamp back
        if 'timestamp' in portfolio_dict and isinstance(portfolio_dict['timestamp'], str):
            portfolio_dict['timestamp'] = datetime.fromisoformat(portfolio_dict['timestamp'])
        
        # Convert positions back
        if 'positions' in portfolio_dict:
            portfolio_dict['positions'] = [
                PositionRecord.from_dict(pos_dict) for pos_dict in portfolio_dict['positions']
            ]
        
        return cls(**portfolio_dict)
