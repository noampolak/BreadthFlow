"""
Standard Execution Engine

Implements standard trade execution for regular market conditions.
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from execution_engine import ExecutionEngine
from backtest_config import BacktestConfig, ExecutionType
from trade_record import TradeRecord, TradeType, TradeStatus

logger = logging.getLogger(__name__)

class StandardExecutionEngine(ExecutionEngine):
    """Standard execution engine for regular market conditions"""
    
    def __init__(self, name: str = "standard_execution", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Default configuration
        self.default_config = {
            'min_commission': 1.0,
            'max_slippage': 0.01,  # 1% maximum slippage
            'execution_delay': 0,  # No execution delay
            'partial_fills': False,  # No partial fills
            'market_impact': 0.0001,  # 0.01% market impact
            'volume_threshold': 1000,  # Minimum volume for execution
            'price_improvement': 0.0001  # 0.01% price improvement
        }
        
        # Update config with defaults
        self.config.update(self.default_config)
    
    def get_name(self) -> str:
        """Get the name of this execution engine"""
        return self.name
    
    def get_supported_execution_types(self) -> List[ExecutionType]:
        """Get list of supported execution types"""
        return [
            ExecutionType.MARKET,
            ExecutionType.LIMIT,
            ExecutionType.STOP,
            ExecutionType.STOP_LIMIT
        ]
    
    def execute_trade(self, signal: Dict[str, Any], 
                     current_price: float,
                     config: BacktestConfig) -> TradeRecord:
        """Execute a trade based on signal and current price"""
        
        # Validate signal
        if not self.validate_signal(signal):
            logger.error(f"Invalid signal: {signal}")
            return self._create_rejected_trade(signal, "Invalid signal")
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, config)
        if position_size <= 0:
            logger.warning(f"Invalid position size: {position_size}")
            return self._create_rejected_trade(signal, "Invalid position size")
        
        # Check volume threshold
        if not self._check_volume_threshold(signal, config):
            logger.warning(f"Volume below threshold for {signal['symbol']}")
            return self._create_rejected_trade(signal, "Insufficient volume")
        
        # Calculate execution price
        execution_price = self.calculate_execution_price(signal, current_price, config)
        if execution_price is None:
            logger.warning(f"Order not triggered for {signal['symbol']}")
            return self._create_cancelled_trade(signal, "Order not triggered")
        
        # Apply market impact
        execution_price = self._apply_market_impact(execution_price, position_size, signal)
        
        # Apply price improvement
        execution_price = self._apply_price_improvement(execution_price, signal)
        
        # Calculate trade value and costs
        trade_value = position_size * execution_price
        commission = self.calculate_commission(trade_value, config)
        slippage = self.calculate_slippage(trade_value, config)
        
        # Determine trade type
        trade_type = self.determine_trade_type(signal)
        
        # Create trade record
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal['symbol'],
            trade_type=trade_type,
            timestamp=signal['timestamp'],
            quantity=position_size,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            status=TradeStatus.EXECUTED,
            signal_strength=signal.get('signal_strength'),
            signal_confidence=signal.get('confidence'),
            signal_type=signal.get('signal_type'),
            strategy_name=signal.get('strategy_name'),
            notes=f"Executed by {self.name}"
        )
        
        # Update execution stats
        self.update_execution_stats(trade, True)
        
        logger.info(f"✅ Executed trade: {trade.symbol} {trade.trade_type.value} {trade.quantity} @ {trade.price}")
        
        return trade
    
    def _calculate_position_size(self, signal: Dict[str, Any], config: BacktestConfig) -> float:
        """Calculate position size for the trade"""
        
        # Get portfolio value (simplified - in real implementation this would come from portfolio)
        portfolio_value = config.initial_capital
        
        # Calculate base position size
        if config.position_sizing_method == "fixed":
            position_size = portfolio_value * config.fixed_position_size
        elif config.position_sizing_method == "kelly":
            # Kelly criterion position sizing
            win_prob = signal.get('confidence', 0.5)
            avg_win = 0.1  # 10% average win
            avg_loss = 0.05  # 5% average loss
            
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, config.kelly_fraction))
            position_size = portfolio_value * kelly_fraction
        else:
            # Default to fixed position sizing
            position_size = portfolio_value * config.fixed_position_size
        
        # Apply position size limits
        min_size = config.min_trade_size
        max_size = config.max_trade_size
        
        position_size = max(min_size, min(position_size, max_size))
        
        return position_size
    
    def _check_volume_threshold(self, signal: Dict[str, Any], config: BacktestConfig) -> bool:
        """Check if volume meets minimum threshold"""
        volume_threshold = self.config.get('volume_threshold', 1000)
        
        # In a real implementation, this would check actual market volume
        # For now, we'll assume volume is sufficient
        return True
    
    def _apply_market_impact(self, price: float, quantity: float, signal: Dict[str, Any]) -> float:
        """Apply market impact to execution price"""
        market_impact = self.config.get('market_impact', 0.0001)
        
        # Simple linear market impact model
        impact_factor = 1 + (market_impact * quantity / 1000)  # Impact increases with quantity
        
        if signal['signal_type'] in ['buy', 'cover']:
            # Buy orders push price up
            return price * impact_factor
        else:
            # Sell orders push price down
            return price / impact_factor
    
    def _apply_price_improvement(self, price: float, signal: Dict[str, Any]) -> float:
        """Apply price improvement to execution price"""
        price_improvement = self.config.get('price_improvement', 0.0001)
        
        if signal['signal_type'] in ['buy', 'cover']:
            # Buy orders get slightly better price
            return price * (1 - price_improvement)
        else:
            # Sell orders get slightly better price
            return price * (1 + price_improvement)
    
    def _create_rejected_trade(self, signal: Dict[str, Any], reason: str) -> TradeRecord:
        """Create a rejected trade record"""
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal['symbol'],
            trade_type=self.determine_trade_type(signal),
            timestamp=signal['timestamp'],
            quantity=0,
            price=0,
            status=TradeStatus.REJECTED,
            notes=f"Rejected: {reason}"
        )
        
        self.update_execution_stats(trade, False)
        return trade
    
    def _create_cancelled_trade(self, signal: Dict[str, Any], reason: str) -> TradeRecord:
        """Create a cancelled trade record"""
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal['symbol'],
            trade_type=self.determine_trade_type(signal),
            timestamp=signal['timestamp'],
            quantity=0,
            price=0,
            status=TradeStatus.CANCELLED,
            notes=f"Cancelled: {reason}"
        )
        
        self.update_execution_stats(trade, False)
        return trade
    
    def get_execution_latency(self) -> float:
        """Get typical execution latency in seconds"""
        return self.config.get('execution_delay', 0)
    
    def get_market_impact_model(self) -> str:
        """Get the market impact model used"""
        return "linear"
    
    def get_price_improvement_model(self) -> str:
        """Get the price improvement model used"""
        return "fixed_percentage"
