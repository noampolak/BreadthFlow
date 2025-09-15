"""
High-Frequency Execution Engine

Implements high-frequency trade execution with minimal latency and advanced order types.
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from .execution_engine import ExecutionEngine
from ..backtest_config import BacktestConfig, ExecutionType
from ..trade_record import TradeRecord, TradeType, TradeStatus

logger = logging.getLogger(__name__)


class HighFrequencyExecutionEngine(ExecutionEngine):
    """High-frequency execution engine for low-latency trading"""

    def __init__(self, name: str = "high_frequency_execution", config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Default configuration for HFT
        self.default_config = {
            "min_commission": 0.1,  # Lower commission for HFT
            "max_slippage": 0.0005,  # 0.05% maximum slippage
            "execution_delay": 0.001,  # 1ms execution delay
            "partial_fills": True,  # Allow partial fills
            "market_impact": 0.00005,  # 0.005% market impact
            "volume_threshold": 100,  # Lower volume threshold
            "price_improvement": 0.00005,  # 0.005% price improvement
            "order_routing": "smart",  # Smart order routing
            "co_location": True,  # Co-location enabled
            "direct_market_access": True,  # Direct market access
            "iceberg_orders": True,  # Support iceberg orders
            "twap_orders": True,  # Support TWAP orders
            "vwap_orders": True,  # Support VWAP orders
            "microsecond_latency": True,  # Microsecond precision
        }

        # Update config with defaults
        self.config.update(self.default_config)

    def get_name(self) -> str:
        """Get the name of this execution engine"""
        return self.name

    def get_supported_execution_types(self) -> List[ExecutionType]:
        """Get list of supported execution types"""
        return [ExecutionType.MARKET, ExecutionType.LIMIT, ExecutionType.STOP, ExecutionType.STOP_LIMIT]

    def execute_trade(self, signal: Dict[str, Any], current_price: float, config: BacktestConfig) -> TradeRecord:
        """Execute a trade with high-frequency optimizations"""

        # Validate signal
        if not self.validate_signal(signal):
            logger.error(f"Invalid signal: {signal}")
            return self._create_rejected_trade(signal, "Invalid signal")

        # Check for HFT-specific validations
        if not self._validate_hft_conditions(signal, config):
            logger.warning(f"HFT conditions not met for {signal['symbol']}")
            return self._create_rejected_trade(signal, "HFT conditions not met")

        # Calculate position size with HFT considerations
        position_size = self._calculate_hft_position_size(signal, config)
        if position_size <= 0:
            logger.warning(f"Invalid HFT position size: {position_size}")
            return self._create_rejected_trade(signal, "Invalid position size")

        # Check volume and liquidity
        if not self._check_hft_volume_threshold(signal, config):
            logger.warning(f"Insufficient liquidity for HFT: {signal['symbol']}")
            return self._create_rejected_trade(signal, "Insufficient liquidity")

        # Calculate execution price with HFT optimizations
        execution_price = self._calculate_hft_execution_price(signal, current_price, config)
        if execution_price is None:
            logger.warning(f"HFT order not triggered for {signal['symbol']}")
            return self._create_cancelled_trade(signal, "Order not triggered")

        # Apply HFT-specific market impact
        execution_price = self._apply_hft_market_impact(execution_price, position_size, signal)

        # Apply HFT price improvement
        execution_price = self._apply_hft_price_improvement(execution_price, signal)

        # Calculate trade value and costs
        trade_value = position_size * execution_price
        commission = self._calculate_hft_commission(trade_value, config)
        slippage = self._calculate_hft_slippage(trade_value, config)

        # Determine trade type
        trade_type = self.determine_trade_type(signal)

        # Create trade record with HFT metadata
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal["symbol"],
            trade_type=trade_type,
            timestamp=signal["timestamp"],
            quantity=position_size,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            status=TradeStatus.EXECUTED,
            signal_strength=signal.get("signal_strength"),
            signal_confidence=signal.get("confidence"),
            signal_type=signal.get("signal_type"),
            strategy_name=signal.get("strategy_name"),
            notes=f"HFT executed by {self.name}",
            metadata={
                "execution_latency": self.get_execution_latency(),
                "order_routing": self.config.get("order_routing"),
                "co_location": self.config.get("co_location"),
                "direct_market_access": self.config.get("direct_market_access"),
            },
        )

        # Update execution stats
        self.update_execution_stats(trade, True)

        logger.info(f"⚡ HFT executed: {trade.symbol} {trade.trade_type.value} {trade.quantity} @ {trade.price}")

        return trade

    def _validate_hft_conditions(self, signal: Dict[str, Any], config: BacktestConfig) -> bool:
        """Validate HFT-specific conditions"""

        # Check if symbol is suitable for HFT
        symbol = signal["symbol"]

        # In a real implementation, this would check:
        # - Symbol has sufficient liquidity
        # - Symbol has tight bid-ask spreads
        # - Symbol has high trading volume
        # - Symbol is available for HFT trading

        # For now, we'll assume all symbols are suitable
        return True

    def _calculate_hft_position_size(self, signal: Dict[str, Any], config: BacktestConfig) -> float:
        """Calculate position size optimized for HFT"""

        # Get portfolio value
        portfolio_value = config.initial_capital

        # HFT typically uses smaller position sizes for faster execution
        base_position_size = portfolio_value * config.fixed_position_size

        # Apply HFT-specific adjustments
        hft_multiplier = self.config.get("hft_position_multiplier", 0.5)  # Smaller positions
        base_position_size *= hft_multiplier

        # Apply position size limits
        min_size = config.min_trade_size * 0.1  # Lower minimum for HFT
        max_size = config.max_trade_size * 0.5  # Lower maximum for HFT

        position_size = max(min_size, min(base_position_size, max_size))

        return position_size

    def _check_hft_volume_threshold(self, signal: Dict[str, Any], config: BacktestConfig) -> bool:
        """Check if volume meets HFT requirements"""
        volume_threshold = self.config.get("volume_threshold", 100)

        # HFT requires higher volume and tighter spreads
        # In a real implementation, this would check:
        # - Current market volume
        # - Bid-ask spread
        # - Order book depth
        # - Market maker activity

        return True

    def _calculate_hft_execution_price(self, signal: Dict[str, Any], current_price: float, config: BacktestConfig) -> float:
        """Calculate execution price with HFT optimizations"""

        execution_type = config.execution_type
        slippage_rate = self.config.get("max_slippage", 0.0005)  # Lower slippage for HFT

        if execution_type == ExecutionType.MARKET:
            # HFT market orders get better prices due to direct market access
            if signal["signal_type"] in ["buy", "cover"]:
                execution_price = current_price * (1 + slippage_rate * 0.5)  # Reduced slippage
            else:
                execution_price = current_price * (1 - slippage_rate * 0.5)  # Reduced slippage

        elif execution_type == ExecutionType.LIMIT:
            # HFT limit orders get better fills
            limit_price = signal.get("limit_price", current_price)
            if signal["signal_type"] in ["buy", "cover"]:
                execution_price = min(limit_price, current_price * 0.9999)  # Slight improvement
            else:
                execution_price = max(limit_price, current_price * 1.0001)  # Slight improvement

        else:
            # Default to market execution
            execution_price = current_price

        return execution_price

    def _apply_hft_market_impact(self, price: float, quantity: float, signal: Dict[str, Any]) -> float:
        """Apply HFT-optimized market impact"""
        market_impact = self.config.get("market_impact", 0.00005)  # Lower impact for HFT

        # HFT uses more sophisticated market impact models
        # For now, we'll use a reduced linear model
        impact_factor = 1 + (market_impact * quantity / 10000)  # Reduced impact

        if signal["signal_type"] in ["buy", "cover"]:
            return price * impact_factor
        else:
            return price / impact_factor

    def _apply_hft_price_improvement(self, price: float, signal: Dict[str, Any]) -> float:
        """Apply HFT price improvement"""
        price_improvement = self.config.get("price_improvement", 0.00005)  # Lower improvement for HFT

        # HFT gets smaller but more consistent price improvements
        if signal["signal_type"] in ["buy", "cover"]:
            return price * (1 - price_improvement)
        else:
            return price * (1 + price_improvement)

    def _calculate_hft_commission(self, trade_value: float, config: BacktestConfig) -> float:
        """Calculate HFT-optimized commission"""
        commission_rate = config.commission_rate * 0.5  # Lower commission for HFT
        min_commission = self.config.get("min_commission", 0.1)

        commission = max(trade_value * commission_rate, min_commission)
        return commission

    def _calculate_hft_slippage(self, trade_value: float, config: BacktestConfig) -> float:
        """Calculate HFT-optimized slippage"""
        slippage_rate = config.slippage_rate * 0.5  # Lower slippage for HFT
        slippage = trade_value * slippage_rate
        return slippage

    def _create_rejected_trade(self, signal: Dict[str, Any], reason: str) -> TradeRecord:
        """Create a rejected HFT trade record"""
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal["symbol"],
            trade_type=self.determine_trade_type(signal),
            timestamp=signal["timestamp"],
            quantity=0,
            price=0,
            status=TradeStatus.REJECTED,
            notes=f"HFT Rejected: {reason}",
            metadata={"execution_engine": "hft"},
        )

        self.update_execution_stats(trade, False)
        return trade

    def _create_cancelled_trade(self, signal: Dict[str, Any], reason: str) -> TradeRecord:
        """Create a cancelled HFT trade record"""
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=signal["symbol"],
            trade_type=self.determine_trade_type(signal),
            timestamp=signal["timestamp"],
            quantity=0,
            price=0,
            status=TradeStatus.CANCELLED,
            notes=f"HFT Cancelled: {reason}",
            metadata={"execution_engine": "hft"},
        )

        self.update_execution_stats(trade, False)
        return trade

    def get_execution_latency(self) -> float:
        """Get HFT execution latency in seconds"""
        return self.config.get("execution_delay", 0.001)

    def get_market_impact_model(self) -> str:
        """Get the HFT market impact model used"""
        return "hft_optimized_linear"

    def get_price_improvement_model(self) -> str:
        """Get the HFT price improvement model used"""
        return "hft_consistent_improvement"

    def get_order_routing_strategy(self) -> str:
        """Get the order routing strategy used"""
        return self.config.get("order_routing", "smart")

    def is_co_located(self) -> bool:
        """Check if co-location is enabled"""
        return self.config.get("co_location", True)

    def has_direct_market_access(self) -> bool:
        """Check if direct market access is enabled"""
        return self.config.get("direct_market_access", True)
