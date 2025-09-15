"""
Execution Engine Interface

Abstract interface for trade execution engines in the BreadthFlow system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from ..backtest_config import BacktestConfig, ExecutionType
from ..trade_record import TradeRecord, TradeType, TradeStatus


class ExecutionEngine(ABC):
    """Abstract interface for trade execution engines"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

        # Performance tracking
        self.execution_stats = {
            "total_orders": 0,
            "executed_orders": 0,
            "cancelled_orders": 0,
            "rejected_orders": 0,
            "average_execution_time": 0.0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
        }

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this execution engine"""
        pass

    @abstractmethod
    def get_supported_execution_types(self) -> List[ExecutionType]:
        """Get list of supported execution types"""
        pass

    @abstractmethod
    def execute_trade(self, signal: Dict[str, Any], current_price: float, config: BacktestConfig) -> TradeRecord:
        """Execute a trade based on signal and current price"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get execution engine configuration"""
        return self.config

    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trade signal"""
        required_fields = ["symbol", "signal_type", "timestamp"]

        for field in required_fields:
            if field not in signal:
                return False

        if signal["signal_type"] not in ["buy", "sell", "short", "cover"]:
            return False

        return True

    def calculate_execution_price(self, signal: Dict[str, Any], current_price: float, config: BacktestConfig) -> float:
        """Calculate execution price with slippage"""

        execution_type = config.execution_type
        slippage_rate = config.slippage_rate

        if execution_type == ExecutionType.MARKET:
            # Market orders execute at current price with slippage
            if signal["signal_type"] in ["buy", "cover"]:
                # Buy orders execute at slightly higher price
                execution_price = current_price * (1 + slippage_rate)
            else:
                # Sell orders execute at slightly lower price
                execution_price = current_price * (1 - slippage_rate)

        elif execution_type == ExecutionType.LIMIT:
            # Limit orders execute at specified price or better
            limit_price = signal.get("limit_price", current_price)
            if signal["signal_type"] in ["buy", "cover"]:
                # Buy limit orders execute at or below limit price
                execution_price = min(limit_price, current_price)
            else:
                # Sell limit orders execute at or above limit price
                execution_price = max(limit_price, current_price)

        elif execution_type == ExecutionType.STOP:
            # Stop orders execute when price crosses trigger
            stop_price = signal.get("stop_price", current_price)
            if signal["signal_type"] in ["sell", "cover"]:
                # Stop loss orders
                if current_price <= stop_price:
                    execution_price = stop_price * (1 - slippage_rate)
                else:
                    execution_price = None  # Order not triggered
            else:
                # Stop buy orders
                if current_price >= stop_price:
                    execution_price = stop_price * (1 + slippage_rate)
                else:
                    execution_price = None  # Order not triggered

        else:
            # Default to market execution
            execution_price = current_price

        return execution_price

    def calculate_commission(self, trade_value: float, config: BacktestConfig) -> float:
        """Calculate commission for trade"""
        commission_rate = config.commission_rate
        min_commission = self.config.get("min_commission", 1.0)

        commission = max(trade_value * commission_rate, min_commission)
        return commission

    def calculate_slippage(self, trade_value: float, config: BacktestConfig) -> float:
        """Calculate slippage cost for trade"""
        slippage_rate = config.slippage_rate
        slippage = trade_value * slippage_rate
        return slippage

    def determine_trade_type(self, signal: Dict[str, Any], current_position: Optional[float] = None) -> TradeType:
        """Determine trade type based on signal and current position"""

        signal_type = signal["signal_type"]

        if signal_type == "buy":
            if current_position is None or current_position == 0:
                return TradeType.BUY
            elif current_position < 0:
                return TradeType.COVER  # Cover short position
            else:
                return TradeType.BUY  # Add to long position

        elif signal_type == "sell":
            if current_position is None or current_position == 0:
                return TradeType.SELL
            elif current_position > 0:
                return TradeType.SELL  # Reduce long position
            else:
                return TradeType.SELL  # Add to short position

        elif signal_type == "short":
            return TradeType.SHORT

        elif signal_type == "cover":
            return TradeType.COVER

        else:
            # Default to buy
            return TradeType.BUY

    def update_execution_stats(self, trade: TradeRecord, success: bool):
        """Update execution statistics"""
        self.execution_stats["total_orders"] += 1

        if success:
            self.execution_stats["executed_orders"] += 1
            self.execution_stats["total_commission"] += trade.commission
            self.execution_stats["total_slippage"] += trade.slippage
        else:
            if trade.status == TradeStatus.CANCELLED:
                self.execution_stats["cancelled_orders"] += 1
            elif trade.status == TradeStatus.REJECTED:
                self.execution_stats["rejected_orders"] += 1

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        metrics = self.execution_stats.copy()

        if metrics["total_orders"] > 0:
            metrics["execution_rate"] = metrics["executed_orders"] / metrics["total_orders"]
            metrics["cancellation_rate"] = metrics["cancelled_orders"] / metrics["total_orders"]
            metrics["rejection_rate"] = metrics["rejected_orders"] / metrics["total_orders"]
        else:
            metrics["execution_rate"] = 0.0
            metrics["cancellation_rate"] = 0.0
            metrics["rejection_rate"] = 0.0

        return metrics

    def get_execution_info(self) -> Dict[str, Any]:
        """Get comprehensive execution engine information"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "supported_execution_types": [et.value for et in self.get_supported_execution_types()],
            "execution_metrics": self.get_execution_metrics(),
        }
