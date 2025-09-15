"""
High-Frequency Backtest Engine

High-frequency trading backtesting engine with microsecond precision and advanced features.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest_config import BacktestConfig
from ..trade_record import PortfolioRecord, PositionRecord, TradeRecord
from .base_backtest_engine import BaseBacktestEngine

logger = logging.getLogger(__name__)


class HighFrequencyBacktestEngine(BaseBacktestEngine):
    """High-frequency backtesting engine with microsecond precision"""

    def __init__(self, name: str = "hft_backtest", config: BacktestConfig = None):
        if config is None:
            config = BacktestConfig()

        super().__init__(name, config)

        # HFT-specific features
        self.tick_data = {}
        self.order_book = {}
        self.latency_model = config.latency_model if hasattr(config, "latency_model") else "constant"
        self.execution_latency = config.execution_latency if hasattr(config, "execution_latency") else 0.001  # 1ms

        # Market microstructure
        self.bid_ask_spreads = {}
        self.market_impact_model = config.market_impact_model if hasattr(config, "market_impact_model") else "linear"
        self.tick_size = config.tick_size if hasattr(config, "tick_size") else 0.01

        # Order management
        self.pending_orders = []
        self.order_id_counter = 0
        self.order_status = {}

        # Performance tracking
        self.tick_returns = []
        self.millisecond_metrics = {}
        self.latency_stats = {"min_latency": float("inf"), "max_latency": 0.0, "avg_latency": 0.0, "latency_count": 0}

        logger.info(f"High-frequency backtest engine initialized: {name}")

    def get_name(self) -> str:
        """Get the name of this backtest engine"""
        return self.name

    def get_supported_execution_types(self) -> List[str]:
        """Get supported execution types for HFT engine"""
        return ["MARKET", "LIMIT", "STOP", "STOP_LIMIT", "IOC", "FOK"]

    def add_tick_data(
        self, symbol: str, timestamp: datetime, bid: float, ask: float, volume: float, trade_price: Optional[float] = None
    ):
        """Add tick data for high-frequency simulation"""

        if symbol not in self.tick_data:
            self.tick_data[symbol] = []

        tick = {
            "timestamp": timestamp,
            "bid": bid,
            "ask": ask,
            "spread": ask - bid,
            "mid_price": (bid + ask) / 2,
            "volume": volume,
            "trade_price": trade_price,
        }

        self.tick_data[symbol].append(tick)

        # Update bid-ask spread statistics
        if symbol not in self.bid_ask_spreads:
            self.bid_ask_spreads[symbol] = []
        self.bid_ask_spreads[symbol].append(tick["spread"])

        # Keep only recent tick data (last 1000 ticks)
        if len(self.tick_data[symbol]) > 1000:
            self.tick_data[symbol] = self.tick_data[symbol][-1000:]

        # Update order book
        self._update_order_book(symbol, tick)

    def _update_order_book(self, symbol: str, tick: Dict[str, Any]):
        """Update order book with new tick data"""

        if symbol not in self.order_book:
            self.order_book[symbol] = {"bids": [], "asks": [], "last_update": None}

        # Simplified order book update
        # In practice, you'd have full order book depth
        self.order_book[symbol]["bids"] = [{"price": tick["bid"], "size": 1000}]
        self.order_book[symbol]["asks"] = [{"price": tick["ask"], "size": 1000}]
        self.order_book[symbol]["last_update"] = tick["timestamp"]

    def _run_backtest_loop(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the main backtest loop with HFT features"""

        results = {
            "trades": [],
            "portfolio_history": [],
            "performance_metrics": {},
            "risk_metrics": {},
            "tick_returns": [],
            "latency_stats": {},
            "order_book_stats": {},
        }

        # Sort signals by timestamp (microsecond precision)
        sorted_signals = sorted(signals, key=lambda x: x.get("timestamp", datetime.min))

        # Process signals with HFT considerations
        for signal in sorted_signals:
            try:
                # Simulate market data at signal timestamp
                current_tick = self._get_tick_at_time(signal.get("symbol"), signal.get("timestamp"))

                if current_tick:
                    # Generate trade with HFT considerations
                    trade = self._generate_hft_trade(signal, current_tick)

                    if trade:
                        # Apply HFT-specific position sizing
                        trade = self._apply_hft_position_sizing(trade, current_tick)

                        if trade and trade.quantity > 0:
                            # Simulate execution latency
                            execution_delay = self._simulate_execution_latency()

                            # Get execution price with market impact
                            execution_price = self._calculate_execution_price(trade, current_tick)
                            trade.price = execution_price

                            # Validate trade with risk manager
                            if self.risk_manager.validate_trade(trade, self.current_portfolio, self.config):
                                # Execute trade
                                executed_trade = self.execution_engine.execute_trade(trade, {"tick": current_tick})

                                if executed_trade:
                                    # Update portfolio
                                    self._update_portfolio(executed_trade)

                                    # Record trade with HFT metadata
                                    executed_trade.execution_latency = execution_delay
                                    self.trades_history.append(executed_trade)
                                    results["trades"].append(executed_trade)

                                    # Update performance analyzer
                                    self.performance_analyzer.add_trade(executed_trade)

                                    logger.info(
                                        f"HFT Trade executed: {executed_trade.symbol} {executed_trade.trade_type.value} {executed_trade.quantity} at ${executed_trade.price:.4f}"
                                    )
                            else:
                                logger.warning(f"Trade rejected by risk manager: {signal}")

                # Update portfolio snapshot
                self._update_portfolio_snapshot()
                results["portfolio_history"].append(self.current_portfolio)

                # Calculate tick returns
                self._calculate_tick_returns(current_tick)

            except Exception as e:
                logger.error(f"Error processing HFT signal: {e}")
                continue

        # Calculate final metrics
        results["performance_metrics"] = self.performance_analyzer.generate_performance_report(self.config)
        results["risk_metrics"] = self.risk_manager.get_risk_report(self.current_portfolio, self.config)
        results["tick_returns"] = self.tick_returns
        results["latency_stats"] = self.latency_stats
        results["order_book_stats"] = self._calculate_order_book_stats()

        return results

    def _generate_hft_trade(self, signal: Dict[str, Any], tick: Dict[str, Any]) -> Optional[TradeRecord]:
        """Generate a trade record with HFT considerations"""

        symbol = signal.get("symbol")
        signal_type = signal.get("signal_type")
        timestamp = signal.get("timestamp")

        if not all([symbol, signal_type, timestamp]):
            return None

        # Use mid price for HFT
        current_price = tick["mid_price"]

        # Determine trade type and quantity
        if signal_type == "BUY":
            trade_type = "BUY"
            quantity = self._calculate_hft_position_size(symbol, current_price, tick)
        elif signal_type == "SELL":
            trade_type = "SELL"
            quantity = self._calculate_hft_position_size(symbol, current_price, tick)
        else:
            return None

        if quantity <= 0:
            return None

        # Create trade record with HFT metadata
        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            trade_type=trade_type,
            quantity=quantity,
            price=current_price,
            commission=0.0,
            realized_pnl=0.0,
            status="PENDING",
        )

        return trade

    def _calculate_hft_position_size(self, symbol: str, price: float, tick: Dict[str, Any]) -> float:
        """Calculate position size for HFT trading"""

        # HFT typically uses smaller position sizes
        available_cash = self.current_portfolio.cash if self.current_portfolio else 0

        # Use smaller position size for HFT (1% instead of 10%)
        position_value = available_cash * 0.01

        if position_value <= 0:
            return 0

        quantity = position_value / price

        # Apply tick size constraints
        quantity = self._round_to_tick_size(quantity, self.tick_size)

        # Apply minimum trade size constraint
        if quantity * price < self.config.min_trade_size:
            return 0

        return quantity

    def _apply_hft_position_sizing(self, trade: TradeRecord, tick: Dict[str, Any]) -> Optional[TradeRecord]:
        """Apply HFT-specific position sizing rules"""

        if not trade or trade.quantity <= 0:
            return trade

        symbol = trade.symbol
        current_price = trade.price
        desired_quantity = trade.quantity

        # Check for market impact
        market_impact = self._calculate_market_impact(desired_quantity, tick)
        if market_impact > 0.001:  # 0.1% market impact threshold
            # Reduce position size to minimize market impact
            desired_quantity *= 0.5

        # Check order book depth
        order_book_depth = self._get_order_book_depth(symbol, trade.trade_type)
        if order_book_depth < desired_quantity:
            # Reduce quantity to available liquidity
            desired_quantity = order_book_depth * 0.8  # Use 80% of available liquidity

        # Apply tick size constraints
        desired_quantity = self._round_to_tick_size(desired_quantity, self.tick_size)

        # Check minimum trade size
        if desired_quantity * current_price < self.config.min_trade_size:
            return None

        # Update trade quantity
        trade.quantity = desired_quantity

        return trade

    def _simulate_execution_latency(self) -> float:
        """Simulate execution latency"""

        if self.latency_model == "constant":
            latency = self.execution_latency
        elif self.latency_model == "normal":
            # Normal distribution around mean latency
            latency = np.random.normal(self.execution_latency, self.execution_latency * 0.1)
        elif self.latency_model == "exponential":
            # Exponential distribution (more realistic for HFT)
            latency = np.random.exponential(self.execution_latency)
        else:
            latency = self.execution_latency

        # Update latency statistics
        self.latency_stats["min_latency"] = min(self.latency_stats["min_latency"], latency)
        self.latency_stats["max_latency"] = max(self.latency_stats["max_latency"], latency)
        self.latency_stats["latency_count"] += 1

        # Update average latency
        total_latency = self.latency_stats["avg_latency"] * (self.latency_stats["latency_count"] - 1) + latency
        self.latency_stats["avg_latency"] = total_latency / self.latency_stats["latency_count"]

        return latency

    def _calculate_execution_price(self, trade: TradeRecord, tick: Dict[str, Any]) -> float:
        """Calculate execution price with market impact"""

        base_price = tick["mid_price"]

        if self.market_impact_model == "linear":
            # Linear market impact model
            impact_factor = trade.quantity / 1000  # Impact per 1000 shares
            price_impact = impact_factor * 0.001  # 0.1% impact per 1000 shares

        elif self.market_impact_model == "square_root":
            # Square root market impact model (more realistic)
            impact_factor = np.sqrt(trade.quantity / 1000)
            price_impact = impact_factor * 0.001

        else:
            price_impact = 0.0

        # Apply impact based on trade direction
        if trade.trade_type == "BUY":
            execution_price = base_price * (1 + price_impact)
        else:  # SELL
            execution_price = base_price * (1 - price_impact)

        # Round to tick size
        execution_price = self._round_to_tick_size(execution_price, self.tick_size)

        return execution_price

    def _calculate_market_impact(self, quantity: float, tick: Dict[str, Any]) -> float:
        """Calculate market impact for a given quantity"""

        # Simplified market impact calculation
        # In practice, you'd use more sophisticated models

        spread = tick["spread"]
        mid_price = tick["mid_price"]

        # Impact as percentage of spread
        impact_pct = (quantity / 1000) * 0.1  # 10% of spread per 1000 shares

        return spread * impact_pct / mid_price

    def _get_order_book_depth(self, symbol: str, trade_type: str) -> float:
        """Get available liquidity from order book"""

        if symbol not in self.order_book:
            return float("inf")

        if trade_type == "BUY":
            # Available ask liquidity
            return sum(order["size"] for order in self.order_book[symbol]["asks"])
        else:  # SELL
            # Available bid liquidity
            return sum(order["size"] for order in self.order_book[symbol]["bids"])

    def _round_to_tick_size(self, value: float, tick_size: float) -> float:
        """Round value to nearest tick size"""

        return round(value / tick_size) * tick_size

    def _get_tick_at_time(self, symbol: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Get tick data at specific timestamp"""

        if symbol not in self.tick_data:
            return None

        # Find closest tick to timestamp
        closest_tick = None
        min_diff = float("inf")

        for tick in self.tick_data[symbol]:
            diff = abs((tick["timestamp"] - timestamp).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_tick = tick

        return closest_tick

    def _calculate_tick_returns(self, tick: Dict[str, Any]):
        """Calculate returns at tick level"""

        if not tick or "mid_price" not in tick:
            return

        # Store tick return for analysis
        self.tick_returns.append(
            {
                "timestamp": tick["timestamp"],
                "mid_price": tick["mid_price"],
                "spread": tick["spread"],
                "volume": tick["volume"],
            }
        )

    def _calculate_order_book_stats(self) -> Dict[str, Any]:
        """Calculate order book statistics"""

        stats = {}

        for symbol, order_book in self.order_book.items():
            if order_book["bids"] and order_book["asks"]:
                bid_depth = sum(order["size"] for order in order_book["bids"])
                ask_depth = sum(order["size"] for order in order_book["asks"])
                spread = order_book["asks"][0]["price"] - order_book["bids"][0]["price"]

                stats[symbol] = {
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "spread": spread,
                    "depth_imbalance": (bid_depth - ask_depth) / (bid_depth + ask_depth),
                }

        return stats

    def get_hft_metrics(self) -> Dict[str, Any]:
        """Get HFT-specific performance metrics"""

        if not self.trades_history:
            return {}

        # Latency metrics
        latency_metrics = {
            "avg_latency_ms": self.latency_stats["avg_latency"] * 1000,
            "min_latency_ms": self.latency_stats["min_latency"] * 1000,
            "max_latency_ms": self.latency_stats["max_latency"] * 1000,
            "latency_std_ms": np.std([t.execution_latency for t in self.trades_history if hasattr(t, "execution_latency")])
            * 1000,
        }

        # Spread analysis
        avg_spreads = {}
        for symbol in self.bid_ask_spreads:
            if self.bid_ask_spreads[symbol]:
                avg_spreads[symbol] = np.mean(self.bid_ask_spreads[symbol])

        # Trade frequency
        if self.start_time and self.end_time:
            duration_hours = (self.end_time - self.start_time).total_seconds() / 3600
            trades_per_hour = len(self.trades_history) / duration_hours if duration_hours > 0 else 0
        else:
            trades_per_hour = 0

        return {
            "latency_metrics": latency_metrics,
            "avg_spreads": avg_spreads,
            "trades_per_hour": trades_per_hour,
            "total_ticks_processed": sum(len(ticks) for ticks in self.tick_data.values()),
            "order_book_stats": self._calculate_order_book_stats(),
        }
