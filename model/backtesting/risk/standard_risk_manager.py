"""
Standard Risk Manager

Implements standard risk management with position limits, portfolio risk control,
and basic risk metrics.
"""

from typing import Dict, List, Any, Optional
import logging
from .risk_manager import RiskManager
from ..backtest_config import BacktestConfig
from ..trade_record import TradeRecord, PositionRecord, PortfolioRecord

logger = logging.getLogger(__name__)


class StandardRiskManager(RiskManager):
    """Standard risk manager for regular trading strategies"""

    def __init__(self, name: str = "standard_risk", config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Default configuration
        self.default_config = {
            "max_single_position_risk": 0.02,  # 2% max risk per position
            "max_correlation_risk": 0.3,  # 30% max correlation between positions
            "max_sector_exposure": 0.25,  # 25% max exposure to any sector
            "max_currency_exposure": 0.5,  # 50% max exposure to any currency
            "position_sizing_method": "risk_based",  # 'risk_based', 'equal_weight', 'volatility_weight'
            "risk_adjustment_factor": 1.0,  # Risk adjustment multiplier
            "enable_stop_loss": True,
            "enable_take_profit": True,
            "enable_trailing_stop": False,
            "trailing_stop_distance": 0.05,  # 5% trailing stop distance
        }

        # Update config with defaults
        self.config.update(self.default_config)

    def get_name(self) -> str:
        """Get the name of this risk manager"""
        return self.name

    def calculate_position_risk(self, position: PositionRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> float:
        """Calculate risk for a specific position"""

        # Calculate position risk based on market value and volatility
        position_value = position.market_value
        portfolio_value = portfolio.total_value

        if portfolio_value <= 0:
            return 0.0

        # Position risk as percentage of portfolio
        position_risk = position_value / portfolio_value

        # Apply volatility adjustment if available
        if hasattr(position, "volatility") and position.volatility:
            volatility_factor = position.volatility / 0.02  # Normalize to 2% volatility
            position_risk *= volatility_factor

        # Apply risk adjustment factor
        risk_adjustment = self.config.get("risk_adjustment_factor", 1.0)
        position_risk *= risk_adjustment

        return position_risk

    def calculate_portfolio_risk(self, portfolio: PortfolioRecord, config: BacktestConfig) -> float:
        """Calculate total portfolio risk"""

        total_risk = 0.0

        # Calculate individual position risks
        for position in portfolio.positions:
            position_risk = self.calculate_position_risk(position, portfolio, config)
            total_risk += position_risk

        # Apply diversification benefit (simplified correlation adjustment)
        diversification_factor = self._calculate_diversification_factor(portfolio)
        total_risk *= diversification_factor

        # Update risk stats
        self.risk_stats["current_portfolio_risk"] = total_risk
        self.risk_stats["max_portfolio_risk"] = max(self.risk_stats["max_portfolio_risk"], total_risk)

        return total_risk

    def validate_trade(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Validate if a trade meets risk requirements"""

        # Check basic risk limits
        if not self._check_basic_risk_limits(trade, portfolio, config):
            logger.warning(f"Trade rejected: Basic risk limits exceeded")
            return False

        # Check position concentration limits
        if not self._check_concentration_limits(trade, portfolio, config):
            logger.warning(f"Trade rejected: Concentration limits exceeded")
            return False

        # Check portfolio risk limits
        if not self._check_portfolio_risk_limits(trade, portfolio, config):
            logger.warning(f"Trade rejected: Portfolio risk limits exceeded")
            return False

        # Check drawdown limits
        if not self.check_drawdown_limit(portfolio, config):
            logger.warning(f"Trade rejected: Drawdown limit exceeded")
            return False

        # All checks passed
        logger.info(f"✅ Trade validated by {self.name}")
        return True

    def _check_basic_risk_limits(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check basic risk limits"""

        # Check minimum trade size
        if trade.quantity * trade.price < config.min_trade_size:
            return False

        # Check maximum trade size
        if trade.quantity * trade.price > config.max_trade_size:
            return False

        # Check maximum single position risk
        max_single_risk = self.config.get("max_single_position_risk", 0.02)
        trade_value = trade.quantity * trade.price
        portfolio_value = portfolio.total_value

        if portfolio_value > 0:
            position_risk = trade_value / portfolio_value
            if position_risk > max_single_risk:
                return False

        return True

    def _check_concentration_limits(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check concentration limits"""

        # Check maximum position size per symbol
        max_position_size = config.max_position_size

        # Calculate new position size for this symbol
        current_position_value = 0
        for position in portfolio.positions:
            if position.symbol == trade.symbol:
                current_position_value = position.market_value
                break

        # Add new trade value
        new_position_value = current_position_value + (trade.quantity * trade.price)
        new_position_size = new_position_value / portfolio.total_value if portfolio.total_value > 0 else 0

        if new_position_size > max_position_size:
            return False

        return True

    def _check_portfolio_risk_limits(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check portfolio risk limits"""

        # Calculate current portfolio risk
        current_risk = self.calculate_portfolio_risk(portfolio, config)

        # Estimate new portfolio risk with this trade
        trade_value = trade.quantity * trade.price
        portfolio_value = portfolio.total_value

        if portfolio_value > 0:
            trade_risk = trade_value / portfolio_value
            new_risk = current_risk + trade_risk

            # Apply diversification benefit
            diversification_factor = self._calculate_diversification_factor(portfolio)
            adjusted_risk = new_risk * diversification_factor

            if adjusted_risk > config.max_portfolio_risk:
                return False

        return True

    def _calculate_diversification_factor(self, portfolio: PortfolioRecord) -> float:
        """Calculate diversification benefit factor"""

        num_positions = len(portfolio.positions)

        if num_positions <= 1:
            return 1.0

        # Simple diversification model
        # More positions = more diversification = lower risk
        # This is a simplified approach - real implementation would use correlation matrix

        # Square root of n rule for uncorrelated assets
        diversification_factor = 1.0 / (num_positions**0.5)

        # Apply minimum diversification factor
        min_factor = 0.5  # At least 50% of original risk
        diversification_factor = max(diversification_factor, min_factor)

        return diversification_factor

    def calculate_optimal_position_size(
        self, signal: Dict[str, Any], portfolio: PortfolioRecord, config: BacktestConfig
    ) -> float:
        """Calculate optimal position size based on risk management rules"""

        position_sizing_method = self.config.get("position_sizing_method", "risk_based")

        if position_sizing_method == "risk_based":
            return self._calculate_risk_based_position_size(signal, portfolio, config)
        elif position_sizing_method == "equal_weight":
            return self._calculate_equal_weight_position_size(signal, portfolio, config)
        elif position_sizing_method == "volatility_weight":
            return self._calculate_volatility_weight_position_size(signal, portfolio, config)
        else:
            return self._calculate_risk_based_position_size(signal, portfolio, config)

    def _calculate_risk_based_position_size(
        self, signal: Dict[str, Any], portfolio: PortfolioRecord, config: BacktestConfig
    ) -> float:
        """Calculate position size based on risk budget"""

        # Get available risk budget
        current_risk = self.calculate_portfolio_risk(portfolio, config)
        max_risk = config.max_portfolio_risk
        available_risk = max_risk - current_risk

        if available_risk <= 0:
            return 0.0

        # Calculate position size based on risk per trade
        risk_per_trade = config.risk_per_trade
        portfolio_value = portfolio.total_value

        # Use the smaller of available risk or risk per trade
        risk_budget = min(available_risk, risk_per_trade)
        position_value = portfolio_value * risk_budget

        return position_value

    def _calculate_equal_weight_position_size(
        self, signal: Dict[str, Any], portfolio: PortfolioRecord, config: BacktestConfig
    ) -> float:
        """Calculate position size using equal weight allocation"""

        portfolio_value = portfolio.total_value
        target_positions = self.config.get("target_positions", 10)

        # Equal weight per position
        position_value = portfolio_value / target_positions

        return position_value

    def _calculate_volatility_weight_position_size(
        self, signal: Dict[str, Any], portfolio: PortfolioRecord, config: BacktestConfig
    ) -> float:
        """Calculate position size using volatility weighting"""

        portfolio_value = portfolio.total_value

        # Get signal volatility (simplified)
        signal_volatility = signal.get("volatility", 0.02)  # Default 2%

        # Inverse volatility weighting
        # Lower volatility = larger position size
        volatility_factor = 0.02 / signal_volatility  # Normalize to 2%
        volatility_factor = max(0.5, min(volatility_factor, 2.0))  # Limit range

        base_position_value = portfolio_value * config.fixed_position_size
        position_value = base_position_value * volatility_factor

        return position_value

    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk limits"""
        return {
            "max_single_position_risk": self.config.get("max_single_position_risk", 0.02),
            "max_portfolio_risk": self.config.get("max_portfolio_risk", 0.02),
            "max_drawdown": self.config.get("max_drawdown", 0.2),
            "max_position_size": self.config.get("max_position_size", 0.1),
        }

    def get_current_risk_status(self, portfolio: PortfolioRecord, config: BacktestConfig) -> Dict[str, Any]:
        """Get current risk status"""

        current_risk = self.calculate_portfolio_risk(portfolio, config)
        max_risk = config.max_portfolio_risk

        return {
            "current_portfolio_risk": current_risk,
            "max_portfolio_risk": max_risk,
            "risk_utilization": current_risk / max_risk if max_risk > 0 else 0,
            "available_risk_budget": max_risk - current_risk,
            "num_positions": len(portfolio.positions),
            "diversification_factor": self._calculate_diversification_factor(portfolio),
        }
