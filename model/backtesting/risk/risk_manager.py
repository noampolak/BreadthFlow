"""
Risk Manager Interface

Abstract interface for risk management in the BreadthFlow system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from ..backtest_config import BacktestConfig
from ..trade_record import TradeRecord, PositionRecord, PortfolioRecord


class RiskManager(ABC):
    """Abstract interface for risk management"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}

        # Risk tracking
        self.risk_stats = {
            "total_positions": 0,
            "risk_limited_trades": 0,
            "risk_rejected_trades": 0,
            "max_portfolio_risk": 0.0,
            "current_portfolio_risk": 0.0,
        }

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this risk manager"""
        pass

    @abstractmethod
    def calculate_position_risk(self, position: PositionRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> float:
        """Calculate risk for a specific position"""
        pass

    @abstractmethod
    def calculate_portfolio_risk(self, portfolio: PortfolioRecord, config: BacktestConfig) -> float:
        """Calculate total portfolio risk"""
        pass

    @abstractmethod
    def validate_trade(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Validate if a trade meets risk requirements"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get risk manager configuration"""
        return self.config

    def calculate_position_size_limit(
        self, signal: Dict[str, Any], portfolio: PortfolioRecord, config: BacktestConfig
    ) -> float:
        """Calculate maximum position size based on risk limits"""

        # Get current portfolio risk
        current_risk = self.calculate_portfolio_risk(portfolio, config)
        max_risk = config.max_portfolio_risk

        # Calculate available risk budget
        available_risk = max_risk - current_risk

        if available_risk <= 0:
            return 0.0

        # Calculate position size based on risk per trade
        risk_per_trade = config.risk_per_trade
        portfolio_value = portfolio.total_value

        # Simple risk-based position sizing
        max_position_value = portfolio_value * min(available_risk, risk_per_trade)

        return max_position_value

    def calculate_stop_loss(self, entry_price: float, signal: Dict[str, Any], config: BacktestConfig) -> float:
        """Calculate stop loss price"""

        stop_loss_rate = config.stop_loss_rate

        if signal["signal_type"] in ["buy", "cover"]:
            # Long position - stop loss below entry
            stop_loss = entry_price * (1 - stop_loss_rate)
        else:
            # Short position - stop loss above entry
            stop_loss = entry_price * (1 + stop_loss_rate)

        return stop_loss

    def calculate_take_profit(self, entry_price: float, signal: Dict[str, Any], config: BacktestConfig) -> float:
        """Calculate take profit price"""

        take_profit_rate = config.take_profit_rate

        if signal["signal_type"] in ["buy", "cover"]:
            # Long position - take profit above entry
            take_profit = entry_price * (1 + take_profit_rate)
        else:
            # Short position - take profit below entry
            take_profit = entry_price * (1 - take_profit_rate)

        return take_profit

    def check_drawdown_limit(self, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check if portfolio is within drawdown limits"""

        max_drawdown = config.max_drawdown

        # Calculate current drawdown
        if hasattr(portfolio, "max_drawdown") and portfolio.max_drawdown is not None:
            current_drawdown = abs(portfolio.max_drawdown)
        else:
            # Calculate drawdown from portfolio history
            current_drawdown = self._calculate_current_drawdown(portfolio)

        return current_drawdown <= max_drawdown

    def _calculate_current_drawdown(self, portfolio: PortfolioRecord) -> float:
        """Calculate current drawdown from portfolio value"""

        # This is a simplified calculation
        # In a real implementation, you'd track the peak portfolio value

        # For now, we'll assume no drawdown if we don't have historical data
        return 0.0

    def check_concentration_limit(self, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check if portfolio meets concentration limits"""

        max_position_size = config.max_position_size

        for position in portfolio.positions:
            position_size = position.position_size
            if position_size > max_position_size:
                return False

        return True

    def calculate_var(self, portfolio: PortfolioRecord, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk for portfolio"""

        # This is a simplified VaR calculation
        # In a real implementation, you'd use historical returns or Monte Carlo simulation

        portfolio_value = portfolio.total_value
        volatility = portfolio.volatility if portfolio.volatility else 0.02  # 2% default

        # Parametric VaR (assuming normal distribution)
        import numpy as np
        from scipy.stats import norm

        z_score = norm.ppf(1 - confidence_level)
        var = portfolio_value * volatility * z_score

        return abs(var)

    def calculate_expected_shortfall(self, portfolio: PortfolioRecord, confidence_level: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR) for portfolio"""

        # This is a simplified ES calculation
        # In a real implementation, you'd use historical returns or Monte Carlo simulation

        var = self.calculate_var(portfolio, confidence_level)

        # For normal distribution, ES = E[X|X>VaR] = μ + σ * φ(α) / (1-α)
        # where φ is the standard normal PDF and α is the confidence level

        import numpy as np
        from scipy.stats import norm

        alpha = 1 - confidence_level
        phi_alpha = norm.pdf(norm.ppf(alpha))
        es_factor = phi_alpha / alpha

        portfolio_value = portfolio.total_value
        volatility = portfolio.volatility if portfolio.volatility else 0.02

        expected_shortfall = portfolio_value * volatility * es_factor

        return abs(expected_shortfall)

    def update_risk_stats(self, trade: TradeRecord, accepted: bool):
        """Update risk management statistics"""
        self.risk_stats["total_positions"] += 1

        if accepted:
            self.risk_stats["risk_limited_trades"] += 1
        else:
            self.risk_stats["risk_rejected_trades"] += 1

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk management performance metrics"""
        metrics = self.risk_stats.copy()

        total_positions = metrics["total_positions"]
        if total_positions > 0:
            metrics["risk_acceptance_rate"] = metrics["risk_limited_trades"] / total_positions
            metrics["risk_rejection_rate"] = metrics["risk_rejected_trades"] / total_positions
        else:
            metrics["risk_acceptance_rate"] = 0.0
            metrics["risk_rejection_rate"] = 0.0

        return metrics

    def get_risk_info(self) -> Dict[str, Any]:
        """Get comprehensive risk manager information"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "config": self.config,
            "risk_metrics": self.get_risk_metrics(),
        }
