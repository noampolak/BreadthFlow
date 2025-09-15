"""
VaR Risk Manager

Implements Value at Risk (VaR) based risk management with advanced risk metrics
and portfolio optimization.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..backtest_config import BacktestConfig
from ..trade_record import PortfolioRecord, PositionRecord, TradeRecord
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class VaRRiskManager(RiskManager):
    """VaR-based risk manager for advanced risk control"""

    def __init__(self, name: str = "var_risk", config: Dict[str, Any] = None):
        super().__init__(name, config)

        # Default configuration for VaR risk management
        self.default_config = {
            "var_confidence_level": 0.95,  # 95% VaR
            "var_time_horizon": 1,  # 1-day VaR
            "max_var_limit": 0.02,  # 2% max VaR
            "var_method": "parametric",  # 'parametric', 'historical', 'monte_carlo'
            "correlation_window": 252,  # Days for correlation calculation
            "volatility_window": 60,  # Days for volatility calculation
            "stress_test_scenarios": ["market_crash", "volatility_spike", "correlation_breakdown"],
            "enable_stress_testing": True,
            "enable_scenario_analysis": True,
            "var_adjustment_factor": 1.0,  # VaR adjustment multiplier
            "enable_conditional_var": True,  # Enable Expected Shortfall
            "max_conditional_var": 0.03,  # 3% max Expected Shortfall
        }

        # Update config with defaults
        self.config.update(self.default_config)

        # Historical data storage for VaR calculations
        self.return_history = {}
        self.correlation_matrix = None
        self.volatility_vector = None

    def get_name(self) -> str:
        """Get the name of this risk manager"""
        return self.name

    def calculate_position_risk(self, position: PositionRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> float:
        """Calculate VaR-based risk for a specific position"""

        # Get position volatility
        position_volatility = self._get_position_volatility(position.symbol)

        # Calculate position VaR
        position_value = position.market_value
        confidence_level = self.config.get("var_confidence_level", 0.95)

        # Parametric VaR calculation
        z_score = self._get_z_score(confidence_level)
        position_var = position_value * position_volatility * z_score

        return abs(position_var)

    def calculate_portfolio_risk(self, portfolio: PortfolioRecord, config: BacktestConfig) -> float:
        """Calculate portfolio VaR"""

        if not portfolio.positions:
            return 0.0

        # Get portfolio weights and volatilities
        weights = []
        volatilities = []
        symbols = []

        total_value = portfolio.total_value
        if total_value <= 0:
            return 0.0

        for position in portfolio.positions:
            weight = position.market_value / total_value
            volatility = self._get_position_volatility(position.symbol)

            weights.append(weight)
            volatilities.append(volatility)
            symbols.append(position.symbol)

        weights = np.array(weights)
        volatilities = np.array(volatilities)

        # Get correlation matrix
        correlation_matrix = self._get_correlation_matrix(symbols)

        # Calculate portfolio VaR
        var_method = self.config.get("var_method", "parametric")

        if var_method == "parametric":
            portfolio_var = self._calculate_parametric_var(weights, volatilities, correlation_matrix)
        elif var_method == "historical":
            portfolio_var = self._calculate_historical_var(portfolio)
        elif var_method == "monte_carlo":
            portfolio_var = self._calculate_monte_carlo_var(weights, volatilities, correlation_matrix)
        else:
            portfolio_var = self._calculate_parametric_var(weights, volatilities, correlation_matrix)

        # Apply VaR adjustment factor
        var_adjustment = self.config.get("var_adjustment_factor", 1.0)
        portfolio_var *= var_adjustment

        # Update risk stats
        self.risk_stats["current_portfolio_risk"] = portfolio_var
        self.risk_stats["max_portfolio_risk"] = max(self.risk_stats["max_portfolio_risk"], portfolio_var)

        return portfolio_var

    def validate_trade(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Validate trade using VaR-based risk limits"""

        # Check basic risk limits
        if not self._check_basic_risk_limits(trade, portfolio, config):
            logger.warning(f"Trade rejected: Basic risk limits exceeded")
            return False

        # Check VaR limits
        if not self._check_var_limits(trade, portfolio, config):
            logger.warning(f"Trade rejected: VaR limits exceeded")
            return False

        # Check Expected Shortfall limits
        if not self._check_conditional_var_limits(trade, portfolio, config):
            logger.warning(f"Trade rejected: Expected Shortfall limits exceeded")
            return False

        # Run stress tests if enabled
        if self.config.get("enable_stress_testing", True):
            if not self._run_stress_tests(trade, portfolio, config):
                logger.warning(f"Trade rejected: Failed stress tests")
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

        return True

    def _check_var_limits(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check VaR limits"""

        # Calculate current portfolio VaR
        current_var = self.calculate_portfolio_risk(portfolio, config)
        max_var = self.config.get("max_var_limit", 0.02)

        # Estimate new portfolio VaR with this trade
        # This is a simplified estimation - in practice, you'd recalculate the full portfolio
        trade_value = trade.quantity * trade.price
        portfolio_value = portfolio.total_value

        if portfolio_value > 0:
            # Simple linear approximation of VaR impact
            trade_weight = trade_value / portfolio_value
            trade_volatility = self._get_position_volatility(trade.symbol)

            # Estimate VaR contribution of the trade
            confidence_level = self.config.get("var_confidence_level", 0.95)
            z_score = self._get_z_score(confidence_level)
            trade_var_contribution = trade_value * trade_volatility * z_score

            # Estimate new portfolio VaR (simplified)
            new_var = current_var + trade_var_contribution * 0.5  # Assume some diversification benefit

            if new_var > max_var:
                return False

        return True

    def _check_conditional_var_limits(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Check Expected Shortfall (Conditional VaR) limits"""

        if not self.config.get("enable_conditional_var", True):
            return True

        # Calculate Expected Shortfall
        es = self.calculate_expected_shortfall(portfolio)
        max_es = self.config.get("max_conditional_var", 0.03)

        # Estimate new ES with trade (simplified)
        trade_value = trade.quantity * trade.price
        portfolio_value = portfolio.total_value

        if portfolio_value > 0:
            trade_weight = trade_value / portfolio_value
            trade_volatility = self._get_position_volatility(trade.symbol)

            # Estimate ES contribution
            confidence_level = self.config.get("var_confidence_level", 0.95)
            alpha = 1 - confidence_level

            # Expected Shortfall factor for normal distribution
            from scipy.stats import norm

            phi_alpha = norm.pdf(norm.ppf(alpha))
            es_factor = phi_alpha / alpha

            trade_es_contribution = trade_value * trade_volatility * es_factor
            new_es = es + trade_es_contribution * 0.5  # Assume diversification benefit

            if new_es > max_es:
                return False

        return True

    def _run_stress_tests(self, trade: TradeRecord, portfolio: PortfolioRecord, config: BacktestConfig) -> bool:
        """Run stress tests on the portfolio"""

        scenarios = self.config.get("stress_test_scenarios", [])

        for scenario in scenarios:
            stress_var = self._calculate_stress_var(portfolio, scenario)
            max_stress_var = self.config.get("max_var_limit", 0.02) * 2  # Double the normal limit

            if stress_var > max_stress_var:
                logger.warning(f"Failed stress test: {scenario}")
                return False

        return True

    def _calculate_stress_var(self, portfolio: PortfolioRecord, scenario: str) -> float:
        """Calculate VaR under stress scenarios"""

        if scenario == "market_crash":
            # Market crash scenario: increase volatilities by 2x
            stress_multiplier = 2.0
        elif scenario == "volatility_spike":
            # Volatility spike scenario: increase volatilities by 1.5x
            stress_multiplier = 1.5
        elif scenario == "correlation_breakdown":
            # Correlation breakdown: assume higher correlations
            stress_multiplier = 1.3
        else:
            stress_multiplier = 1.0

        # Calculate stressed portfolio VaR
        # This is a simplified implementation
        base_var = self.calculate_portfolio_risk(portfolio, None)
        stress_var = base_var * stress_multiplier

        return stress_var

    def _calculate_parametric_var(
        self, weights: np.ndarray, volatilities: np.ndarray, correlation_matrix: np.ndarray
    ) -> float:
        """Calculate parametric VaR"""

        confidence_level = self.config.get("var_confidence_level", 0.95)
        z_score = self._get_z_score(confidence_level)

        # Calculate portfolio volatility
        portfolio_variance = np.dot(weights.T, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Calculate VaR
        portfolio_var = portfolio_volatility * z_score

        return abs(portfolio_var)

    def _calculate_historical_var(self, portfolio: PortfolioRecord) -> float:
        """Calculate historical VaR"""

        # This is a simplified implementation
        # In practice, you'd use actual historical returns

        confidence_level = self.config.get("var_confidence_level", 0.95)

        # Generate synthetic returns for demonstration
        num_days = 252
        returns = np.random.normal(0, 0.02, num_days)  # 2% daily volatility

        # Calculate historical VaR
        var_percentile = (1 - confidence_level) * 100
        historical_var = np.percentile(returns, var_percentile)

        # Scale by portfolio value
        portfolio_var = abs(historical_var * portfolio.total_value)

        return portfolio_var

    def _calculate_monte_carlo_var(
        self, weights: np.ndarray, volatilities: np.ndarray, correlation_matrix: np.ndarray
    ) -> float:
        """Calculate Monte Carlo VaR"""

        confidence_level = self.config.get("var_confidence_level", 0.95)
        num_simulations = 10000

        # Generate correlated random returns
        portfolio_returns = []

        for _ in range(num_simulations):
            # Generate correlated random variables
            correlated_returns = np.random.multivariate_normal(
                np.zeros(len(weights)), correlation_matrix * np.outer(volatilities, volatilities)
            )

            # Calculate portfolio return
            portfolio_return = np.dot(weights, correlated_returns)
            portfolio_returns.append(portfolio_return)

        # Calculate VaR from simulated returns
        var_percentile = (1 - confidence_level) * 100
        monte_carlo_var = np.percentile(portfolio_returns, var_percentile)

        return abs(monte_carlo_var)

    def _get_position_volatility(self, symbol: str) -> float:
        """Get volatility for a position"""

        # In a real implementation, this would calculate historical volatility
        # For now, we'll use a default value
        return 0.02  # 2% daily volatility

    def _get_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """Get correlation matrix for symbols"""

        # In a real implementation, this would calculate historical correlations
        # For now, we'll use a simplified correlation matrix
        n = len(symbols)
        correlation_matrix = np.eye(n) * 0.3  # Base correlation of 0.3
        np.fill_diagonal(correlation_matrix, 1.0)  # Diagonal is 1.0

        return correlation_matrix

    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level"""

        from scipy.stats import norm

        return norm.ppf(confidence_level)

    def update_return_history(self, symbol: str, returns: List[float]):
        """Update return history for VaR calculations"""

        if symbol not in self.return_history:
            self.return_history[symbol] = []

        self.return_history[symbol].extend(returns)

        # Keep only recent history
        max_history = self.config.get("correlation_window", 252)
        if len(self.return_history[symbol]) > max_history:
            self.return_history[symbol] = self.return_history[symbol][-max_history:]

    def get_var_report(self, portfolio: PortfolioRecord, config: BacktestConfig) -> Dict[str, Any]:
        """Generate comprehensive VaR report"""

        portfolio_var = self.calculate_portfolio_risk(portfolio, config)
        expected_shortfall = self.calculate_expected_shortfall(portfolio)

        # Calculate component VaR for each position
        component_var = {}
        for position in portfolio.positions:
            position_var = self.calculate_position_risk(position, portfolio, config)
            component_var[position.symbol] = position_var

        # Run stress tests
        stress_results = {}
        if self.config.get("enable_stress_testing", True):
            scenarios = self.config.get("stress_test_scenarios", [])
            for scenario in scenarios:
                stress_var = self._calculate_stress_var(portfolio, scenario)
                stress_results[scenario] = stress_var

        return {
            "portfolio_var": portfolio_var,
            "expected_shortfall": expected_shortfall,
            "var_confidence_level": self.config.get("var_confidence_level", 0.95),
            "var_time_horizon": self.config.get("var_time_horizon", 1),
            "component_var": component_var,
            "stress_test_results": stress_results,
            "var_method": self.config.get("var_method", "parametric"),
            "risk_limits": {
                "max_var": self.config.get("max_var_limit", 0.02),
                "max_conditional_var": self.config.get("max_conditional_var", 0.03),
            },
        }
