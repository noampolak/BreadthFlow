"""
Performance Metrics for Breadth/Thrust Signals POC

Implements comprehensive trading performance metrics:
- Sharpe ratio and risk-adjusted returns
- Hit rate and win/loss analysis
- Maximum drawdown calculation
- Value at Risk (VaR) and other risk metrics
- Statistical significance testing
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, sum as spark_sum, avg, stddev, min as spark_min, max as spark_max,
    window, expr, row_number, rank, dense_rank, lit, udf, lag, lead
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    LongType, TimestampType, BooleanType
)

from features.common.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Performance Metrics Calculator.
    
    Calculates comprehensive trading performance metrics:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
    - Win/loss analysis and hit rates
    - Drawdown analysis
    - Risk metrics (VaR, CVaR, volatility)
    - Statistical significance testing
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PerformanceMetrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
        
        logger.info(f"PerformanceMetrics initialized with {risk_free_rate:.1%} risk-free rate")
    
    def calculate_all_metrics(
        self,
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        trade_pnls: Optional[List[float]] = None,
        portfolio_values: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            returns: List of daily returns
            benchmark_returns: List of benchmark daily returns
            trade_pnls: List of individual trade P&Ls
            portfolio_values: List of portfolio values over time
            
        Returns:
            Dictionary with all performance metrics
        """
        logger.info("Calculating comprehensive performance metrics")
        
        metrics = {}
        
        # Basic return metrics
        metrics.update(self._calculate_return_metrics(returns))
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(returns))
        
        # Risk-adjusted return metrics
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Drawdown analysis
        if portfolio_values:
            metrics.update(self._calculate_drawdown_metrics(portfolio_values))
        
        # Trading statistics
        if trade_pnls:
            metrics.update(self._calculate_trading_metrics(trade_pnls))
        
        # Benchmark comparison
        if benchmark_returns:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        # Statistical significance
        metrics.update(self._calculate_statistical_metrics(returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate basic return metrics.
        
        Args:
            returns: List of daily returns
            
        Returns:
            Dictionary with return metrics
        """
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Total return
        total_return = (1 + returns_array).prod() - 1
        
        # Annualized return
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(returns)) - 1
        
        # Average daily return
        avg_daily_return = np.mean(returns_array)
        
        # Best and worst days
        best_day = np.max(returns_array)
        worst_day = np.min(returns_array)
        
        # Positive and negative days
        positive_days = np.sum(returns_array > 0)
        negative_days = np.sum(returns_array < 0)
        total_days = len(returns_array)
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "avg_daily_return": avg_daily_return,
            "best_day": best_day,
            "worst_day": worst_day,
            "positive_days": positive_days,
            "negative_days": negative_days,
            "total_days": total_days,
            "positive_day_rate": positive_days / total_days if total_days > 0 else 0.0
        }
    
    def _calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate risk metrics.
        
        Args:
            returns: List of daily returns
            
        Returns:
            Dictionary with risk metrics
        """
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Volatility
        volatility = np.std(returns_array)
        annualized_volatility = volatility * np.sqrt(self.trading_days_per_year)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns_array, 5)  # 95% VaR
        var_99 = np.percentile(returns_array, 1)  # 99% VaR
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(returns_array[returns_array <= var_95])
        cvar_99 = np.mean(returns_array[returns_array <= var_99])
        
        # Downside deviation
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        # Skewness and kurtosis
        skewness = stats.skew(returns_array)
        kurtosis = stats.kurtosis(returns_array)
        
        return {
            "volatility": volatility,
            "annualized_volatility": annualized_volatility,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "downside_deviation": downside_deviation,
            "skewness": skewness,
            "kurtosis": kurtosis
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            returns: List of daily returns
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Excess returns
        daily_risk_free = (1 + self.risk_free_rate) ** (1 / self.trading_days_per_year) - 1
        excess_returns = returns_array - daily_risk_free
        
        # Sharpe ratio
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year) if np.std(excess_returns) > 0 else 0.0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(self.trading_days_per_year) if downside_deviation > 0 else 0.0
        
        # Information ratio (assuming zero benchmark)
        information_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(self.trading_days_per_year) if np.std(returns_array) > 0 else 0.0
        
        # Treynor ratio (assuming beta = 1 for simplicity)
        treynor_ratio = np.mean(excess_returns) * self.trading_days_per_year if np.std(returns_array) > 0 else 0.0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "information_ratio": information_ratio,
            "treynor_ratio": treynor_ratio
        }
    
    def _calculate_drawdown_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """
        Calculate drawdown metrics.
        
        Args:
            portfolio_values: List of portfolio values over time
            
        Returns:
            Dictionary with drawdown metrics
        """
        if not portfolio_values:
            return {}
        
        values_array = np.array(portfolio_values)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values_array)
        
        # Calculate drawdowns
        drawdowns = (values_array - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Average drawdown
        avg_drawdown = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0.0
        
        # Drawdown duration
        drawdown_periods = np.sum(drawdowns < 0)
        total_periods = len(drawdowns)
        drawdown_frequency = drawdown_periods / total_periods if total_periods > 0 else 0.0
        
        # Calmar ratio (annualized return / max drawdown)
        total_return = (values_array[-1] - values_array[0]) / values_array[0]
        annualized_return = (1 + total_return) ** (self.trading_days_per_year / len(values_array)) - 1
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "drawdown_frequency": drawdown_frequency,
            "calmar_ratio": calmar_ratio
        }
    
    def _calculate_trading_metrics(self, trade_pnls: List[float]) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.
        
        Args:
            trade_pnls: List of individual trade P&Ls
            
        Returns:
            Dictionary with trading metrics
        """
        if not trade_pnls:
            return {}
        
        pnls_array = np.array(trade_pnls)
        
        # Basic statistics
        total_trades = len(pnls_array)
        winning_trades = np.sum(pnls_array > 0)
        losing_trades = np.sum(pnls_array < 0)
        
        # Hit rate
        hit_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Average win and loss
        avg_win = np.mean(pnls_array[pnls_array > 0]) if winning_trades > 0 else 0.0
        avg_loss = np.mean(pnls_array[pnls_array < 0]) if losing_trades > 0 else 0.0
        
        # Largest win and loss
        largest_win = np.max(pnls_array) if winning_trades > 0 else 0.0
        largest_loss = np.min(pnls_array) if losing_trades > 0 else 0.0
        
        # Profit factor
        total_wins = np.sum(pnls_array[pnls_array > 0])
        total_losses = abs(np.sum(pnls_array[pnls_array < 0]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Win/loss ratio
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
        
        # Expected value
        expected_value = np.mean(pnls_array)
        
        # Risk of ruin (simplified)
        std_pnl = np.std(pnls_array)
        risk_of_ruin = np.exp(-2 * expected_value / std_pnl) if std_pnl > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "hit_rate": hit_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "win_loss_ratio": win_loss_ratio,
            "expected_value": expected_value,
            "risk_of_ruin": risk_of_ruin
        }
    
    def _calculate_benchmark_metrics(
        self, 
        returns: List[float], 
        benchmark_returns: List[float]
    ) -> Dict[str, float]:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            returns: List of strategy returns
            benchmark_returns: List of benchmark returns
            
        Returns:
            Dictionary with benchmark metrics
        """
        if not returns or not benchmark_returns:
            return {}
        
        # Ensure same length
        min_length = min(len(returns), len(benchmark_returns))
        strategy_returns = np.array(returns[:min_length])
        benchmark_returns = np.array(benchmark_returns[:min_length])
        
        # Excess returns
        excess_returns = strategy_returns - benchmark_returns
        
        # Alpha (excess return)
        alpha = np.mean(excess_returns) * self.trading_days_per_year
        
        # Beta (market sensitivity)
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        
        # Information ratio
        tracking_error = np.std(excess_returns)
        information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(self.trading_days_per_year) if tracking_error > 0 else 0.0
        
        # Correlation
        correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
        
        # R-squared
        r_squared = correlation ** 2
        
        # Jensen's alpha
        risk_free_daily = (1 + self.risk_free_rate) ** (1 / self.trading_days_per_year) - 1
        jensen_alpha = np.mean(strategy_returns - risk_free_daily) - beta * np.mean(benchmark_returns - risk_free_daily)
        jensen_alpha *= self.trading_days_per_year
        
        return {
            "alpha": alpha,
            "beta": beta,
            "information_ratio": information_ratio,
            "correlation": correlation,
            "r_squared": r_squared,
            "jensen_alpha": jensen_alpha,
            "tracking_error": tracking_error * np.sqrt(self.trading_days_per_year)
        }
    
    def _calculate_statistical_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate statistical significance metrics.
        
        Args:
            returns: List of daily returns
            
        Returns:
            Dictionary with statistical metrics
        """
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Test for normality
        _, normality_p_value = stats.normaltest(returns_array)
        
        # Test for zero mean
        _, mean_p_value = stats.ttest_1samp(returns_array, 0)
        
        # Jarque-Bera test for normality
        _, jarque_bera_p_value = stats.jarque_bera(returns_array)
        
        # Ljung-Box test for autocorrelation
        if len(returns_array) > 10:
            _, ljung_box_p_value = stats.acf(returns_array, nlags=min(10, len(returns_array)//4), fft=False)
            ljung_box_p_value = ljung_box_p_value[0] if len(ljung_box_p_value) > 0 else 1.0
        else:
            ljung_box_p_value = 1.0
        
        # Confidence intervals for mean return
        mean_ci_lower, mean_ci_upper = stats.t.interval(
            0.95, len(returns_array)-1, 
            loc=np.mean(returns_array), 
            scale=stats.sem(returns_array)
        )
        
        return {
            "normality_p_value": normality_p_value,
            "mean_p_value": mean_p_value,
            "jarque_bera_p_value": jarque_bera_p_value,
            "ljung_box_p_value": ljung_box_p_value,
            "mean_ci_lower": mean_ci_lower,
            "mean_ci_upper": mean_ci_upper,
            "is_normal": normality_p_value > 0.05,
            "is_significant": mean_p_value < 0.05
        }
    
    def calculate_rolling_metrics(
        self,
        returns: List[float],
        window_size: int = 252,
        step_size: int = 21
    ) -> Dict[str, List[float]]:
        """
        Calculate rolling performance metrics.
        
        Args:
            returns: List of daily returns
            window_size: Rolling window size in days
            step_size: Step size for rolling windows
            
        Returns:
            Dictionary with rolling metrics
        """
        if not returns or len(returns) < window_size:
            return {}
        
        rolling_metrics = {
            "sharpe_ratio": [],
            "volatility": [],
            "total_return": [],
            "max_drawdown": [],
            "hit_rate": []
        }
        
        for i in range(0, len(returns) - window_size + 1, step_size):
            window_returns = returns[i:i + window_size]
            
            # Calculate metrics for this window
            metrics = self.calculate_all_metrics(window_returns)
            
            rolling_metrics["sharpe_ratio"].append(metrics.get("sharpe_ratio", 0.0))
            rolling_metrics["volatility"].append(metrics.get("annualized_volatility", 0.0))
            rolling_metrics["total_return"].append(metrics.get("total_return", 0.0))
            rolling_metrics["max_drawdown"].append(metrics.get("max_drawdown", 0.0))
            rolling_metrics["hit_rate"].append(metrics.get("hit_rate", 0.0))
        
        return rolling_metrics
    
    def generate_performance_report(
        self,
        metrics: Dict[str, float],
        strategy_name: str = "Breadth/Thrust Strategy"
    ) -> str:
        """
        Generate a formatted performance report.
        
        Args:
            metrics: Dictionary with performance metrics
            strategy_name: Name of the strategy
            
        Returns:
            Formatted performance report string
        """
        report = f"""
{'='*60}
{strategy_name} - Performance Report
{'='*60}

RETURN METRICS:
  Total Return: {metrics.get('total_return', 0):.2%}
  Annualized Return: {metrics.get('annualized_return', 0):.2%}
  Average Daily Return: {metrics.get('avg_daily_return', 0):.4%}
  Best Day: {metrics.get('best_day', 0):.2%}
  Worst Day: {metrics.get('worst_day', 0):.2%}

RISK METRICS:
  Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}
  Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}
  VaR (95%): {metrics.get('var_95', 0):.2%}
  CVaR (95%): {metrics.get('cvar_95', 0):.2%}

RISK-ADJUSTED METRICS:
  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}
  Information Ratio: {metrics.get('information_ratio', 0):.2f}

TRADING METRICS:
  Total Trades: {metrics.get('total_trades', 0)}
  Hit Rate: {metrics.get('hit_rate', 0):.2%}
  Profit Factor: {metrics.get('profit_factor', 0):.2f}
  Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}
  Average Win: ${metrics.get('avg_win', 0):,.2f}
  Average Loss: ${metrics.get('avg_loss', 0):,.2f}

STATISTICAL SIGNIFICANCE:
  Is Normal Distribution: {metrics.get('is_normal', False)}
  Is Statistically Significant: {metrics.get('is_significant', False)}
  Mean P-Value: {metrics.get('mean_p_value', 1):.4f}

{'='*60}
        """
        
        return report


def create_performance_metrics(risk_free_rate: float = 0.02) -> PerformanceMetrics:
    """
    Factory function to create PerformanceMetrics instance.
    
    Args:
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Configured PerformanceMetrics instance
    """
    return PerformanceMetrics(risk_free_rate)


def calculate_metrics(
    returns: List[float],
    benchmark_returns: Optional[List[float]] = None,
    trade_pnls: Optional[List[float]] = None,
    portfolio_values: Optional[List[float]] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """
    Calculate performance metrics for backtesting results.
    
    Args:
        returns: List of daily returns
        benchmark_returns: List of benchmark daily returns
        trade_pnls: List of individual trade P&Ls
        portfolio_values: List of portfolio values over time
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary with all performance metrics
    """
    logger.info("Calculating performance metrics for backtest results")
    
    # Create metrics calculator
    metrics_calc = create_performance_metrics(risk_free_rate=risk_free_rate)
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_all_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        trade_pnls=trade_pnls,
        portfolio_values=portfolio_values
    )
    
    return metrics


# Example usage and testing
if __name__ == "__main__":
    # Create performance metrics calculator
    metrics_calc = create_performance_metrics(risk_free_rate=0.02)
    
    # Generate sample returns
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    sample_trades = np.random.normal(100, 500, 50)  # 50 trades
    sample_portfolio = [100000 * (1 + np.cumsum(sample_returns))]
    
    # Calculate all metrics
    metrics = metrics_calc.calculate_all_metrics(
        returns=sample_returns.tolist(),
        trade_pnls=sample_trades.tolist(),
        portfolio_values=sample_portfolio[0].tolist()
    )
    
    # Generate report
    report = metrics_calc.generate_performance_report(metrics)
    print(report)
