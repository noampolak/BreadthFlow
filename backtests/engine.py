"""
Backtesting Engine for Breadth/Thrust Signals POC

Implements comprehensive backtesting framework:
- Portfolio simulation with realistic constraints
- Transaction cost modeling (commissions, slippage)
- Performance tracking and metrics calculation
- Risk management and position sizing
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
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
from features.common.io import read_delta, write_delta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionType(Enum):
    """Position types for portfolio management."""
    LONG = "long"
    SHORT = "short"
    CASH = "cash"


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    # Portfolio settings
    initial_capital: float = 100000.0
    position_size_pct: float = 0.1  # 10% per position
    max_positions: int = 10
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005   # 0.05% slippage
    min_commission: float = 1.0     # Minimum commission per trade
    
    # Risk management
    stop_loss_pct: float = 0.05     # 5% stop loss
    take_profit_pct: float = 0.15   # 15% take profit
    max_drawdown_pct: float = 0.20  # 20% max drawdown
    
    # Signal filters
    min_signal_confidence: float = 70.0
    min_signal_strength: str = "strong"
    
    # Benchmark settings
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02    # 2% annual risk-free rate


@dataclass
class PortfolioPosition:
    """Represents a single portfolio position."""
    symbol: str
    position_type: PositionType
    shares: float
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    current_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    is_active: bool = True


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0  # 95% Value at Risk
    calmar_ratio: float = 0.0
    
    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Portfolio evolution
    portfolio_values: List[float] = field(default_factory=list)
    benchmark_values: List[float] = field(default_factory=list)
    dates: List[datetime] = field(default_factory=list)
    
    # Signal analysis
    signals_generated: int = 0
    signals_acted_upon: int = 0
    signal_accuracy: float = 0.0


class BacktestEngine:
    """
    Main Backtesting Engine for Breadth/Thrust Signals.
    
    Implements comprehensive backtesting with:
    - Portfolio simulation with realistic constraints
    - Transaction cost modeling
    - Performance tracking and risk management
    - Signal evaluation and accuracy measurement
    """
    
    def __init__(self, spark: SparkSession, config: Optional[BacktestConfig] = None):
        """
        Initialize BacktestEngine.
        
        Args:
            spark: PySpark session for distributed processing
            config: Backtesting configuration
        """
        self.spark = spark
        self.config = config or BacktestConfig()
        
        # Portfolio state
        self.cash = self.config.initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.peak_portfolio_value = self.config.initial_capital
        self.current_drawdown = 0.0
        
        # Signal tracking
        self.signals_processed = 0
        self.signals_acted_upon = 0
        self.correct_signals = 0
        
        logger.info(f"BacktestEngine initialized with ${self.config.initial_capital:,.2f} initial capital")
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        save_results: bool = True
    ) -> BacktestResult:
        """
        Run complete backtest for the specified period.
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            symbols: List of symbols to trade
            save_results: Save results to Delta Lake
            
        Returns:
            BacktestResult with comprehensive performance metrics
        """
        logger.info(f"Running backtest from {start_date} to {end_date}")
        start_time = datetime.now()
        
        try:
            # Load market data
            market_data = self._load_market_data(start_date, end_date, symbols)
            
            # Load trading signals
            signals = self._load_trading_signals(start_date, end_date)
            
            # Load benchmark data
            benchmark_data = self._load_benchmark_data(start_date, end_date)
            
            # Run portfolio simulation
            portfolio_evolution = self._simulate_portfolio(market_data, signals)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(
                portfolio_evolution, benchmark_data
            )
            
            # Save results if requested
            if save_results:
                self._save_backtest_results(results, start_date, end_date)
            
            # Log completion
            backtest_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Backtest completed in {backtest_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise
    
    def _load_market_data(
        self, 
        start_date: str, 
        end_date: str, 
        symbols: Optional[List[str]]
    ) -> DataFrame:
        """
        Load market data for backtesting.
        
        Args:
            start_date: Start date
            end_date: End date
            symbols: List of symbols
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info("Loading market data")
        
        # Load from Delta Lake (same as data fetcher and signal generator)
        market_data = read_delta(self.spark, "data/ohlcv")
        
        # Apply date filters
        market_data = market_data.filter(
            (col("date") >= start_date) & (col("date") <= end_date)
        )
        
        # Apply symbol filter if provided
        if symbols:
            market_data = market_data.filter(col("symbol").isin(symbols))
        
        # Add price change calculations
        window_spec = Window.partitionBy("symbol").orderBy("date")
        market_data = market_data.withColumn(
            "price_change_pct",
            (col("close") - lag("close", 1).over(window_spec)) / lag("close", 1).over(window_spec)
        ).withColumn(
            "high_low_range",
            (col("high") - col("low")) / col("close")
        )
        
        return market_data.orderBy("date")
    
    def _load_trading_signals(self, start_date: str, end_date: str) -> DataFrame:
        """
        Load trading signals for backtesting.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with trading signals
        """
        logger.info("Loading trading signals")
        
        try:
            signals = read_delta(self.spark, "data/trading_signals")
            
            # Apply date filters
            signals = signals.filter(
                (col("date") >= start_date) & (col("date") <= end_date)
            )
            
            # Filter by signal quality
            signals = signals.filter(
                (col("confidence_score") >= self.config.min_signal_confidence) &
                (col("signal_strength").isin(["strong", "very_strong"]))
            )
            
            return signals.orderBy("date")
            
        except Exception as e:
            logger.warning(f"Could not load signals: {e}")
            # Return empty DataFrame with correct schema
            return self.spark.createDataFrame([], schema=self._get_signals_schema())
    
    def _load_benchmark_data(self, start_date: str, end_date: str) -> DataFrame:
        """
        Load benchmark data for comparison.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with benchmark data
        """
        logger.info(f"Loading benchmark data for {self.config.benchmark_symbol}")
        
        try:
            benchmark_data = read_delta(self.spark, "data/ohlcv")
            
            # Filter for benchmark symbol
            benchmark_data = benchmark_data.filter(
                col("symbol") == self.config.benchmark_symbol
            )
            
            # Apply date filters
            benchmark_data = benchmark_data.filter(
                (col("date") >= start_date) & (col("date") <= end_date)
            )
            
            return benchmark_data.orderBy("date")
            
        except Exception as e:
            logger.warning(f"Could not load benchmark data: {e}")
            return self.spark.createDataFrame([], schema=self._get_ohlcv_schema())
    
    def _get_signals_schema(self) -> StructType:
        """Get signals schema."""
        return StructType([
            StructField("date", TimestampType(), False),
            StructField("composite_score_0_100", DoubleType(), True),
            StructField("signal_direction", StringType(), True),
            StructField("signal_strength", StringType(), True),
            StructField("confidence_score", DoubleType(), True),
            StructField("buy_signal", BooleanType(), True),
            StructField("sell_signal", BooleanType(), True),
            StructField("strong_buy_signal", BooleanType(), True),
            StructField("strong_sell_signal", BooleanType(), True)
        ])
    
    def _get_ohlcv_schema(self) -> StructType:
        """Get OHLCV schema."""
        return StructType([
            StructField("symbol", StringType(), False),
            StructField("date", TimestampType(), False),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", LongType(), True)
        ])
    
    def _simulate_portfolio(self, market_data: DataFrame, signals: DataFrame) -> List[Dict[str, Any]]:
        """
        Simulate portfolio evolution based on signals and market data.
        
        Args:
            market_data: Market OHLCV data
            signals: Trading signals
            
        Returns:
            List of portfolio states over time
        """
        logger.info("Simulating portfolio evolution")
        
        # Convert to pandas for easier iteration
        market_pdf = market_data.toPandas()
        signals_pdf = signals.toPandas()
        
        portfolio_evolution = []
        
        # Group market data by date
        market_by_date = market_pdf.groupby("date")
        
        for date, daily_data in market_by_date:
            # Update portfolio with current market prices
            self._update_portfolio_prices(daily_data)
            
            # Check for signal on this date
            daily_signals = signals_pdf[signals_pdf["date"] == date]
            
            if not daily_signals.empty:
                signal = daily_signals.iloc[0]
                self._process_signal(signal, daily_data)
            
            # Check for stop losses and take profits
            self._check_risk_management(daily_data)
            
            # Record portfolio state
            portfolio_state = self._record_portfolio_state(date, daily_data)
            portfolio_evolution.append(portfolio_state)
            
            # Update peak value and drawdown
            current_value = portfolio_state["total_value"]
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value
            
            self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        return portfolio_evolution
    
    def _update_portfolio_prices(self, daily_data: pd.DataFrame):
        """Update portfolio positions with current market prices."""
        for symbol, position in self.positions.items():
            if not position.is_active:
                continue
            
            # Find current price for this symbol
            symbol_data = daily_data[daily_data["symbol"] == symbol]
            if not symbol_data.empty:
                position.current_price = symbol_data.iloc[0]["close"]
                position.current_value = position.shares * position.current_price
                
                # Calculate unrealized P&L
                if position.position_type == PositionType.LONG:
                    position.unrealized_pnl = (position.current_price - position.entry_price) * position.shares
                else:  # SHORT
                    position.unrealized_pnl = (position.entry_price - position.current_price) * position.shares
    
    def _process_signal(self, signal: pd.Series, daily_data: pd.DataFrame):
        """Process trading signal and execute trades."""
        self.signals_processed += 1
        
        # Determine action based on signal
        action = None
        if signal["strong_buy_signal"] or signal["buy_signal"]:
            action = "buy"
        elif signal["strong_sell_signal"] or signal["sell_signal"]:
            action = "sell"
        
        if not action:
            return
        
        # Find available symbols for trading
        available_symbols = daily_data["symbol"].unique()
        
        # Execute trades based on signal strength and available capital
        if action == "buy":
            self._execute_buy_signals(available_symbols, daily_data, signal)
        elif action == "sell":
            self._execute_sell_signals(available_symbols, daily_data, signal)
        
        self.signals_acted_upon += 1
    
    def _execute_buy_signals(self, symbols: List[str], daily_data: pd.DataFrame, signal: pd.Series):
        """Execute buy signals based on available capital and position limits."""
        available_capital = self.cash * self.config.position_size_pct
        
        # Check if we can take new positions
        if len(self.positions) >= self.config.max_positions:
            return
        
        for symbol in symbols:
            if symbol in self.positions and self.positions[symbol].is_active:
                continue  # Already have position
            
            symbol_data = daily_data[daily_data["symbol"] == symbol]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[0]["close"]
            
            # Calculate position size
            shares = available_capital / current_price
            
            # Apply transaction costs
            total_cost = shares * current_price
            commission = max(total_cost * self.config.commission_rate, self.config.min_commission)
            slippage = total_cost * self.config.slippage_rate
            total_cost += commission + slippage
            
            if total_cost <= self.cash:
                # Execute trade
                self._open_position(
                    symbol, PositionType.LONG, shares, current_price, 
                    commission + slippage
                )
                self.cash -= total_cost
    
    def _execute_sell_signals(self, symbols: List[str], daily_data: pd.DataFrame, signal: pd.Series):
        """Execute sell signals for existing positions."""
        for symbol in symbols:
            if symbol not in self.positions or not self.positions[symbol].is_active:
                continue
            
            position = self.positions[symbol]
            if position.position_type != PositionType.LONG:
                continue
            
            symbol_data = daily_data[daily_data["symbol"] == symbol]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[0]["close"]
            
            # Calculate proceeds
            proceeds = position.shares * current_price
            commission = max(proceeds * self.config.commission_rate, self.config.min_commission)
            slippage = proceeds * self.config.slippage_rate
            net_proceeds = proceeds - commission - slippage
            
            # Close position
            self._close_position(symbol, current_price, commission + slippage)
            self.cash += net_proceeds
    
    def _open_position(
        self, 
        symbol: str, 
        position_type: PositionType, 
        shares: float, 
        price: float, 
        transaction_cost: float
    ):
        """Open a new position."""
        position = PortfolioPosition(
            symbol=symbol,
            position_type=position_type,
            shares=shares,
            entry_price=price,
            entry_date=datetime.now(),
            current_price=price,
            current_value=shares * price,
            stop_loss=price * (1 - self.config.stop_loss_pct),
            take_profit=price * (1 + self.config.take_profit_pct)
        )
        
        self.positions[symbol] = position
        
        # Record trade
        self.trade_history.append({
            "date": datetime.now(),
            "symbol": symbol,
            "action": "buy",
            "shares": shares,
            "price": price,
            "transaction_cost": transaction_cost,
            "signal_strength": "strong"
        })
    
    def _close_position(self, symbol: str, price: float, transaction_cost: float):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.is_active = False
        
        # Calculate realized P&L
        if position.position_type == PositionType.LONG:
            position.realized_pnl = (price - position.entry_price) * position.shares - transaction_cost
        else:  # SHORT
            position.realized_pnl = (position.entry_price - price) * position.shares - transaction_cost
        
        # Record trade
        self.trade_history.append({
            "date": datetime.now(),
            "symbol": symbol,
            "action": "sell",
            "shares": position.shares,
            "price": price,
            "transaction_cost": transaction_cost,
            "realized_pnl": position.realized_pnl
        })
    
    def _check_risk_management(self, daily_data: pd.DataFrame):
        """Check and execute stop losses and take profits."""
        for symbol, position in list(self.positions.items()):
            if not position.is_active:
                continue
            
            symbol_data = daily_data[daily_data["symbol"] == symbol]
            if symbol_data.empty:
                continue
            
            current_price = symbol_data.iloc[0]["close"]
            
            # Check stop loss
            if current_price <= position.stop_loss:
                self._close_position(symbol, current_price, 0.0)  # Market order
                self.correct_signals += 1  # Stop loss hit
            
            # Check take profit
            elif current_price >= position.take_profit:
                self._close_position(symbol, current_price, 0.0)  # Market order
                self.correct_signals += 1  # Take profit hit
    
    def _record_portfolio_state(self, date: datetime, daily_data: pd.DataFrame) -> Dict[str, Any]:
        """Record current portfolio state."""
        # Calculate total portfolio value
        total_value = self.cash
        for position in self.positions.values():
            if position.is_active:
                total_value += position.current_value
        
        # Calculate total P&L
        total_pnl = total_value - self.config.initial_capital
        
        return {
            "date": date,
            "cash": self.cash,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / self.config.initial_capital) * 100,
            "drawdown_pct": self.current_drawdown * 100,
            "active_positions": len([p for p in self.positions.values() if p.is_active]),
            "signals_processed": self.signals_processed,
            "signals_acted_upon": self.signals_acted_upon
        }
    
    def _calculate_performance_metrics(
        self, 
        portfolio_evolution: List[Dict[str, Any]], 
        benchmark_data: DataFrame
    ) -> BacktestResult:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_evolution: Portfolio states over time
            benchmark_data: Benchmark data for comparison
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info("Calculating performance metrics")
        
        if not portfolio_evolution:
            return BacktestResult()
        
        # Extract time series data
        dates = [state["date"] for state in portfolio_evolution]
        portfolio_values = [state["total_value"] for state in portfolio_evolution]
        
        # Calculate basic metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Calculate returns
        returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(daily_return)
        
        # Calculate risk metrics
        volatility = np.std(returns) * np.sqrt(252) if returns else 0.0
        annualized_return = total_return * (252 / len(returns)) if returns else 0.0
        
        # Calculate Sharpe ratio
        excess_returns = [r - self.config.risk_free_rate/252 for r in returns]
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if returns and np.std(excess_returns) > 0 else 0.0
        
        # Calculate maximum drawdown
        peak = initial_value
        max_drawdown = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Calculate trading statistics
        winning_trades = [t for t in self.trade_history if t.get("realized_pnl", 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get("realized_pnl", 0) < 0]
        
        total_trades = len(self.trade_history)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)
        
        hit_rate = winning_count / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t["realized_pnl"] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t["realized_pnl"] for t in losing_trades]) if losing_trades else 0.0
        
        profit_factor = abs(avg_win * winning_count / (avg_loss * losing_count)) if losing_count > 0 and avg_loss != 0 else 0.0
        
        largest_win = max([t["realized_pnl"] for t in winning_trades]) if winning_trades else 0.0
        largest_loss = min([t["realized_pnl"] for t in losing_trades]) if losing_trades else 0.0
        
        # Calculate signal accuracy
        signal_accuracy = self.correct_signals / self.signals_acted_upon if self.signals_acted_upon > 0 else 0.0
        
        # Calculate Value at Risk (95%)
        var_95 = np.percentile(returns, 5) if returns else 0.0
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            volatility=volatility,
            var_95=var_95,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            portfolio_values=portfolio_values,
            dates=dates,
            signals_generated=self.signals_processed,
            signals_acted_upon=self.signals_acted_upon,
            signal_accuracy=signal_accuracy
        )
    
    def _save_backtest_results(self, results: BacktestResult, start_date: str, end_date: str):
        """Save backtest results to Delta Lake."""
        logger.info("Saving backtest results")
        
        # Create results DataFrame
        results_data = []
        for i, date in enumerate(results.dates):
            results_data.append({
                "date": date,
                "portfolio_value": results.portfolio_values[i],
                "total_return_pct": results.total_return * 100,
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown_pct": results.max_drawdown * 100,
                "hit_rate": results.hit_rate,
                "total_trades": results.total_trades,
                "signals_processed": results.signals_generated,
                "signals_acted_upon": results.signals_acted_upon,
                "signal_accuracy": results.signal_accuracy
            })
        
        results_df = self.spark.createDataFrame(results_data)
        
        # Save to Delta Lake
        output_path = f"backtests/out/results_{start_date}_{end_date}"
        write_delta(
            df=results_df,
            path=output_path,
            partition_cols=["year", "month"],
            mode="overwrite"
        )
        
        logger.info(f"Results saved to {output_path}")


def create_backtest_engine(spark: SparkSession, config: Optional[BacktestConfig] = None) -> BacktestEngine:
    """
    Factory function to create BacktestEngine instance.
    
    Args:
        spark: PySpark session
        config: Backtesting configuration
        
    Returns:
        Configured BacktestEngine instance
    """
    return BacktestEngine(spark, config)


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("BacktestEngineTest") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=100000.0,
        position_size_pct=0.1,
        max_positions=5,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # Create backtest engine
    engine = create_backtest_engine(spark, config)
    
    try:
        # Run backtest
        results = engine.run_backtest(
            start_date="2024-01-01",
            end_date="2024-12-31",
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        )
        
        print(f"Backtest Results:")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annualized Return: {results.annualized_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Hit Rate: {results.hit_rate:.2%}")
        print(f"Total Trades: {results.total_trades}")
        print(f"Signal Accuracy: {results.signal_accuracy:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        spark.stop()
