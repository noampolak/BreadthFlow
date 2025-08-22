#!/usr/bin/env python3
"""
Timeframe-Agnostic Backtesting Engine

This module provides backtesting capabilities across multiple timeframes
while maintaining backward compatibility with existing daily backtesting.

Key Features:
- Support for multiple timeframes: 1min, 5min, 15min, 1hour, 1day
- Timeframe-aware trading hours and execution logic
- Realistic slippage and commission modeling by timeframe
- Comprehensive performance metrics per timeframe
- Backward compatibility with existing backtesting
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeAgnosticBacktestEngine:
    """
    Backtesting engine that adapts execution logic and risk parameters based on timeframe.
    
    This class maintains backward compatibility while adding multi-timeframe support.
    """
    
    def __init__(self, timeframe: str = '1day', initial_capital: float = 100000.0):
        """
        Initialize the backtesting engine for a specific timeframe.
        
        Args:
            timeframe: Target timeframe ('1min', '5min', '15min', '1hour', '1day')
            initial_capital: Initial capital for backtesting
        """
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        self.trading_hours = self.get_trading_hours(timeframe)
        self.execution_params = self.get_execution_parameters(timeframe)
        self.supported_timeframes = ['1min', '5min', '15min', '1hour', '1day']
        
        if timeframe not in self.supported_timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}")
        
        # Portfolio tracking
        self.positions = {}  # symbol -> {'quantity': int, 'avg_price': float, 'entry_time': datetime}
        self.trades = []     # List of executed trades
        self.portfolio_value_history = []
        self.cash_history = []
        
        logger.info(f"TimeframeAgnosticBacktestEngine initialized for {timeframe}")
        logger.info(f"Trading hours: {self.trading_hours}")
        logger.info(f"Execution parameters: {self.execution_params}")
    
    def get_trading_hours(self, timeframe: str) -> Dict[str, Any]:
        """
        Get trading hours and market session information for the timeframe.
        
        Args:
            timeframe: Target timeframe
            
        Returns:
            Dictionary with trading hours configuration
        """
        # Base trading hours (US market)
        base_config = {
            'market_open': time(9, 30),   # 9:30 AM
            'market_close': time(16, 0),  # 4:00 PM
            'timezone': 'US/Eastern',
            'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        }
        
        # Timeframe-specific adjustments
        timeframe_configs = {
            '1day': {
                **base_config,
                'session_type': 'daily',
                'execution_time': 'market_close',  # Execute at market close
                'allow_partial_days': False
            },
            '1hour': {
                **base_config,
                'session_type': 'intraday',
                'execution_time': 'any_hour',  # Execute at any trading hour
                'allow_partial_days': True,
                'min_session_hours': 6.5  # Full trading day is 6.5 hours
            },
            '15min': {
                **base_config,
                'session_type': 'intraday',
                'execution_time': 'any_time',
                'allow_partial_days': True,
                'min_session_minutes': 15
            },
            '5min': {
                **base_config,
                'session_type': 'intraday',
                'execution_time': 'any_time',
                'allow_partial_days': True,
                'min_session_minutes': 5
            },
            '1min': {
                **base_config,
                'session_type': 'intraday',
                'execution_time': 'any_time',
                'allow_partial_days': True,
                'min_session_minutes': 1
            }
        }
        
        return timeframe_configs.get(timeframe, timeframe_configs['1day'])
    
    def get_execution_parameters(self, timeframe: str) -> Dict[str, Any]:
        """
        Get execution parameters optimized for the timeframe.
        
        Different timeframes have different liquidity characteristics and execution costs.
        
        Args:
            timeframe: Target timeframe
            
        Returns:
            Dictionary of execution parameters
        """
        parameters = {
            '1day': {
                # Daily timeframe - lower costs, better execution
                'commission_rate': 0.001,      # 0.1% commission
                'slippage_rate': 0.0005,       # 0.05% slippage
                'market_impact': 0.0002,       # 0.02% market impact
                'max_position_size': 0.1,      # 10% of portfolio per position
                'execution_delay_bars': 1,      # Execute next day
                'bid_ask_spread': 0.0001,      # 0.01% spread
                'liquidity_requirement': 1000000  # Min volume for execution
            },
            '1hour': {
                # Hourly timeframe - moderate costs
                'commission_rate': 0.0015,     # 0.15% commission
                'slippage_rate': 0.001,        # 0.1% slippage
                'market_impact': 0.0005,       # 0.05% market impact
                'max_position_size': 0.08,     # 8% of portfolio per position
                'execution_delay_bars': 0,      # Execute same bar
                'bid_ask_spread': 0.0002,      # 0.02% spread
                'liquidity_requirement': 500000
            },
            '15min': {
                # 15-minute timeframe - higher costs
                'commission_rate': 0.002,      # 0.2% commission
                'slippage_rate': 0.0015,       # 0.15% slippage
                'market_impact': 0.001,        # 0.1% market impact
                'max_position_size': 0.06,     # 6% of portfolio per position
                'execution_delay_bars': 0,
                'bid_ask_spread': 0.0003,      # 0.03% spread
                'liquidity_requirement': 250000
            },
            '5min': {
                # 5-minute timeframe - high costs
                'commission_rate': 0.0025,     # 0.25% commission
                'slippage_rate': 0.002,        # 0.2% slippage
                'market_impact': 0.0015,       # 0.15% market impact
                'max_position_size': 0.05,     # 5% of portfolio per position
                'execution_delay_bars': 0,
                'bid_ask_spread': 0.0005,      # 0.05% spread
                'liquidity_requirement': 100000
            },
            '1min': {
                # 1-minute timeframe - very high costs
                'commission_rate': 0.003,      # 0.3% commission
                'slippage_rate': 0.0025,       # 0.25% slippage
                'market_impact': 0.002,        # 0.2% market impact
                'max_position_size': 0.03,     # 3% of portfolio per position
                'execution_delay_bars': 0,
                'bid_ask_spread': 0.001,       # 0.1% spread
                'liquidity_requirement': 50000
            }
        }
        
        return parameters.get(timeframe, parameters['1day'])
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """
        Check if the market is open at the given timestamp.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if market is open
        """
        # For daily timeframe, always consider market open (end-of-day execution)
        if self.timeframe == '1day':
            return timestamp.weekday() < 5  # Monday=0, Friday=4
        
        # For intraday timeframes, check actual trading hours
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        current_time = timestamp.time()
        return (self.trading_hours['market_open'] <= current_time <= self.trading_hours['market_close'])
    
    def calculate_execution_price(self, signal_price: float, signal_type: str, 
                                volume: float = 0) -> Tuple[float, Dict[str, float]]:
        """
        Calculate realistic execution price including slippage and costs.
        
        Args:
            signal_price: Original signal price
            signal_type: 'BUY' or 'SELL'
            volume: Trading volume for market impact calculation
            
        Returns:
            Tuple of (execution_price, cost_breakdown)
        """
        params = self.execution_params
        
        # Base price adjustments
        bid_ask_spread = signal_price * params['bid_ask_spread']
        slippage = signal_price * params['slippage_rate']
        
        # Market impact based on volume (simplified)
        market_impact = 0.0
        if volume > 0:
            impact_factor = min(volume / params['liquidity_requirement'], 1.0)
            market_impact = signal_price * params['market_impact'] * impact_factor
        
        # Calculate execution price
        if signal_type == 'BUY':
            # For buys, we pay the ask price plus slippage and market impact
            execution_price = signal_price + (bid_ask_spread / 2) + slippage + market_impact
        else:  # SELL
            # For sells, we receive the bid price minus slippage and market impact
            execution_price = signal_price - (bid_ask_spread / 2) - slippage - market_impact
        
        cost_breakdown = {
            'signal_price': signal_price,
            'bid_ask_spread': bid_ask_spread,
            'slippage': slippage,
            'market_impact': market_impact,
            'total_cost': abs(execution_price - signal_price)
        }
        
        return execution_price, cost_breakdown
    
    def execute_signal(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Execute a trading signal with realistic constraints.
        
        Args:
            signal: Signal dictionary
            market_data: Market data for execution
            
        Returns:
            Trade execution record or None if not executed
        """
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        confidence = signal.get('confidence', 0.5)
        signal_price = signal.get('close', 0.0)
        volume = signal.get('volume', 0)
        
        # Skip HOLD signals
        if signal_type == 'HOLD':
            return None
        
        # Check if we can execute (market open, sufficient confidence, etc.)
        signal_date = pd.to_datetime(signal['date'])
        if not self.is_market_open(signal_date):
            logger.debug(f"Market closed at {signal_date}, skipping signal for {symbol}")
            return None
        
        # Minimum confidence threshold based on timeframe
        min_confidence = {
            '1day': 0.6, '1hour': 0.5, '15min': 0.4, '5min': 0.3, '1min': 0.2
        }.get(self.timeframe, 0.5)
        
        if confidence < min_confidence:
            logger.debug(f"Confidence {confidence} below threshold {min_confidence} for {symbol}")
            return None
        
        # Calculate position size based on confidence and risk parameters
        portfolio_value = self.get_portfolio_value(market_data)
        max_position_value = portfolio_value * self.execution_params['max_position_size']
        
        # Adjust position size by confidence
        position_value = max_position_value * confidence
        
        if signal_price <= 0:
            logger.warning(f"Invalid signal price {signal_price} for {symbol}")
            return None
        
        # Calculate execution price and costs
        execution_price, cost_breakdown = self.calculate_execution_price(signal_price, signal_type, volume)
        
        # Calculate shares to trade
        if signal_type == 'BUY':
            max_shares = int(position_value / execution_price)
            shares_to_buy = max_shares
            
            # Check if we have enough cash
            total_cost = shares_to_buy * execution_price
            commission = total_cost * self.execution_params['commission_rate']
            
            if total_cost + commission > self.current_capital:
                # Reduce position size to fit available capital
                available_for_position = self.current_capital * 0.95  # Keep 5% cash buffer
                shares_to_buy = int(available_for_position / (execution_price * (1 + self.execution_params['commission_rate'])))
            
            if shares_to_buy <= 0:
                logger.debug(f"Insufficient capital for {symbol} BUY signal")
                return None
            
            # Execute BUY
            total_cost = shares_to_buy * execution_price
            commission = total_cost * self.execution_params['commission_rate']
            
            # Update portfolio
            if symbol in self.positions:
                # Add to existing position
                old_quantity = self.positions[symbol]['quantity']
                old_avg_price = self.positions[symbol]['avg_price']
                new_avg_price = ((old_quantity * old_avg_price) + total_cost) / (old_quantity + shares_to_buy)
                
                self.positions[symbol]['quantity'] += shares_to_buy
                self.positions[symbol]['avg_price'] = new_avg_price
            else:
                # New position
                self.positions[symbol] = {
                    'quantity': shares_to_buy,
                    'avg_price': execution_price,
                    'entry_time': signal_date
                }
            
            self.current_capital -= (total_cost + commission)
            
            trade_record = {
                'symbol': symbol,
                'action': 'BUY',
                'quantity': shares_to_buy,
                'price': execution_price,
                'total_value': total_cost,
                'commission': commission,
                'timestamp': signal_date,
                'signal_confidence': confidence,
                'cost_breakdown': cost_breakdown,
                'timeframe': self.timeframe
            }
            
        else:  # SELL
            if symbol not in self.positions or self.positions[symbol]['quantity'] <= 0:
                logger.debug(f"No position to sell for {symbol}")
                return None
            
            # Sell entire position for simplicity (could be partial)
            shares_to_sell = self.positions[symbol]['quantity']
            
            # Execute SELL
            total_proceeds = shares_to_sell * execution_price
            commission = total_proceeds * self.execution_params['commission_rate']
            net_proceeds = total_proceeds - commission
            
            # Calculate P&L
            avg_purchase_price = self.positions[symbol]['avg_price']
            pnl = (execution_price - avg_purchase_price) * shares_to_sell - commission
            
            # Update portfolio
            self.current_capital += net_proceeds
            del self.positions[symbol]
            
            trade_record = {
                'symbol': symbol,
                'action': 'SELL',
                'quantity': shares_to_sell,
                'price': execution_price,
                'total_value': total_proceeds,
                'commission': commission,
                'net_proceeds': net_proceeds,
                'pnl': pnl,
                'timestamp': signal_date,
                'signal_confidence': confidence,
                'cost_breakdown': cost_breakdown,
                'timeframe': self.timeframe
            }
        
        # Record the trade
        self.trades.append(trade_record)
        
        logger.debug(f"Executed {signal_type} for {symbol}: {trade_record['quantity']} shares at ${execution_price:.2f}")
        return trade_record
    
    def get_portfolio_value(self, market_data: pd.DataFrame) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            market_data: Current market data for position valuation
            
        Returns:
            Total portfolio value
        """
        total_value = self.current_capital
        
        # Add value of current positions
        for symbol, position in self.positions.items():
            # Get current price from market data
            symbol_data = market_data[market_data['symbol'] == symbol] if 'symbol' in market_data.columns else market_data
            
            if not symbol_data.empty:
                current_price = symbol_data['Close'].iloc[-1] if 'Close' in symbol_data.columns else position['avg_price']
                position_value = position['quantity'] * current_price
                total_value += position_value
        
        return total_value
    
    def run_backtest(self, signals: List[Dict[str, Any]], market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest for the given signals and market data.
        
        Args:
            signals: List of trading signals
            market_data: Historical market data
            
        Returns:
            Backtest results and performance metrics
        """
        logger.info(f"Starting backtest for {len(signals)} signals on {self.timeframe} timeframe")
        
        # Reset portfolio state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_value_history = []
        self.cash_history = []
        
        # Sort signals by date
        signals_sorted = sorted(signals, key=lambda x: x['date'])
        
        # Execute signals
        executed_trades = 0
        for signal in signals_sorted:
            trade_record = self.execute_signal(signal, market_data)
            if trade_record:
                executed_trades += 1
            
            # Record portfolio value
            portfolio_value = self.get_portfolio_value(market_data)
            self.portfolio_value_history.append({
                'date': signal['date'],
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'positions_value': portfolio_value - self.current_capital
            })
        
        # Calculate final portfolio value
        final_portfolio_value = self.get_portfolio_value(market_data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(final_portfolio_value)
        
        results = {
            'timeframe': self.timeframe,
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'total_return': (final_portfolio_value - self.initial_capital) / self.initial_capital,
            'total_signals': len(signals),
            'executed_trades': executed_trades,
            'execution_rate': executed_trades / len(signals) if signals else 0,
            'trades': self.trades,
            'portfolio_history': self.portfolio_value_history,
            'final_positions': self.positions.copy(),
            'performance_metrics': performance_metrics,
            'execution_parameters': self.execution_params,
            'trading_hours': self.trading_hours
        }
        
        logger.info(f"Backtest completed: {executed_trades}/{len(signals)} signals executed")
        logger.info(f"Final portfolio value: ${final_portfolio_value:,.2f} (Return: {results['total_return']:.2%})")
        
        return results
    
    def _calculate_performance_metrics(self, final_value: float) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {'total_return': 0.0, 'trade_count': 0}
        
        # Basic metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Trade-based metrics
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / len(self.trades) if self.trades else 0
        
        avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Portfolio value metrics
        if self.portfolio_value_history:
            values = [h['portfolio_value'] for h in self.portfolio_value_history]
            returns = np.diff(values) / values[:-1]
            
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (total_return / volatility) if volatility > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        metrics = {
            'total_return': round(total_return, 4),
            'annualized_return': round(total_return, 4),  # Simplified
            'volatility': round(volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'max_drawdown': round(max_drawdown, 4),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 4),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'total_trades': len(self.trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades)
        }
        
        return metrics

# Factory function for backward compatibility
def create_backtest_engine(timeframe: str = '1day', initial_capital: float = 100000.0) -> TimeframeAgnosticBacktestEngine:
    """Create a timeframe-agnostic backtest engine."""
    return TimeframeAgnosticBacktestEngine(timeframe, initial_capital)

# Example usage and testing
if __name__ == "__main__":
    # Test the backtest engine
    print("=== Testing Timeframe-Agnostic Backtest Engine ===")
    
    # Test with daily timeframe (backward compatibility)
    daily_engine = create_backtest_engine('1day', 100000)
    print(f"Daily execution params: {daily_engine.execution_params}")
    
    # Test with hourly timeframe
    hourly_engine = create_backtest_engine('1hour', 100000)
    print(f"Hourly execution params: {hourly_engine.execution_params}")
    
    # Create sample signals for testing
    sample_signals = [
        {
            'symbol': 'AAPL',
            'date': '2024-08-15',
            'signal_type': 'BUY',
            'confidence': 0.8,
            'close': 150.0,
            'volume': 1000000
        },
        {
            'symbol': 'AAPL',
            'date': '2024-08-20',
            'signal_type': 'SELL',
            'confidence': 0.7,
            'close': 155.0,
            'volume': 1200000
        }
    ]
    
    # Create sample market data
    dates = pd.date_range(start='2024-08-01', end='2024-08-31', freq='D')
    sample_market_data = pd.DataFrame({
        'Date': dates,
        'symbol': 'AAPL',
        'Open': np.random.uniform(145, 155, len(dates)),
        'High': np.random.uniform(150, 160, len(dates)),
        'Low': np.random.uniform(140, 150, len(dates)),
        'Close': np.random.uniform(145, 155, len(dates)),
        'Volume': np.random.randint(500000, 2000000, len(dates))
    })
    
    # Test backtest execution
    print("\n=== Testing Backtest Execution ===")
    results = daily_engine.run_backtest(sample_signals, sample_market_data)
    print(f"Backtest results: Total return: {results['total_return']:.2%}")
    print(f"Executed trades: {results['executed_trades']}/{results['total_signals']}")
    
    if results['trades']:
        print(f"Sample trade: {results['trades'][0]}")
    
    print(f"Performance metrics: {results['performance_metrics']}")
