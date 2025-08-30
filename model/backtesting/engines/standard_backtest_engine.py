"""
Standard Backtest Engine

Standard implementation for backtesting with common features and optimizations.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
from .base_backtest_engine import BaseBacktestEngine
from ..backtest_config import BacktestConfig
from ..trade_record import TradeRecord, PositionRecord, PortfolioRecord
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class StandardBacktestEngine(BaseBacktestEngine):
    """Standard backtesting engine with enhanced features"""
    
    def __init__(self, name: str = "standard_backtest", config: BacktestConfig = None):
        if config is None:
            config = BacktestConfig()
        
        super().__init__(name, config)
        
        # Standard engine specific features
        self.rebalance_frequency = config.rebalance_frequency if config else 'daily'
        self.enable_stop_loss = True  # Default to True
        self.enable_take_profit = True  # Default to True
        self.enable_trailing_stop = False  # Default to False
        
        # Position tracking
        self.position_limits = {}
        self.stop_loss_levels = {}
        self.take_profit_levels = {}
        self.trailing_stops = {}
        
        # Performance tracking
        self.daily_returns = []
        self.monthly_returns = []
        self.rolling_metrics = {}
        
        logger.info(f"Standard backtest engine initialized: {name}")
    
    def get_name(self) -> str:
        """Get the name of this backtest engine"""
        return self.name
    
    def get_supported_execution_types(self) -> List[str]:
        """Get supported execution types for standard engine"""
        return ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
    
    def set_position_limit(self, symbol: str, max_position_value: float):
        """Set maximum position value for a symbol"""
        self.position_limits[symbol] = max_position_value
        logger.info(f"Position limit set for {symbol}: ${max_position_value:,.2f}")
    
    def set_stop_loss(self, symbol: str, stop_loss_pct: float):
        """Set stop loss percentage for a symbol"""
        self.stop_loss_levels[symbol] = stop_loss_pct
        logger.info(f"Stop loss set for {symbol}: {stop_loss_pct:.2%}")
    
    def set_take_profit(self, symbol: str, take_profit_pct: float):
        """Set take profit percentage for a symbol"""
        self.take_profit_levels[symbol] = take_profit_pct
        logger.info(f"Take profit set for {symbol}: {take_profit_pct:.2%}")
    
    def set_trailing_stop(self, symbol: str, trailing_pct: float):
        """Set trailing stop percentage for a symbol"""
        self.trailing_stops[symbol] = trailing_pct
        logger.info(f"Trailing stop set for {symbol}: {trailing_pct:.2%}")
    
    def _run_backtest_loop(self, signals: List[Dict[str, Any]], 
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the main backtest loop with standard engine features"""
        
        results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {},
            'risk_metrics': {},
            'daily_returns': [],
            'monthly_returns': []
        }
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda x: x.get('timestamp', datetime.min))
        
        # Process signals chronologically
        for signal in sorted_signals:
            try:
                # Check for stop loss and take profit triggers
                self._check_exit_signals(market_data)
                
                # Generate trade from signal
                trade = self._generate_trade_from_signal(signal, market_data)
                
                if trade:
                    # Apply position sizing rules
                    trade = self._apply_position_sizing(trade)
                    
                    if trade and trade.quantity > 0:
                        # Validate trade with risk manager
                        if self.risk_manager.validate_trade(trade, self.current_portfolio, self.config):
                            # Execute trade
                            executed_trade = self.execution_engine.execute_trade(trade, market_data)
                            
                            if executed_trade:
                                # Update portfolio
                                self._update_portfolio(executed_trade)
                                
                                # Set stop loss and take profit levels
                                self._set_exit_levels(executed_trade, market_data)
                                
                                # Record trade
                                self.trades_history.append(executed_trade)
                                results['trades'].append(executed_trade)
                                
                                # Update performance analyzer
                                self.performance_analyzer.add_trade(executed_trade)
                                
                                logger.info(f"Trade executed: {executed_trade.symbol} {executed_trade.trade_type.value} {executed_trade.quantity}")
                        else:
                            logger.warning(f"Trade rejected by risk manager: {signal}")
                
                # Update portfolio snapshot
                self._update_portfolio_snapshot()
                results['portfolio_history'].append(self.current_portfolio)
                
                # Calculate daily returns
                self._calculate_daily_returns()
                
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
                continue
        
        # Calculate final metrics
        results['performance_metrics'] = self.performance_analyzer.generate_performance_report(self.config)
        results['risk_metrics'] = self.risk_manager.get_risk_report(self.current_portfolio, self.config)
        results['daily_returns'] = self.daily_returns
        results['monthly_returns'] = self._calculate_monthly_returns()
        
        return results
    
    def _apply_position_sizing(self, trade: TradeRecord) -> Optional[TradeRecord]:
        """Apply position sizing rules to a trade"""
        
        if not trade or trade.quantity <= 0:
            return trade
        
        symbol = trade.symbol
        current_price = trade.price
        desired_quantity = trade.quantity
        
        # Check position limits
        if symbol in self.position_limits:
            max_position_value = self.position_limits[symbol]
            max_quantity = max_position_value / current_price
            desired_quantity = min(desired_quantity, max_quantity)
        
        # Check portfolio concentration limits
        portfolio_value = self.current_portfolio.total_value if self.current_portfolio else 0
        if portfolio_value > 0:
            max_concentration = self.config.max_position_concentration if hasattr(self.config, 'max_position_concentration') else 0.1
            max_position_value = portfolio_value * max_concentration
            max_quantity = max_position_value / current_price
            desired_quantity = min(desired_quantity, max_quantity)
        
        # Check minimum trade size
        if desired_quantity * current_price < self.config.min_trade_size:
            return None
        
        # Update trade quantity
        trade.quantity = desired_quantity
        
        return trade
    
    def _check_exit_signals(self, market_data: Dict[str, Any]):
        """Check for stop loss and take profit triggers"""
        
        if not self.current_portfolio:
            return
        
        for position in self.current_portfolio.positions:
            symbol = position.symbol
            current_price = self._get_current_price(symbol, market_data)
            
            if not current_price:
                continue
            
            # Check stop loss
            if symbol in self.stop_loss_levels and self.enable_stop_loss:
                stop_loss_pct = self.stop_loss_levels[symbol]
                stop_loss_price = position.avg_price * (1 - stop_loss_pct)
                
                if current_price <= stop_loss_price:
                    self._create_exit_trade(position, current_price, 'STOP_LOSS')
            
            # Check take profit
            if symbol in self.take_profit_levels and self.enable_take_profit:
                take_profit_pct = self.take_profit_levels[symbol]
                take_profit_price = position.avg_price * (1 + take_profit_pct)
                
                if current_price >= take_profit_price:
                    self._create_exit_trade(position, current_price, 'TAKE_PROFIT')
            
            # Check trailing stop
            if symbol in self.trailing_stops and self.enable_trailing_stop:
                trailing_pct = self.trailing_stops[symbol]
                # This is a simplified trailing stop implementation
                # In practice, you'd track the highest price since entry
                highest_price = position.avg_price * 1.05  # Simplified
                trailing_stop_price = highest_price * (1 - trailing_pct)
                
                if current_price <= trailing_stop_price:
                    self._create_exit_trade(position, current_price, 'TRAILING_STOP')
    
    def _create_exit_trade(self, position: PositionRecord, current_price: float, exit_reason: str):
        """Create an exit trade for a position"""
        
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol=position.symbol,
            trade_type='SELL',
            quantity=position.quantity,
            price=current_price,
            commission=0.0,
            realized_pnl=0.0,
            status='PENDING'
        )
        
        # Execute the exit trade
        if self.risk_manager.validate_trade(trade, self.current_portfolio, self.config):
            executed_trade = self.execution_engine.execute_trade(trade, {})
            
            if executed_trade:
                self._update_portfolio(executed_trade)
                self.trades_history.append(executed_trade)
                self.performance_analyzer.add_trade(executed_trade)
                
                logger.info(f"Exit trade executed: {position.symbol} {exit_reason} at ${current_price:.2f}")
    
    def _set_exit_levels(self, trade: TradeRecord, market_data: Dict[str, Any]):
        """Set stop loss and take profit levels for a new position"""
        
        if trade.trade_type != 'BUY':
            return
        
        symbol = trade.symbol
        entry_price = trade.price
        
        # Set default stop loss if not already set
        if symbol not in self.stop_loss_levels and self.enable_stop_loss:
            default_stop_loss = 0.05  # 5% default stop loss
            self.stop_loss_levels[symbol] = default_stop_loss
        
        # Set default take profit if not already set
        if symbol not in self.take_profit_levels and self.enable_take_profit:
            default_take_profit = 0.10  # 10% default take profit
            self.take_profit_levels[symbol] = default_take_profit
    
    def _calculate_daily_returns(self):
        """Calculate daily returns for performance tracking"""
        
        if len(self.equity_curve) < 2:
            return
        
        current_value = self.equity_curve[-1]['total_value']
        previous_value = self.equity_curve[-2]['total_value']
        
        if previous_value > 0:
            daily_return = (current_value - previous_value) / previous_value
            self.daily_returns.append({
                'date': self.equity_curve[-1]['timestamp'].date(),
                'return': daily_return,
                'portfolio_value': current_value
            })
    
    def _calculate_monthly_returns(self) -> List[Dict[str, Any]]:
        """Calculate monthly returns"""
        
        if not self.daily_returns:
            return []
        
        monthly_returns = {}
        
        for daily_return in self.daily_returns:
            month_key = daily_return['date'].strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            monthly_returns[month_key].append(daily_return['return'])
        
        # Calculate cumulative monthly returns
        result = []
        for month, returns in monthly_returns.items():
            cumulative_return = (1 + pd.Series(returns)).prod() - 1
            result.append({
                'month': month,
                'return': cumulative_return,
                'days': len(returns)
            })
        
        return result
    
    def get_rolling_metrics(self, window: int = 30) -> Dict[str, Any]:
        """Calculate rolling performance metrics"""
        
        if len(self.daily_returns) < window:
            return {}
        
        returns_series = pd.Series([r['return'] for r in self.daily_returns])
        rolling_returns = returns_series.rolling(window=window)
        
        metrics = {
            'rolling_sharpe': (rolling_returns.mean() / rolling_returns.std() * np.sqrt(252)).iloc[-1],
            'rolling_volatility': rolling_returns.std().iloc[-1] * np.sqrt(252),
            'rolling_max_drawdown': self._calculate_rolling_drawdown(window),
            'rolling_win_rate': self._calculate_rolling_win_rate(window)
        }
        
        return metrics
    
    def _calculate_rolling_drawdown(self, window: int) -> float:
        """Calculate rolling maximum drawdown"""
        
        if len(self.daily_returns) < window:
            return 0.0
        
        recent_returns = self.daily_returns[-window:]
        portfolio_values = [r['portfolio_value'] for r in recent_returns]
        
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_rolling_win_rate(self, window: int) -> float:
        """Calculate rolling win rate"""
        
        if len(self.trades_history) < window:
            return 0.0
        
        recent_trades = self.trades_history[-window:]
        winning_trades = sum(1 for trade in recent_trades if trade.realized_pnl > 0)
        
        return winning_trades / len(recent_trades) if recent_trades else 0.0
    
    def get_standard_metrics(self) -> Dict[str, Any]:
        """Get standard performance metrics"""
        
        if not self.current_portfolio:
            return {}
        
        # Basic metrics
        total_return = (self.current_portfolio.total_value / self.config.initial_capital - 1)
        
        # Risk metrics
        volatility = np.std([r['return'] for r in self.daily_returns]) * np.sqrt(252) if self.daily_returns else 0
        max_drawdown = self._calculate_max_drawdown()
        
        # Trade metrics
        total_trades = len(self.trades_history)
        winning_trades = sum(1 for trade in self.trades_history if trade.realized_pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Sharpe ratio (simplified)
        avg_return = np.mean([r['return'] for r in self.daily_returns]) * 252 if self.daily_returns else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_portfolio_value': self.current_portfolio.total_value
        }
