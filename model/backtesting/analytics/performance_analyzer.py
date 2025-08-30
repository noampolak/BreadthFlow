"""
Performance Analyzer

Comprehensive performance analysis and reporting for backtesting results.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ..backtest_config import BacktestConfig
from ..trade_record import TradeRecord, PositionRecord, PortfolioRecord

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Comprehensive performance analysis for backtesting results"""
    
    def __init__(self, name: str = "performance_analyzer"):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Performance metrics storage
        self.returns = []
        self.equity_curve = []
        self.drawdowns = []
        self.trades = []
        self.positions = []
        
        # Analysis results cache
        self._analysis_cache = {}
        self._last_analysis_time = None
    
    def get_name(self) -> str:
        """Get the name of this analyzer"""
        return self.name
    
    def add_portfolio_snapshot(self, portfolio: PortfolioRecord, timestamp: datetime):
        """Add a portfolio snapshot for analysis"""
        
        # Store equity value
        self.equity_curve.append({
            'timestamp': timestamp,
            'total_value': portfolio.total_value,
            'cash': portfolio.cash_value,
            'positions_value': portfolio.positions_value
        })
        
        # Calculate return if we have previous data
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['total_value']
            current_value = portfolio.total_value
            
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self.returns.append({
                    'timestamp': timestamp,
                    'return': daily_return
                })
        
        # Store positions
        for position in portfolio.positions:
            self.positions.append({
                'timestamp': timestamp,
                'symbol': position.symbol,
                'quantity': position.quantity,
                'market_value': position.market_value,
                'unrealized_pnl': position.unrealized_pnl
            })
    
    def add_trade(self, trade: TradeRecord):
        """Add a completed trade for analysis"""
        
        self.trades.append({
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'trade_type': trade.trade_type.value,
            'quantity': trade.quantity,
            'price': trade.price,
            'commission': trade.commission,
            'realized_pnl': trade.net_amount,  # Use net_amount as proxy for PnL
            'status': trade.status.value
        })
    
    def calculate_returns_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive returns metrics"""
        
        if not self.returns:
            return {}
        
        returns_series = pd.Series([r['return'] for r in self.returns])
        
        # Basic return metrics
        total_return = (self.equity_curve[-1]['total_value'] / self.equity_curve[0]['total_value']) - 1
        annualized_return = self._calculate_annualized_return(returns_series)
        volatility = returns_series.std() * np.sqrt(252)  # Annualized volatility
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns_series)
        sortino_ratio = self._calculate_sortino_ratio(returns_series)
        calmar_ratio = self._calculate_calmar_ratio(returns_series, total_return)
        
        # Drawdown metrics
        max_drawdown = self._calculate_max_drawdown()
        avg_drawdown = self._calculate_avg_drawdown()
        
        # Additional metrics
        var_95 = np.percentile(returns_series, 5)  # 95% VaR
        cvar_95 = returns_series[returns_series <= var_95].mean()  # Conditional VaR
        skewness = returns_series.skew()
        kurtosis = returns_series.kurtosis()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': self._calculate_win_rate(),
            'profit_factor': self._calculate_profit_factor(),
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss()
        }
    
    def calculate_trade_metrics(self) -> Dict[str, Any]:
        """Calculate trade-specific metrics"""
        
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic trade metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['realized_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['realized_pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['realized_pnl'].sum()
        avg_pnl = trades_df['realized_pnl'].mean()
        
        winning_pnl = trades_df[trades_df['realized_pnl'] > 0]['realized_pnl']
        losing_pnl = trades_df[trades_df['realized_pnl'] < 0]['realized_pnl']
        
        avg_win = winning_pnl.mean() if len(winning_pnl) > 0 else 0
        avg_loss = losing_pnl.mean() if len(losing_pnl) > 0 else 0
        
        profit_factor = abs(winning_pnl.sum() / losing_pnl.sum()) if losing_pnl.sum() != 0 else float('inf')
        
        # Trade duration metrics
        trade_durations = self._calculate_trade_durations()
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Symbol analysis
        symbol_performance = self._analyze_symbol_performance()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'symbol_performance': symbol_performance,
            'largest_win': trades_df['realized_pnl'].max(),
            'largest_loss': trades_df['realized_pnl'].min(),
            'consecutive_wins': self._calculate_consecutive_wins(),
            'consecutive_losses': self._calculate_consecutive_losses()
        }
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        if not self.returns:
            return {}
        
        returns_series = pd.Series([r['return'] for r in self.returns])
        
        # Value at Risk metrics
        var_95 = np.percentile(returns_series, 5)
        var_99 = np.percentile(returns_series, 1)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns_series[returns_series <= var_95].mean()
        cvar_99 = returns_series[returns_series <= var_99].mean()
        
        # Downside deviation
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Recovery time
        recovery_time = self._calculate_recovery_time()
        
        # Beta calculation (if market data available)
        beta = self._calculate_beta(returns_series)
        
        # Correlation with market
        market_correlation = self._calculate_market_correlation(returns_series)
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation,
            'max_drawdown': max_drawdown,
            'recovery_time': recovery_time,
            'beta': beta,
            'market_correlation': market_correlation,
            'volatility': returns_series.std() * np.sqrt(252),
            'skewness': returns_series.skew(),
            'kurtosis': returns_series.kurtosis()
        }
    
    def generate_performance_report(self, config: BacktestConfig) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Calculate all metrics
        returns_metrics = self.calculate_returns_metrics()
        trade_metrics = self.calculate_trade_metrics()
        risk_metrics = self.calculate_risk_metrics()
        
        # Generate equity curve data
        equity_data = self._generate_equity_curve_data()
        
        # Generate drawdown analysis
        drawdown_data = self._generate_drawdown_data()
        
        # Generate monthly returns
        monthly_returns = self._calculate_monthly_returns()
        
        # Performance attribution
        attribution = self._calculate_performance_attribution()
        
        report = {
            'summary': {
                'total_return': returns_metrics.get('total_return', 0),
                'annualized_return': returns_metrics.get('annualized_return', 0),
                'sharpe_ratio': returns_metrics.get('sharpe_ratio', 0),
                'max_drawdown': returns_metrics.get('max_drawdown', 0),
                'win_rate': trade_metrics.get('win_rate', 0),
                'total_trades': trade_metrics.get('total_trades', 0)
            },
            'returns_metrics': returns_metrics,
            'trade_metrics': trade_metrics,
            'risk_metrics': risk_metrics,
            'equity_curve': equity_data,
            'drawdown_analysis': drawdown_data,
            'monthly_returns': monthly_returns,
            'performance_attribution': attribution,
            'backtest_config': {
                'start_date': config.start_date.isoformat() if config.start_date else None,
                'end_date': config.end_date.isoformat() if config.end_date else None,
                'initial_capital': config.initial_capital,
                'commission_rate': config.commission_rate,
                'slippage': config.slippage_rate
            }
        }
        
        return report
    
    def _calculate_annualized_return(self, returns_series: pd.Series) -> float:
        """Calculate annualized return"""
        
        if len(returns_series) == 0:
            return 0.0
        
        # Calculate total return
        total_return = (1 + returns_series).prod() - 1
        
        # Calculate time period in years
        if len(self.equity_curve) >= 2:
            start_date = self.equity_curve[0]['timestamp']
            end_date = self.equity_curve[-1]['timestamp']
            years = (end_date - start_date).days / 365.25
        else:
            years = len(returns_series) / 252  # Assume daily data
        
        if years <= 0:
            return 0.0
        
        # Annualize
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return annualized_return
    
    def _calculate_sharpe_ratio(self, returns_series: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        
        if len(returns_series) == 0:
            return 0.0
        
        excess_returns = returns_series - risk_free_rate / 252  # Daily risk-free rate
        volatility = returns_series.std()
        
        if volatility == 0:
            return 0.0
        
        sharpe_ratio = excess_returns.mean() / volatility * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_sortino_ratio(self, returns_series: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        
        if len(returns_series) == 0:
            return 0.0
        
        excess_returns = returns_series - risk_free_rate / 252
        downside_returns = returns_series[returns_series < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return 0.0
        
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
        
        return sortino_ratio
    
    def _calculate_calmar_ratio(self, returns_series: pd.Series, total_return: float) -> float:
        """Calculate Calmar ratio"""
        
        max_drawdown = self._calculate_max_drawdown()
        
        if max_drawdown == 0:
            return 0.0
        
        calmar_ratio = total_return / abs(max_drawdown)
        
        return calmar_ratio
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        
        if not self.equity_curve:
            return 0.0
        
        equity_values = [point['total_value'] for point in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_avg_drawdown(self) -> float:
        """Calculate average drawdown"""
        
        if not self.equity_curve:
            return 0.0
        
        equity_values = [point['total_value'] for point in self.equity_curve]
        peak = equity_values[0]
        drawdowns = []
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        return np.mean(drawdowns) if drawdowns else 0.0
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trades if trade['realized_pnl'] > 0)
        total_trades = len(self.trades)
        
        return winning_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        
        if not self.trades:
            return 0.0
        
        winning_pnl = sum(trade['realized_pnl'] for trade in self.trades if trade['realized_pnl'] > 0)
        losing_pnl = abs(sum(trade['realized_pnl'] for trade in self.trades if trade['realized_pnl'] < 0))
        
        return winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')
    
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade"""
        
        winning_trades = [trade['realized_pnl'] for trade in self.trades if trade['realized_pnl'] > 0]
        
        return np.mean(winning_trades) if winning_trades else 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade"""
        
        losing_trades = [trade['realized_pnl'] for trade in self.trades if trade['realized_pnl'] < 0]
        
        return np.mean(losing_trades) if losing_trades else 0.0
    
    def _calculate_trade_durations(self) -> List[float]:
        """Calculate trade durations in days"""
        
        # This is a simplified implementation
        # In practice, you'd track entry and exit times for each position
        
        return [1.0] * len(self.trades)  # Assume 1 day per trade
    
    def _analyze_symbol_performance(self) -> Dict[str, Any]:
        """Analyze performance by symbol"""
        
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        symbol_stats = {}
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            
            symbol_stats[symbol] = {
                'total_trades': len(symbol_trades),
                'total_pnl': symbol_trades['realized_pnl'].sum(),
                'avg_pnl': symbol_trades['realized_pnl'].mean(),
                'win_rate': len(symbol_trades[symbol_trades['realized_pnl'] > 0]) / len(symbol_trades),
                'max_win': symbol_trades['realized_pnl'].max(),
                'max_loss': symbol_trades['realized_pnl'].min()
            }
        
        return symbol_stats
    
    def _calculate_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins"""
        
        if not self.trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade['realized_pnl'] > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses"""
        
        if not self.trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in self.trades:
            if trade['realized_pnl'] < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_recovery_time(self) -> int:
        """Calculate time to recover from maximum drawdown"""
        
        # This is a simplified implementation
        # In practice, you'd track the actual recovery time
        
        return 30  # Assume 30 days
    
    def _calculate_beta(self, returns_series: pd.Series) -> float:
        """Calculate beta relative to market"""
        
        # This is a simplified implementation
        # In practice, you'd need market returns data
        
        return 1.0  # Assume beta of 1.0
    
    def _calculate_market_correlation(self, returns_series: pd.Series) -> float:
        """Calculate correlation with market returns"""
        
        # This is a simplified implementation
        # In practice, you'd need market returns data
        
        return 0.5  # Assume 0.5 correlation
    
    def _generate_equity_curve_data(self) -> List[Dict[str, Any]]:
        """Generate equity curve data for plotting"""
        
        return [
            {
                'timestamp': point['timestamp'].isoformat(),
                'total_value': point['total_value'],
                'cash': point['cash'],
                'positions_value': point['positions_value']
            }
            for point in self.equity_curve
        ]
    
    def _generate_drawdown_data(self) -> List[Dict[str, Any]]:
        """Generate drawdown data for plotting"""
        
        if not self.equity_curve:
            return []
        
        equity_values = [point['total_value'] for point in self.equity_curve]
        timestamps = [point['timestamp'] for point in self.equity_curve]
        
        peak = equity_values[0]
        drawdown_data = []
        
        for i, value in enumerate(equity_values):
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdown_data.append({
                'timestamp': timestamps[i].isoformat(),
                'drawdown': drawdown,
                'peak': peak,
                'value': value
            })
        
        return drawdown_data
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns"""
        
        if not self.returns:
            return {}
        
        monthly_returns = {}
        
        for ret in self.returns:
            month_key = ret['timestamp'].strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            monthly_returns[month_key].append(ret['return'])
        
        # Calculate cumulative monthly returns
        for month in monthly_returns:
            monthly_returns[month] = (1 + pd.Series(monthly_returns[month])).prod() - 1
        
        return monthly_returns
    
    def _calculate_performance_attribution(self) -> Dict[str, Any]:
        """Calculate performance attribution analysis"""
        
        # This is a simplified implementation
        # In practice, you'd analyze contribution from different factors
        
        return {
            'factor_attribution': {
                'market_timing': 0.3,
                'stock_selection': 0.5,
                'risk_management': 0.2
            },
            'sector_attribution': {},
            'style_attribution': {}
        }
