"""
Backtest Engine Interface

Abstract interface for backtest engines in the BreadthFlow system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from backtest_config import BacktestConfig
from trade_record import TradeRecord, PositionRecord, PortfolioRecord

class BacktestEngineInterface(ABC):
    """Abstract interface for backtest engines"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this backtest engine"""
        pass
    
    @abstractmethod
    def get_supported_execution_types(self) -> List[str]:
        """Get list of supported execution types"""
        pass
    
    @abstractmethod
    def run_backtest(self, config: BacktestConfig, 
                    price_data: Dict[str, pd.DataFrame],
                    signal_data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with given configuration and data"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get backtest engine configuration"""
        pass
    
    def validate_config(self, config: BacktestConfig) -> bool:
        """Validate backtest configuration"""
        if not config.validate():
            return False
        
        # Check if execution type is supported
        if config.execution_type.value not in self.get_supported_execution_types():
            return False
        
        return True
    
    def validate_data(self, price_data: Dict[str, pd.DataFrame], 
                     signal_data: pd.DataFrame) -> bool:
        """Validate that required data is available"""
        if not price_data:
            return False
        
        if signal_data.empty:
            return False
        
        # Check if all required symbols have price data
        for symbol in signal_data['symbol'].unique() if 'symbol' in signal_data.columns else []:
            if symbol not in price_data:
                return False
            
            if price_data[symbol].empty:
                return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this backtest engine"""
        return {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'average_execution_time': 0.0
        }
    
    def get_backtest_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of backtest results"""
        if not results:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        trades = results.get('trades', [])
        portfolio_history = results.get('portfolio_history', [])
        
        summary = {
            'total_trades': len(trades),
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        if portfolio_history:
            # Calculate total return
            initial_value = portfolio_history[0].total_value if portfolio_history else 0
            final_value = portfolio_history[-1].total_value if portfolio_history else 0
            if initial_value > 0:
                summary['total_return'] = (final_value - initial_value) / initial_value
        
        if trades:
            # Calculate win rate
            winning_trades = [t for t in trades if t.total_pnl > 0] if hasattr(trades[0], 'total_pnl') else []
            summary['win_rate'] = len(winning_trades) / len(trades) if trades else 0
        
        return summary
    
    def calculate_risk_metrics(self, portfolio_history: List[PortfolioRecord]) -> Dict[str, float]:
        """Calculate risk metrics from portfolio history"""
        if not portfolio_history or len(portfolio_history) < 2:
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Calculate daily returns
        returns = []
        for i in range(1, len(portfolio_history)):
            prev_value = portfolio_history[i-1].total_value
            curr_value = portfolio_history[i].total_value
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                returns.append(daily_return)
        
        if not returns:
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
        import numpy as np
        
        # Calculate metrics
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array, ddof=1)
        
        # Sharpe ratio (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns_array, 5)
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns, ddof=1) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (annualized return / max drawdown)
        annualized_return = mean_return * 252  # Assuming daily data
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def generate_trade_signals(self, signal_data: pd.DataFrame, 
                             config: BacktestConfig) -> List[Dict[str, Any]]:
        """Generate trade signals from signal data"""
        signals = []
        
        if signal_data.empty:
            return signals
        
        # Filter signals based on thresholds
        filtered_signals = signal_data[
            (signal_data['signal_strength'].abs() >= config.signal_threshold) &
            (signal_data['confidence'] >= config.confidence_threshold)
        ]
        
        for _, row in filtered_signals.iterrows():
            signal = {
                'timestamp': row.get('date', row.get('timestamp')),
                'symbol': row.get('symbol'),
                'signal_type': row.get('signal_type'),
                'signal_strength': row.get('signal_strength'),
                'confidence': row.get('confidence'),
                'price': row.get('close', row.get('price'))
            }
            signals.append(signal)
        
        return signals
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              portfolio_value: float,
                              config: BacktestConfig) -> float:
        """Calculate position size based on signal and configuration"""
        
        if config.position_sizing_method == "fixed":
            return portfolio_value * config.fixed_position_size
        
        elif config.position_sizing_method == "kelly":
            # Kelly criterion position sizing
            win_prob = signal.get('confidence', 0.5)
            avg_win = 0.1  # 10% average win
            avg_loss = 0.05  # 5% average loss
            
            kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, config.kelly_fraction))
            
            return portfolio_value * kelly_fraction
        
        elif config.position_sizing_method == "risk_parity":
            # Risk parity position sizing
            risk_per_trade = portfolio_value * config.risk_per_trade
            stop_loss = signal.get('stop_loss', 0.05)
            
            if stop_loss > 0:
                return risk_per_trade / stop_loss
            else:
                return portfolio_value * config.fixed_position_size
        
        else:
            # Default to fixed position sizing
            return portfolio_value * config.fixed_position_size
