"""
Base Backtest Engine

Abstract base class for all backtesting engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from datetime import datetime
from ..backtest_engine_interface import BacktestEngineInterface
from ..backtest_config import BacktestConfig
from ..trade_record import TradeRecord, PositionRecord, PortfolioRecord
from ..execution.execution_engine import ExecutionEngine
from ..risk.risk_manager import RiskManager
from ..analytics.performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)

class BaseBacktestEngine(BacktestEngineInterface):
    """Base implementation for backtesting engines"""
    
    def __init__(self, name: str, config: BacktestConfig):
        self.name = name
        self.config = config
        
        # Initialize components
        self.execution_engine: Optional[ExecutionEngine] = None
        self.risk_manager: Optional[RiskManager] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        
        # Backtest state
        self.current_portfolio: Optional[PortfolioRecord] = None
        self.trades_history: List[TradeRecord] = []
        self.positions_history: List[PositionRecord] = []
        self.portfolio_history: List[PortfolioRecord] = []
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
    
    def get_name(self) -> str:
        """Get the name of this backtest engine"""
        return self.name
    
    def get_supported_execution_types(self) -> List[str]:
        """Get supported execution types"""
        return ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
    
    def get_config(self) -> Dict[str, Any]:
        """Get backtest engine configuration"""
        return {
            'name': self.name,
            'config': self.config.to_dict() if self.config else {},
            'components': {
                'execution_engine': self.execution_engine.get_name() if self.execution_engine else None,
                'risk_manager': self.risk_manager.get_name() if self.risk_manager else None,
                'performance_analyzer': self.performance_analyzer.get_name() if self.performance_analyzer else None
            }
        }
    
    def set_execution_engine(self, execution_engine: ExecutionEngine):
        """Set the execution engine for this backtest"""
        self.execution_engine = execution_engine
        logger.info(f"Execution engine set: {execution_engine.get_name()}")
    
    def set_risk_manager(self, risk_manager: RiskManager):
        """Set the risk manager for this backtest"""
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set: {risk_manager.get_name()}")
    
    def set_performance_analyzer(self, performance_analyzer: PerformanceAnalyzer):
        """Set the performance analyzer for this backtest"""
        self.performance_analyzer = performance_analyzer
        logger.info(f"Performance analyzer set: {performance_analyzer.get_name()}")
    
    def run_backtest(self, config: BacktestConfig, 
                    price_data: Dict[str, pd.DataFrame],
                    signal_data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with given configuration and data"""
        
        # Convert DataFrame to list of signals for compatibility
        signals = []
        if not signal_data.empty and 'symbol' in signal_data.columns:
            for _, row in signal_data.iterrows():
                signal = {
                    'symbol': row['symbol'],
                    'signal_type': row.get('signal_type', 'BUY'),
                    'timestamp': row.get('timestamp', datetime.now()),
                    'strength': row.get('strength', 0.5)
                }
                signals.append(signal)
        
        # Convert price data to market data format
        market_data = {
            'prices': {},
            'timestamps': []
        }
        
        for symbol, df in price_data.items():
            if not df.empty and 'close' in df.columns:
                market_data['prices'][symbol] = df['close'].tolist()
                market_data['timestamps'] = df.index.tolist()
        
        # Call the existing implementation
        return self._run_backtest_impl(signals, market_data)
    
    def _run_backtest_impl(self, signals: List[Dict[str, Any]], 
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the backtest with the given signals and market data"""
        
        logger.info(f"Starting backtest: {self.name}")
        self.start_time = datetime.now()
        self.is_running = True
        
        try:
            # Initialize portfolio
            self._initialize_portfolio()
            
            # Validate components
            self._validate_components()
            
            # Preprocess data
            processed_signals = self._preprocess_signals(signals)
            processed_market_data = self._preprocess_market_data(market_data)
            
            # Run the main backtest loop
            results = self._run_backtest_loop(processed_signals, processed_market_data)
            
            # Post-process results
            final_results = self._postprocess_results(results)
            
            logger.info(f"Backtest completed: {self.name}")
            return final_results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            self.end_time = datetime.now()
            self.is_running = False
    
    def _initialize_portfolio(self):
        """Initialize the portfolio for backtesting"""
        
        self.current_portfolio = PortfolioRecord(
            timestamp=datetime.now(),
            cash=self.config.initial_capital,
            positions=[],
            total_value=self.config.initial_capital,
            positions_value=0.0
        )
        
        # Initialize performance analyzer
        if self.performance_analyzer:
            self.performance_analyzer.add_portfolio_snapshot(
                self.current_portfolio, 
                self.current_portfolio.timestamp
            )
        
        logger.info(f"Portfolio initialized with ${self.config.initial_capital:,.2f}")
    
    def _validate_components(self):
        """Validate that all required components are set"""
        
        if not self.execution_engine:
            raise ValueError("Execution engine not set")
        
        if not self.risk_manager:
            raise ValueError("Risk manager not set")
        
        if not self.performance_analyzer:
            raise ValueError("Performance analyzer not set")
        
        logger.info("All components validated")
    
    def _preprocess_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess signals for backtesting"""
        
        processed_signals = []
        
        for signal in signals:
            # Validate signal structure
            if not self._validate_signal(signal):
                logger.warning(f"Invalid signal skipped: {signal}")
                continue
            
            # Add signal metadata
            processed_signal = {
                **signal,
                'processed_timestamp': datetime.now(),
                'signal_id': f"signal_{len(processed_signals)}"
            }
            
            processed_signals.append(processed_signal)
        
        logger.info(f"Preprocessed {len(processed_signals)} signals")
        return processed_signals
    
    def _preprocess_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess market data for backtesting"""
        
        # Validate market data structure
        if not self._validate_market_data(market_data):
            raise ValueError("Invalid market data structure")
        
        # Add metadata
        processed_data = {
            **market_data,
            'processed_timestamp': datetime.now(),
            'data_points': len(market_data.get('prices', {}))
        }
        
        logger.info(f"Preprocessed market data with {processed_data['data_points']} data points")
        return processed_data
    
    def _run_backtest_loop(self, signals: List[Dict[str, Any]], 
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the main backtest loop"""
        
        # This is the core backtest logic that subclasses should implement
        # For the base class, we'll provide a simple implementation
        
        results = {
            'trades': [],
            'portfolio_history': [],
            'performance_metrics': {},
            'risk_metrics': {}
        }
        
        # Process each signal
        for signal in signals:
            try:
                # Generate trade from signal
                trade = self._generate_trade_from_signal(signal, market_data)
                
                if trade:
                    # Validate trade with risk manager
                    if self.risk_manager.validate_trade(trade, self.current_portfolio, self.config):
                        # Execute trade
                        executed_trade = self.execution_engine.execute_trade(trade, market_data)
                        
                        if executed_trade:
                            # Update portfolio
                            self._update_portfolio(executed_trade)
                            
                            # Record trade
                            self.trades_history.append(executed_trade)
                            results['trades'].append(executed_trade)
                            
                            # Update performance analyzer
                            self.performance_analyzer.add_trade(executed_trade)
                            
                            logger.info(f"Trade executed: {executed_trade.symbol} {executed_trade.trade_type.value}")
                    else:
                        logger.warning(f"Trade rejected by risk manager: {signal}")
                
                # Update portfolio snapshot
                self._update_portfolio_snapshot()
                results['portfolio_history'].append(self.current_portfolio)
                
            except Exception as e:
                logger.error(f"Error processing signal: {e}")
                continue
        
        # Calculate final metrics
        results['performance_metrics'] = self.performance_analyzer.generate_performance_report(self.config)
        results['risk_metrics'] = self.risk_manager.get_risk_report(self.current_portfolio, self.config)
        
        return results
    
    def _postprocess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process backtest results"""
        
        # Add backtest metadata
        results['backtest_metadata'] = {
            'engine_name': self.name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
            'config': self.config.to_dict()
        }
        
        # Add summary statistics
        results['summary'] = {
            'total_trades': len(results['trades']),
            'final_portfolio_value': self.current_portfolio.total_value if self.current_portfolio else 0,
            'total_return': (self.current_portfolio.total_value / self.config.initial_capital - 1) if self.current_portfolio else 0,
            'max_drawdown': results['performance_metrics'].get('summary', {}).get('max_drawdown', 0),
            'sharpe_ratio': results['performance_metrics'].get('summary', {}).get('sharpe_ratio', 0)
        }
        
        logger.info(f"Backtest results post-processed")
        return results
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal structure"""
        
        required_fields = ['symbol', 'signal_type', 'timestamp']
        
        for field in required_fields:
            if field not in signal:
                return False
        
        return True
    
    def _validate_market_data(self, market_data: Dict[str, Any]) -> bool:
        """Validate market data structure"""
        
        required_fields = ['prices', 'timestamps']
        
        for field in required_fields:
            if field not in market_data:
                return False
        
        return True
    
    def _generate_trade_from_signal(self, signal: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> Optional[TradeRecord]:
        """Generate a trade record from a signal"""
        
        # This is a simplified implementation
        # Subclasses should override this with more sophisticated logic
        
        symbol = signal.get('symbol')
        signal_type = signal.get('signal_type')
        timestamp = signal.get('timestamp')
        
        if not all([symbol, signal_type, timestamp]):
            return None
        
        # Get current price
        current_price = self._get_current_price(symbol, market_data)
        if not current_price:
            return None
        
        # Determine trade type and quantity
        if signal_type == 'BUY':
            trade_type = 'BUY'
            quantity = self._calculate_position_size(symbol, current_price)
        elif signal_type == 'SELL':
            trade_type = 'SELL'
            quantity = self._calculate_position_size(symbol, current_price)
        else:
            return None
        
        if quantity <= 0:
            return None
        
        # Create trade record
        trade = TradeRecord(
            timestamp=timestamp,
            symbol=symbol,
            trade_type=trade_type,
            quantity=quantity,
            price=current_price,
            commission=0.0,
            realized_pnl=0.0,
            status='PENDING'
        )
        
        return trade
    
    def _get_current_price(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """Get current price for a symbol"""
        
        prices = market_data.get('prices', {}).get(symbol, [])
        if prices:
            return prices[-1]  # Return latest price
        
        return None
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size for a trade"""
        
        # Simple position sizing: use 10% of available cash
        available_cash = self.current_portfolio.cash if self.current_portfolio else 0
        position_value = available_cash * 0.1
        
        if position_value <= 0:
            return 0
        
        quantity = position_value / price
        
        # Apply minimum trade size constraint
        if quantity * price < self.config.min_trade_size:
            return 0
        
        return quantity
    
    def _update_portfolio(self, trade: TradeRecord):
        """Update portfolio after a trade"""
        
        if not self.current_portfolio:
            return
        
        # Calculate trade value
        trade_value = trade.quantity * trade.price
        total_cost = trade_value + trade.commission
        
        # Update cash
        if trade.trade_type == 'BUY':
            self.current_portfolio.cash -= total_cost
        else:  # SELL
            self.current_portfolio.cash += (trade_value - trade.commission)
        
        # Update positions
        self._update_positions(trade)
        
        # Update total values
        self._recalculate_portfolio_values()
    
    def _update_positions(self, trade: TradeRecord):
        """Update positions after a trade"""
        
        # Find existing position
        existing_position = None
        for position in self.current_portfolio.positions:
            if position.symbol == trade.symbol:
                existing_position = position
                break
        
        if trade.trade_type == 'BUY':
            if existing_position:
                # Add to existing position
                existing_position.quantity += trade.quantity
                existing_position.avg_price = (
                    (existing_position.avg_price * existing_position.quantity + 
                     trade.price * trade.quantity) / 
                    (existing_position.quantity + trade.quantity)
                )
            else:
                # Create new position
                new_position = PositionRecord(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    avg_price=trade.price,
                    market_value=trade.quantity * trade.price,
                    unrealized_pnl=0.0
                )
                self.current_portfolio.positions.append(new_position)
        
        else:  # SELL
            if existing_position:
                # Reduce position
                existing_position.quantity -= trade.quantity
                
                if existing_position.quantity <= 0:
                    # Remove position if fully sold
                    self.current_portfolio.positions.remove(existing_position)
    
    def _recalculate_portfolio_values(self):
        """Recalculate portfolio total values"""
        
        if not self.current_portfolio:
            return
        
        positions_value = sum(position.market_value for position in self.current_portfolio.positions)
        self.current_portfolio.positions_value = positions_value
        self.current_portfolio.total_value = self.current_portfolio.cash + positions_value
    
    def _update_portfolio_snapshot(self):
        """Update portfolio snapshot for analysis"""
        
        if self.performance_analyzer and self.current_portfolio:
            self.performance_analyzer.add_portfolio_snapshot(
                self.current_portfolio,
                self.current_portfolio.timestamp
            )
    
    def get_backtest_status(self) -> Dict[str, Any]:
        """Get current backtest status"""
        
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_trades': len(self.trades_history),
            'current_portfolio_value': self.current_portfolio.total_value if self.current_portfolio else 0,
            'components': {
                'execution_engine': self.execution_engine.get_name() if self.execution_engine else None,
                'risk_manager': self.risk_manager.get_name() if self.risk_manager else None,
                'performance_analyzer': self.performance_analyzer.get_name() if self.performance_analyzer else None
            }
        }
