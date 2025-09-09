"""
Test data factory for creating realistic test data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_ohlcv_data(
        symbols: List[str] = ['AAPL', 'MSFT'], 
        days: int = 30,
        start_date: datetime = None
    ) -> pd.DataFrame:
        """Create realistic OHLCV data for testing"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        
        data = []
        for symbol in symbols:
            base_price = 150 if symbol == 'AAPL' else 300
            for date in dates:
                # Generate realistic price movement
                price_change = np.random.normal(0, 2)
                price = base_price + price_change
                
                # Ensure high >= low and close is within range
                high = max(price + np.random.uniform(0, 3), price + 1)
                low = min(price - np.random.uniform(0, 3), price - 1)
                open_price = price + np.random.uniform(-1, 1)
                close_price = price + np.random.uniform(-1, 1)
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': np.random.randint(1000000, 10000000)
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_signal_data(
        symbols: List[str] = ['AAPL', 'MSFT'],
        days: int = 30,
        start_date: datetime = None
    ) -> pd.DataFrame:
        """Create signal data for testing"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
        
        data = []
        for symbol in symbols:
            for date in dates:
                # Generate realistic signal data
                signal_strength = np.random.uniform(0, 1)
                confidence = np.random.uniform(0.5, 1.0)
                signal_type = np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4])
                
                data.append({
                    'symbol': symbol,
                    'date': date,
                    'signal_type': signal_type,
                    'signal_strength': round(signal_strength, 3),
                    'confidence': round(confidence, 3),
                    'strategy': np.random.choice(['technical', 'fundamental', 'sentiment']),
                    'created_at': date
                })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_user_data() -> Dict[str, Any]:
        """Create user data for testing"""
        return {
            'username': 'test_user',
            'email': 'test@example.com',
            'password': 'test_password',
            'first_name': 'Test',
            'last_name': 'User',
            'is_active': True,
            'created_at': datetime.now()
        }
    
    @staticmethod
    def create_pipeline_run_data(
        symbols: List[str] = ['AAPL'],
        status: str = 'running'
    ) -> Dict[str, Any]:
        """Create pipeline run data for testing"""
        return {
            'symbols': symbols,
            'timeframe': '1day',
            'status': status,
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=1) if status == 'completed' else None,
            'config': {
                'strategy': 'technical',
                'parameters': {
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26
                }
            }
        }
    
    @staticmethod
    def create_backtest_data(
        symbols: List[str] = ['AAPL'],
        days: int = 30
    ) -> Dict[str, Any]:
        """Create backtest data for testing"""
        return {
            'symbols': symbols,
            'timeframe': '1day',
            'start_date': datetime.now() - timedelta(days=days),
            'end_date': datetime.now(),
            'initial_capital': 100000,
            'strategy': 'technical',
            'parameters': {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26
            },
            'results': {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.05,
                'win_rate': 0.65,
                'total_trades': 25
            }
        }
    
    @staticmethod
    def create_technical_indicators_data(
        symbols: List[str] = ['AAPL'],
        days: int = 30
    ) -> pd.DataFrame:
        """Create technical indicators data for testing"""
        ohlcv_data = TestDataFactory.create_ohlcv_data(symbols, days)
        
        data = []
        for symbol in symbols:
            symbol_data = ohlcv_data[ohlcv_data['symbol'] == symbol].copy()
            
            # Calculate basic technical indicators
            symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
            symbol_data['ema_12'] = symbol_data['close'].ewm(span=12).mean()
            symbol_data['ema_26'] = symbol_data['close'].ewm(span=26).mean()
            symbol_data['rsi'] = TestDataFactory._calculate_rsi(symbol_data['close'])
            symbol_data['macd'] = symbol_data['ema_12'] - symbol_data['ema_26']
            symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9).mean()
            symbol_data['bollinger_upper'] = symbol_data['close'].rolling(window=20).mean() + (symbol_data['close'].rolling(window=20).std() * 2)
            symbol_data['bollinger_lower'] = symbol_data['close'].rolling(window=20).mean() - (symbol_data['close'].rolling(window=20).std() * 2)
            
            data.append(symbol_data)
        
        return pd.concat(data, ignore_index=True)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for testing"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def create_market_data(
        symbols: List[str] = ['AAPL', 'MSFT', 'GOOGL'],
        days: int = 30
    ) -> Dict[str, pd.DataFrame]:
        """Create comprehensive market data for testing"""
        data = {}
        
        for symbol in symbols:
            data[symbol] = {
                'ohlcv': TestDataFactory.create_ohlcv_data([symbol], days),
                'signals': TestDataFactory.create_signal_data([symbol], days),
                'indicators': TestDataFactory.create_technical_indicators_data([symbol], days)
            }
        
        return data
