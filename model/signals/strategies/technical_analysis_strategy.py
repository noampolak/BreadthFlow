"""
Technical Analysis Strategy

Implements technical analysis-based signal generation strategies.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
import time
from base_signal_strategy import BaseSignalStrategy
from signal_config import SignalConfig
from components.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class TechnicalAnalysisStrategy(BaseSignalStrategy):
    """Technical analysis-based signal generation strategy"""
    
    def __init__(self, name: str = "technical_analysis", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Set required data and timeframes
        self.required_data = ["stock_price"]
        self.supported_timeframes = ["1min", "5min", "15min", "1hour", "1day"]
        
        # Initialize technical indicators component
        self.technical_indicators = TechnicalIndicators()
        
        # Default strategy configuration
        self.default_config = {
            'indicators': ['rsi', 'macd', 'bollinger_bands'],
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'signal_threshold': 0.5,
            'confidence_threshold': 0.7
        }
        
        # Update config with defaults
        self.config.update(self.default_config)
    
    def generate_signals(self, data: Dict[str, Any], 
                        config: SignalConfig):
        """Generate technical analysis signals"""
        
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for technical analysis but not available")
        
        start_time = time.time()
        success = False
        
        try:
            # Validate data
            if not self.validate_data(data):
                raise ValueError("Invalid data for technical analysis strategy")
            
            # Preprocess data
            processed_data = self.preprocess_data(data, config)
            
            if not processed_data:
                logger.error("No valid data after preprocessing")
                if PANDAS_AVAILABLE:
                    return pd.DataFrame()
                else:
                    return {}
            
            # Get stock price data
            stock_data = processed_data.get("stock_price")
            if stock_data is None or stock_data.empty:
                logger.error("No stock price data available")
                if PANDAS_AVAILABLE:
                    return pd.DataFrame()
                else:
                    return {}
            
            # Generate technical indicators
            indicators = self.config.get('indicators', ['rsi', 'macd', 'bollinger_bands'])
            signals = self.technical_indicators.generate_technical_signals(
                stock_data, indicators, self.config
            )
            
            # Generate trading signals based on indicators
            signals = self._generate_trading_signals(signals)
            
            # Calculate signal strength and confidence
            signals = self._calculate_signal_metrics(signals)
            
            # Postprocess signals
            signals = self.postprocess_signals(signals, config)
            
            success = True
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical analysis signals: {e}")
            if PANDAS_AVAILABLE:
                return pd.DataFrame()
            else:
                return {}
        
        finally:
            # Update performance stats
            generation_time = time.time() - start_time
            self.update_performance_stats(success, generation_time)
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate that required data is available"""
        if "stock_price" not in data:
            logger.error("Technical analysis strategy requires stock_price data")
            return False
        
        stock_data = data["stock_price"]
        if stock_data.empty:
            logger.error("Stock price data is empty")
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on technical indicators"""
        signals = data.copy()
        
        # RSI signals
        if 'rsi' in signals.columns:
            rsi_oversold = self.config.get('rsi_oversold', 30)
            rsi_overbought = self.config.get('rsi_overbought', 70)
            
            signals['rsi_buy'] = (signals['rsi'] < rsi_oversold).astype(int)
            signals['rsi_sell'] = (signals['rsi'] > rsi_overbought).astype(int)
            signals['rsi_neutral'] = ((signals['rsi'] >= rsi_oversold) & 
                                    (signals['rsi'] <= rsi_overbought)).astype(int)
        
        # MACD signals
        if 'macd_line' in signals.columns and 'macd_signal' in signals.columns:
            signals['macd_buy'] = (signals['macd_line'] > signals['macd_signal']).astype(int)
            signals['macd_sell'] = (signals['macd_line'] < signals['macd_signal']).astype(int)
            signals['macd_crossover'] = ((signals['macd_line'] > signals['macd_signal']) & 
                                       (signals['macd_line'].shift(1) <= signals['macd_signal'].shift(1))).astype(int)
        
        # Bollinger Bands signals
        if 'bb_upper' in signals.columns and 'bb_lower' in signals.columns:
            signals['bb_buy'] = (signals['close'] < signals['bb_lower']).astype(int)
            signals['bb_sell'] = (signals['close'] > signals['bb_upper']).astype(int)
            signals['bb_squeeze'] = ((signals['bb_upper'] - signals['bb_lower']) / 
                                   signals['bb_middle'] < 0.1).astype(int)
        
        # Moving average signals
        if 'sma' in signals.columns:
            sma_period = self.config.get('sma_period', 20)
            signals['ma_buy'] = (signals['close'] > signals['sma']).astype(int)
            signals['ma_sell'] = (signals['close'] < signals['sma']).astype(int)
        
        # Volume signals
        if 'volume' in signals.columns:
            volume_sma = signals['volume'].rolling(window=20).mean()
            signals['volume_spike'] = (signals['volume'] > volume_sma * 1.5).astype(int)
            signals['volume_low'] = (signals['volume'] < volume_sma * 0.5).astype(int)
        
        return signals
    
    def _calculate_signal_metrics(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate signal strength and confidence metrics"""
        
        # Calculate composite signal strength
        buy_signals = [col for col in signals.columns if col.endswith('_buy')]
        sell_signals = [col for col in signals.columns if col.endswith('_sell')]
        
        buy_strength = signals[buy_signals].sum(axis=1) if buy_signals else pd.Series(0, index=signals.index)
        sell_strength = signals[sell_signals].sum(axis=1) if sell_signals else pd.Series(0, index=signals.index)
        
        # Calculate net signal strength (-1 to 1)
        total_signals = buy_strength + sell_strength
        signal_strength = np.where(total_signals > 0, 
                                 (buy_strength - sell_strength) / total_signals, 0)
        
        signals['signal_strength'] = signal_strength
        
        # Calculate signal confidence based on multiple factors
        confidence_factors = []
        
        # RSI confidence (closer to extremes = higher confidence)
        if 'rsi' in signals.columns:
            rsi_confidence = 1 - abs(signals['rsi'] - 50) / 50
            confidence_factors.append(rsi_confidence)
        
        # MACD confidence (stronger divergence = higher confidence)
        if 'macd_histogram' in signals.columns:
            macd_confidence = abs(signals['macd_histogram']) / signals['macd_histogram'].abs().max()
            confidence_factors.append(macd_confidence)
        
        # Volume confidence (higher volume = higher confidence)
        if 'volume' in signals.columns:
            volume_confidence = signals['volume'] / signals['volume'].max()
            confidence_factors.append(volume_confidence)
        
        # Calculate average confidence
        if confidence_factors:
            confidence = pd.concat(confidence_factors, axis=1).mean(axis=1)
        else:
            confidence = pd.Series(0.5, index=signals.index)
        
        signals['confidence'] = confidence
        
        # Determine signal type
        signals['signal_type'] = np.where(signal_strength > 0.3, 'buy',
                                        np.where(signal_strength < -0.3, 'sell', 'hold'))
        
        return signals
    
    def get_strategy_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Get individual strategy signals"""
        signals = {}
        
        # RSI strategy
        if 'rsi' in data.columns:
            rsi_oversold = self.config.get('rsi_oversold', 30)
            rsi_overbought = self.config.get('rsi_overbought', 70)
            
            signals['rsi_strategy'] = np.where(data['rsi'] < rsi_oversold, 1,
                                             np.where(data['rsi'] > rsi_overbought, -1, 0))
        
        # MACD strategy
        if 'macd_line' in data.columns and 'macd_signal' in data.columns:
            signals['macd_strategy'] = np.where(data['macd_line'] > data['macd_signal'], 1, -1)
        
        # Bollinger Bands strategy
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            signals['bb_strategy'] = np.where(data['close'] < data['bb_lower'], 1,
                                            np.where(data['close'] > data['bb_upper'], -1, 0))
        
        # Moving average strategy
        if 'sma' in data.columns:
            signals['ma_strategy'] = np.where(data['close'] > data['sma'], 1, -1)
        
        return signals
    
    def get_signal_summary(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of generated signals"""
        if signals.empty:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'average_strength': 0.0,
                'average_confidence': 0.0
            }
        
        summary = {
            'total_signals': len(signals),
            'buy_signals': len(signals[signals['signal_type'] == 'buy']),
            'sell_signals': len(signals[signals['signal_type'] == 'sell']),
            'hold_signals': len(signals[signals['signal_type'] == 'hold']),
            'average_strength': signals['signal_strength'].mean() if 'signal_strength' in signals.columns else 0.0,
            'average_confidence': signals['confidence'].mean() if 'confidence' in signals.columns else 0.0
        }
        
        # Calculate signal distribution
        if 'signal_type' in signals.columns:
            summary['signal_distribution'] = signals['signal_type'].value_counts().to_dict()
        
        return summary
