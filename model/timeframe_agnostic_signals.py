#!/usr/bin/env python3
"""
Timeframe-Agnostic Signal Generator

This module provides signal generation capabilities across multiple timeframes
while maintaining backward compatibility with existing daily signal generation.

Key Features:
- Support for multiple timeframes: 1min, 5min, 15min, 1hour, 1day
- Timeframe-optimized technical indicators and parameters
- Consistent signal format across all timeframes
- Backward compatibility with existing signal generation
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeframeAgnosticSignalGenerator:
    """
    Signal generator that adapts technical analysis parameters based on timeframe.
    
    This class maintains backward compatibility while adding multi-timeframe support.
    """
    
    def __init__(self, timeframe: str = '1day'):
        """
        Initialize the signal generator for a specific timeframe.
        
        Args:
            timeframe: Target timeframe ('1min', '5min', '15min', '1hour', '1day')
        """
        self.timeframe = timeframe
        self.parameters = self.get_timeframe_parameters(timeframe)
        self.supported_timeframes = ['1min', '5min', '15min', '1hour', '1day']
        
        if timeframe not in self.supported_timeframes:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.supported_timeframes}")
        
        logger.info(f"TimeframeAgnosticSignalGenerator initialized for {timeframe}")
        logger.info(f"Using parameters: {self.parameters}")
    
    def get_timeframe_parameters(self, timeframe: str) -> Dict[str, Any]:
        """
        Get optimized technical analysis parameters for the specified timeframe.
        
        Different timeframes require different parameter settings for optimal signal generation.
        
        Args:
            timeframe: Target timeframe
            
        Returns:
            Dictionary of technical analysis parameters
        """
        parameters = {
            '1day': {
                # Daily timeframe - traditional parameters
                'ma_short': 20,
                'ma_long': 50,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'bb_period': 20,
                'bb_std': 2,
                'volume_ma': 20,
                'min_volume_ratio': 1.5,
                'price_change_threshold': 0.02,  # 2%
                'confidence_base': 0.6,
                'lookback_period': 50
            },
            '1hour': {
                # Hourly timeframe - adjusted for intraday trading
                'ma_short': 12,
                'ma_long': 24,
                'rsi_period': 14,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'bb_period': 20,
                'bb_std': 1.8,
                'volume_ma': 24,
                'min_volume_ratio': 1.3,
                'price_change_threshold': 0.015,  # 1.5%
                'confidence_base': 0.5,
                'lookback_period': 48
            },
            '15min': {
                # 15-minute timeframe - faster reaction
                'ma_short': 8,
                'ma_long': 16,
                'rsi_period': 14,
                'rsi_oversold': 20,
                'rsi_overbought': 80,
                'bb_period': 16,
                'bb_std': 1.6,
                'volume_ma': 20,
                'min_volume_ratio': 1.2,
                'price_change_threshold': 0.01,  # 1%
                'confidence_base': 0.4,
                'lookback_period': 32
            },
            '5min': {
                # 5-minute timeframe - very responsive
                'ma_short': 6,
                'ma_long': 12,
                'rsi_period': 10,
                'rsi_oversold': 15,
                'rsi_overbought': 85,
                'bb_period': 12,
                'bb_std': 1.4,
                'volume_ma': 15,
                'min_volume_ratio': 1.1,
                'price_change_threshold': 0.008,  # 0.8%
                'confidence_base': 0.3,
                'lookback_period': 24
            },
            '1min': {
                # 1-minute timeframe - highly responsive
                'ma_short': 5,
                'ma_long': 10,
                'rsi_period': 8,
                'rsi_oversold': 10,
                'rsi_overbought': 90,
                'bb_period': 10,
                'bb_std': 1.2,
                'volume_ma': 10,
                'min_volume_ratio': 1.05,
                'price_change_threshold': 0.005,  # 0.5%
                'confidence_base': 0.2,
                'lookback_period': 20
            }
        }
        
        return parameters.get(timeframe, parameters['1day'])
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators optimized for the current timeframe.
        
        Args:
            data: OHLCV DataFrame with columns [Date, Open, High, Low, Close, Volume]
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        if data.empty:
            return data
        
        df = data.copy()
        params = self.parameters
        
        try:
            # Moving Averages
            df['MA_Short'] = df['Close'].rolling(window=params['ma_short']).mean()
            df['MA_Long'] = df['Close'].rolling(window=params['ma_long']).mean()
            
            # RSI (Relative Strength Index)
            df['RSI'] = self._calculate_rsi(df['Close'], params['rsi_period'])
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['Close'], params['bb_period'], params['bb_std'])
            df['BB_Upper'] = bb_data['upper']
            df['BB_Lower'] = bb_data['lower']
            df['BB_Middle'] = bb_data['middle']
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=params['volume_ma']).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Price change indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_MA'] = df['Price_Change'].rolling(window=params['ma_short']).mean()
            
            # Momentum indicators
            df['Momentum'] = df['Close'] / df['Close'].shift(params['ma_short']) - 1
            
            # Support/Resistance levels (simplified)
            df['High_MA'] = df['High'].rolling(window=params['ma_short']).mean()
            df['Low_MA'] = df['Low'].rolling(window=params['ma_short']).mean()
            
            logger.debug(f"Calculated technical indicators for {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'middle': middle,
            'upper': middle + (std * std_dev),
            'lower': middle - (std * std_dev)
        }
    
    def generate_signals(self, data: pd.DataFrame, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate trading signals for the given data and timeframe.
        
        Args:
            data: OHLCV DataFrame
            symbols: List of symbols (optional, extracted from data if not provided)
            
        Returns:
            List of signal dictionaries
        """
        if data.empty:
            logger.warning("Empty data provided for signal generation")
            return []
        
        # If symbols not provided, extract from data
        if symbols is None:
            if 'symbol' in data.columns:
                symbols = data['symbol'].unique().tolist()
            else:
                symbols = ['UNKNOWN']
        
        signals = []
        
        for symbol in symbols:
            try:
                # Filter data for this symbol
                if 'symbol' in data.columns:
                    symbol_data = data[data['symbol'] == symbol].copy()
                else:
                    symbol_data = data.copy()
                
                if symbol_data.empty:
                    logger.warning(f"No data found for symbol {symbol}")
                    continue
                
                # Sort by date
                date_col = 'Date' if 'Date' in symbol_data.columns else 'date'
                if date_col in symbol_data.columns:
                    symbol_data = symbol_data.sort_values(date_col)
                
                # Calculate technical indicators
                symbol_data = self.calculate_technical_indicators(symbol_data)
                
                # Generate signals for this symbol
                symbol_signals = self._generate_symbol_signals(symbol_data, symbol)
                signals.extend(symbol_signals)
                
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                continue
        
        # Add timeframe information to all signals
        for signal in signals:
            signal['timeframe'] = self.timeframe
            signal['create_time'] = datetime.now().isoformat()
        
        logger.info(f"Generated {len(signals)} signals for timeframe {self.timeframe}")
        return signals
    
    def _generate_symbol_signals(self, data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """
        Generate signals for a single symbol.
        
        Args:
            data: Symbol-specific OHLCV data with technical indicators
            symbol: Stock symbol
            
        Returns:
            List of signal dictionaries for this symbol
        """
        if len(data) < self.parameters['lookback_period']:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} rows, need {self.parameters['lookback_period']}")
            return []
        
        signals = []
        params = self.parameters
        
        # Get the latest data point for signal generation
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        # Extract date
        date_col = 'Date' if 'Date' in data.columns else 'date'
        signal_date = latest[date_col] if date_col in data.columns else datetime.now().date()
        
        # Convert timestamp to date string if needed
        if hasattr(signal_date, 'strftime'):
            signal_date_str = signal_date.strftime('%Y-%m-%d')
        else:
            signal_date_str = str(signal_date)
        
        # Signal generation logic
        signal_type = "HOLD"
        confidence = params['confidence_base']
        strength = "WEAK"
        
        # Check if we have the required indicators
        required_indicators = ['MA_Short', 'MA_Long', 'RSI', 'BB_Upper', 'BB_Lower']
        if not all(indicator in latest.index for indicator in required_indicators):
            logger.warning(f"Missing technical indicators for {symbol}")
            return []
        
        # BUY signals
        buy_signals = 0
        buy_factors = []
        
        # MA Crossover (bullish)
        if (latest['MA_Short'] > latest['MA_Long'] and 
            previous['MA_Short'] <= previous['MA_Long']):
            buy_signals += 1
            buy_factors.append("MA_CROSSOVER_BULL")
        
        # RSI oversold
        if latest['RSI'] < params['rsi_oversold']:
            buy_signals += 1
            buy_factors.append("RSI_OVERSOLD")
        
        # Price below lower Bollinger Band
        if latest['Close'] < latest['BB_Lower']:
            buy_signals += 1
            buy_factors.append("BB_OVERSOLD")
        
        # Volume confirmation
        if 'Volume_Ratio' in latest.index and latest['Volume_Ratio'] > params['min_volume_ratio']:
            buy_signals += 0.5
            buy_factors.append("VOLUME_CONFIRM")
        
        # SELL signals
        sell_signals = 0
        sell_factors = []
        
        # MA Crossover (bearish)
        if (latest['MA_Short'] < latest['MA_Long'] and 
            previous['MA_Short'] >= previous['MA_Long']):
            sell_signals += 1
            sell_factors.append("MA_CROSSOVER_BEAR")
        
        # RSI overbought
        if latest['RSI'] > params['rsi_overbought']:
            sell_signals += 1
            sell_factors.append("RSI_OVERBOUGHT")
        
        # Price above upper Bollinger Band
        if latest['Close'] > latest['BB_Upper']:
            sell_signals += 1
            sell_factors.append("BB_OVERBOUGHT")
        
        # Volume confirmation for sell
        if 'Volume_Ratio' in latest.index and latest['Volume_Ratio'] > params['min_volume_ratio']:
            sell_signals += 0.5
            sell_factors.append("VOLUME_CONFIRM")
        
        # Determine final signal
        if buy_signals >= 1.0 and buy_signals > sell_signals:
            signal_type = "BUY"
            confidence = min(0.95, params['confidence_base'] + (buy_signals * 0.15))
            strength = "STRONG" if buy_signals >= 2.0 else "MODERATE"
        elif sell_signals >= 1.0 and sell_signals > buy_signals:
            signal_type = "SELL"
            confidence = min(0.95, params['confidence_base'] + (sell_signals * 0.15))
            strength = "STRONG" if sell_signals >= 2.0 else "MODERATE"
        else:
            # Default to HOLD with adjusted confidence
            confidence = max(0.1, params['confidence_base'] * 0.5)
        
        # Calculate additional metrics
        price_change = latest['Price_Change'] if 'Price_Change' in latest.index else 0.0
        volume_change = (latest['Volume_Ratio'] - 1.0) if 'Volume_Ratio' in latest.index else 0.0
        
        # Create signal record
        signal = {
            'symbol': symbol,
            'date': signal_date_str,
            'signal_type': signal_type,
            'confidence': round(float(confidence), 3),
            'strength': strength,
            'price_change': round(float(price_change), 6) if not pd.isna(price_change) else 0.0,
            'volume_change': round(float(volume_change), 6) if not pd.isna(volume_change) else 0.0,
            'close': round(float(latest['Close']), 2),
            'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
            'rsi': round(float(latest['RSI']), 2) if not pd.isna(latest['RSI']) else 50.0,
            'ma_short': round(float(latest['MA_Short']), 2) if not pd.isna(latest['MA_Short']) else latest['Close'],
            'ma_long': round(float(latest['MA_Long']), 2) if not pd.isna(latest['MA_Long']) else latest['Close'],
            'buy_factors': buy_factors,
            'sell_factors': sell_factors,
            'timeframe': self.timeframe,
            'parameters_used': self.parameters.copy()
        }
        
        signals.append(signal)
        
        logger.debug(f"Generated {signal_type} signal for {symbol} with confidence {confidence:.3f}")
        return signals
    
    def get_signal_summary(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of the signals.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Summary dictionary
        """
        if not signals:
            return {
                'total_signals': 0,
                'timeframe': self.timeframe,
                'signal_distribution': {},
                'avg_confidence': 0.0
            }
        
        # Count signals by type
        signal_counts = {}
        confidence_sum = 0.0
        
        for signal in signals:
            signal_type = signal.get('signal_type', 'UNKNOWN')
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            confidence_sum += signal.get('confidence', 0.0)
        
        avg_confidence = confidence_sum / len(signals) if len(signals) > 0 else 0.0
        
        summary = {
            'total_signals': len(signals),
            'timeframe': self.timeframe,
            'signal_distribution': signal_counts,
            'avg_confidence': round(avg_confidence, 3),
            'parameters_used': self.parameters,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return summary

class MultiTimeframeSignalGenerator:
    """
    Generator that can create signals across multiple timeframes simultaneously.
    """
    
    def __init__(self, timeframes: List[str] = None):
        """
        Initialize multi-timeframe signal generator.
        
        Args:
            timeframes: List of timeframes to support (default: all supported)
        """
        if timeframes is None:
            timeframes = ['1min', '5min', '15min', '1hour', '1day']
        
        self.timeframes = timeframes
        self.generators = {}
        
        # Create generators for each timeframe
        for timeframe in timeframes:
            self.generators[timeframe] = TimeframeAgnosticSignalGenerator(timeframe)
        
        logger.info(f"MultiTimeframeSignalGenerator initialized for timeframes: {timeframes}")
    
    def generate_all_timeframes(self, data_by_timeframe: Dict[str, pd.DataFrame], 
                               symbols: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate signals for all timeframes.
        
        Args:
            data_by_timeframe: Dictionary mapping timeframe to OHLCV DataFrame
            symbols: List of symbols (optional)
            
        Returns:
            Dictionary mapping timeframe to list of signals
        """
        all_signals = {}
        
        for timeframe in self.timeframes:
            if timeframe in data_by_timeframe:
                data = data_by_timeframe[timeframe]
                generator = self.generators[timeframe]
                signals = generator.generate_signals(data, symbols)
                all_signals[timeframe] = signals
            else:
                logger.warning(f"No data provided for timeframe {timeframe}")
                all_signals[timeframe] = []
        
        return all_signals
    
    def get_consensus_signals(self, timeframe_signals: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Generate consensus signals by combining signals from multiple timeframes.
        
        Args:
            timeframe_signals: Dictionary of signals by timeframe
            
        Returns:
            List of consensus signals
        """
        # This is a simplified consensus algorithm
        # In practice, you might want more sophisticated logic
        
        consensus_signals = []
        
        # Get all symbols that have signals
        all_symbols = set()
        for signals in timeframe_signals.values():
            for signal in signals:
                all_symbols.add(signal['symbol'])
        
        for symbol in all_symbols:
            # Collect signals for this symbol across timeframes
            symbol_signals = {}
            for timeframe, signals in timeframe_signals.items():
                for signal in signals:
                    if signal['symbol'] == symbol:
                        symbol_signals[timeframe] = signal
                        break
            
            if symbol_signals:
                consensus = self._calculate_consensus(symbol_signals)
                if consensus:
                    consensus_signals.append(consensus)
        
        return consensus_signals
    
    def _calculate_consensus(self, symbol_signals: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Calculate consensus signal for a symbol across timeframes."""
        if not symbol_signals:
            return None
        
        # Weight timeframes (longer timeframes have more weight)
        timeframe_weights = {
            '1day': 1.0,
            '1hour': 0.8,
            '15min': 0.6,
            '5min': 0.4,
            '1min': 0.2
        }
        
        # Count weighted votes
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        total_confidence = 0.0
        total_weight = 0.0
        
        symbol = None
        latest_date = None
        latest_close = None
        
        for timeframe, signal in symbol_signals.items():
            weight = timeframe_weights.get(timeframe, 0.5)
            confidence = signal.get('confidence', 0.5)
            weighted_confidence = weight * confidence
            
            signal_type = signal.get('signal_type', 'HOLD')
            if signal_type == 'BUY':
                buy_weight += weighted_confidence
            elif signal_type == 'SELL':
                sell_weight += weighted_confidence
            else:
                hold_weight += weighted_confidence
            
            total_confidence += confidence
            total_weight += weight
            
            # Store symbol info
            symbol = signal['symbol']
            if latest_date is None or signal.get('date', '') > latest_date:
                latest_date = signal.get('date')
                latest_close = signal.get('close')
        
        # Determine consensus
        max_weight = max(buy_weight, sell_weight, hold_weight)
        if max_weight == buy_weight and buy_weight > 0:
            consensus_type = 'BUY'
        elif max_weight == sell_weight and sell_weight > 0:
            consensus_type = 'SELL'
        else:
            consensus_type = 'HOLD'
        
        avg_confidence = total_confidence / len(symbol_signals) if symbol_signals else 0.0
        
        consensus = {
            'symbol': symbol,
            'date': latest_date,
            'signal_type': consensus_type,
            'confidence': round(avg_confidence, 3),
            'consensus_weight': round(max_weight, 3),
            'close': latest_close,
            'timeframe': 'CONSENSUS',
            'contributing_timeframes': list(symbol_signals.keys()),
            'create_time': datetime.now().isoformat()
        }
        
        return consensus

# Factory functions for backward compatibility
def create_signal_generator(timeframe: str = '1day') -> TimeframeAgnosticSignalGenerator:
    """Create a timeframe-agnostic signal generator."""
    return TimeframeAgnosticSignalGenerator(timeframe)

def create_multi_timeframe_generator(timeframes: List[str] = None) -> MultiTimeframeSignalGenerator:
    """Create a multi-timeframe signal generator."""
    return MultiTimeframeSignalGenerator(timeframes)

# Example usage and testing
if __name__ == "__main__":
    # Test single timeframe generator
    print("=== Testing Single Timeframe Generator ===")
    
    # Test with daily timeframe (backward compatibility)
    daily_generator = create_signal_generator('1day')
    print(f"Daily parameters: {daily_generator.parameters}")
    
    # Test with hourly timeframe
    hourly_generator = create_signal_generator('1hour')
    print(f"Hourly parameters: {hourly_generator.parameters}")
    
    # Test multi-timeframe generator
    print("\n=== Testing Multi-Timeframe Generator ===")
    multi_generator = create_multi_timeframe_generator(['1day', '1hour', '15min'])
    print(f"Initialized for timeframes: {multi_generator.timeframes}")
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-08-01', end='2024-08-31', freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 110, len(dates)),
        'High': np.random.uniform(110, 120, len(dates)),
        'Low': np.random.uniform(90, 100, len(dates)),
        'Close': np.random.uniform(95, 115, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Test signal generation
    print("\n=== Testing Signal Generation ===")
    signals = daily_generator.generate_signals(sample_data, ['TEST_SYMBOL'])
    print(f"Generated {len(signals)} signals")
    
    if signals:
        print(f"Sample signal: {signals[0]}")
        
        # Test signal summary
        summary = daily_generator.get_signal_summary(signals)
        print(f"Signal summary: {summary}")
