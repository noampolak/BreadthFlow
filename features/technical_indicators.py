"""
Technical Indicators for Financial Time Series

Comprehensive collection of technical indicators commonly used
in financial analysis and machine learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from scipy import stats

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Technical indicators calculator for financial time series data.
    
    Features:
    - Moving averages (SMA, EMA, WMA)
    - Momentum indicators (RSI, MACD, Stochastic)
    - Volatility indicators (Bollinger Bands, ATR)
    - Trend indicators (ADX, Parabolic SAR)
    - Volume indicators (OBV, VWAP)
    """
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("TechnicalIndicators initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with all technical indicators added
        """
        try:
            result_df = df.copy()
            
            # Moving averages
            result_df = self._add_moving_averages(result_df)
            
            # Momentum indicators
            result_df = self._add_momentum_indicators(result_df)
            
            # Volatility indicators
            result_df = self._add_volatility_indicators(result_df)
            
            # Trend indicators
            result_df = self._add_trend_indicators(result_df)
            
            # Volume indicators
            result_df = self._add_volume_indicators(result_df)
            
            # Price patterns
            result_df = self._add_price_patterns(result_df)
            
            self.logger.info(f"Calculated technical indicators for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages."""
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential Moving Averages
            for period in [5, 10, 20, 50, 100]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # Weighted Moving Average
            for period in [10, 20]:
                weights = np.arange(1, period + 1)
                df[f'wma_{period}'] = df['close'].rolling(window=period).apply(
                    lambda x: np.average(x, weights=weights), raw=True
                )
            
            # Hull Moving Average
            df['hma_9'] = self._hull_moving_average(df['close'], 9)
            df['hma_21'] = self._hull_moving_average(df['close'], 21)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding moving averages: {str(e)}")
            return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        try:
            # RSI (Relative Strength Index)
            df['rsi_14'] = self._rsi(df['close'], 14)
            df['rsi_21'] = self._rsi(df['close'], 21)
            
            # MACD (Moving Average Convergence Divergence)
            macd_line, signal_line, histogram = self._macd(df['close'])
            df['macd'] = macd_line
            df['macd_signal'] = signal_line
            df['macd_histogram'] = histogram
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self._stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Williams %R
            df['williams_r'] = self._williams_r(df['high'], df['low'], df['close'])
            
            # Rate of Change
            for period in [5, 10, 20]:
                df[f'roc_{period}'] = df['close'].pct_change(period) * 100
            
            # Momentum
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding momentum indicators: {str(e)}")
            return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._bollinger_bands(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Average True Range
            df['atr_14'] = self._atr(df['high'], df['low'], df['close'], 14)
            df['atr_21'] = self._atr(df['high'], df['low'], df['close'], 21)
            
            # Volatility (standard deviation)
            for period in [10, 20, 30]:
                df[f'volatility_{period}'] = df['close'].rolling(window=period).std()
            
            # Historical Volatility
            for period in [10, 20, 30]:
                returns = df['close'].pct_change()
                df[f'hist_vol_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volatility indicators: {str(e)}")
            return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        try:
            # ADX (Average Directional Index)
            df['adx_14'] = self._adx(df['high'], df['low'], df['close'], 14)
            
            # Parabolic SAR
            df['psar'] = self._parabolic_sar(df['high'], df['low'], df['close'])
            
            # Ichimoku Cloud
            ichimoku = self._ichimoku_cloud(df['high'], df['low'], df['close'])
            df['tenkan_sen'] = ichimoku['tenkan_sen']
            df['kijun_sen'] = ichimoku['kijun_sen']
            df['senkou_span_a'] = ichimoku['senkou_span_a']
            df['senkou_span_b'] = ichimoku['senkou_span_b']
            df['chikou_span'] = ichimoku['chikou_span']
            
            # Aroon Oscillator
            aroon_up, aroon_down = self._aroon(df['high'], df['low'], 14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down
            df['aroon_oscillator'] = aroon_up - aroon_down
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding trend indicators: {str(e)}")
            return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        try:
            # On-Balance Volume
            df['obv'] = self._obv(df['close'], df['volume'])
            
            # Volume Weighted Average Price
            df['vwap'] = self._vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # Accumulation/Distribution Line
            df['ad_line'] = self._ad_line(df['high'], df['low'], df['close'], df['volume'])
            
            # Money Flow Index
            df['mfi_14'] = self._mfi(df['high'], df['low'], df['close'], df['volume'], 14)
            
            # Volume Rate of Change
            for period in [5, 10, 20]:
                df[f'volume_roc_{period}'] = df['volume'].pct_change(period) * 100
            
            # Volume Moving Averages
            for period in [5, 10, 20]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volume indicators: {str(e)}")
            return df
    
    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern indicators."""
        try:
            # Price position relative to moving averages
            for period in [20, 50, 200]:
                df[f'price_vs_sma_{period}'] = (df['close'] / df[f'sma_{period}'] - 1) * 100
            
            # Price channels
            df['price_channel_high'] = df['high'].rolling(window=20).max()
            df['price_channel_low'] = df['low'].rolling(window=20).min()
            df['price_channel_position'] = (df['close'] - df['price_channel_low']) / (df['price_channel_high'] - df['price_channel_low'])
            
            # Support and resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            
            # Gap analysis
            df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
            df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding price patterns: {str(e)}")
            return df
    
    # Individual indicator calculation methods
    
    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = minus_dm.abs()
        
        tr = self._atr(high, low, close, period)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                      af: float = 0.02, max_af: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR."""
        psar = pd.Series(index=close.index, dtype=float)
        psar.iloc[0] = low.iloc[0]
        
        uptrend = True
        af_current = af
        ep = high.iloc[0]
        
        for i in range(1, len(close)):
            if uptrend:
                psar.iloc[i] = psar.iloc[i-1] + af_current * (ep - psar.iloc[i-1])
                if low.iloc[i] <= psar.iloc[i]:
                    uptrend = False
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    af_current = af
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af_current = min(af_current + af, max_af)
            else:
                psar.iloc[i] = psar.iloc[i-1] + af_current * (ep - psar.iloc[i-1])
                if high.iloc[i] >= psar.iloc[i]:
                    uptrend = True
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    af_current = af
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af_current = min(af_current + af, max_af)
        
        return psar
    
    def _ichimoku_cloud(self, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
        """Calculate Ichimoku Cloud indicators."""
        tenkan_period = 9
        kijun_period = 26
        senkou_span_b_period = 52
        
        tenkan_sen = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
        kijun_sen = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
        senkou_span_b = ((high.rolling(window=senkou_span_b_period).max() + 
                         low.rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
        chikou_span = close.shift(-kijun_period)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def _aroon(self, high: pd.Series, low: pd.Series, period: int = 14) -> tuple:
        """Calculate Aroon indicators."""
        aroon_up = high.rolling(window=period).apply(lambda x: (period - x.argmax()) / period * 100, raw=True)
        aroon_down = low.rolling(window=period).apply(lambda x: (period - x.argmin()) / period * 100, raw=True)
        return aroon_up, aroon_down
    
    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _vwap(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    def _ad_line(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad_line = (clv * volume).cumsum()
        return ad_line
    
    def _mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _hull_moving_average(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average."""
        wma_half = prices.rolling(window=period//2).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
        wma_full = prices.rolling(window=period).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
        hma = (2 * wma_half - wma_full).rolling(window=int(np.sqrt(period))).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)
        return hma
