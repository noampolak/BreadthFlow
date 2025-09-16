"""
Technical Indicators Component

Provides technical analysis indicators for signal generation.
"""

from typing import Any, Dict, List, Optional

import numpy as np

# Optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical analysis indicators for signal generation"""

    def __init__(self):
        self.indicators = {
            "sma": self.simple_moving_average,
            "ema": self.exponential_moving_average,
            "rsi": self.relative_strength_index,
            "macd": self.moving_average_convergence_divergence,
            "bollinger_bands": self.bollinger_bands,
            "stochastic": self.stochastic_oscillator,
            "williams_r": self.williams_percent_r,
            "cci": self.commodity_channel_index,
            "adx": self.average_directional_index,
            "atr": self.average_true_range,
        }

    def calculate_indicator(self, indicator_name: str, data, **kwargs):
        """Calculate a specific technical indicator"""
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return self.indicators[indicator_name](data, **kwargs)

    def simple_moving_average(self, data, period: int = 20, column: str = "close"):
        """Calculate Simple Moving Average"""
        return data[column].rolling(window=period).mean()

    def exponential_moving_average(self, data, period: int = 20, column: str = "close"):
        """Calculate Exponential Moving Average"""
        return data[column].ewm(span=period).mean()

    def relative_strength_index(self, data, period: int = 14, column: str = "close"):
        """Calculate Relative Strength Index"""
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def moving_average_convergence_divergence(
        self, data, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = "close"
    ):
        """Calculate MACD"""
        ema_fast = data[column].ewm(span=fast_period).mean()
        ema_slow = data[column].ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def bollinger_bands(self, data, period: int = 20, std_dev: float = 2, column: str = "close"):
        """Calculate Bollinger Bands"""
        sma = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {"upper": upper_band, "middle": sma, "lower": lower_band}

    def stochastic_oscillator(self, data, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        low_min = data["low"].rolling(window=k_period).min()
        high_max = data["high"].rolling(window=k_period).max()
        k_percent = 100 * ((data["close"] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {"k_percent": k_percent, "d_percent": d_percent}

    def williams_percent_r(self, data, period: int = 14):
        """Calculate Williams %R"""
        high_max = data["high"].rolling(window=period).max()
        low_min = data["low"].rolling(window=period).min()
        williams_r = -100 * ((high_max - data["close"]) / (high_max - low_min))
        return williams_r

    def commodity_channel_index(self, data, period: int = 20):
        """Calculate Commodity Channel Index"""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci

    def average_directional_index(self, data, period: int = 14):
        """Calculate Average Directional Index"""
        high_diff = data["high"].diff()
        low_diff = data["low"].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        tr = np.maximum(
            data["high"] - data["low"],
            np.maximum(np.abs(data["high"] - data["close"].shift(1)), np.abs(data["low"] - data["close"].shift(1))),
        )

        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean())
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / pd.Series(tr).rolling(window=period).mean())

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period).mean()

        return {"plus_di": plus_di, "minus_di": minus_di, "adx": adx}

    def average_true_range(self, data, period: int = 14):
        """Calculate Average True Range"""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift(1))
        low_close = np.abs(data["low"] - data["close"].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = pd.Series(true_range).rolling(window=period).mean()

        return atr

    def generate_technical_signals(self, data, indicators: List[str], parameters: Dict[str, Any] = None):
        """Generate technical analysis signals"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for technical indicators but not available")

        if parameters is None:
            parameters = {}

        signals = data.copy()

        for indicator in indicators:
            try:
                if indicator == "macd":
                    macd_data = self.moving_average_convergence_divergence(data, **parameters.get("macd", {}))
                    signals["macd_line"] = macd_data["macd"]
                    signals["macd_signal"] = macd_data["signal"]
                    signals["macd_histogram"] = macd_data["histogram"]

                elif indicator == "bollinger_bands":
                    bb_data = self.bollinger_bands(data, **parameters.get("bollinger_bands", {}))
                    signals["bb_upper"] = bb_data["upper"]
                    signals["bb_middle"] = bb_data["middle"]
                    signals["bb_lower"] = bb_data["lower"]

                elif indicator == "stochastic":
                    stoch_data = self.stochastic_oscillator(data, **parameters.get("stochastic", {}))
                    signals["stoch_k"] = stoch_data["k_percent"]
                    signals["stoch_d"] = stoch_data["d_percent"]

                elif indicator == "adx":
                    adx_data = self.average_directional_index(data, **parameters.get("adx", {}))
                    signals["adx"] = adx_data["adx"]
                    signals["plus_di"] = adx_data["plus_di"]
                    signals["minus_di"] = adx_data["minus_di"]

                else:
                    # Single value indicators
                    indicator_data = self.calculate_indicator(indicator, data, **parameters.get(indicator, {}))
                    signals[indicator] = indicator_data

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
                continue

        return signals

    def get_signal_strength(self, data, buy_conditions: List[str], sell_conditions: List[str]):
        """Calculate signal strength based on technical conditions"""
        signal_strength = pd.Series(0.0, index=data.index)

        # Buy signals
        for condition in buy_conditions:
            if condition in data.columns:
                signal_strength += np.where(data[condition] > 0, 1, 0)

        # Sell signals
        for condition in sell_conditions:
            if condition in data.columns:
                signal_strength -= np.where(data[condition] > 0, 1, 0)

        return signal_strength

    # Test compatibility methods
    def calculate_rsi(self, data, period: int = 14):
        """Calculate RSI - test compatibility method"""
        if hasattr(data, 'name'):  # It's a Series
            # Convert Series to DataFrame for the method
            df = pd.DataFrame({'close': data})
            return self.relative_strength_index(df, period=period)
        else:  # It's a DataFrame
            return self.relative_strength_index(data, period=period)

    def calculate_macd(self, data, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Calculate MACD - test compatibility method"""
        if hasattr(data, 'name'):  # It's a Series
            # Convert Series to DataFrame for the method
            df = pd.DataFrame({'close': data})
            result = self.moving_average_convergence_divergence(df, fast_period, slow_period, signal_period)
            return result["macd"], result["signal"], result["histogram"]
        else:  # It's a DataFrame
            result = self.moving_average_convergence_divergence(data, fast_period, slow_period, signal_period)
            return result["macd"], result["signal"], result["histogram"]

    def calculate_bollinger_bands(self, data, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands - test compatibility method"""
        if hasattr(data, 'name'):  # It's a Series
            # Convert Series to DataFrame for the method
            df = pd.DataFrame({'close': data})
            result = self.bollinger_bands(df, period=period, std_dev=std_dev)
            return result["upper"], result["middle"], result["lower"]
        else:  # It's a DataFrame
            result = self.bollinger_bands(data, period=period, std_dev=std_dev)
            return result["upper"], result["middle"], result["lower"]

    def calculate_sma(self, data, period: int = 20):
        """Calculate SMA - test compatibility method"""
        if hasattr(data, 'name'):  # It's a Series
            # Convert Series to DataFrame for the method
            df = pd.DataFrame({'close': data})
            return self.simple_moving_average(df, period=period)
        else:  # It's a DataFrame
            return self.simple_moving_average(data, period=period)
