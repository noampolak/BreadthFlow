"""
Technical Indicators Module

Generic, reusable technical analysis indicators that can be used
across multiple experiments and trading strategies.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

# import talib  # Temporarily disabled for easier deployment


class TechnicalIndicators:
    """Generic technical indicators calculator."""

    @staticmethod
    def calculate_rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_macd(prices: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        # Calculate MACD line
        macd = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        signal_line = macd.ewm(span=signal).mean()

        # Calculate histogram
        histogram = macd - signal_line

        return {
            "macd": macd,
            "signal": signal_line,
            "histogram": histogram,
        }

    @staticmethod
    def calculate_bollinger_bands(prices: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2.0) -> dict:
        """Calculate Bollinger Bands."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        # Calculate middle band (SMA)
        middle = prices.rolling(window=period).mean()

        # Calculate standard deviation
        std = prices.rolling(window=period).std()

        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }

    @staticmethod
    def get_all_indicators(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for OHLCV dataset."""
        results = pd.DataFrame(index=ohlcv_data.index)

        # RSI
        results["rsi"] = TechnicalIndicators.calculate_rsi(ohlcv_data["Close"])

        # MACD
        macd_data = TechnicalIndicators.calculate_macd(ohlcv_data["Close"])
        results["macd"] = macd_data["macd"]
        results["macd_signal"] = macd_data["signal"]
        results["macd_histogram"] = macd_data["histogram"]

        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(ohlcv_data["Close"])
        results["bb_upper"] = bb_data["upper"]
        results["bb_middle"] = bb_data["middle"]
        results["bb_lower"] = bb_data["lower"]
        results["bb_width"] = (bb_data["upper"] - bb_data["lower"]) / bb_data["middle"]
        results["bb_position"] = (ohlcv_data["Close"] - bb_data["lower"]) / (bb_data["upper"] - bb_data["lower"])

        return results
