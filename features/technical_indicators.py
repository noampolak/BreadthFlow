"""
Technical Indicators Module

Generic, reusable technical analysis indicators that can be used
across multiple experiments and trading strategies.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import talib


class TechnicalIndicators:
    """Generic technical indicators calculator."""

    @staticmethod
    def calculate_rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)

    @staticmethod
    def calculate_macd(prices: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        macd, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return {
            "macd": pd.Series(macd, index=prices.index),
            "signal": pd.Series(signal_line, index=prices.index),
            "histogram": pd.Series(histogram, index=prices.index),
        }

    @staticmethod
    def calculate_bollinger_bands(prices: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2.0) -> dict:
        """Calculate Bollinger Bands."""
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
        return {
            "upper": pd.Series(upper, index=prices.index),
            "middle": pd.Series(middle, index=prices.index),
            "lower": pd.Series(lower, index=prices.index),
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
