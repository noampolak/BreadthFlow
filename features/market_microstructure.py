"""
Market Microstructure Module

Generic, reusable market microstructure indicators that can be used
across multiple experiments and trading strategies.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd


class MarketMicrostructure:
    """Generic market microstructure calculator."""

    @staticmethod
    def calculate_volume_indicators(
        volume: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], period: int = 20
    ) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators."""
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        # Volume moving average
        volume_ma = volume.rolling(window=period).mean()

        # Volume rate of change
        volume_roc = volume.pct_change(periods=period)

        # Price-volume trend
        pvt = (volume * close.pct_change()).cumsum()

        # On-Balance Volume
        obv = np.where(close.diff() > 0, volume, np.where(close.diff() < 0, -volume, 0)).cumsum()
        obv = pd.Series(obv, index=close.index)

        return {"volume_ma": volume_ma, "volume_roc": volume_roc, "pvt": pvt, "obv": obv}

    @staticmethod
    def calculate_price_impact(
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray],
    ) -> Dict[str, pd.Series]:
        """Calculate price impact indicators."""
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Average True Range
        atr = true_range.rolling(window=14).mean()

        # Price impact per volume
        price_impact = true_range / volume
        price_impact = price_impact.replace([np.inf, -np.inf], np.nan)

        return {"true_range": true_range, "atr": atr, "price_impact": price_impact}

    @staticmethod
    def calculate_volatility_indicators(close: Union[pd.Series, np.ndarray], period: int = 20) -> Dict[str, pd.Series]:
        """Calculate volatility indicators."""
        if isinstance(close, np.ndarray):
            close = pd.Series(close)

        # Rolling volatility (standard deviation of returns)
        returns = close.pct_change()
        volatility = returns.rolling(window=period).std()

        # Parkinson volatility (using high-low range)
        # Note: This would need high-low data, simplified here
        parkinson_vol = volatility * np.sqrt(2 * np.log(2))  # Approximation

        # Garman-Klass volatility
        gk_vol = volatility * np.sqrt(0.5)  # Approximation

        return {"volatility": volatility, "parkinson_vol": parkinson_vol, "gk_vol": gk_vol}

    @staticmethod
    def get_all_microstructure(ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all market microstructure indicators."""
        results = pd.DataFrame(index=ohlcv_data.index)

        # Volume indicators
        volume_data = MarketMicrostructure.calculate_volume_indicators(ohlcv_data["Volume"], ohlcv_data["Close"])
        results["volume_ma"] = volume_data["volume_ma"]
        results["volume_roc"] = volume_data["volume_roc"]
        results["pvt"] = volume_data["pvt"]
        results["obv"] = volume_data["obv"]

        # Price impact
        if "High" in ohlcv_data.columns and "Low" in ohlcv_data.columns:
            price_impact_data = MarketMicrostructure.calculate_price_impact(
                ohlcv_data["High"], ohlcv_data["Low"], ohlcv_data["Close"], ohlcv_data["Volume"]
            )
            results["true_range"] = price_impact_data["true_range"]
            results["atr"] = price_impact_data["atr"]
            results["price_impact"] = price_impact_data["price_impact"]

        # Volatility indicators
        volatility_data = MarketMicrostructure.calculate_volatility_indicators(ohlcv_data["Close"])
        results["volatility"] = volatility_data["volatility"]
        results["parkinson_vol"] = volatility_data["parkinson_vol"]
        results["gk_vol"] = volatility_data["gk_vol"]

        return results
