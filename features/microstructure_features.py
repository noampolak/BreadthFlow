"""
Market Microstructure Features

Extract features related to market microstructure, order flow,
and trading patterns for financial machine learning.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MicrostructureFeatures:
    """
    Market microstructure feature engineering for financial time series.

    Features:
    - Volume-based features (volume patterns, VWAP, volume ratios)
    - Price-volume relationships (price impact, volume-weighted features)
    - Order flow indicators (bid-ask spread proxies, market depth)
    - Trading intensity and patterns
    - Market quality indicators
    """

    def __init__(self):
        """Initialize microstructure features calculator."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("MicrostructureFeatures initialized")

    def calculate_all_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all microstructure features for a DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all microstructure features added
        """
        try:
            result_df = df.copy()

            # Volume-based features
            result_df = self._add_volume_features(result_df)

            # Price-volume relationship features
            result_df = self._add_price_volume_features(result_df)

            # Order flow indicators
            result_df = self._add_order_flow_features(result_df)

            # Trading intensity features
            result_df = self._add_trading_intensity_features(result_df)

            # Market quality indicators
            result_df = self._add_market_quality_features(result_df)

            # Liquidity indicators
            result_df = self._add_liquidity_features(result_df)

            self.logger.info(f"Calculated microstructure features for {len(result_df)} records")
            return result_df

        except Exception as e:
            self.logger.error(f"Error calculating microstructure features: {str(e)}")
            return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        try:
            # Volume moving averages
            for period in [5, 10, 20, 50]:
                df[f"volume_sma_{period}"] = df["volume"].rolling(window=period).mean()
                df[f"volume_ema_{period}"] = df["volume"].ewm(span=period).mean()

            # Volume ratios
            for period in [5, 10, 20]:
                df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_sma_{period}"]
                df[f"volume_zscore_{period}"] = (df["volume"] - df[f"volume_sma_{period}"]) / df["volume"].rolling(
                    window=period
                ).std()

            # Volume momentum
            for period in [1, 5, 10]:
                df[f"volume_momentum_{period}"] = df["volume"].pct_change(period)

            # Volume volatility
            for period in [5, 10, 20]:
                df[f"volume_volatility_{period}"] = df["volume"].rolling(window=period).std()
                df[f"volume_cv_{period}"] = df[f"volume_volatility_{period}"] / df[f"volume_sma_{period}"]

            # Volume percentiles
            for period in [20, 50]:
                df[f"volume_percentile_{period}"] = df["volume"].rolling(window=period).rank(pct=True)

            # Volume acceleration
            df["volume_acceleration"] = df["volume"].diff().diff()

            return df

        except Exception as e:
            self.logger.error(f"Error adding volume features: {str(e)}")
            return df

    def _add_price_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-volume relationship features."""
        try:
            # Volume Weighted Average Price (VWAP)
            df["vwap"] = self._calculate_vwap(df)

            # Price-VWAP relationships
            df["price_vs_vwap"] = (df["close"] / df["vwap"] - 1) * 100
            df["price_vs_vwap_zscore"] = (df["price_vs_vwap"] - df["price_vs_vwap"].rolling(window=20).mean()) / df[
                "price_vs_vwap"
            ].rolling(window=20).std()

            # Volume-weighted price features
            df["volume_weighted_price"] = (df["close"] * df["volume"]).rolling(window=20).sum() / df["volume"].rolling(
                window=20
            ).sum()
            df["price_volume_trend"] = ((df["close"] - df["close"].shift(1)) * df["volume"]).rolling(window=20).sum()

            # Price impact indicators
            df["price_impact"] = df["close"].pct_change() / df["volume"].pct_change()
            df["price_impact_ma"] = df["price_impact"].rolling(window=20).mean()

            # Volume-price correlation
            for period in [5, 10, 20]:
                df[f"volume_price_corr_{period}"] = df["close"].rolling(window=period).corr(df["volume"])

            # Volume-weighted returns
            df["volume_weighted_return"] = (df["close"].pct_change() * df["volume"]).rolling(window=20).sum() / df[
                "volume"
            ].rolling(window=20).sum()

            # Price-volume divergence
            price_trend = df["close"].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
            volume_trend = df["volume"].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
            df["price_volume_divergence"] = np.sign(price_trend) != np.sign(volume_trend)

            return df

        except Exception as e:
            self.logger.error(f"Error adding price-volume features: {str(e)}")
            return df

    def _add_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add order flow indicators."""
        try:
            # Bid-ask spread proxies (using high-low as proxy)
            df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
            df["spread_proxy_ma"] = df["spread_proxy"].rolling(window=20).mean()
            df["spread_proxy_zscore"] = (df["spread_proxy"] - df["spread_proxy_ma"]) / df["spread_proxy"].rolling(
                window=20
            ).std()

            # Price range features
            df["price_range"] = df["high"] - df["low"]
            df["price_range_pct"] = df["price_range"] / df["close"]
            df["price_range_ma"] = df["price_range_pct"].rolling(window=20).mean()

            # Intraday price patterns
            df["open_close_range"] = abs(df["open"] - df["close"]) / df["close"]
            df["high_close_range"] = (df["high"] - df["close"]) / df["close"]
            df["low_close_range"] = (df["close"] - df["low"]) / df["close"]

            # Price efficiency (how close close is to high/low)
            df["price_efficiency"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
            df["price_efficiency_ma"] = df["price_efficiency"].rolling(window=20).mean()

            # Order flow imbalance (simplified)
            df["order_flow_imbalance"] = (df["close"] - df["open"]) / (df["high"] - df["low"])
            df["order_flow_imbalance"] = df["order_flow_imbalance"].fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Error adding order flow features: {str(e)}")
            return df

    def _add_trading_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trading intensity and pattern features."""
        try:
            # Trading intensity (volume per unit time)
            df["trading_intensity"] = df["volume"] / df["volume"].rolling(window=20).mean()

            # Volume bursts
            volume_threshold = df["volume"].rolling(window=20).quantile(0.8)
            df["volume_burst"] = df["volume"] > volume_threshold
            df["volume_burst_intensity"] = df["volume"] / volume_threshold

            # Volume clustering
            df["volume_clustering"] = df["volume_burst"].rolling(window=5).sum()

            # Volume persistence
            df["volume_persistence"] = (
                df["volume_burst"].rolling(window=10).apply(lambda x: self._longest_consecutive(x), raw=True)
            )

            # Volume acceleration patterns
            df["volume_acceleration_ma"] = df["volume_acceleration"].rolling(window=5).mean()
            df["volume_acceleration_std"] = df["volume_acceleration"].rolling(window=10).std()

            # Volume momentum indicators
            df["volume_momentum_ma"] = df["volume_momentum_1"].rolling(window=5).mean()
            df["volume_momentum_std"] = df["volume_momentum_1"].rolling(window=10).std()

            # Volume regime indicators
            volume_ma_short = df["volume"].rolling(window=5).mean()
            volume_ma_long = df["volume"].rolling(window=20).mean()
            df["volume_regime"] = (volume_ma_short > volume_ma_long).astype(int)

            return df

        except Exception as e:
            self.logger.error(f"Error adding trading intensity features: {str(e)}")
            return df

    def _add_market_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market quality indicators."""
        try:
            # Price stability indicators
            df["price_stability"] = 1 / (df["close"].pct_change().abs().rolling(window=20).mean() + 1e-8)
            df["price_consistency"] = 1 / (df["close"].pct_change().rolling(window=20).std() + 1e-8)

            # Volume stability
            df["volume_stability"] = 1 / (df["volume"].pct_change().abs().rolling(window=20).mean() + 1e-8)
            df["volume_consistency"] = 1 / (df["volume"].pct_change().rolling(window=20).std() + 1e-8)

            # Market efficiency indicators
            df["price_efficiency_ratio"] = abs(df["close"].pct_change()) / (df["high"] - df["low"]) / df["close"]
            df["price_efficiency_ratio"] = df["price_efficiency_ratio"].fillna(0)

            # Volume efficiency
            df["volume_efficiency"] = df["volume"] / (df["high"] - df["low"]) / df["close"]
            df["volume_efficiency"] = df["volume_efficiency"].fillna(0)

            # Market depth proxies
            df["market_depth_proxy"] = df["volume"] / df["spread_proxy"]
            df["market_depth_proxy"] = df["market_depth_proxy"].fillna(0)

            # Price impact measures
            df["price_impact_measure"] = abs(df["close"].pct_change()) / (
                df["volume"] / df["volume"].rolling(window=20).mean()
            )
            df["price_impact_measure"] = df["price_impact_measure"].fillna(0)

            return df

        except Exception as e:
            self.logger.error(f"Error adding market quality features: {str(e)}")
            return df

    def _add_liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity indicators."""
        try:
            # Liquidity ratios
            df["liquidity_ratio"] = df["volume"] / df["spread_proxy"]
            df["liquidity_ratio"] = df["liquidity_ratio"].fillna(0)

            # Amihud illiquidity measure (simplified)
            returns = df["close"].pct_change().abs()
            volume = df["volume"]
            df["amihud_illiquidity"] = returns / volume
            df["amihud_illiquidity"] = df["amihud_illiquidity"].fillna(0)
            df["amihud_illiquidity_ma"] = df["amihud_illiquidity"].rolling(window=20).mean()

            # Volume-weighted liquidity
            df["volume_weighted_liquidity"] = df["liquidity_ratio"] * df["volume"]
            df["volume_weighted_liquidity_ma"] = df["volume_weighted_liquidity"].rolling(window=20).mean()

            # Liquidity persistence
            liquidity_threshold = df["liquidity_ratio"].rolling(window=20).quantile(0.5)
            df["high_liquidity"] = df["liquidity_ratio"] > liquidity_threshold
            df["liquidity_persistence"] = df["high_liquidity"].rolling(window=5).sum()

            # Liquidity volatility
            df["liquidity_volatility"] = df["liquidity_ratio"].rolling(window=20).std()

            return df

        except Exception as e:
            self.logger.error(f"Error adding liquidity features: {str(e)}")
            return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return vwap

    def _longest_consecutive(self, arr: np.ndarray) -> int:
        """Find the longest consecutive sequence of True values."""
        if len(arr) == 0:
            return 0

        max_count = 0
        current_count = 0

        for val in arr:
            if val:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def get_microstructure_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Get summary of microstructure features in the dataset.

        Args:
            df: DataFrame with microstructure features

        Returns:
            Dictionary with microstructure feature summary
        """
        try:
            summary = {
                "volume_features": {
                    "volume_mean": df["volume"].mean() if "volume" in df.columns else 0,
                    "volume_std": df["volume"].std() if "volume" in df.columns else 0,
                    "volume_skewness": df["volume"].skew() if "volume" in df.columns else 0,
                    "volume_kurtosis": df["volume"].kurtosis() if "volume" in df.columns else 0,
                },
                "price_volume_relationship": {
                    "price_volume_correlation": df["close"].corr(df["volume"])
                    if "close" in df.columns and "volume" in df.columns
                    else 0,
                    "vwap_deviation_mean": df["price_vs_vwap"].mean() if "price_vs_vwap" in df.columns else 0,
                    "vwap_deviation_std": df["price_vs_vwap"].std() if "price_vs_vwap" in df.columns else 0,
                },
                "market_quality": {
                    "spread_proxy_mean": df["spread_proxy"].mean() if "spread_proxy" in df.columns else 0,
                    "price_stability_mean": df["price_stability"].mean() if "price_stability" in df.columns else 0,
                    "volume_stability_mean": df["volume_stability"].mean() if "volume_stability" in df.columns else 0,
                },
                "liquidity": {
                    "liquidity_ratio_mean": df["liquidity_ratio"].mean() if "liquidity_ratio" in df.columns else 0,
                    "amihud_illiquidity_mean": df["amihud_illiquidity"].mean() if "amihud_illiquidity" in df.columns else 0,
                },
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting microstructure summary: {str(e)}")
            return {"error": str(e)}
