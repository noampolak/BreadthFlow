"""
Time Features Module

Generic, reusable time-based feature engineering that can be used
across multiple experiments and trading strategies.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd


class TimeFeatures:
    """Generic time-based feature calculator."""

    @staticmethod
    def extract_time_features(datetime_index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Extract time-based features from datetime index."""
        features = {}

        # Basic time features
        features["hour"] = datetime_index.hour
        features["day_of_week"] = datetime_index.dayofweek
        features["day_of_month"] = datetime_index.day
        features["month"] = datetime_index.month
        features["quarter"] = datetime_index.quarter
        features["year"] = datetime_index.year

        # Cyclical encoding for time features
        features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
        features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
        features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
        features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)

        # Business day features
        features["is_weekend"] = (datetime_index.dayofweek >= 5).astype(int)
        features["is_month_start"] = datetime_index.is_month_start.astype(int)
        features["is_month_end"] = datetime_index.is_month_end.astype(int)
        features["is_quarter_start"] = datetime_index.is_quarter_start.astype(int)
        features["is_quarter_end"] = datetime_index.is_quarter_end.astype(int)
        features["is_year_start"] = datetime_index.is_year_start.astype(int)
        features["is_year_end"] = datetime_index.is_year_end.astype(int)

        # Convert to Series with proper index
        for key, value in features.items():
            features[key] = pd.Series(value, index=datetime_index)

        return features

    @staticmethod
    def calculate_seasonality_features(datetime_index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Calculate seasonality features."""
        features = {}

        # Day of year
        features["day_of_year"] = datetime_index.dayofyear

        # Week of year
        features["week_of_year"] = datetime_index.isocalendar().week

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        month = datetime_index.month
        season = np.where(month.isin([12, 1, 2]), 1, np.where(month.isin([3, 4, 5]), 2, np.where(month.isin([6, 7, 8]), 3, 4)))
        features["season"] = pd.Series(season, index=datetime_index)

        # Cyclical encoding for seasonality
        features["day_of_year_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365)
        features["day_of_year_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365)
        features["week_of_year_sin"] = np.sin(2 * np.pi * features["week_of_year"] / 52)
        features["week_of_year_cos"] = np.cos(2 * np.pi * features["week_of_year"] / 52)

        return features

    @staticmethod
    def calculate_trading_session_features(datetime_index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
        """Calculate trading session features."""
        features = {}

        # Market session indicators (assuming US market hours)
        hour = datetime_index.hour
        features["is_premarket"] = ((hour >= 4) & (hour < 9)).astype(int)
        features["is_regular_hours"] = ((hour >= 9) & (hour < 16)).astype(int)
        features["is_afterhours"] = ((hour >= 16) & (hour < 20)).astype(int)
        features["is_overnight"] = ((hour >= 20) | (hour < 4)).astype(int)

        # Market open/close proximity
        features["minutes_to_open"] = np.where(
            hour < 9,
            (9 - hour) * 60 - datetime_index.minute,
            np.where(hour >= 16, (24 - hour + 9) * 60 - datetime_index.minute, 0),
        )
        features["minutes_to_close"] = np.where(hour < 16, (16 - hour) * 60 - datetime_index.minute, 0)

        # Convert to Series
        for key, value in features.items():
            features[key] = pd.Series(value, index=datetime_index)

        return features

    @staticmethod
    def get_all_time_features(datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate all time-based features."""
        results = pd.DataFrame(index=datetime_index)

        # Basic time features
        time_features = TimeFeatures.extract_time_features(datetime_index)
        for key, value in time_features.items():
            results[key] = value

        # Seasonality features
        seasonality_features = TimeFeatures.calculate_seasonality_features(datetime_index)
        for key, value in seasonality_features.items():
            results[key] = value

        # Trading session features
        session_features = TimeFeatures.calculate_trading_session_features(datetime_index)
        for key, value in session_features.items():
            results[key] = value

        return results
