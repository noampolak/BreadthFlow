"""
Time-based Features for Financial Time Series

Extract temporal patterns and cyclical features from timestamps
for financial machine learning models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)


class TimeFeatures:
    """
    Time-based feature engineering for financial time series.
    
    Features:
    - Cyclical time features (hour, day, week, month, year)
    - Business calendar features (trading days, holidays)
    - Seasonal patterns and trends
    - Time-based aggregations
    - Market session indicators
    """
    
    def __init__(self):
        """Initialize time features calculator."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("TimeFeatures initialized")
    
    def calculate_all_time_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate all time-based features for a DataFrame.
        
        Args:
            df: DataFrame with timestamp data
            date_column: Name of the date/timestamp column
            
        Returns:
            DataFrame with all time features added
        """
        try:
            result_df = df.copy()
            
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
                result_df[date_column] = pd.to_datetime(result_df[date_column])
            
            # Basic time features
            result_df = self._add_basic_time_features(result_df, date_column)
            
            # Cyclical time features
            result_df = self._add_cyclical_features(result_df, date_column)
            
            # Business calendar features
            result_df = self._add_business_calendar_features(result_df, date_column)
            
            # Seasonal features
            result_df = self._add_seasonal_features(result_df, date_column)
            
            # Market session features
            result_df = self._add_market_session_features(result_df, date_column)
            
            # Time-based aggregations
            result_df = self._add_time_aggregations(result_df, date_column)
            
            self.logger.info(f"Calculated time features for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating time features: {str(e)}")
            return df
    
    def _add_basic_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add basic time features."""
        try:
            dt = df[date_column]
            
            # Basic time components
            df['year'] = dt.dt.year
            df['month'] = dt.dt.month
            df['day'] = dt.dt.day
            df['day_of_year'] = dt.dt.dayofyear
            df['week_of_year'] = dt.dt.isocalendar().week
            df['day_of_week'] = dt.dt.dayofweek  # Monday=0, Sunday=6
            df['hour'] = dt.dt.hour
            df['minute'] = dt.dt.minute
            
            # Quarter information
            df['quarter'] = dt.dt.quarter
            df['is_quarter_start'] = dt.dt.is_quarter_start
            df['is_quarter_end'] = dt.dt.is_quarter_end
            
            # Month information
            df['is_month_start'] = dt.dt.is_month_start
            df['is_month_end'] = dt.dt.is_month_end
            
            # Week information
            df['is_week_start'] = dt.dt.dayofweek == 0  # Monday
            df['is_week_end'] = dt.dt.dayofweek == 6    # Sunday
            
            # Weekend indicator
            df['is_weekend'] = dt.dt.dayofweek >= 5
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding basic time features: {str(e)}")
            return df
    
    def _add_cyclical_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add cyclical time features using sine/cosine encoding."""
        try:
            dt = df[date_column]
            
            # Hour cyclical features (24-hour cycle)
            df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
            
            # Day of week cyclical features (7-day cycle)
            df['day_of_week_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
            
            # Day of year cyclical features (365-day cycle)
            df['day_of_year_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
            df['day_of_year_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
            
            # Month cyclical features (12-month cycle)
            df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)
            
            # Week of year cyclical features (52-week cycle)
            df['week_of_year_sin'] = np.sin(2 * np.pi * dt.dt.isocalendar().week / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * dt.dt.isocalendar().week / 52)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding cyclical features: {str(e)}")
            return df
    
    def _add_business_calendar_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add business calendar features."""
        try:
            dt = df[date_column]
            
            # Business day indicators
            df['is_business_day'] = dt.dt.weekday < 5  # Monday-Friday
            
            # Days since/until business events
            df['days_since_month_start'] = (dt - dt.dt.to_period('M').dt.start_time).dt.days
            df['days_until_month_end'] = (dt.dt.to_period('M').dt.end_time - dt).dt.days
            df['days_since_quarter_start'] = (dt - dt.dt.to_period('Q').dt.start_time).dt.days
            df['days_until_quarter_end'] = (dt.dt.to_period('Q').dt.end_time - dt).dt.days
            
            # Business day of month
            business_days = dt[dt.dt.weekday < 5]
            if len(business_days) > 0:
                df['business_day_of_month'] = business_days.groupby(business_days.dt.to_period('M')).cumcount() + 1
                df['business_day_of_month'] = df['business_day_of_month'].fillna(0)
            else:
                df['business_day_of_month'] = 0
            
            # Month-end proximity (for month-end effects)
            month_end = dt.dt.to_period('M').dt.end_time
            df['days_to_month_end'] = (month_end - dt).dt.days
            df['is_month_end_week'] = df['days_to_month_end'] <= 5
            
            # Quarter-end proximity
            quarter_end = dt.dt.to_period('Q').dt.end_time
            df['days_to_quarter_end'] = (quarter_end - dt).dt.days
            df['is_quarter_end_week'] = df['days_to_quarter_end'] <= 5
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding business calendar features: {str(e)}")
            return df
    
    def _add_seasonal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add seasonal pattern features."""
        try:
            dt = df[date_column]
            
            # Season indicators
            df['is_spring'] = dt.dt.month.isin([3, 4, 5])
            df['is_summer'] = dt.dt.month.isin([6, 7, 8])
            df['is_autumn'] = dt.dt.month.isin([9, 10, 11])
            df['is_winter'] = dt.dt.month.isin([12, 1, 2])
            
            # Holiday proximity (simplified)
            df['is_holiday_season'] = dt.dt.month.isin([11, 12])  # November-December
            df['is_earnings_season'] = dt.dt.month.isin([1, 4, 7, 10])  # Quarterly earnings
            
            # Tax season effects
            df['is_tax_season'] = dt.dt.month.isin([3, 4])  # March-April
            
            # Summer vacation effects
            df['is_summer_vacation'] = dt.dt.month.isin([7, 8])  # July-August
            
            # Year-end effects
            df['is_year_end'] = dt.dt.month == 12
            df['is_year_start'] = dt.dt.month == 1
            
            # Mid-month effects (15th of each month)
            df['is_mid_month'] = dt.dt.day == 15
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding seasonal features: {str(e)}")
            return df
    
    def _add_market_session_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add market session indicators."""
        try:
            dt = df[date_column]
            
            # Market session indicators (simplified for US markets)
            df['is_pre_market'] = (dt.dt.hour >= 4) & (dt.dt.hour < 9) & (dt.dt.weekday < 5)
            df['is_regular_hours'] = (dt.dt.hour >= 9) & (dt.dt.hour < 16) & (dt.dt.weekday < 5)
            df['is_after_hours'] = (dt.dt.hour >= 16) & (dt.dt.hour < 20) & (dt.dt.weekday < 5)
            df['is_overnight'] = (dt.dt.hour >= 20) | (dt.dt.hour < 4) | (dt.dt.weekday >= 5)
            
            # Market open/close proximity
            market_open = dt.dt.normalize() + pd.Timedelta(hours=9)  # 9 AM
            market_close = dt.dt.normalize() + pd.Timedelta(hours=16)  # 4 PM
            
            df['minutes_to_market_open'] = ((market_open - dt).dt.total_seconds() / 60).clip(lower=0)
            df['minutes_to_market_close'] = ((market_close - dt).dt.total_seconds() / 60).clip(lower=0)
            
            # Market session progress (0-1)
            session_duration = 7 * 60  # 7 hours in minutes
            df['market_session_progress'] = (session_duration - df['minutes_to_market_close']) / session_duration
            df['market_session_progress'] = df['market_session_progress'].clip(0, 1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding market session features: {str(e)}")
            return df
    
    def _add_time_aggregations(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add time-based aggregation features."""
        try:
            dt = df[date_column]
            
            # Rolling time windows
            if 'close' in df.columns:
                # Price changes over different time periods
                for period in [1, 5, 10, 20]:
                    df[f'price_change_{period}d'] = df['close'].pct_change(period)
                    df[f'price_volatility_{period}d'] = df['close'].pct_change().rolling(window=period).std()
                
                # Time-based moving averages
                for period in [5, 10, 20]:
                    df[f'price_ma_{period}d'] = df['close'].rolling(window=period).mean()
                    df[f'price_ema_{period}d'] = df['close'].ewm(span=period).mean()
            
            # Volume aggregations
            if 'volume' in df.columns:
                for period in [5, 10, 20]:
                    df[f'volume_ma_{period}d'] = df['volume'].rolling(window=period).mean()
                    df[f'volume_ratio_{period}d'] = df['volume'] / df[f'volume_ma_{period}d']
            
            # Time-based statistics
            df['days_since_start'] = (dt - dt.min()).dt.days
            df['days_until_end'] = (dt.max() - dt).dt.days
            
            # Relative time position (0-1)
            total_days = (dt.max() - dt.min()).days
            if total_days > 0:
                df['relative_time_position'] = (dt - dt.min()).dt.days / total_days
            else:
                df['relative_time_position'] = 0.5
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding time aggregations: {str(e)}")
            return df
    
    def get_time_feature_summary(self, df: pd.DataFrame, date_column: str = 'date') -> Dict[str, any]:
        """
        Get summary of time features in the dataset.
        
        Args:
            df: DataFrame with time features
            date_column: Name of the date column
            
        Returns:
            Dictionary with time feature summary
        """
        try:
            if date_column not in df.columns:
                return {"error": f"Date column '{date_column}' not found"}
            
            dt = df[date_column]
            
            summary = {
                "date_range": {
                    "start": dt.min(),
                    "end": dt.max(),
                    "duration_days": (dt.max() - dt.min()).days
                },
                "time_coverage": {
                    "total_records": len(df),
                    "unique_dates": dt.nunique(),
                    "date_frequency": len(df) / dt.nunique() if dt.nunique() > 0 else 0
                },
                "temporal_distribution": {
                    "weekdays": df['day_of_week'].value_counts().to_dict() if 'day_of_week' in df.columns else {},
                    "months": df['month'].value_counts().to_dict() if 'month' in df.columns else {},
                    "hours": df['hour'].value_counts().to_dict() if 'hour' in df.columns else {}
                },
                "business_calendar": {
                    "business_days": df['is_business_day'].sum() if 'is_business_day' in df.columns else 0,
                    "weekend_days": df['is_weekend'].sum() if 'is_weekend' in df.columns else 0
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting time feature summary: {str(e)}")
            return {"error": str(e)}
