"""
Feature Utilities Module

Generic, reusable utilities for feature engineering that can be used
across multiple experiments and trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class FeatureUtils:
    """Generic feature engineering utilities."""
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, 
                            method: str = 'forward_fill',
                            limit: Optional[int] = None) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        if method == 'forward_fill':
            return data.fillna(method='ffill', limit=limit)
        elif method == 'backward_fill':
            return data.fillna(method='bfill', limit=limit)
        elif method == 'mean':
            return data.fillna(data.mean())
        elif method == 'median':
            return data.fillna(data.median())
        elif method == 'zero':
            return data.fillna(0)
        elif method == 'drop':
            return data.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def remove_outliers(data: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from the dataset."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Keep data within bounds
            mask = ((data >= lower_bound) & (data <= upper_bound)).all(axis=1)
            return data[mask]
        
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            mask = (z_scores < threshold).all(axis=1)
            return data[mask]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def scale_features(data: pd.DataFrame,
                      method: str = 'standard',
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale features using various methods."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        scaled_data = data.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        scaled_data[columns] = scaler.fit_transform(data[columns])
        return scaled_data
    
    @staticmethod
    def create_lag_features(data: pd.Series, 
                          lags: List[int],
                          name_prefix: str = 'lag') -> pd.DataFrame:
        """Create lag features from a time series."""
        result = pd.DataFrame(index=data.index)
        
        for lag in lags:
            result[f'{name_prefix}_{lag}'] = data.shift(lag)
        
        return result
    
    @staticmethod
    def create_rolling_features(data: pd.Series,
                              windows: List[int],
                              functions: List[str] = ['mean', 'std', 'min', 'max'],
                              name_prefix: str = 'rolling') -> pd.DataFrame:
        """Create rolling window features."""
        result = pd.DataFrame(index=data.index)
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).mean()
                elif func == 'std':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).std()
                elif func == 'min':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).min()
                elif func == 'max':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).max()
                elif func == 'median':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).median()
                elif func == 'skew':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).skew()
                elif func == 'kurt':
                    result[f'{name_prefix}_{window}_{func}'] = data.rolling(window).kurt()
        
        return result
    
    @staticmethod
    def create_interaction_features(data: pd.DataFrame,
                                  feature_pairs: List[tuple],
                                  operations: List[str] = ['multiply', 'divide', 'add', 'subtract']) -> pd.DataFrame:
        """Create interaction features between pairs of features."""
        result = data.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in data.columns and feature2 in data.columns:
                for op in operations:
                    if op == 'multiply':
                        result[f'{feature1}_x_{feature2}'] = data[feature1] * data[feature2]
                    elif op == 'divide':
                        result[f'{feature1}_div_{feature2}'] = data[feature1] / data[feature2]
                    elif op == 'add':
                        result[f'{feature1}_plus_{feature2}'] = data[feature1] + data[feature2]
                    elif op == 'subtract':
                        result[f'{feature1}_minus_{feature2}'] = data[feature1] - data[feature2]
        
        return result
    
    @staticmethod
    def create_polynomial_features(data: pd.DataFrame,
                                 columns: List[str],
                                 degree: int = 2) -> pd.DataFrame:
        """Create polynomial features."""
        result = data.copy()
        
        for col in columns:
            if col in data.columns:
                for d in range(2, degree + 1):
                    result[f'{col}_pow_{d}'] = data[col] ** d
        
        return result
    
    @staticmethod
    def select_features_by_correlation(data: pd.DataFrame,
                                     target: pd.Series,
                                     threshold: float = 0.1) -> List[str]:
        """Select features based on correlation with target."""
        correlations = data.corrwith(target).abs()
        selected_features = correlations[correlations > threshold].index.tolist()
        return selected_features
    
    @staticmethod
    def create_feature_summary(data: pd.DataFrame) -> pd.DataFrame:
        """Create a summary of features in the dataset."""
        summary = pd.DataFrame({
            'feature': data.columns,
            'dtype': data.dtypes,
            'non_null_count': data.count(),
            'null_count': data.isnull().sum(),
            'null_percentage': (data.isnull().sum() / len(data)) * 100,
            'unique_count': data.nunique(),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        })
        
        return summary
