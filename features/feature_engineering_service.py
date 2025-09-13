"""
Feature Engineering Service for ML Pipeline

Orchestrates all feature engineering components including technical indicators,
time features, microstructure features, and automated feature selection.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .technical_indicators import TechnicalIndicators
from .time_features import TimeFeatures
from .microstructure_features import MicrostructureFeatures
from .automated_feature_engineering import AutomatedFeatureEngineering

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """
    Main feature engineering service that orchestrates all feature types.
    
    Features:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Time-based features (cyclical, seasonal, business calendar)
    - Market microstructure features (volume, liquidity, order flow)
    - Automated feature selection and engineering
    - Feature validation and quality checks
    """
    
    def __init__(self, spark_session=None):
        """
        Initialize the feature engineering service.
        
        Args:
            spark_session: Spark session for distributed processing
        """
        self.spark = spark_session
        self.technical_indicators = TechnicalIndicators()
        self.time_features = TimeFeatures()
        self.microstructure_features = MicrostructureFeatures()
        self.automated_fe = AutomatedFeatureEngineering()
        
        # Feature engineering configuration
        self.config = {
            "technical_indicators": True,
            "time_features": True,
            "microstructure_features": True,
            "automated_feature_engineering": True,
            "feature_selection": True,
            "max_features": 1000,
            "correlation_threshold": 0.95,
            "variance_threshold": 0.01
        }
        
        logger.info("FeatureEngineeringService initialized")
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        feature_types: List[str] = None,
        target_column: str = None
    ) -> Dict[str, Any]:
        """
        Engineer features for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            feature_types: List of feature types to engineer (default: all)
            target_column: Target column for supervised feature selection
            
        Returns:
            Dictionary with engineered features and metadata
        """
        try:
            if feature_types is None:
                feature_types = ['technical', 'time', 'microstructure']
            
            logger.info(f"Starting feature engineering for {len(df)} records")
            start_time = datetime.now()
            
            # Start with original data
            result_df = df.copy()
            feature_metadata = {
                "original_features": list(df.columns),
                "feature_types": feature_types,
                "engineering_steps": []
            }
            
            # Technical indicators
            if 'technical' in feature_types and self.config["technical_indicators"]:
                logger.info("Engineering technical indicators...")
                step_start = datetime.now()
                result_df = self.technical_indicators.calculate_all_indicators(result_df)
                step_duration = (datetime.now() - step_start).total_seconds()
                feature_metadata["engineering_steps"].append({
                    "step": "technical_indicators",
                    "duration_seconds": step_duration,
                    "features_added": len(result_df.columns) - len(df.columns)
                })
                logger.info(f"Technical indicators completed in {step_duration:.2f}s")
            
            # Time features
            if 'time' in feature_types and self.config["time_features"]:
                logger.info("Engineering time features...")
                step_start = datetime.now()
                result_df = self.time_features.calculate_all_time_features(result_df)
                step_duration = (datetime.now() - step_start).total_seconds()
                feature_metadata["engineering_steps"].append({
                    "step": "time_features",
                    "duration_seconds": step_duration,
                    "features_added": len(result_df.columns) - len(df.columns) - feature_metadata["engineering_steps"][-1]["features_added"] if feature_metadata["engineering_steps"] else 0
                })
                logger.info(f"Time features completed in {step_duration:.2f}s")
            
            # Microstructure features
            if 'microstructure' in feature_types and self.config["microstructure_features"]:
                logger.info("Engineering microstructure features...")
                step_start = datetime.now()
                result_df = self.microstructure_features.calculate_all_microstructure_features(result_df)
                step_duration = (datetime.now() - step_start).total_seconds()
                feature_metadata["engineering_steps"].append({
                    "step": "microstructure_features",
                    "duration_seconds": step_duration,
                    "features_added": len(result_df.columns) - len(df.columns) - sum(step["features_added"] for step in feature_metadata["engineering_steps"])
                })
                logger.info(f"Microstructure features completed in {step_duration:.2f}s")
            
            # Automated feature engineering
            if 'automated' in feature_types and self.config["automated_feature_engineering"]:
                logger.info("Performing automated feature engineering...")
                step_start = datetime.now()
                automated_result = self.automated_fe.engineer_features_automated(
                    result_df, 
                    target_column=target_column
                )
                step_duration = (datetime.now() - step_start).total_seconds()
                
                if automated_result["success"]:
                    result_df = automated_result["data"]
                    feature_metadata["engineering_steps"].append({
                        "step": "automated_feature_engineering",
                        "duration_seconds": step_duration,
                        "features_added": automated_result["metadata"]["features_added"],
                        "libraries_used": automated_result["metadata"]["libraries_used"]
                    })
                    logger.info(f"Automated feature engineering completed in {step_duration:.2f}s")
                else:
                    logger.warning(f"Automated feature engineering failed: {automated_result.get('error', 'Unknown error')}")
            
            # Feature selection
            if self.config["feature_selection"] and target_column:
                logger.info("Performing feature selection...")
                step_start = datetime.now()
                result_df, selection_metadata = self._select_features(result_df, target_column)
                step_duration = (datetime.now() - step_start).total_seconds()
                feature_metadata["engineering_steps"].append({
                    "step": "feature_selection",
                    "duration_seconds": step_duration,
                    "features_selected": selection_metadata["features_selected"],
                    "features_removed": selection_metadata["features_removed"]
                })
                logger.info(f"Feature selection completed in {step_duration:.2f}s")
            
            # Feature validation
            validation_result = self._validate_features(result_df)
            feature_metadata["validation"] = validation_result
            
            # Calculate final statistics
            total_duration = (datetime.now() - start_time).total_seconds()
            feature_metadata.update({
                "total_duration_seconds": total_duration,
                "total_features": len(result_df.columns),
                "features_added": len(result_df.columns) - len(df.columns),
                "feature_columns": list(result_df.columns),
                "completion_time": datetime.now().isoformat()
            })
            
            logger.info(f"Feature engineering completed in {total_duration:.2f}s")
            logger.info(f"Total features: {len(result_df.columns)} (added: {len(result_df.columns) - len(df.columns)})")
            
            return {
                "success": True,
                "data": result_df,
                "metadata": feature_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": df
            }
    
    def _select_features(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Select the most relevant features for machine learning.
        
        Args:
            df: DataFrame with features
            target_column: Target column for supervised selection
            
        Returns:
            Tuple of (selected_features_df, selection_metadata)
        """
        try:
            if target_column not in df.columns:
                logger.warning(f"Target column '{target_column}' not found, skipping feature selection")
                return df, {"features_selected": len(df.columns), "features_removed": 0}
            
            # Get feature columns (exclude target and non-numeric columns)
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
            
            # Remove columns with too many missing values
            missing_threshold = 0.5
            valid_columns = []
            for col in feature_columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio < missing_threshold:
                    valid_columns.append(col)
            
            logger.info(f"Valid features for selection: {len(valid_columns)}")
            
            # Remove highly correlated features
            correlation_matrix = df[valid_columns].corr().abs()
            upper_tri = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.config["correlation_threshold"])]
            selected_columns = [col for col in valid_columns if col not in to_drop]
            
            logger.info(f"Removed {len(to_drop)} highly correlated features")
            
            # Remove low variance features
            variances = df[selected_columns].var()
            low_variance_columns = variances[variances < self.config["variance_threshold"]].index.tolist()
            selected_columns = [col for col in selected_columns if col not in low_variance_columns]
            
            logger.info(f"Removed {len(low_variance_columns)} low variance features")
            
            # Limit number of features
            if len(selected_columns) > self.config["max_features"]:
                # Select top features by correlation with target
                correlations = df[selected_columns + [target_column]].corr()[target_column].abs()
                correlations = correlations.drop(target_column)
                selected_columns = correlations.nlargest(self.config["max_features"]).index.tolist()
                logger.info(f"Limited to top {self.config['max_features']} features by correlation")
            
            # Create final DataFrame
            final_columns = selected_columns + [target_column] if target_column in df.columns else selected_columns
            result_df = df[final_columns].copy()
            
            selection_metadata = {
                "features_selected": len(selected_columns),
                "features_removed": len(feature_columns) - len(selected_columns),
                "correlation_removed": len(to_drop),
                "variance_removed": len(low_variance_columns),
                "selected_features": selected_columns
            }
            
            return result_df, selection_metadata
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return df, {"features_selected": len(df.columns), "features_removed": 0}
    
    def _validate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate engineered features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result = {
                "total_features": len(df.columns),
                "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
                "missing_values": df.isnull().sum().sum(),
                "infinite_values": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
                "duplicate_rows": df.duplicated().sum(),
                "feature_types": {}
            }
            
            # Analyze feature types
            for col in df.columns:
                dtype = str(df[col].dtype)
                if dtype not in validation_result["feature_types"]:
                    validation_result["feature_types"][dtype] = 0
                validation_result["feature_types"][dtype] += 1
            
            # Check for constant features
            constant_features = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    constant_features.append(col)
            
            validation_result["constant_features"] = constant_features
            validation_result["constant_features_count"] = len(constant_features)
            
            # Check for highly skewed features
            skewed_features = []
            for col in df.select_dtypes(include=[np.number]).columns:
                if abs(df[col].skew()) > 3:  # Highly skewed
                    skewed_features.append(col)
            
            validation_result["skewed_features"] = skewed_features
            validation_result["skewed_features_count"] = len(skewed_features)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating features: {str(e)}")
            return {"error": str(e)}
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of features in the dataset.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary with feature summary
        """
        try:
            summary = {
                "basic_info": {
                    "total_features": len(df.columns),
                    "total_records": len(df),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                },
                "data_types": df.dtypes.value_counts().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_features": {
                    "count": len(df.select_dtypes(include=[np.number]).columns),
                    "mean_values": df.select_dtypes(include=[np.number]).mean().to_dict(),
                    "std_values": df.select_dtypes(include=[np.number]).std().to_dict(),
                    "skewness": df.select_dtypes(include=[np.number]).skew().to_dict()
                },
                "categorical_features": {
                    "count": len(df.select_dtypes(include=['object', 'category']).columns),
                    "unique_values": df.select_dtypes(include=['object', 'category']).nunique().to_dict()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting feature summary: {str(e)}")
            return {"error": str(e)}
    
    def engineer_features_for_symbols(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        feature_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Engineer features for multiple symbols.
        
        Args:
            data_dict: Dictionary with symbol as key and DataFrame as value
            feature_types: List of feature types to engineer
            
        Returns:
            Dictionary with results for each symbol
        """
        try:
            results = {}
            total_start_time = datetime.now()
            
            for symbol, df in data_dict.items():
                logger.info(f"Engineering features for {symbol}")
                symbol_start_time = datetime.now()
                
                result = self.engineer_features(df, feature_types)
                
                symbol_duration = (datetime.now() - symbol_start_time).total_seconds()
                result["symbol"] = symbol
                result["duration_seconds"] = symbol_duration
                
                results[symbol] = result
                
                logger.info(f"Features for {symbol} completed in {symbol_duration:.2f}s")
            
            total_duration = (datetime.now() - total_start_time).total_seconds()
            
            return {
                "success": True,
                "results": results,
                "total_symbols": len(data_dict),
                "total_duration_seconds": total_duration,
                "completion_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error engineering features for symbols: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of the feature engineering service.
        
        Returns:
            Dictionary with service status
        """
        try:
            return {
                "status": "healthy",
                "components": {
                    "technical_indicators": "available",
                    "time_features": "available", 
                    "microstructure_features": "available"
                },
                "configuration": self.config,
                "spark_available": self.spark is not None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
