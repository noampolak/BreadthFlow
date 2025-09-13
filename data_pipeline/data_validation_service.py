"""
Data Validation Service for ML Pipeline

Handles data quality validation, schema validation, and data drift detection
for the ML training pipeline.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, isnan, when, isnull, avg, stddev, min as spark_min, max as spark_max

logger = logging.getLogger(__name__)


class DataValidationService:
    """
    Service for validating data quality and detecting data drift.
    
    Features:
    - Schema validation
    - Data quality checks
    - Statistical validation
    - Data drift detection
    - Missing data analysis
    - Outlier detection
    """
    
    def __init__(self, spark_session=None):
        """
        Initialize the data validation service.
        
        Args:
            spark_session: Spark session for data processing
        """
        self.spark = spark_session
        self.quality_thresholds = {
            "min_completeness": 0.95,  # 95% data completeness
            "max_missing_ratio": 0.05,  # 5% missing data
            "max_outlier_ratio": 0.01,  # 1% outliers
            "min_data_points": 10,  # Minimum data points per symbol
            "max_price_change": 0.5,  # 50% maximum price change
            "min_volume": 1000,  # Minimum volume threshold
        }
        
        logger.info("DataValidationService initialized")
    
    def validate_schema(self, df: DataFrame, expected_schema: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate DataFrame schema against expected schema.
        
        Args:
            df: DataFrame to validate
            expected_schema: Expected column names and types
            
        Returns:
            Dictionary with validation results
        """
        try:
            actual_columns = df.columns
            expected_columns = list(expected_schema.keys())
            
            # Check for missing columns
            missing_columns = set(expected_columns) - set(actual_columns)
            extra_columns = set(actual_columns) - set(expected_columns)
            
            # Check data types
            type_mismatches = []
            for col_name, expected_type in expected_schema.items():
                if col_name in actual_columns:
                    actual_type = str(df.schema[col_name].dataType)
                    if expected_type not in actual_type.lower():
                        type_mismatches.append({
                            "column": col_name,
                            "expected": expected_type,
                            "actual": actual_type
                        })
            
            is_valid = len(missing_columns) == 0 and len(type_mismatches) == 0
            
            return {
                "is_valid": is_valid,
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns),
                "type_mismatches": type_mismatches,
                "total_columns": len(actual_columns),
                "expected_columns": len(expected_columns)
            }
            
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def validate_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """
        Validate data quality using various checks.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with quality validation results
        """
        try:
            # Basic statistics
            total_records = df.count()
            
            if total_records == 0:
                return {
                    "is_valid": False,
                    "error": "No data to validate",
                    "total_records": 0
                }
            
            # Missing data analysis
            missing_stats = self._analyze_missing_data(df)
            
            # Price validation
            price_stats = self._validate_prices(df)
            
            # Volume validation
            volume_stats = self._validate_volumes(df)
            
            # Data completeness
            completeness = 1.0 - (missing_stats["total_missing"] / (total_records * len(df.columns)))
            
            # Overall quality score
            quality_score = self._calculate_quality_score(missing_stats, price_stats, volume_stats)
            
            # Determine if data is valid
            is_valid = (
                completeness >= self.quality_thresholds["min_completeness"] and
                missing_stats["missing_ratio"] <= self.quality_thresholds["max_missing_ratio"] and
                price_stats["is_valid"] and
                volume_stats["is_valid"]
            )
            
            return {
                "is_valid": is_valid,
                "quality_score": quality_score,
                "completeness": completeness,
                "total_records": total_records,
                "missing_data": missing_stats,
                "price_validation": price_stats,
                "volume_validation": volume_stats,
                "thresholds": self.quality_thresholds
            }
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def detect_data_drift(
        self, 
        current_df: DataFrame, 
        reference_df: DataFrame,
        columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between current and reference datasets.
        
        Args:
            current_df: Current dataset
            reference_df: Reference dataset
            columns: Columns to check for drift (default: all numeric columns)
            
        Returns:
            Dictionary with drift detection results
        """
        try:
            if columns is None:
                # Get numeric columns
                numeric_columns = [field.name for field in current_df.schema.fields 
                                 if str(field.dataType) in ['DoubleType', 'FloatType', 'IntegerType', 'LongType']]
            else:
                numeric_columns = columns
            
            drift_results = {}
            
            for col_name in numeric_columns:
                if col_name in current_df.columns and col_name in reference_df.columns:
                    # Calculate statistics for both datasets
                    current_stats = self._calculate_column_stats(current_df, col_name)
                    reference_stats = self._calculate_column_stats(reference_df, col_name)
                    
                    # Calculate drift metrics
                    drift_metrics = self._calculate_drift_metrics(current_stats, reference_stats)
                    
                    drift_results[col_name] = {
                        "current_stats": current_stats,
                        "reference_stats": reference_stats,
                        "drift_metrics": drift_metrics,
                        "has_drift": drift_metrics["ks_statistic"] > 0.1 or drift_metrics["mean_shift"] > 0.1
                    }
            
            # Overall drift assessment
            columns_with_drift = [col for col, result in drift_results.items() if result["has_drift"]]
            overall_drift = len(columns_with_drift) > 0
            
            return {
                "has_drift": overall_drift,
                "columns_with_drift": columns_with_drift,
                "drift_details": drift_results,
                "total_columns_checked": len(numeric_columns)
            }
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {str(e)}")
            return {
                "has_drift": False,
                "error": str(e)
            }
    
    def validate_training_data(self, df: DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Validate data specifically for ML training.
        
        Args:
            df: DataFrame to validate
            target_column: Target column for supervised learning
            
        Returns:
            Dictionary with training data validation results
        """
        try:
            # Basic validation
            basic_validation = self.validate_data_quality(df)
            
            if not basic_validation["is_valid"]:
                return basic_validation
            
            # Check for sufficient data
            total_records = df.count()
            unique_symbols = df.select("symbol").distinct().count()
            
            # Check date range
            date_range = df.agg({"date": "min", "date": "max"}).collect()[0]
            date_span = (date_range[1] - date_range[0]).days
            
            # Check for data leakage (future data in training set)
            current_date = datetime.now()
            future_data = df.filter(col("date") > current_date).count()
            
            # Check target variable if specified
            target_validation = {}
            if target_column and target_column in df.columns:
                target_stats = self._calculate_column_stats(df, target_column)
                target_validation = {
                    "is_valid": target_stats["missing_ratio"] < 0.1,
                    "missing_ratio": target_stats["missing_ratio"],
                    "unique_values": target_stats["unique_count"],
                    "stats": target_stats
                }
            
            # Overall training data validity
            is_valid = (
                basic_validation["is_valid"] and
                total_records >= 1000 and  # Minimum records for training
                unique_symbols >= 10 and  # Minimum symbols
                date_span >= 30 and  # Minimum time span
                future_data == 0 and  # No future data
                (not target_column or target_validation.get("is_valid", True))
            )
            
            return {
                "is_valid": is_valid,
                "basic_validation": basic_validation,
                "data_sufficiency": {
                    "total_records": total_records,
                    "unique_symbols": unique_symbols,
                    "date_span_days": date_span,
                    "future_data_count": future_data
                },
                "target_validation": target_validation,
                "recommendations": self._generate_training_recommendations(
                    total_records, unique_symbols, date_span, future_data
                )
            }
            
        except Exception as e:
            logger.error(f"Error validating training data: {str(e)}")
            return {
                "is_valid": False,
                "error": str(e)
            }
    
    def _analyze_missing_data(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        try:
            # Count missing values per column
            missing_counts = {}
            total_records = df.count()
            
            for col_name in df.columns:
                missing_count = df.filter(col(col_name).isNull() | col(col_name).isnan()).count()
                missing_counts[col_name] = missing_count
            
            total_missing = sum(missing_counts.values())
            missing_ratio = total_missing / (total_records * len(df.columns))
            
            return {
                "missing_counts": missing_counts,
                "total_missing": total_missing,
                "missing_ratio": missing_ratio,
                "columns_with_missing": [col for col, count in missing_counts.items() if count > 0]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing missing data: {str(e)}")
            return {"error": str(e)}
    
    def _validate_prices(self, df: DataFrame) -> Dict[str, Any]:
        """Validate price data."""
        try:
            price_columns = ['open', 'high', 'low', 'close', 'adj_close']
            price_stats = {}
            
            for col_name in price_columns:
                if col_name in df.columns:
                    stats = self._calculate_column_stats(df, col_name)
                    price_stats[col_name] = stats
            
            # Check for negative prices
            negative_prices = 0
            for col_name in price_columns:
                if col_name in df.columns:
                    negative_count = df.filter(col(col_name) <= 0).count()
                    negative_prices += negative_count
            
            # Check for extreme price changes
            extreme_changes = 0
            if 'close' in df.columns:
                # This would need to be implemented with window functions
                # For now, we'll use a simple approach
                extreme_changes = 0  # Placeholder
            
            is_valid = negative_prices == 0 and extreme_changes < df.count() * 0.01
            
            return {
                "is_valid": is_valid,
                "negative_prices": negative_prices,
                "extreme_changes": extreme_changes,
                "price_stats": price_stats
            }
            
        except Exception as e:
            logger.error(f"Error validating prices: {str(e)}")
            return {"is_valid": False, "error": str(e)}
    
    def _validate_volumes(self, df: DataFrame) -> Dict[str, Any]:
        """Validate volume data."""
        try:
            if 'volume' not in df.columns:
                return {"is_valid": True, "message": "No volume column"}
            
            volume_stats = self._calculate_column_stats(df, 'volume')
            
            # Check for negative volumes
            negative_volumes = df.filter(col('volume') < 0).count()
            
            # Check for zero volumes
            zero_volumes = df.filter(col('volume') == 0).count()
            
            is_valid = negative_volumes == 0 and zero_volumes < df.count() * 0.1
            
            return {
                "is_valid": is_valid,
                "negative_volumes": negative_volumes,
                "zero_volumes": zero_volumes,
                "volume_stats": volume_stats
            }
            
        except Exception as e:
            logger.error(f"Error validating volumes: {str(e)}")
            return {"is_valid": False, "error": str(e)}
    
    def _calculate_column_stats(self, df: DataFrame, col_name: str) -> Dict[str, Any]:
        """Calculate statistics for a column."""
        try:
            stats = df.agg({
                col_name: "count",
                col_name: "mean",
                col_name: "stddev",
                col_name: "min",
                col_name: "max"
            }).collect()[0]
            
            total_count = df.count()
            missing_count = df.filter(col(col_name).isNull() | col(col_name).isnan()).count()
            
            return {
                "count": stats[0],
                "mean": stats[1],
                "stddev": stats[2],
                "min": stats[3],
                "max": stats[4],
                "missing_count": missing_count,
                "missing_ratio": missing_count / total_count if total_count > 0 else 0,
                "unique_count": df.select(col_name).distinct().count()
            }
            
        except Exception as e:
            logger.error(f"Error calculating column stats: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_drift_metrics(self, current_stats: Dict, reference_stats: Dict) -> Dict[str, float]:
        """Calculate drift metrics between current and reference statistics."""
        try:
            # Mean shift
            mean_shift = abs(current_stats["mean"] - reference_stats["mean"]) / reference_stats["mean"] if reference_stats["mean"] != 0 else 0
            
            # Standard deviation change
            std_change = abs(current_stats["stddev"] - reference_stats["stddev"]) / reference_stats["stddev"] if reference_stats["stddev"] != 0 else 0
            
            # Kolmogorov-Smirnov statistic (simplified)
            ks_statistic = min(1.0, mean_shift + std_change)
            
            return {
                "mean_shift": mean_shift,
                "std_change": std_change,
                "ks_statistic": ks_statistic
            }
            
        except Exception as e:
            logger.error(f"Error calculating drift metrics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_quality_score(
        self, 
        missing_stats: Dict, 
        price_stats: Dict, 
        volume_stats: Dict
    ) -> float:
        """Calculate overall data quality score."""
        try:
            # Completeness score
            completeness_score = 1.0 - missing_stats.get("missing_ratio", 0)
            
            # Price validity score
            price_score = 1.0 if price_stats.get("is_valid", False) else 0.5
            
            # Volume validity score
            volume_score = 1.0 if volume_stats.get("is_valid", False) else 0.5
            
            # Overall quality score (weighted average)
            quality_score = (completeness_score * 0.5 + price_score * 0.3 + volume_score * 0.2)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def _generate_training_recommendations(
        self, 
        total_records: int, 
        unique_symbols: int, 
        date_span: int, 
        future_data: int
    ) -> List[str]:
        """Generate recommendations for training data."""
        recommendations = []
        
        if total_records < 1000:
            recommendations.append("Consider collecting more data (minimum 1000 records recommended)")
        
        if unique_symbols < 10:
            recommendations.append("Consider including more symbols (minimum 10 recommended)")
        
        if date_span < 30:
            recommendations.append("Consider extending the date range (minimum 30 days recommended)")
        
        if future_data > 0:
            recommendations.append("Remove future data to prevent data leakage")
        
        if not recommendations:
            recommendations.append("Data appears suitable for training")
        
        return recommendations
