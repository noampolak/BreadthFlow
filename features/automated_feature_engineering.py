"""
Automated Feature Engineering with Featuretools and Tsfresh

Integrates open-source automated feature engineering tools
for comprehensive feature generation and selection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Try to import optional dependencies
try:
    import featuretools as ft

    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
    ft = None

try:
    import tsfresh
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute

    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    tsfresh = None

try:
    from feature_engine import selection

    FEATURE_ENGINE_AVAILABLE = True
except ImportError:
    FEATURE_ENGINE_AVAILABLE = False
    selection = None

logger = logging.getLogger(__name__)


class AutomatedFeatureEngineering:
    """
    Automated feature engineering using open-source tools.

    Features:
    - Featuretools for relational feature engineering
    - Tsfresh for time series feature extraction
    - Feature-engine for feature selection
    - Automated feature validation and quality checks
    """

    def __init__(self):
        """Initialize automated feature engineering."""
        self.logger = logging.getLogger(__name__)

        # Check availability of libraries
        self.libraries_available = {
            "featuretools": FEATURETOOLS_AVAILABLE,
            "tsfresh": TSFRESH_AVAILABLE,
            "feature_engine": FEATURE_ENGINE_AVAILABLE,
        }

        # Configuration
        self.config = {
            "featuretools": {"max_depth": 2, "n_jobs": 1, "chunk_size": 0.1},
            "tsfresh": {"default_fc_parameters": None, "n_jobs": 1, "chunk_size": None},  # Use default
            "feature_engine": {"k_best": 50, "correlation_threshold": 0.95},
        }

        self.logger.info(f"AutomatedFeatureEngineering initialized. Libraries available: {self.libraries_available}")

    def engineer_features_automated(
        self, df: pd.DataFrame, target_column: str = None, entity_id_column: str = "symbol", time_index_column: str = "date"
    ) -> Dict[str, Any]:
        """
        Perform automated feature engineering using all available tools.

        Args:
            df: DataFrame with time series data
            target_column: Target column for supervised feature selection
            entity_id_column: Column identifying different entities (e.g., symbols)
            time_index_column: Column with time index

        Returns:
            Dictionary with engineered features and metadata
        """
        try:
            logger.info("Starting automated feature engineering")
            start_time = datetime.now()

            result_df = df.copy()
            metadata = {
                "original_features": list(df.columns),
                "original_records": len(df),
                "engineering_steps": [],
                "libraries_used": [],
            }

            # Tsfresh feature extraction
            if self.libraries_available["tsfresh"] and TSFRESH_AVAILABLE:
                logger.info("Extracting features with Tsfresh...")
                step_start = datetime.now()

                tsfresh_result = self._extract_tsfresh_features(result_df, entity_id_column, time_index_column)

                if tsfresh_result["success"]:
                    result_df = tsfresh_result["data"]
                    metadata["engineering_steps"].append(
                        {
                            "step": "tsfresh_extraction",
                            "duration_seconds": (datetime.now() - step_start).total_seconds(),
                            "features_added": tsfresh_result["features_added"],
                        }
                    )
                    metadata["libraries_used"].append("tsfresh")
                    logger.info(f"Tsfresh completed: {tsfresh_result['features_added']} features added")

            # Featuretools feature engineering
            if self.libraries_available["featuretools"] and FEATURETOOLS_AVAILABLE:
                logger.info("Engineering features with Featuretools...")
                step_start = datetime.now()

                featuretools_result = self._extract_featuretools_features(result_df, entity_id_column, time_index_column)

                if featuretools_result["success"]:
                    result_df = featuretools_result["data"]
                    metadata["engineering_steps"].append(
                        {
                            "step": "featuretools_extraction",
                            "duration_seconds": (datetime.now() - step_start).total_seconds(),
                            "features_added": featuretools_result["features_added"],
                        }
                    )
                    metadata["libraries_used"].append("featuretools")
                    logger.info(f"Featuretools completed: {featuretools_result['features_added']} features added")

            # Feature selection with Feature-engine
            if self.libraries_available["feature_engine"] and FEATURE_ENGINE_AVAILABLE and target_column:
                logger.info("Selecting features with Feature-engine...")
                step_start = datetime.now()

                selection_result = self._select_features_feature_engine(result_df, target_column)

                if selection_result["success"]:
                    result_df = selection_result["data"]
                    metadata["engineering_steps"].append(
                        {
                            "step": "feature_selection",
                            "duration_seconds": (datetime.now() - step_start).total_seconds(),
                            "features_selected": selection_result["features_selected"],
                            "features_removed": selection_result["features_removed"],
                        }
                    )
                    metadata["libraries_used"].append("feature_engine")
                    logger.info(f"Feature selection completed: {selection_result['features_selected']} features selected")

            # Calculate final statistics
            total_duration = (datetime.now() - start_time).total_seconds()
            metadata.update(
                {
                    "total_duration_seconds": total_duration,
                    "final_features": len(result_df.columns),
                    "features_added": len(result_df.columns) - len(df.columns),
                    "completion_time": datetime.now().isoformat(),
                }
            )

            logger.info(f"Automated feature engineering completed in {total_duration:.2f}s")
            logger.info(f"Total features: {len(result_df.columns)} (added: {len(result_df.columns) - len(df.columns)})")

            return {"success": True, "data": result_df, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error in automated feature engineering: {str(e)}")
            return {"success": False, "error": str(e), "data": df}

    def _extract_tsfresh_features(self, df: pd.DataFrame, entity_id_column: str, time_index_column: str) -> Dict[str, Any]:
        """Extract features using Tsfresh."""
        try:
            if not TSFRESH_AVAILABLE:
                return {"success": False, "error": "Tsfresh not available"}

            # Prepare data for Tsfresh
            tsfresh_df = df.copy()

            # Ensure proper data types
            if time_index_column in tsfresh_df.columns:
                tsfresh_df[time_index_column] = pd.to_datetime(tsfresh_df[time_index_column])

            # Extract features
            features_before = len(tsfresh_df.columns)

            # Use default feature extraction parameters
            extracted_features = extract_features(
                tsfresh_df,
                column_id=entity_id_column,
                column_sort=time_index_column,
                n_jobs=self.config["tsfresh"]["n_jobs"],
                chunk_size=self.config["tsfresh"]["chunk_size"],
            )

            # Impute missing values
            extracted_features = impute(extracted_features)

            # Merge with original data
            if not extracted_features.empty:
                # Reset index to merge properly
                extracted_features = extracted_features.reset_index()

                # Merge with original data
                result_df = tsfresh_df.merge(
                    extracted_features, left_on=entity_id_column, right_on=entity_id_column, how="left"
                )
            else:
                result_df = tsfresh_df

            features_added = len(result_df.columns) - features_before

            return {"success": True, "data": result_df, "features_added": features_added}

        except Exception as e:
            logger.error(f"Error in Tsfresh feature extraction: {str(e)}")
            return {"success": False, "error": str(e)}

    def _extract_featuretools_features(
        self, df: pd.DataFrame, entity_id_column: str, time_index_column: str
    ) -> Dict[str, Any]:
        """Extract features using Featuretools."""
        try:
            if not FEATURETOOLS_AVAILABLE:
                return {"success": False, "error": "Featuretools not available"}

            # Prepare data for Featuretools
            featuretools_df = df.copy()

            # Ensure proper data types
            if time_index_column in featuretools_df.columns:
                featuretools_df[time_index_column] = pd.to_datetime(featuretools_df[time_index_column])

            # Create EntitySet
            es = ft.EntitySet(id="financial_data")

            # Add dataframe to entity set
            es = es.add_dataframe(
                dataframe_name="financial_data",
                dataframe=featuretools_df,
                index="id",  # Create a unique index
                time_index=time_index_column,
                logical_types={entity_id_column: ft.variable_types.Categorical, time_index_column: ft.variable_types.Datetime},
            )

            # Generate features
            features_before = len(featuretools_df.columns)

            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="financial_data",
                max_depth=self.config["featuretools"]["max_depth"],
                n_jobs=self.config["featuretools"]["n_jobs"],
                chunk_size=self.config["featuretools"]["chunk_size"],
            )

            # Merge with original data
            if not feature_matrix.empty:
                # Reset index to merge properly
                feature_matrix = feature_matrix.reset_index()

                # Merge with original data
                result_df = featuretools_df.merge(feature_matrix, left_index=True, right_index=True, how="left")
            else:
                result_df = featuretools_df

            features_added = len(result_df.columns) - features_before

            return {"success": True, "data": result_df, "features_added": features_added}

        except Exception as e:
            logger.error(f"Error in Featuretools feature extraction: {str(e)}")
            return {"success": False, "error": str(e)}

    def _select_features_feature_engine(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Select features using Feature-engine."""
        try:
            if not FEATURE_ENGINE_AVAILABLE:
                return {"success": False, "error": "Feature-engine not available"}

            if target_column not in df.columns:
                return {"success": False, "error": f"Target column '{target_column}' not found"}

            # Get feature columns (exclude target and non-numeric columns)
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)

            if len(feature_columns) == 0:
                return {"success": False, "error": "No numeric features found for selection"}

            # Prepare data
            X = df[feature_columns].fillna(0)  # Fill missing values
            y = df[target_column]

            # Remove constant features
            constant_selector = selection.DropConstantFeatures()
            X_constant_removed = constant_selector.fit_transform(X)

            # Remove highly correlated features
            correlation_selector = selection.DropCorrelatedFeatures(
                threshold=self.config["feature_engine"]["correlation_threshold"]
            )
            X_corr_removed = correlation_selector.fit_transform(X_constant_removed)

            # Select k best features
            k_best_selector = selection.SelectKBestFeatures(k=self.config["feature_engine"]["k_best"])
            X_selected = k_best_selector.fit_transform(X_corr_removed, y)

            # Create result DataFrame
            selected_columns = X_selected.columns.tolist() + [target_column]
            result_df = df[selected_columns].copy()

            features_selected = len(selected_columns) - 1  # Exclude target
            features_removed = len(feature_columns) - features_selected

            return {
                "success": True,
                "data": result_df,
                "features_selected": features_selected,
                "features_removed": features_removed,
            }

        except Exception as e:
            logger.error(f"Error in Feature-engine feature selection: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_library_status(self) -> Dict[str, Any]:
        """Get status of available libraries."""
        return {
            "libraries_available": self.libraries_available,
            "configuration": self.config,
            "recommendations": self._get_installation_recommendations(),
        }

    def _get_installation_recommendations(self) -> List[str]:
        """Get installation recommendations for missing libraries."""
        recommendations = []

        if not self.libraries_available["featuretools"]:
            recommendations.append("Install Featuretools: pip install featuretools")

        if not self.libraries_available["tsfresh"]:
            recommendations.append("Install Tsfresh: pip install tsfresh")

        if not self.libraries_available["feature_engine"]:
            recommendations.append("Install Feature-engine: pip install feature-engine")

        if not recommendations:
            recommendations.append("All automated feature engineering libraries are available")

        return recommendations

    def validate_automated_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate automatically generated features."""
        try:
            validation_result = {
                "total_features": len(df.columns),
                "numeric_features": len(df.select_dtypes(include=[np.number]).columns),
                "missing_values": df.isnull().sum().sum(),
                "infinite_values": np.isinf(df.select_dtypes(include=[np.number])).sum().sum(),
                "constant_features": [],
                "highly_correlated_pairs": [],
            }

            # Find constant features
            for col in df.select_dtypes(include=[np.number]).columns:
                if df[col].nunique() <= 1:
                    validation_result["constant_features"].append(col)

            # Find highly correlated feature pairs
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                high_corr_pairs = []
                for col in upper_tri.columns:
                    for idx in upper_tri.index:
                        if pd.notna(upper_tri.loc[idx, col]) and upper_tri.loc[idx, col] > 0.95:
                            high_corr_pairs.append((idx, col, upper_tri.loc[idx, col]))

                validation_result["highly_correlated_pairs"] = high_corr_pairs

            return validation_result

        except Exception as e:
            logger.error(f"Error validating automated features: {str(e)}")
            return {"error": str(e)}
