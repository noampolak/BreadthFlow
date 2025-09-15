"""
Universal Data Fetcher

Orchestrates data fetching across multiple sources and resources,
providing a unified interface for the BreadthFlow system.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
from ..logging.enhanced_logger import EnhancedLogger
from ..logging.error_handler import ErrorHandler
from .resources.data_resources import get_resource_by_name, validate_resource_data
from .sources.data_source_interface import DataSourceInterface

logger = logging.getLogger(__name__)


class UniversalDataFetcher:
    """Universal data fetcher that orchestrates multiple data sources"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_sources: Dict[str, DataSourceInterface] = {}
        self.logger = EnhancedLogger("universal_data_fetcher", "data_fetcher")
        self.error_handler = ErrorHandler()

        # Performance tracking
        self.fetch_stats = {"total_requests": 0, "successful_requests": 0, "failed_requests": 0, "total_data_points": 0}

        # Register default data sources
        self._register_default_sources()

    def _register_default_sources(self):
        """Register default data sources"""
        try:
            from .sources.yfinance_source import YFinanceDataSource

            yfinance_source = YFinanceDataSource()
            self.register_data_source("yfinance", yfinance_source)
            logger.info("✅ Registered YFinance data source")
        except Exception as e:
            logger.warning(f"Could not register YFinance data source: {e}")

    def register_data_source(self, source_name: str, data_source: DataSourceInterface):
        """Register a data source"""
        self.data_sources[source_name] = data_source
        logger.info(f"✅ Registered data source: {source_name}")

    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return list(self.data_sources.keys())

    def get_supported_resources(self) -> Dict[str, List[str]]:
        """Get supported resources for each data source"""
        supported = {}
        for source_name, source in self.data_sources.items():
            supported[source_name] = source.get_supported_resources()
        return supported

    def fetch_data(
        self,
        resource_name: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        source_name: str = None,
        **kwargs,
    ):
        """Fetch data for specified resource and symbols"""

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for data fetching but not available")

        with self.logger.log_performance("fetch_data"):
            try:
                # Validate resource
                resource = get_resource_by_name(resource_name)
                if not resource:
                    raise ValueError(f"Unknown resource: {resource_name}")

                # Determine data source
                if source_name:
                    if source_name not in self.data_sources:
                        raise ValueError(f"Unknown data source: {source_name}")
                    sources_to_try = [source_name]
                else:
                    # Find sources that support this resource
                    sources_to_try = []
                    for name, source in self.data_sources.items():
                        if source.validate_resource_support(resource_name):
                            sources_to_try.append(name)

                    if not sources_to_try:
                        raise ValueError(f"No data source supports resource: {resource_name}")

                # Try sources in order
                for source_name in sources_to_try:
                    try:
                        source = self.data_sources[source_name]

                        self.logger.log_operation(
                            "fetch_attempt",
                            {
                                "resource": resource_name,
                                "source": source_name,
                                "symbols_count": len(symbols),
                                "date_range": f"{start_date} to {end_date}",
                            },
                        )

                        # Fetch data
                        data = source.fetch_data(resource_name, symbols, start_date, end_date, **kwargs)

                        if not data.empty:
                            # Validate data quality
                            validation_result = self._validate_data_quality(data, resource)

                            # Update stats
                            self.fetch_stats["successful_requests"] += 1
                            self.fetch_stats["total_data_points"] += len(data)

                            self.logger.log_operation(
                                "fetch_success",
                                {
                                    "resource": resource_name,
                                    "source": source_name,
                                    "data_points": len(data),
                                    "quality_score": validation_result.get("quality_score", 0.0),
                                },
                            )

                            return data

                    except Exception as e:
                        self.error_handler.handle_error(
                            e,
                            {"resource": resource_name, "source": source_name, "symbols": symbols},
                            "universal_data_fetcher",
                            "fetch_data",
                        )
                        logger.warning(f"Failed to fetch from {source_name}: {e}")
                        continue

                # If we get here, all sources failed
                self.fetch_stats["failed_requests"] += 1
                raise Exception(f"All data sources failed for resource: {resource_name}")

            except Exception as e:
                self.error_handler.handle_error(
                    e,
                    {"resource": resource_name, "symbols": symbols, "start_date": start_date, "end_date": end_date},
                    "universal_data_fetcher",
                    "fetch_data",
                )
                raise

    def fetch_multiple_resources(
        self, resources: List[str], symbols: List[str], start_date: datetime, end_date: datetime, **kwargs
    ):
        """Fetch multiple resources at once"""

        results = {}

        for resource_name in resources:
            try:
                data = self.fetch_data(resource_name, symbols, start_date, end_date, **kwargs)
                results[resource_name] = data
            except Exception as e:
                logger.error(f"Failed to fetch {resource_name}: {e}")
                if PANDAS_AVAILABLE:
                    results[resource_name] = pd.DataFrame()  # Empty DataFrame for failed resources
                else:
                    results[resource_name] = {}  # Empty dict for failed resources

        return results

    def get_data_summary(self, data, resource_name: str) -> Dict[str, Any]:
        """Get summary statistics for fetched data"""

        if not PANDAS_AVAILABLE:
            return {
                "resource": resource_name,
                "total_records": 0,
                "symbols": [],
                "date_range": None,
                "completeness": 0.0,
                "error": "pandas not available",
            }

        if data.empty:
            return {"resource": resource_name, "total_records": 0, "symbols": [], "date_range": None, "completeness": 0.0}

        summary = {
            "resource": resource_name,
            "total_records": len(data),
            "symbols": data["symbol"].unique().tolist() if "symbol" in data.columns else [],
            "date_range": {
                "start": data["date"].min() if "date" in data.columns else None,
                "end": data["date"].max() if "date" in data.columns else None,
            },
            "completeness": 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            "columns": list(data.columns),
        }

        return summary

    def _validate_data_quality(self, data, resource) -> Dict[str, Any]:
        """Validate data quality"""

        if not PANDAS_AVAILABLE:
            return {"quality_score": 0.0, "issues": ["pandas not available"]}

        if data.empty:
            return {"quality_score": 0.0, "issues": ["No data"]}

        issues = []
        score = 1.0

        # Check completeness
        completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        if completeness < 0.9:
            issues.append(f"Low completeness: {completeness:.2%}")
            score *= completeness

        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate records")
            score *= 0.9

        # Validate against resource schema
        invalid_records = 0
        for _, row in data.iterrows():
            if not validate_resource_data(row.to_dict(), resource):
                invalid_records += 1

        if invalid_records > 0:
            issues.append(f"Found {invalid_records} invalid records")
            score *= 1.0 - (invalid_records / len(data))

        return {
            "quality_score": score,
            "issues": issues,
            "completeness": completeness,
            "duplicates": duplicates,
            "invalid_records": invalid_records,
        }

    def get_fetch_statistics(self) -> Dict[str, Any]:
        """Get fetch statistics"""
        stats = self.fetch_stats.copy()
        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def get_source_health(self) -> Dict[str, Any]:
        """Get health status for all data sources"""
        health = {}

        for source_name, source in self.data_sources.items():
            try:
                # Test source connectivity
                test_symbols = ["AAPL"]
                test_start = datetime.now() - timedelta(days=1)
                test_end = datetime.now()

                # Try to fetch a small amount of data
                test_data = source.fetch_data("stock_price", test_symbols, test_start, test_end)

                health[source_name] = {
                    "status": "healthy",
                    "last_test": datetime.now(),
                    "test_result": "success" if not test_data.empty else "no_data",
                }

            except Exception as e:
                health[source_name] = {"status": "unhealthy", "last_test": datetime.now(), "error": str(e)}

        return health
