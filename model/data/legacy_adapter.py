"""
Legacy Adapter for Backward Compatibility

Provides backward compatibility with the existing TimeframeAgnosticFetcher
while integrating with the new universal data fetching system.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .resources.data_resources import STOCK_PRICE, validate_resource_data
from .sources.data_source_interface import DataSourceInterface

logger = logging.getLogger(__name__)


class LegacyTimeframeAdapter(DataSourceInterface):
    """Adapter for existing TimeframeAgnosticFetcher"""

    def __init__(self, legacy_fetcher=None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = "legacy_timeframe_adapter"
        self.supported_resources = ["stock_price"]
        self.legacy_fetcher = legacy_fetcher

        # Mapping from new resource names to legacy parameters
        self.resource_mapping = {
            "stock_price": {
                "legacy_method": "fetch_ohlcv",
                "timeframe_mapping": {"1min": "1m", "5min": "5m", "15min": "15m", "1hour": "1h", "1day": "1d"},
            }
        }

    def get_name(self) -> str:
        """Get the name of this data source"""
        return self.name

    def get_supported_resources(self) -> List[str]:
        """Get list of supported resource names"""
        return self.supported_resources

    def get_config(self) -> Dict[str, Any]:
        """Get data source configuration"""
        return self.config

    def fetch_data(
        self, resource_name: str, symbols: List[str], start_date: datetime, end_date: datetime, **kwargs
    ) -> pd.DataFrame:
        """Fetch data using legacy fetcher"""

        if resource_name not in self.supported_resources:
            raise ValueError(f"Unsupported resource: {resource_name}")

        if not self.legacy_fetcher:
            raise ValueError("Legacy fetcher not available")

        try:
            # Map resource to legacy method
            resource_config = self.resource_mapping[resource_name]
            legacy_method = resource_config["legacy_method"]

            # Get timeframe from kwargs or use default
            timeframe = kwargs.get("timeframe", "1day")
            legacy_timeframe = resource_config["timeframe_mapping"].get(timeframe, "1d")

            # Call legacy fetcher
            if legacy_method == "fetch_ohlcv":
                data = self._fetch_ohlcv_legacy(symbols, start_date, end_date, legacy_timeframe, **kwargs)
            else:
                raise ValueError(f"Unknown legacy method: {legacy_method}")

            # Validate and transform data
            if not data.empty:
                data = self._transform_to_new_format(data, resource_name)

            return data

        except Exception as e:
            logger.error(f"Error in legacy adapter: {e}")
            raise

    def _fetch_ohlcv_legacy(
        self, symbols: List[str], start_date: datetime, end_date: datetime, timeframe: str, **kwargs
    ) -> pd.DataFrame:
        """Fetch OHLCV data using legacy fetcher"""

        all_data = []

        for symbol in symbols:
            try:
                # Call legacy fetcher method
                # This assumes the legacy fetcher has a method like fetch_ohlcv
                if hasattr(self.legacy_fetcher, "fetch_ohlcv"):
                    symbol_data = self.legacy_fetcher.fetch_ohlcv(
                        symbol=symbol, start_date=start_date, end_date=end_date, timeframe=timeframe, **kwargs
                    )
                elif hasattr(self.legacy_fetcher, "fetch_data"):
                    # Alternative method name
                    symbol_data = self.legacy_fetcher.fetch_data(
                        symbol=symbol, start_date=start_date, end_date=end_date, timeframe=timeframe, **kwargs
                    )
                else:
                    logger.warning(f"Legacy fetcher doesn't have expected methods for {symbol}")
                    continue

                if symbol_data is not None and not symbol_data.empty:
                    # Add symbol column if not present
                    if "symbol" not in symbol_data.columns:
                        symbol_data["symbol"] = symbol

                    all_data.append(symbol_data)

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        return result

    def _transform_to_new_format(self, data: pd.DataFrame, resource_name: str) -> pd.DataFrame:
        """Transform legacy data format to new format"""

        if resource_name == "stock_price":
            # Ensure required columns exist
            required_columns = ["symbol", "date", "open", "high", "low", "close", "volume"]

            # Map common column names
            column_mapping = {
                "Date": "date",
                "Time": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
                "Symbol": "symbol",
            }

            # Rename columns if needed
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data = data.rename(columns={old_col: new_col})

            # Ensure date column is datetime
            if "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])

            # Select only the columns we need
            available_columns = [col for col in required_columns if col in data.columns]
            data = data[available_columns]

            # Validate data
            for _, row in data.iterrows():
                if not validate_resource_data(row.to_dict(), STOCK_PRICE):
                    logger.warning(f"Invalid stock price data for {row.get('symbol', 'unknown')}")

        return data

    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting information for this data source"""
        return {
            "requests_per_minute": 30,  # Conservative for legacy system
            "requests_per_hour": 500,
            "requests_per_day": 5000,
        }

    def get_data_quality_info(self) -> Dict[str, Any]:
        """Get data quality information for this data source"""
        return {
            "completeness": 0.85,  # Legacy system might have lower quality
            "accuracy": 0.90,
            "latency": "medium",
            "update_frequency": "batch",
            "legacy_system": True,
        }


class LegacyAdapterManager:
    """Manages legacy adapters for backward compatibility"""

    def __init__(self):
        self.adapters = {}
        self.enabled = True

    def register_legacy_fetcher(self, name: str, legacy_fetcher, config: Dict[str, Any] = None):
        """Register a legacy fetcher with adapter"""
        adapter = LegacyTimeframeAdapter(legacy_fetcher, config)
        self.adapters[name] = adapter
        logger.info(f"✅ Registered legacy adapter: {name}")

    def get_adapter(self, name: str) -> Optional[LegacyTimeframeAdapter]:
        """Get a legacy adapter by name"""
        return self.adapters.get(name)

    def list_adapters(self) -> List[str]:
        """List all registered adapters"""
        return list(self.adapters.keys())

    def enable_legacy_support(self):
        """Enable legacy adapter support"""
        self.enabled = True
        logger.info("✅ Legacy adapter support enabled")

    def disable_legacy_support(self):
        """Disable legacy adapter support"""
        self.enabled = False
        logger.info("⚠️ Legacy adapter support disabled")

    def is_enabled(self) -> bool:
        """Check if legacy support is enabled"""
        return self.enabled
