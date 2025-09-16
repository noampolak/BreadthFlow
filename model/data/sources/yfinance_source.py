"""
YFinance Data Source Implementation

Provides YFinance integration for the BreadthFlow abstraction system.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from ..resources.data_resources import STOCK_PRICE, DataResource
from .data_source_interface import DataSourceInterface

logger = logging.getLogger(__name__)


class YFinanceDataSource(DataSourceInterface):
    """YFinance data source implementation"""

    def __init__(self):
        self.name = "yfinance"
        self.supported_resources = [STOCK_PRICE.name]
        self.rate_limit_delay = 0.1  # 100ms between requests

    def get_name(self) -> str:
        """Get the name of this data source"""
        return self.name

    def get_config(self) -> Dict[str, Any]:
        """Get data source configuration"""
        return {"name": self.name, "supported_resources": self.supported_resources, "rate_limit_delay": self.rate_limit_delay}

    def get_supported_resources(self) -> List[str]:
        """Get list of supported resources"""
        return self.supported_resources

    def validate_resource_support(self, resource_name: str) -> bool:
        """Validate if this source supports the given resource"""
        return resource_name in self.supported_resources

    def fetch_data(
        self, resource_name: str, symbols: List[str], start_date: datetime, end_date: datetime, **kwargs
    ) -> pd.DataFrame:
        """Fetch data from YFinance"""

        try:
            logger.info(f"Fetching {resource_name} data for {symbols} from {start_date} to {end_date}")

            all_data = []

            for symbol in symbols:
                # Create YFinance ticker
                ticker = yf.Ticker(symbol)

                # Fetch historical data
                data = ticker.history(
                    start=start_date, end=end_date, interval="1d" if resource_name == "stock_price" else "1d"
                )

                if data.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue

                # Reset index to get date as column
                data = data.reset_index()

                # Add symbol column
                data["symbol"] = symbol

                # Rename columns to match our schema
                column_mapping = {
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }

                # Only rename columns that exist
                existing_columns = {k: v for k, v in column_mapping.items() if k in data.columns}
                data = data.rename(columns=existing_columns)

                # Add adj_close if it exists, otherwise use close
                if "Adj Close" in data.columns:
                    data = data.rename(columns={"Adj Close": "adj_close"})
                else:
                    data["adj_close"] = data["close"]

                # Select only the columns we need
                columns = ["symbol", "date", "open", "high", "low", "close", "volume", "adj_close"]
                data = data[columns]

                all_data.append(data)

                logger.info(f"Successfully fetched {len(data)} rows for {symbol}")

            if not all_data:
                return pd.DataFrame()

            # Combine all data
            result = pd.concat(all_data, ignore_index=True)

            return result

        except Exception as e:
            logger.error(f"Error fetching data for {symbols}: {e}")
            raise

    def get_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score"""
        if data.empty:
            return 0.0

        try:
            # Calculate quality metrics
            total_rows = len(data)
            null_count = data.isnull().sum().sum()
            quality_score = 1.0 - (null_count / (total_rows * len(data.columns)))

            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
