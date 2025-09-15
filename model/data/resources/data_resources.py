"""
Data Resources Definitions

Defines the core data resource types, frequencies, and field specifications
for the BreadthFlow data management system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    """Types of data resources"""

    TRADING = "trading"
    FUNDAMENTAL = "fundamental"
    ALTERNATIVE = "alternative"
    CUSTOM = "custom"


class DataFrequency(Enum):
    """Data update frequencies"""

    REALTIME = "realtime"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    HOUR_1 = "1hour"
    DAY_1 = "1day"
    WEEK_1 = "1week"
    MONTH_1 = "1month"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class ResourceField:
    """Definition of a data field within a resource"""

    name: str
    type: str  # 'string', 'integer', 'float', 'datetime', 'boolean'
    description: str
    required: bool = True
    default: Any = None
    validation_rules: Dict[str, Any] = None


@dataclass
class DataResource:
    """Definition of a data resource"""

    name: str
    type: ResourceType
    frequency: DataFrequency
    fields: List[ResourceField]
    description: str
    source_requirements: Dict[str, Any] = None
    validation_rules: Dict[str, Any] = None


# Predefined data resources

# Stock Price Resource
STOCK_PRICE = DataResource(
    name="stock_price",
    type=ResourceType.TRADING,
    frequency=DataFrequency.MINUTE_1,
    description="OHLCV stock price data",
    fields=[
        ResourceField("symbol", "string", "Stock symbol", True),
        ResourceField("date", "datetime", "Timestamp", True),
        ResourceField("open", "float", "Opening price", True),
        ResourceField("high", "float", "Highest price", True),
        ResourceField("low", "float", "Lowest price", True),
        ResourceField("close", "float", "Closing price", True),
        ResourceField("volume", "integer", "Trading volume", True),
        ResourceField("adj_close", "float", "Adjusted closing price", False),
    ],
    validation_rules={"price_positive": True, "high_ge_low": True, "volume_positive": True},
)

# Revenue Resource
REVENUE = DataResource(
    name="revenue",
    type=ResourceType.FUNDAMENTAL,
    frequency=DataFrequency.QUARTERLY,
    description="Company revenue data",
    fields=[
        ResourceField("symbol", "string", "Stock symbol", True),
        ResourceField("date", "datetime", "Report date", True),
        ResourceField("revenue", "float", "Revenue amount", True),
        ResourceField("currency", "string", "Currency code", False, "USD"),
        ResourceField("period", "string", "Reporting period", True),
        ResourceField("source", "string", "Data source", False),
    ],
    validation_rules={"revenue_positive": True, "date_valid": True},
)

# Market Cap Resource
MARKET_CAP = DataResource(
    name="market_cap",
    type=ResourceType.FUNDAMENTAL,
    frequency=DataFrequency.DAY_1,
    description="Market capitalization data",
    fields=[
        ResourceField("symbol", "string", "Stock symbol", True),
        ResourceField("date", "datetime", "Date", True),
        ResourceField("market_cap", "float", "Market capitalization", True),
        ResourceField("shares_outstanding", "integer", "Shares outstanding", False),
        ResourceField("currency", "string", "Currency code", False, "USD"),
        ResourceField("source", "string", "Data source", False),
    ],
    validation_rules={"market_cap_positive": True, "shares_positive": True},
)

# News Sentiment Resource
NEWS_SENTIMENT = DataResource(
    name="news_sentiment",
    type=ResourceType.ALTERNATIVE,
    frequency=DataFrequency.HOUR_1,
    description="News sentiment analysis data",
    fields=[
        ResourceField("symbol", "string", "Stock symbol", True),
        ResourceField("date", "datetime", "Timestamp", True),
        ResourceField("sentiment_score", "float", "Sentiment score (-1 to 1)", True),
        ResourceField("sentiment_label", "string", "Sentiment label", False),
        ResourceField("news_count", "integer", "Number of news articles", False),
        ResourceField("source", "string", "Data source", False),
        ResourceField("confidence", "float", "Confidence score", False),
    ],
    validation_rules={"sentiment_range": (-1.0, 1.0), "confidence_range": (0.0, 1.0)},
)

# Additional predefined resources can be added here
# P/E Ratio, P/B Ratio, Earnings, etc.


def get_resource_by_name(name: str) -> Optional[DataResource]:
    """Get a predefined resource by name"""
    resources = {"stock_price": STOCK_PRICE, "revenue": REVENUE, "market_cap": MARKET_CAP, "news_sentiment": NEWS_SENTIMENT}
    return resources.get(name)


def list_available_resources() -> List[str]:
    """List all available predefined resources"""
    return ["stock_price", "revenue", "market_cap", "news_sentiment"]


def validate_resource_data(data: Dict[str, Any], resource: DataResource) -> bool:
    """Validate data against resource definition"""
    # Check required fields
    for field in resource.fields:
        if field.required and field.name not in data:
            return False

    # Check field types
    for field in resource.fields:
        if field.name in data:
            value = data[field.name]
            if not _validate_field_type(value, field.type):
                return False

    # Check validation rules
    if resource.validation_rules:
        if not _validate_resource_rules(data, resource.validation_rules):
            return False

    return True


def _validate_field_type(value: Any, expected_type: str) -> bool:
    """Validate field type"""
    type_mapping = {
        "string": str,
        "integer": int,
        "float": (int, float),
        "datetime": str,  # Simplified for now
        "boolean": bool,
    }

    expected_class = type_mapping.get(expected_type)
    if expected_class is None:
        return True

    return isinstance(value, expected_class)


def _validate_resource_rules(data: Dict[str, Any], rules: Dict[str, Any]) -> bool:
    """Validate resource-specific rules"""
    for rule, value in rules.items():
        if rule == "price_positive":
            price_fields = ["open", "high", "low", "close"]
            for field in price_fields:
                if field in data and data[field] <= 0:
                    return False

        elif rule == "high_ge_low":
            if "high" in data and "low" in data:
                if data["high"] < data["low"]:
                    return False

        elif rule == "volume_positive":
            if "volume" in data and data["volume"] < 0:
                return False

        elif rule == "revenue_positive":
            if "revenue" in data and data["revenue"] <= 0:
                return False

        elif rule == "market_cap_positive":
            if "market_cap" in data and data["market_cap"] <= 0:
                return False

        elif rule == "sentiment_range":
            if "sentiment_score" in data:
                min_val, max_val = value
                if not (min_val <= data["sentiment_score"] <= max_val):
                    return False

        elif rule == "confidence_range":
            if "confidence" in data:
                min_val, max_val = value
                if not (min_val <= data["confidence"] <= max_val):
                    return False

    return True
