"""
Fundamental Indicators Component

Provides fundamental analysis indicators for signal generation.
"""

# Optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FundamentalIndicators:
    """Fundamental analysis indicators for signal generation"""

    def __init__(self):
        self.indicators = {
            "pe_ratio": self.price_to_earnings_ratio,
            "pb_ratio": self.price_to_book_ratio,
            "ps_ratio": self.price_to_sales_ratio,
            "ev_ebitda": self.enterprise_value_to_ebitda,
            "debt_to_equity": self.debt_to_equity_ratio,
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "roe": self.return_on_equity,
            "roa": self.return_on_assets,
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "revenue_growth": self.revenue_growth,
            "earnings_growth": self.earnings_growth,
            "dividend_yield": self.dividend_yield,
            "payout_ratio": self.payout_ratio,
        }

    def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate a specific fundamental indicator"""
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return self.indicators[indicator_name](data, **kwargs)

    def price_to_earnings_ratio(
        self, data: pd.DataFrame, price_column: str = "close", earnings_column: str = "earnings"
    ) -> pd.Series:
        """Calculate Price-to-Earnings Ratio"""
        if earnings_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[price_column] / data[earnings_column]

    def price_to_book_ratio(
        self, data: pd.DataFrame, price_column: str = "close", book_value_column: str = "book_value"
    ) -> pd.Series:
        """Calculate Price-to-Book Ratio"""
        if book_value_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[price_column] / data[book_value_column]

    def price_to_sales_ratio(
        self, data: pd.DataFrame, price_column: str = "close", revenue_column: str = "revenue"
    ) -> pd.Series:
        """Calculate Price-to-Sales Ratio"""
        if revenue_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[price_column] / data[revenue_column]

    def enterprise_value_to_ebitda(
        self, data: pd.DataFrame, ev_column: str = "enterprise_value", ebitda_column: str = "ebitda"
    ) -> pd.Series:
        """Calculate Enterprise Value to EBITDA"""
        if ev_column not in data.columns or ebitda_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[ev_column] / data[ebitda_column]

    def debt_to_equity_ratio(
        self, data: pd.DataFrame, debt_column: str = "total_debt", equity_column: str = "total_equity"
    ) -> pd.Series:
        """Calculate Debt-to-Equity Ratio"""
        if debt_column not in data.columns or equity_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[debt_column] / data[equity_column]

    def current_ratio(
        self,
        data: pd.DataFrame,
        current_assets_column: str = "current_assets",
        current_liabilities_column: str = "current_liabilities",
    ) -> pd.Series:
        """Calculate Current Ratio"""
        if current_assets_column not in data.columns or current_liabilities_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[current_assets_column] / data[current_liabilities_column]

    def quick_ratio(
        self,
        data: pd.DataFrame,
        current_assets_column: str = "current_assets",
        inventory_column: str = "inventory",
        current_liabilities_column: str = "current_liabilities",
    ) -> pd.Series:
        """Calculate Quick Ratio"""
        if (
            current_assets_column not in data.columns
            or inventory_column not in data.columns
            or current_liabilities_column not in data.columns
        ):
            return pd.Series(np.nan, index=data.index)

        return (data[current_assets_column] - data[inventory_column]) / data[current_liabilities_column]

    def return_on_equity(
        self, data: pd.DataFrame, net_income_column: str = "net_income", equity_column: str = "total_equity"
    ) -> pd.Series:
        """Calculate Return on Equity"""
        if net_income_column not in data.columns or equity_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[net_income_column] / data[equity_column]

    def return_on_assets(
        self, data: pd.DataFrame, net_income_column: str = "net_income", assets_column: str = "total_assets"
    ) -> pd.Series:
        """Calculate Return on Assets"""
        if net_income_column not in data.columns or assets_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[net_income_column] / data[assets_column]

    def gross_margin(
        self, data: pd.DataFrame, gross_profit_column: str = "gross_profit", revenue_column: str = "revenue"
    ) -> pd.Series:
        """Calculate Gross Margin"""
        if gross_profit_column not in data.columns or revenue_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[gross_profit_column] / data[revenue_column]

    def operating_margin(
        self, data: pd.DataFrame, operating_income_column: str = "operating_income", revenue_column: str = "revenue"
    ) -> pd.Series:
        """Calculate Operating Margin"""
        if operating_income_column not in data.columns or revenue_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[operating_income_column] / data[revenue_column]

    def net_margin(
        self, data: pd.DataFrame, net_income_column: str = "net_income", revenue_column: str = "revenue"
    ) -> pd.Series:
        """Calculate Net Margin"""
        if net_income_column not in data.columns or revenue_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[net_income_column] / data[revenue_column]

    def revenue_growth(self, data: pd.DataFrame, revenue_column: str = "revenue", period: int = 4) -> pd.Series:
        """Calculate Revenue Growth Rate"""
        if revenue_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[revenue_column].pct_change(periods=period)

    def earnings_growth(self, data: pd.DataFrame, earnings_column: str = "earnings", period: int = 4) -> pd.Series:
        """Calculate Earnings Growth Rate"""
        if earnings_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[earnings_column].pct_change(periods=period)

    def dividend_yield(self, data: pd.DataFrame, dividend_column: str = "dividend", price_column: str = "close") -> pd.Series:
        """Calculate Dividend Yield"""
        if dividend_column not in data.columns or price_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[dividend_column] / data[price_column]

    def payout_ratio(
        self, data: pd.DataFrame, dividend_column: str = "dividend", earnings_column: str = "earnings"
    ) -> pd.Series:
        """Calculate Payout Ratio"""
        if dividend_column not in data.columns or earnings_column not in data.columns:
            return pd.Series(np.nan, index=data.index)

        return data[dividend_column] / data[earnings_column]

    def generate_fundamental_signals(
        self, data: pd.DataFrame, indicators: List[str], parameters: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """Generate fundamental analysis signals"""
        if parameters is None:
            parameters = {}

        signals = data.copy()

        for indicator in indicators:
            try:
                indicator_data = self.calculate_indicator(indicator, data, **parameters.get(indicator, {}))
                signals[indicator] = indicator_data

            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
                continue

        return signals

    def get_fundamental_score(
        self,
        data: pd.DataFrame,
        positive_indicators: List[str],
        negative_indicators: List[str],
        weights: Dict[str, float] = None,
    ) -> pd.Series:
        """Calculate fundamental score based on indicators"""
        if weights is None:
            weights = {}

        score = pd.Series(0.0, index=data.index)

        # Positive indicators (higher is better)
        for indicator in positive_indicators:
            if indicator in data.columns:
                weight = weights.get(indicator, 1.0)
                # Normalize to 0-1 range
                normalized = (data[indicator] - data[indicator].min()) / (data[indicator].max() - data[indicator].min())
                score += weight * normalized

        # Negative indicators (lower is better)
        for indicator in negative_indicators:
            if indicator in data.columns:
                weight = weights.get(indicator, 1.0)
                # Normalize and invert (lower becomes higher)
                normalized = 1 - ((data[indicator] - data[indicator].min()) / (data[indicator].max() - data[indicator].min()))
                score += weight * normalized

        return score

    def get_valuation_signals(
        self, data: pd.DataFrame, valuation_indicators: List[str], thresholds: Dict[str, Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Generate valuation-based signals"""
        if thresholds is None:
            thresholds = {}

        signals = data.copy()

        for indicator in valuation_indicators:
            if indicator not in data.columns:
                continue

            indicator_thresholds = thresholds.get(indicator, {})
            undervalued_threshold = indicator_thresholds.get("undervalued", 0.3)
            overvalued_threshold = indicator_thresholds.get("overvalued", 0.7)

            # Calculate percentile rank
            percentile_rank = data[indicator].rank(pct=True)

            # Generate signals
            signals[f"{indicator}_undervalued"] = (percentile_rank < undervalued_threshold).astype(int)
            signals[f"{indicator}_overvalued"] = (percentile_rank > overvalued_threshold).astype(int)
            signals[f"{indicator}_fair_value"] = (
                (percentile_rank >= undervalued_threshold) & (percentile_rank <= overvalued_threshold)
            ).astype(int)

        return signals
