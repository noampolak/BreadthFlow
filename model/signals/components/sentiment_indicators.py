"""
Sentiment Indicators Component

Provides sentiment analysis indicators for signal generation.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    DataFrame = DataFrame
    Series = Series
except ImportError:
    PANDAS_AVAILABLE = False
    # Create dummy types for type hints when pandas is not available
    DataFrame = Any
    Series = Any

logger = logging.getLogger(__name__)


class SentimentIndicators:
    """Sentiment analysis indicators for signal generation"""

    def __init__(self):
        self.indicators = {
            "news_sentiment": self.news_sentiment_score,
            "social_sentiment": self.social_media_sentiment,
            "analyst_rating": self.analyst_rating_score,
            "insider_trading": self.insider_trading_signal,
            "options_flow": self.options_flow_sentiment,
            "short_interest": self.short_interest_ratio,
            "put_call_ratio": self.put_call_ratio,
            "fear_greed_index": self.fear_greed_index,
            "volatility_index": self.volatility_index,
            "momentum_sentiment": self.momentum_sentiment,
        }

    def calculate_indicator(self, indicator_name: str, data: DataFrame, **kwargs) -> Series:
        """Calculate a specific sentiment indicator"""
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return self.indicators[indicator_name](data, **kwargs)

    def news_sentiment_score(
        self, data: DataFrame, sentiment_column: str = "sentiment_score", volume_column: str = "news_count"
    ) -> Series:
        """Calculate weighted news sentiment score"""
        if sentiment_column not in data.columns:
            return Series(np.nan, index=data.index)

        sentiment = data[sentiment_column]

        # Apply volume weighting if available
        if volume_column in data.columns:
            volume = data[volume_column]
            # Normalize volume to 0-1 range
            volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())
            # Weight sentiment by volume
            weighted_sentiment = sentiment * volume_normalized
            return weighted_sentiment

        return sentiment

    def social_media_sentiment(
        self,
        data: DataFrame,
        positive_mentions_column: str = "positive_mentions",
        negative_mentions_column: str = "negative_mentions",
        total_mentions_column: str = "total_mentions",
    ) -> Series:
        """Calculate social media sentiment score"""
        if positive_mentions_column not in data.columns or negative_mentions_column not in data.columns:
            return Series(np.nan, index=data.index)

        positive = data[positive_mentions_column]
        negative = data[negative_mentions_column]

        # Calculate sentiment ratio
        total = positive + negative
        sentiment = np.where(total > 0, (positive - negative) / total, 0)

        return Series(sentiment, index=data.index)

    def analyst_rating_score(
        self,
        data: DataFrame,
        buy_ratings_column: str = "buy_ratings",
        hold_ratings_column: str = "hold_ratings",
        sell_ratings_column: str = "sell_ratings",
    ) -> Series:
        """Calculate analyst rating score"""
        if (
            buy_ratings_column not in data.columns
            or hold_ratings_column not in data.columns
            or sell_ratings_column not in data.columns
        ):
            return Series(np.nan, index=data.index)

        buy = data[buy_ratings_column]
        hold = data[hold_ratings_column]
        sell = data[sell_ratings_column]

        # Calculate weighted score (Buy=1, Hold=0.5, Sell=0)
        total = buy + hold + sell
        score = np.where(total > 0, (buy + 0.5 * hold) / total, 0.5)

        return Series(score, index=data.index)

    def insider_trading_signal(
        self,
        data: DataFrame,
        insider_buy_volume_column: str = "insider_buy_volume",
        insider_sell_volume_column: str = "insider_sell_volume",
        market_volume_column: str = "volume",
    ) -> Series:
        """Calculate insider trading sentiment"""
        if insider_buy_volume_column not in data.columns or insider_sell_volume_column not in data.columns:
            return Series(np.nan, index=data.index)

        buy_volume = data[insider_buy_volume_column]
        sell_volume = data[insider_sell_volume_column]

        # Calculate insider trading ratio
        total_insider = buy_volume + sell_volume
        insider_ratio = np.where(total_insider > 0, (buy_volume - sell_volume) / total_insider, 0)

        # Normalize to market volume if available
        if market_volume_column in data.columns:
            market_volume = data[market_volume_column]
            normalized_ratio = insider_ratio * (total_insider / market_volume)
            return Series(normalized_ratio, index=data.index)

        return Series(insider_ratio, index=data.index)

    def options_flow_sentiment(
        self,
        data: DataFrame,
        call_volume_column: str = "call_volume",
        put_volume_column: str = "put_volume",
        call_premium_column: str = "call_premium",
        put_premium_column: str = "put_premium",
    ) -> Series:
        """Calculate options flow sentiment"""
        if call_volume_column not in data.columns or put_volume_column not in data.columns:
            return Series(np.nan, index=data.index)

        call_volume = data[call_volume_column]
        put_volume = data[put_volume_column]

        # Calculate put-call ratio
        put_call_ratio = np.where(call_volume > 0, put_volume / call_volume, 1)

        # Calculate premium flow if available
        if call_premium_column in data.columns and put_premium_column in data.columns:
            call_premium = data[call_premium_column]
            put_premium = data[put_premium_column]
            premium_flow = call_premium - put_premium
            # Combine volume and premium flow
            sentiment = (1 / put_call_ratio) + (premium_flow / (call_premium + put_premium))
        else:
            sentiment = 1 / put_call_ratio

        return Series(sentiment, index=data.index)

    def short_interest_ratio(
        self,
        data: DataFrame,
        short_interest_column: str = "short_interest",
        shares_outstanding_column: str = "shares_outstanding",
        avg_volume_column: str = "avg_volume",
    ) -> Series:
        """Calculate short interest ratio"""
        if short_interest_column not in data.columns:
            return Series(np.nan, index=data.index)

        short_interest = data[short_interest_column]

        # Calculate short interest as percentage of shares outstanding
        if shares_outstanding_column in data.columns:
            shares_outstanding = data[shares_outstanding_column]
            short_ratio = short_interest / shares_outstanding
        else:
            short_ratio = short_interest

        # Calculate days to cover if average volume is available
        if avg_volume_column in data.columns:
            avg_volume = data[avg_volume_column]
            days_to_cover = np.where(avg_volume > 0, short_interest / avg_volume, 0)
            # Combine both metrics
            sentiment = short_ratio * days_to_cover
        else:
            sentiment = short_ratio

        return Series(sentiment, index=data.index)

    def put_call_ratio(
        self, data: DataFrame, put_volume_column: str = "put_volume", call_volume_column: str = "call_volume"
    ) -> Series:
        """Calculate put-call ratio"""
        if put_volume_column not in data.columns or call_volume_column not in data.columns:
            return Series(np.nan, index=data.index)

        put_volume = data[put_volume_column]
        call_volume = data[call_volume_column]

        ratio = np.where(call_volume > 0, put_volume / call_volume, 1)

        return Series(ratio, index=data.index)

    def fear_greed_index(
        self,
        data: DataFrame,
        volatility_column: str = "volatility",
        momentum_column: str = "momentum",
        market_volume_column: str = "volume",
        put_call_column: str = "put_call_ratio",
    ) -> Series:
        """Calculate fear and greed index"""
        components = []

        # Volatility component (inverse relationship)
        if volatility_column in data.columns:
            vol = data[volatility_column]
            vol_normalized = 1 - ((vol - vol.min()) / (vol.max() - vol.min()))
            components.append(vol_normalized)

        # Momentum component
        if momentum_column in data.columns:
            mom = data[momentum_column]
            mom_normalized = (mom - mom.min()) / (mom.max() - mom.min())
            components.append(mom_normalized)

        # Market volume component
        if market_volume_column in data.columns:
            vol = data[market_volume_column]
            vol_normalized = (vol - vol.min()) / (vol.max() - vol.min())
            components.append(vol_normalized)

        # Put-call ratio component (inverse relationship)
        if put_call_column in data.columns:
            pc_ratio = data[put_call_column]
            pc_normalized = 1 - ((pc_ratio - pc_ratio.min()) / (pc_ratio.max() - pc_ratio.min()))
            components.append(pc_normalized)

        if not components:
            return Series(np.nan, index=data.index)

        # Calculate average of all components
        fear_greed = pd.concat(components, axis=1).mean(axis=1)

        return fear_greed

    def volatility_index(
        self,
        data: DataFrame,
        high_column: str = "high",
        low_column: str = "low",
        close_column: str = "close",
        period: int = 20,
    ) -> Series:
        """Calculate volatility index"""
        if high_column not in data.columns or low_column not in data.columns or close_column not in data.columns:
            return Series(np.nan, index=data.index)

        high = data[high_column]
        low = data[low_column]
        close = data[close_column]

        # Calculate true range
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))

        # Calculate average true range
        atr = true_range.rolling(window=period).mean()

        # Normalize to price
        volatility = atr / close

        return volatility

    def momentum_sentiment(
        self, data: DataFrame, price_column: str = "close", volume_column: str = "volume", period: int = 14
    ) -> Series:
        """Calculate momentum-based sentiment"""
        if price_column not in data.columns:
            return Series(np.nan, index=data.index)

        price = data[price_column]

        # Calculate price momentum
        momentum = price.pct_change(periods=period)

        # Calculate volume momentum if available
        if volume_column in data.columns:
            volume = data[volume_column]
            volume_momentum = volume.pct_change(periods=period)

            # Combine price and volume momentum
            sentiment = momentum * volume_momentum
        else:
            sentiment = momentum

        return sentiment

    def generate_sentiment_signals(
        self, data: DataFrame, indicators: List[str], parameters: Dict[str, Any] = None
    ) -> DataFrame:
        """Generate sentiment analysis signals"""
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

    def get_sentiment_score(
        self, data: DataFrame, sentiment_indicators: List[str], weights: Dict[str, float] = None
    ) -> Series:
        """Calculate composite sentiment score"""
        if weights is None:
            weights = {}

        score = Series(0.0, index=data.index)
        total_weight = 0

        for indicator in sentiment_indicators:
            if indicator in data.columns:
                weight = weights.get(indicator, 1.0)
                # Normalize to -1 to 1 range
                indicator_data = data[indicator]
                normalized = 2 * ((indicator_data - indicator_data.min()) / (indicator_data.max() - indicator_data.min())) - 1
                score += weight * normalized
                total_weight += weight

        if total_weight > 0:
            score = score / total_weight

        return score

    def get_sentiment_signals(
        self, data: DataFrame, sentiment_indicators: List[str], thresholds: Dict[str, float] = None
    ) -> DataFrame:
        """Generate sentiment-based trading signals"""
        if thresholds is None:
            thresholds = {"bullish": 0.3, "bearish": -0.3, "neutral": 0.1}

        signals = data.copy()

        # Calculate composite sentiment score
        sentiment_score = self.get_sentiment_score(data, sentiment_indicators)

        # Generate signals based on thresholds
        signals["sentiment_bullish"] = (sentiment_score > thresholds["bullish"]).astype(int)
        signals["sentiment_bearish"] = (sentiment_score < thresholds["bearish"]).astype(int)
        signals["sentiment_neutral"] = (
            (sentiment_score >= thresholds["bearish"]) & (sentiment_score <= thresholds["bullish"])
        ).astype(int)
        signals["sentiment_score"] = sentiment_score

        return signals

    # Test compatibility methods
    def calculate_news_sentiment(self, news_data: List[Dict[str, Any]]) -> float:
        """Calculate news sentiment - test compatibility method"""
        return self.analyze_news_sentiment(news_data)

    def calculate_social_sentiment(self, social_data: List[Dict[str, Any]]) -> float:
        """Calculate social sentiment - test compatibility method"""
        return self.analyze_social_sentiment(social_data)

    def analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> float:
        """Analyze news sentiment"""
        if not news_data:
            return 0.0

        # Simple sentiment analysis based on news data
        total_sentiment = 0.0
        count = 0

        for news_item in news_data:
            if "sentiment" in news_item:
                sentiment_value = news_item["sentiment"]
                # Handle both numeric and string sentiment values
                if isinstance(sentiment_value, str):
                    # Convert string sentiment to numeric
                    sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                    sentiment_value = sentiment_map.get(sentiment_value.lower(), 0.0)
                total_sentiment += sentiment_value
                count += 1

        return total_sentiment / count if count > 0 else 0.0

    def analyze_social_sentiment(self, social_data: List[Dict[str, Any]]) -> float:
        """Analyze social media sentiment"""
        if not social_data:
            return 0.0

        # Simple sentiment analysis based on social data
        total_sentiment = 0.0
        count = 0

        for social_item in social_data:
            if "sentiment" in social_item:
                sentiment_value = social_item["sentiment"]
                # Handle both numeric and string sentiment values
                if isinstance(sentiment_value, str):
                    # Convert string sentiment to numeric
                    sentiment_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                    sentiment_value = sentiment_map.get(sentiment_value.lower(), 0.0)
                total_sentiment += sentiment_value
                count += 1

        return total_sentiment / count if count > 0 else 0.0
