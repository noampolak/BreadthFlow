"""
Unit tests for signal generation components.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from model.signals.components.fundamental_indicators import FundamentalIndicators
from model.signals.components.sentiment_indicators import SentimentIndicators

# Import signal generation components
from model.signals.components.technical_indicators import TechnicalIndicators
from model.signals.composite_signal_generator import CompositeSignalGenerator
from model.signals.signal_config import SignalConfig
from model.signals.strategies.fundamental_analysis_strategy import FundamentalAnalysisStrategy
from model.signals.strategies.technical_analysis_strategy import TechnicalAnalysisStrategy


class TestTechnicalIndicators:
    """Test technical indicators calculations"""

    @pytest.fixture
    def sample_data(self, sample_ohlcv_data):
        """Sample OHLCV data for testing"""
        return sample_ohlcv_data

    @pytest.fixture
    def indicators(self):
        """Technical indicators instance"""
        return TechnicalIndicators()

    def test_rsi_calculation(self, indicators, sample_data):
        """Test RSI calculation accuracy"""
        symbol_data = sample_data[sample_data["symbol"] == "AAPL"]
        rsi = indicators.calculate_rsi(symbol_data["close"], period=14)

        assert len(rsi) == len(symbol_data)
        assert all(0 <= val <= 100 for val in rsi.dropna())
        assert not rsi.isna().all()  # Should have some valid values

    def test_macd_calculation(self, indicators, sample_data):
        """Test MACD calculation"""
        symbol_data = sample_data[sample_data["symbol"] == "AAPL"]
        macd, signal, histogram = indicators.calculate_macd(symbol_data["close"])

        assert len(macd) == len(symbol_data)
        assert len(signal) == len(symbol_data)
        assert len(histogram) == len(symbol_data)
        assert not macd.isna().all()

    def test_bollinger_bands(self, indicators, sample_data):
        """Test Bollinger Bands calculation"""
        symbol_data = sample_data[sample_data["symbol"] == "AAPL"]
        upper, middle, lower = indicators.calculate_bollinger_bands(symbol_data["close"], period=20, std_dev=2)

        assert len(upper) == len(symbol_data)
        assert len(middle) == len(symbol_data)
        assert len(lower) == len(symbol_data)
        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_sma_calculation(self, indicators, sample_data):
        """Test Simple Moving Average calculation"""
        symbol_data = sample_data[sample_data["symbol"] == "AAPL"]
        sma = indicators.calculate_sma(symbol_data["close"], period=20)

        assert len(sma) == len(symbol_data)
        assert not sma.isna().all()
        assert sma.iloc[-1] == symbol_data["close"].tail(20).mean()


class TestFundamentalIndicators:
    """Test fundamental indicators calculations"""

    @pytest.fixture
    def indicators(self):
        """Fundamental indicators instance"""
        return FundamentalIndicators()

    def test_pe_ratio_calculation(self, indicators):
        """Test P/E ratio calculation"""
        price = 150.0
        earnings_per_share = 5.0
        pe_ratio = indicators.calculate_pe_ratio(price, earnings_per_share)

        assert pe_ratio == 30.0

    def test_pb_ratio_calculation(self, indicators):
        """Test P/B ratio calculation"""
        price = 150.0
        book_value_per_share = 50.0
        pb_ratio = indicators.calculate_pb_ratio(price, book_value_per_share)

        assert pb_ratio == 3.0

    def test_roe_calculation(self, indicators):
        """Test ROE calculation"""
        net_income = 1000000
        shareholders_equity = 5000000
        roe = indicators.calculate_roe(net_income, shareholders_equity)

        assert roe == 0.2


class TestSentimentIndicators:
    """Test sentiment indicators calculations"""

    @pytest.fixture
    def indicators(self):
        """Sentiment indicators instance"""
        return SentimentIndicators()

    def test_news_sentiment(self, indicators):
        """Test news sentiment calculation"""
        news_data = [
            {"sentiment": "positive", "confidence": 0.8},
            {"sentiment": "negative", "confidence": 0.6},
            {"sentiment": "neutral", "confidence": 0.5},
        ]

        sentiment_score = indicators.calculate_news_sentiment(news_data)

        assert -1 <= sentiment_score <= 1
        assert isinstance(sentiment_score, float)

    def test_social_sentiment(self, indicators):
        """Test social media sentiment calculation"""
        social_data = [
            {"sentiment": "positive", "engagement": 100},
            {"sentiment": "negative", "engagement": 50},
            {"sentiment": "neutral", "engagement": 25},
        ]

        sentiment_score = indicators.calculate_social_sentiment(social_data)

        assert -1 <= sentiment_score <= 1
        assert isinstance(sentiment_score, float)


class TestTechnicalAnalysisStrategy:
    """Test technical analysis strategy"""

    @pytest.fixture
    def strategy(self):
        """Technical analysis strategy instance"""
        config = SignalConfig(rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, bollinger_period=20, bollinger_std=2)
        return TechnicalAnalysisStrategy(config)

    def test_signal_generation(self, strategy, sample_ohlcv_data):
        """Test signal generation"""
        symbol_data = sample_ohlcv_data[sample_ohlcv_data["symbol"] == "AAPL"]
        signals = strategy.generate_signals(symbol_data)

        assert len(signals) == len(symbol_data)
        assert "signal_type" in signals.columns
        assert "signal_strength" in signals.columns
        assert "confidence" in signals.columns
        assert signals["signal_type"].isin(["buy", "sell", "hold"]).all()

    def test_signal_strength_range(self, strategy, sample_ohlcv_data):
        """Test signal strength is within valid range"""
        symbol_data = sample_ohlcv_data[sample_ohlcv_data["symbol"] == "AAPL"]
        signals = strategy.generate_signals(symbol_data)

        assert (signals["signal_strength"] >= 0).all()
        assert (signals["signal_strength"] <= 1).all()

    def test_confidence_range(self, strategy, sample_ohlcv_data):
        """Test confidence is within valid range"""
        symbol_data = sample_ohlcv_data[sample_ohlcv_data["symbol"] == "AAPL"]
        signals = strategy.generate_signals(symbol_data)

        assert (signals["confidence"] >= 0).all()
        assert (signals["confidence"] <= 1).all()


class TestFundamentalAnalysisStrategy:
    """Test fundamental analysis strategy"""

    @pytest.fixture
    def strategy(self):
        """Fundamental analysis strategy instance"""
        config = SignalConfig(pe_threshold=20, pb_threshold=3, roe_threshold=0.15)
        return FundamentalAnalysisStrategy(config)

    def test_signal_generation(self, strategy):
        """Test signal generation with fundamental data"""
        fundamental_data = pd.DataFrame({"pe_ratio": [15, 25, 30], "pb_ratio": [2, 4, 5], "roe": [0.2, 0.1, 0.05]})

        signals = strategy.generate_signals(fundamental_data)

        assert len(signals) == len(fundamental_data)
        assert "signal_type" in signals.columns
        assert "signal_strength" in signals.columns
        assert "confidence" in signals.columns


class TestCompositeSignalGenerator:
    """Test composite signal generator"""

    @pytest.fixture
    def generator(self):
        """Composite signal generator instance"""
        config = SignalConfig(rsi_period=14, macd_fast=12, macd_slow=26, combination_method="weighted_average")
        return CompositeSignalGenerator(config)

    def test_signal_combination(self, generator, sample_ohlcv_data):
        """Test signal combination from multiple strategies"""
        symbol_data = sample_ohlcv_data[sample_ohlcv_data["symbol"] == "AAPL"]
        signals = generator.generate_signals(symbol_data)

        assert len(signals) == len(symbol_data)
        assert "signal_type" in signals.columns
        assert "signal_strength" in signals.columns
        assert "confidence" in signals.columns
        assert "consensus_score" in signals.columns

    def test_consensus_filtering(self, generator, sample_ohlcv_data):
        """Test consensus filtering removes conflicting signals"""
        symbol_data = sample_ohlcv_data[sample_ohlcv_data["symbol"] == "AAPL"]
        signals = generator.generate_signals(symbol_data)

        # Check that consensus score is calculated
        assert "consensus_score" in signals.columns
        assert (signals["consensus_score"] >= 0).all()
        assert (signals["consensus_score"] <= 1).all()


class TestSignalConfig:
    """Test signal configuration"""

    def test_config_creation(self):
        """Test signal configuration creation"""
        config = SignalConfig(rsi_period=14, macd_fast=12, macd_slow=26)

        assert config.rsi_period == 14
        assert config.macd_fast == 12
        assert config.macd_slow == 26

    def test_config_validation(self):
        """Test signal configuration validation"""
        with pytest.raises(ValueError):
            SignalConfig(rsi_period=-1)  # Invalid period

        with pytest.raises(ValueError):
            SignalConfig(macd_fast=30, macd_slow=20)  # Fast > Slow

    def test_config_serialization(self):
        """Test signal configuration serialization"""
        config = SignalConfig(rsi_period=14, macd_fast=12, macd_slow=26)

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["rsi_period"] == 14

        # Test deserialization
        new_config = SignalConfig.from_dict(config_dict)
        assert new_config.rsi_period == 14
        assert new_config.macd_fast == 12
        assert new_config.macd_slow == 26
