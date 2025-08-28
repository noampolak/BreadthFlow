#!/usr/bin/env python3
"""
Signal Generation System Test Script

Tests the Phase 3 signal generation abstraction components.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the model subdirectories to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'signals'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'signals', 'components'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'signals', 'strategies'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'logging'))

# Import our signal generation components
from signal_config import SignalConfig
from signal_generator_interface import SignalGeneratorInterface
from components.technical_indicators import TechnicalIndicators
from components.fundamental_indicators import FundamentalIndicators
from components.sentiment_indicators import SentimentIndicators
from strategies.base_signal_strategy import BaseSignalStrategy
from strategies.technical_analysis_strategy import TechnicalAnalysisStrategy
from strategies.fundamental_analysis_strategy import FundamentalAnalysisStrategy
from composite_signal_generator import CompositeSignalGenerator

def create_mock_data():
    """Create mock data for testing"""
    # Create mock stock price data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    symbols = ['AAPL', 'MSFT']
    
    stock_data = []
    for symbol in symbols:
        for date in dates:
            # Generate realistic price data
            base_price = 150 if symbol == 'AAPL' else 300
            price_change = np.random.normal(0, 2)
            price = base_price + price_change
            
            stock_data.append({
                'symbol': symbol,
                'date': date,
                'open': price - 1,
                'high': price + 2,
                'low': price - 2,
                'close': price,
                'volume': np.random.randint(1000000, 5000000)
            })
    
    stock_df = pd.DataFrame(stock_data)
    
    # Create mock fundamental data
    fundamental_data = []
    for symbol in symbols:
        for date in dates[::7]:  # Weekly data
            fundamental_data.append({
                'symbol': symbol,
                'date': date,
                'revenue': np.random.uniform(50000, 100000),
                'earnings': np.random.uniform(5000, 15000),
                'market_cap': np.random.uniform(1000000, 3000000),
                'pe_ratio': np.random.uniform(10, 30),
                'pb_ratio': np.random.uniform(1, 5),
                'roe': np.random.uniform(0.05, 0.25)
            })
    
    fundamental_df = pd.DataFrame(fundamental_data)
    
    return {
        'stock_price': stock_df,
        'revenue': fundamental_df[['symbol', 'date', 'revenue']],
        'market_cap': fundamental_df[['symbol', 'date', 'market_cap']]
    }

def test_signal_config():
    """Test signal configuration"""
    print("🧪 Testing Signal Configuration...")
    
    # Create signal config
    config = SignalConfig(
        strategy_name="technical_analysis",
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        parameters={
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26
        },
        required_resources=["stock_price"],
        signal_threshold=0.5,
        confidence_threshold=0.7
    )
    
    # Test validation
    is_valid = config.validate()
    print(f"✅ Config validation: {is_valid}")
    
    # Test serialization
    config_dict = config.to_dict()
    config_from_dict = SignalConfig.from_dict(config_dict)
    print(f"✅ Config serialization: {config_from_dict.strategy_name}")
    
    return True

def test_technical_indicators():
    """Test technical indicators component"""
    print("🧪 Testing Technical Indicators...")
    
    # Create mock data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(140, 160, len(dates)),
        'high': np.random.uniform(145, 165, len(dates)),
        'low': np.random.uniform(135, 155, len(dates)),
        'close': np.random.uniform(140, 160, len(dates)),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Initialize technical indicators
    tech_indicators = TechnicalIndicators()
    
    # Test individual indicators
    rsi = tech_indicators.relative_strength_index(data)
    print(f"✅ RSI calculated: {len(rsi)} values, range: {rsi.min():.2f} - {rsi.max():.2f}")
    
    macd = tech_indicators.moving_average_convergence_divergence(data)
    print(f"✅ MACD calculated: {len(macd['macd'])} values")
    
    bb = tech_indicators.bollinger_bands(data)
    print(f"✅ Bollinger Bands calculated: {len(bb['upper'])} values")
    
    # Test signal generation
    indicators = ['rsi', 'macd', 'bollinger_bands']
    signals = tech_indicators.generate_technical_signals(data, indicators)
    print(f"✅ Technical signals generated: {len(signals)} records, {len(signals.columns)} columns")
    
    return True

def test_fundamental_indicators():
    """Test fundamental indicators component"""
    print("🧪 Testing Fundamental Indicators...")
    
    # Create mock fundamental data
    data = pd.DataFrame({
        'close': np.random.uniform(140, 160, 100),
        'earnings': np.random.uniform(5000, 15000, 100),
        'revenue': np.random.uniform(50000, 100000, 100),
        'book_value': np.random.uniform(20000, 50000, 100),
        'net_income': np.random.uniform(3000, 12000, 100),
        'total_equity': np.random.uniform(80000, 200000, 100)
    })
    
    # Initialize fundamental indicators
    fund_indicators = FundamentalIndicators()
    
    # Test individual indicators
    pe_ratio = fund_indicators.price_to_earnings_ratio(data)
    print(f"✅ P/E Ratio calculated: {len(pe_ratio)} values, range: {pe_ratio.min():.2f} - {pe_ratio.max():.2f}")
    
    pb_ratio = fund_indicators.price_to_book_ratio(data)
    print(f"✅ P/B Ratio calculated: {len(pb_ratio)} values")
    
    roe = fund_indicators.return_on_equity(data)
    print(f"✅ ROE calculated: {len(roe)} values")
    
    # Test signal generation
    indicators = ['pe_ratio', 'pb_ratio', 'roe']
    signals = fund_indicators.generate_fundamental_signals(data, indicators)
    print(f"✅ Fundamental signals generated: {len(signals)} records, {len(signals.columns)} columns")
    
    return True

def test_sentiment_indicators():
    """Test sentiment indicators component"""
    print("🧪 Testing Sentiment Indicators...")
    
    # Create mock sentiment data
    data = pd.DataFrame({
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'news_count': np.random.randint(10, 100, 100),
        'positive_mentions': np.random.randint(0, 50, 100),
        'negative_mentions': np.random.randint(0, 50, 100),
        'buy_ratings': np.random.randint(0, 20, 100),
        'hold_ratings': np.random.randint(0, 15, 100),
        'sell_ratings': np.random.randint(0, 10, 100)
    })
    
    # Initialize sentiment indicators
    sentiment_indicators = SentimentIndicators()
    
    # Test individual indicators
    news_sentiment = sentiment_indicators.news_sentiment_score(data)
    print(f"✅ News sentiment calculated: {len(news_sentiment)} values")
    
    social_sentiment = sentiment_indicators.social_media_sentiment(data)
    print(f"✅ Social sentiment calculated: {len(social_sentiment)} values")
    
    analyst_rating = sentiment_indicators.analyst_rating_score(data)
    print(f"✅ Analyst rating calculated: {len(analyst_rating)} values")
    
    # Test signal generation
    indicators = ['news_sentiment', 'social_sentiment', 'analyst_rating']
    signals = sentiment_indicators.generate_sentiment_signals(data, indicators)
    print(f"✅ Sentiment signals generated: {len(signals)} records, {len(signals.columns)} columns")
    
    return True

def test_technical_analysis_strategy():
    """Test technical analysis strategy"""
    print("🧪 Testing Technical Analysis Strategy...")
    
    # Create mock data
    mock_data = create_mock_data()
    
    # Create signal config
    config = SignalConfig(
        strategy_name="technical_analysis",
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        required_resources=["stock_price"]
    )
    
    # Initialize strategy
    strategy = TechnicalAnalysisStrategy()
    
    # Test data validation
    is_valid = strategy.validate_data(mock_data)
    print(f"✅ Data validation: {is_valid}")
    
    # Test signal generation
    signals = strategy.generate_signals(mock_data, config)
    print(f"✅ Technical analysis signals: {len(signals)} records")
    
    if not signals.empty:
        summary = strategy.get_signal_summary(signals)
        print(f"✅ Signal summary: {summary['total_signals']} total, {summary['buy_signals']} buy, {summary['sell_signals']} sell")
    
    return True

def test_fundamental_analysis_strategy():
    """Test fundamental analysis strategy"""
    print("🧪 Testing Fundamental Analysis Strategy...")
    
    # Create mock data
    mock_data = create_mock_data()
    
    # Create signal config
    config = SignalConfig(
        strategy_name="fundamental_analysis",
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        required_resources=["stock_price", "revenue", "market_cap"]
    )
    
    # Initialize strategy
    strategy = FundamentalAnalysisStrategy()
    
    # Test data validation
    is_valid = strategy.validate_data(mock_data)
    print(f"✅ Data validation: {is_valid}")
    
    # Test signal generation
    signals = strategy.generate_signals(mock_data, config)
    print(f"✅ Fundamental analysis signals: {len(signals)} records")
    
    if not signals.empty:
        summary = strategy.get_signal_summary(signals)
        print(f"✅ Signal summary: {summary['total_signals']} total, {summary['buy_signals']} buy, {summary['sell_signals']} sell")
    
    return True

def test_composite_signal_generator():
    """Test composite signal generator"""
    print("🧪 Testing Composite Signal Generator...")
    
    # Create mock data
    mock_data = create_mock_data()
    
    # Create signal config - use one of the supported strategies
    config = SignalConfig(
        strategy_name="technical_analysis",
        symbols=["AAPL", "MSFT"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        required_resources=["stock_price", "revenue", "market_cap"]
    )
    
    # Initialize composite generator
    generator = CompositeSignalGenerator()
    
    # Test supported strategies
    supported_strategies = generator.get_supported_strategies()
    print(f"✅ Supported strategies: {supported_strategies}")
    
    # Test signal generation
    signals = generator.generate_signals(config, mock_data)
    print(f"✅ Composite signals: {len(signals)} records")
    
    if not signals.empty:
        quality_metrics = generator.get_signal_quality_metrics(signals)
        print(f"✅ Quality metrics: {quality_metrics['total_signals']} signals, avg strength: {quality_metrics['signal_strength_avg']:.3f}")
    
    # Test performance metrics
    performance_metrics = generator.get_performance_metrics()
    print(f"✅ Performance metrics: {performance_metrics['success_rate']:.2%} success rate")
    
    return True

def main():
    """Run all signal generation tests"""
    print("🚀 Starting Signal Generation System Test Suite")
    print("=" * 50)
    
    tests = [
        test_signal_config,
        test_technical_indicators,
        test_fundamental_indicators,
        test_sentiment_indicators,
        test_technical_analysis_strategy,
        test_fundamental_analysis_strategy,
        test_composite_signal_generator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All signal generation components are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
