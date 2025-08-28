# 🏗️ **BreadthFlow Advanced Component Abstraction Plan**

## 🎯 **Overview**

This document outlines a comprehensive plan to transform BreadthFlow from a monolithic trading system into a highly modular, extensible platform that supports:

- **Multi-resource data fetching**: Trading data, fundamental data, alternative data, and custom data sources
- **Flexible signal generation**: Multiple algorithms, strategies, and custom logic
- **Modular backtesting**: Different execution engines and risk models
- **Component-based architecture**: Plug-and-play components with runtime configuration

## 🚀 **Key Objectives**

### **1. Universal Data Fetching**
- Support for any type of financial data (OHLCV, fundamentals, alternative data)
- Resource-based architecture with field-level granularity
- Multiple data source integration (Yahoo Finance, Alpha Vantage, Polygon, custom APIs)
- Real-time and historical data support

### **2. Flexible Signal Generation**
- Multiple signal generation algorithms and strategies
- Custom function composition and chaining
- Machine learning model integration
- Real-time signal processing

### **3. Modular Backtesting**
- Multiple execution engines (standard, high-frequency, options)
- Customizable risk models and position sizing
- Performance analytics and reporting

### **4. Component Registry System**
- Runtime component discovery and registration
- Configuration management and validation
- Version control and dependency management

---

## 📋 **Table of Contents**

1. [Current State Analysis](#current-state-analysis)
2. [Data Fetching Abstraction](#data-fetching-abstraction)
3. [Signal Generation Abstraction](#signal-generation-abstraction)
4. [Backtesting Abstraction](#backtesting-abstraction)
5. [Training & Model Management System](#training--model-management-system)
6. [Component Registry System](#component-registry-system)
7. [Configuration Management](#configuration-management)
8. [Error Handling & Logging](#error-handling--logging)
9. [Data Validation & Quality](#data-validation--quality)
10. [Performance Monitoring](#performance-monitoring)
11. [Security & Authentication](#security--authentication)
12. [Real-time Streaming](#real-time-streaming)
13. [Pipeline Orchestration](#pipeline-orchestration)
14. [Testing Framework](#testing-framework)
15. [Documentation & Examples](#documentation--examples)
16. [Implementation Plan](#implementation-plan)
17. [Migration Guide](#migration-guide)

---

## 🔍 **Current State Analysis**

### **Existing Components**
- **Data Fetching**: `TimeframeAgnosticFetcher` with `YFinanceIntradaySource`
- **Signal Generation**: `TimeframeAgnosticSignalGenerator` with hardcoded technical indicators
- **Backtesting**: `TimeframeAgnosticBacktestEngine` with fixed execution logic

### **Current Limitations**
- Limited to OHLCV data only
- Hardcoded technical indicators
- Single execution model
- Tight coupling between components
- No runtime configuration

### **Target Architecture**
```
Component Registry → Resource Definitions → Data Sources → Signal Algorithms → Backtest Engines
       ↓                    ↓                    ↓              ↓              ↓
Configuration → Validation → Execution → Results → Analytics
```

---

## 📊 **Data Fetching Abstraction**

### **🎯 Universal Data Resource System**

The new data fetching system will support any type of financial data through a resource-based architecture:

#### **Resource Definition Schema**
```python
# model/data/resources.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

class ResourceType(Enum):
    TRADING = "trading"           # OHLCV, tick data
    FUNDAMENTAL = "fundamental"   # Financial statements, ratios
    ALTERNATIVE = "alternative"   # News, sentiment, social media
    CUSTOM = "custom"            # User-defined data

class DataFrequency(Enum):
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    HOUR_1 = "1hour"
    DAY_1 = "1day"
    WEEK_1 = "1week"
    MONTH_1 = "1month"
    QUARTER = "quarter"
    YEAR_1 = "1year"

@dataclass
class ResourceField:
    """Definition of a single data field"""
    name: str
    type: str  # 'float', 'int', 'string', 'datetime', 'boolean'
    description: str
    unit: Optional[str] = None  # 'USD', 'shares', 'percentage', etc.
    required: bool = True
    validation_rules: Optional[Dict[str, Any]] = None

@dataclass
class DataResource:
    """Definition of a data resource"""
    name: str                    # e.g., 'stock_price', 'revenue', 'market_cap'
    type: ResourceType
    symbol_required: bool = True  # Whether symbol is required
    fields: List[ResourceField]
    frequency: DataFrequency
    description: str
    source_compatibility: List[str]  # List of compatible data sources
    validation_schema: Optional[Dict[str, Any]] = None
```

#### **Predefined Resources**
```python
# model/data/predefined_resources.py

# Trading Data Resources
STOCK_PRICE = DataResource(
    name="stock_price",
    type=ResourceType.TRADING,
    symbol_required=True,
    fields=[
        ResourceField("date", "datetime", "Trading date/time"),
        ResourceField("open", "float", "Opening price", "USD"),
        ResourceField("high", "float", "High price", "USD"),
        ResourceField("low", "float", "Low price", "USD"),
        ResourceField("close", "float", "Closing price", "USD"),
        ResourceField("volume", "int", "Trading volume", "shares"),
        ResourceField("adj_close", "float", "Adjusted closing price", "USD", required=False)
    ],
    frequency=DataFrequency.DAY_1,
    description="Stock price OHLCV data",
    source_compatibility=["yfinance", "alpha_vantage", "polygon"]
)

# Fundamental Data Resources
REVENUE = DataResource(
    name="revenue",
    type=ResourceType.FUNDAMENTAL,
    symbol_required=True,
    fields=[
        ResourceField("date", "datetime", "Report date"),
        ResourceField("revenue", "float", "Total revenue", "USD"),
        ResourceField("quarter", "string", "Fiscal quarter", required=False),
        ResourceField("year", "int", "Fiscal year"),
        ResourceField("currency", "string", "Currency", required=False)
    ],
    frequency=DataFrequency.QUARTER,
    description="Company revenue data",
    source_compatibility=["yfinance", "alpha_vantage", "polygon"]
)

MARKET_CAP = DataResource(
    name="market_cap",
    type=ResourceType.FUNDAMENTAL,
    symbol_required=True,
    fields=[
        ResourceField("date", "datetime", "Date"),
        ResourceField("market_cap", "float", "Market capitalization", "USD"),
        ResourceField("shares_outstanding", "int", "Shares outstanding", "shares", required=False),
        ResourceField("price", "float", "Current stock price", "USD", required=False)
    ],
    frequency=DataFrequency.DAY_1,
    description="Market capitalization data",
    source_compatibility=["yfinance", "alpha_vantage", "polygon"]
)

# Alternative Data Resources
NEWS_SENTIMENT = DataResource(
    name="news_sentiment",
    type=ResourceType.ALTERNATIVE,
    symbol_required=True,
    fields=[
        ResourceField("date", "datetime", "News date"),
        ResourceField("sentiment_score", "float", "Sentiment score", "score"),
        ResourceField("sentiment_label", "string", "Sentiment label"),
        ResourceField("news_count", "int", "Number of news articles"),
        ResourceField("source", "string", "Data source", required=False)
    ],
    frequency=DataFrequency.DAY_1,
    description="News sentiment analysis",
    source_compatibility=["alpha_vantage", "polygon", "custom"]
)
```

### **🔌 Data Source Interface**

#### **Abstract Data Source**
```python
# model/data/interfaces/data_source_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from ..resources import DataResource

class DataSourceInterface(ABC):
    """Abstract interface for data sources"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return data source name"""
        pass
    
    @abstractmethod
    def get_supported_resources(self) -> List[str]:
        """Return list of supported resource names"""
        pass
    
    @abstractmethod
    def get_resource_definition(self, resource_name: str) -> Optional[DataResource]:
        """Get resource definition for this source"""
        pass
    
    @abstractmethod
    def fetch_data(self, resource_name: str, symbol: str, 
                   start_date: str, end_date: str, 
                   **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch data for specified resource and symbol"""
        pass
    
    @abstractmethod
    def validate_parameters(self, resource_name: str, **kwargs) -> bool:
        """Validate input parameters"""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, int]:
        """Return rate limits for this source"""
        pass
    
    @abstractmethod
    def get_authentication_requirements(self) -> Dict[str, str]:
        """Return authentication requirements"""
        pass
```

#### **Enhanced YFinance Implementation**
```python
# model/data/sources/yfinance_source.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import yfinance as yf
from ..interfaces.data_source_interface import DataSourceInterface
from ..resources import DataResource, ResourceType, DataFrequency
from ..predefined_resources import STOCK_PRICE, REVENUE, MARKET_CAP

class YFinanceDataSource(DataSourceInterface):
    """Enhanced YFinance data source with multi-resource support"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.name = "yfinance"
        self.api_key = api_key
        self.rate_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 2000
        }
        
        # Map resource names to YFinance methods
        self.resource_mappings = {
            "stock_price": self._fetch_stock_price,
            "revenue": self._fetch_revenue,
            "market_cap": self._fetch_market_cap,
            "earnings": self._fetch_earnings,
            "balance_sheet": self._fetch_balance_sheet,
            "cash_flow": self._fetch_cash_flow
        }
    
    def get_name(self) -> str:
        return self.name
    
    def get_supported_resources(self) -> List[str]:
        return list(self.resource_mappings.keys())
    
    def get_resource_definition(self, resource_name: str) -> Optional[DataResource]:
        resource_definitions = {
            "stock_price": STOCK_PRICE,
            "revenue": REVENUE,
            "market_cap": MARKET_CAP
        }
        return resource_definitions.get(resource_name)
    
    def fetch_data(self, resource_name: str, symbol: str, 
                   start_date: str, end_date: str, 
                   **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch data for specified resource"""
        if resource_name not in self.resource_mappings:
            raise ValueError(f"Unsupported resource: {resource_name}")
        
        fetch_method = self.resource_mappings[resource_name]
        return fetch_method(symbol, start_date, end_date, **kwargs)
    
    def _fetch_stock_price(self, symbol: str, start_date: str, end_date: str, 
                          **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch stock price data"""
        timeframe = kwargs.get('timeframe', '1d')
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=timeframe)
        
        if data.empty:
            return pd.DataFrame(), {"error": "No data returned"}
        
        # Standardize column names
        data = data.reset_index()
        data = data.rename(columns={
            'Datetime': 'date',
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        })
        
        metadata = {
            "symbol": symbol,
            "resource": "stock_price",
            "records_count": len(data),
            "date_range": f"{start_date} to {end_date}",
            "timeframe": timeframe
        }
        
        return data, metadata
    
    def _fetch_revenue(self, symbol: str, start_date: str, end_date: str, 
                      **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch revenue data"""
        ticker = yf.Ticker(symbol)
        
        # Get financial data
        financials = ticker.financials
        if financials.empty:
            return pd.DataFrame(), {"error": "No financial data available"}
        
        # Extract revenue data
        revenue_data = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else pd.Series()
        
        if revenue_data.empty:
            return pd.DataFrame(), {"error": "No revenue data available"}
        
        # Convert to DataFrame
        data = pd.DataFrame({
            'date': revenue_data.index,
            'revenue': revenue_data.values,
            'year': revenue_data.index.year,
            'quarter': revenue_data.index.quarter
        })
        
        metadata = {
            "symbol": symbol,
            "resource": "revenue",
            "records_count": len(data),
            "data_type": "quarterly"
        }
        
        return data, metadata
    
    def _fetch_market_cap(self, symbol: str, start_date: str, end_date: str, 
                         **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch market cap data"""
        ticker = yf.Ticker(symbol)
        
        # Get info
        info = ticker.info
        market_cap = info.get('marketCap', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        current_price = info.get('currentPrice', 0)
        
        data = pd.DataFrame({
            'date': [pd.Timestamp.now().date()],
            'market_cap': [market_cap],
            'shares_outstanding': [shares_outstanding],
            'price': [current_price]
        })
        
        metadata = {
            "symbol": symbol,
            "resource": "market_cap",
            "records_count": 1,
            "data_type": "current"
        }
        
        return data, metadata
```

### **🎛️ Data Fetching Orchestrator**

#### **Universal Data Fetcher**
```python
# model/data/universal_fetcher.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from .interfaces.data_source_interface import DataSourceInterface
from .resources import DataResource
from .component_registry import ComponentRegistry

class UniversalDataFetcher:
    """Orchestrates data fetching across multiple sources and resources"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self.data_sources: Dict[str, DataSourceInterface] = {}
        self._load_data_sources()
    
    def _load_data_sources(self):
        """Load all registered data sources"""
        sources = self.registry.get_components("data_source")
        for source_name, source_class in sources.items():
            self.data_sources[source_name] = source_class()
    
    def get_available_resources(self) -> Dict[str, List[str]]:
        """Get available resources by data source"""
        resources = {}
        for source_name, source in self.data_sources.items():
            resources[source_name] = source.get_supported_resources()
        return resources
    
    def fetch_resource(self, resource_name: str, symbol: str, 
                      start_date: str, end_date: str,
                      data_source: Optional[str] = None,
                      **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch data for a specific resource"""
        
        # Determine data source
        if data_source is None:
            data_source = self._select_best_source(resource_name)
        
        if data_source not in self.data_sources:
            raise ValueError(f"Unknown data source: {data_source}")
        
        source = self.data_sources[data_source]
        
        # Validate resource is supported
        if resource_name not in source.get_supported_resources():
            raise ValueError(f"Resource {resource_name} not supported by {data_source}")
        
        # Fetch data
        return source.fetch_data(resource_name, symbol, start_date, end_date, **kwargs)
    
    def fetch_multiple_resources(self, resources: List[Dict[str, Any]], 
                               symbol: str, start_date: str, end_date: str) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Fetch multiple resources for a symbol"""
        results = {}
        
        for resource_config in resources:
            resource_name = resource_config['name']
            data_source = resource_config.get('source')
            kwargs = resource_config.get('parameters', {})
            
            data, metadata = self.fetch_resource(
                resource_name, symbol, start_date, end_date, data_source, **kwargs
            )
            results[resource_name] = (data, metadata)
        
        return results
    
    def _select_best_source(self, resource_name: str) -> str:
        """Select the best data source for a resource"""
        # Simple selection logic - can be enhanced with cost, quality, availability
        for source_name, source in self.data_sources.items():
            if resource_name in source.get_supported_resources():
                return source_name
        
        raise ValueError(f"No data source found for resource: {resource_name}")
```

---

## 🎯 **Signal Generation Abstraction**

### **🧠 Flexible Signal Generation Framework**

The new signal generation system will support multiple algorithms, strategies, and custom logic through a composable function-based architecture:

#### **Signal Generation Interface**
```python
# model/signals/interfaces/signal_generator_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
from dataclasses import dataclass

@dataclass
class SignalConfig:
    """Configuration for signal generation"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_data: List[str]  # List of required data resources
    output_fields: List[str]  # List of output signal fields

class SignalGeneratorInterface(ABC):
    """Abstract interface for signal generators"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return generator name"""
        pass
    
    @abstractmethod
    def get_config(self) -> SignalConfig:
        """Return generator configuration"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        symbols: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate signals from input data"""
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate input data format"""
        pass
    
    @abstractmethod
    def get_required_resources(self) -> List[str]:
        """Return required data resources"""
        pass
```

#### **Function-Based Signal Components**
```python
# model/signals/components/technical_indicators.py
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    name: str
    parameters: Dict[str, Any]
    description: str
    category: str  # 'trend', 'momentum', 'volatility', 'volume'

class TechnicalIndicators:
    """Library of technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'upper': middle + (std * std_dev),
            'middle': middle,
            'lower': middle - (std * std_dev)
        }
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages"""
        mas = {}
        for period in periods:
            mas[f'ma_{period}'] = prices.rolling(window=period).mean()
        return mas

# model/signals/components/fundamental_indicators.py
class FundamentalIndicators:
    """Library of fundamental indicators"""
    
    @staticmethod
    def calculate_pe_ratio(price: float, earnings: float) -> float:
        """Calculate P/E ratio"""
        return price / earnings if earnings > 0 else None
    
    @staticmethod
    def calculate_pb_ratio(price: float, book_value: float) -> float:
        """Calculate P/B ratio"""
        return price / book_value if book_value > 0 else None
    
    @staticmethod
    def calculate_revenue_growth(revenue_data: pd.DataFrame) -> pd.Series:
        """Calculate revenue growth rate"""
        return revenue_data['revenue'].pct_change()
    
    @staticmethod
    def calculate_market_cap_ratio(market_cap: float, revenue: float) -> float:
        """Calculate market cap to revenue ratio"""
        return market_cap / revenue if revenue > 0 else None

# model/signals/components/sentiment_indicators.py
class SentimentIndicators:
    """Library of sentiment indicators"""
    
    @staticmethod
    def calculate_sentiment_score(sentiment_data: pd.DataFrame) -> pd.Series:
        """Calculate sentiment score"""
        return sentiment_data['sentiment_score']
    
    @staticmethod
    def calculate_sentiment_momentum(sentiment_data: pd.DataFrame, period: int = 5) -> pd.Series:
        """Calculate sentiment momentum"""
        return sentiment_data['sentiment_score'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_news_volume(sentiment_data: pd.DataFrame) -> pd.Series:
        """Calculate news volume"""
        return sentiment_data['news_count']
```

#### **Signal Strategy Framework**
```python
# model/signals/strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from ..interfaces.signal_generator_interface import SignalGeneratorInterface, SignalConfig

class BaseSignalStrategy(SignalGeneratorInterface):
    """Base class for signal strategies"""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self._name = name
        self._description = description
        self._parameters = parameters
        self._config = self._create_config()
    
    def get_name(self) -> str:
        return self._name
    
    def get_config(self) -> SignalConfig:
        return self._config
    
    @abstractmethod
    def _create_config(self) -> SignalConfig:
        """Create strategy configuration"""
        pass
    
    @abstractmethod
    def _generate_signals_for_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate signals for a single symbol"""
        pass
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        symbols: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate signals for all symbols"""
        all_signals = []
        
        for symbol in symbols:
            try:
                symbol_signals = self._generate_signals_for_symbol(symbol, data)
                all_signals.extend(symbol_signals)
            except Exception as e:
                print(f"Error generating signals for {symbol}: {str(e)}")
                continue
        
        return all_signals

# model/signals/strategies/technical_strategy.py
from typing import Dict, List, Any
import pandas as pd
from .base_strategy import BaseSignalStrategy
from ..components.technical_indicators import TechnicalIndicators

class TechnicalAnalysisStrategy(BaseSignalStrategy):
    """Technical analysis-based signal strategy"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'ma_short': 20,
            'ma_long': 50,
            'bb_period': 20,
            'bb_std': 2
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            name="technical_analysis",
            description="Technical analysis strategy using RSI, moving averages, and Bollinger Bands",
            parameters=default_params
        )
    
    def _create_config(self) -> SignalConfig:
        return SignalConfig(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            required_data=['stock_price'],
            output_fields=['signal_type', 'confidence', 'strength', 'indicators']
        )
    
    def get_required_resources(self) -> List[str]:
        return ['stock_price']
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        return 'stock_price' in data and not data['stock_price'].empty
    
    def _generate_signals_for_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate technical analysis signals"""
        price_data = data['stock_price']
        
        # Calculate technical indicators
        close_prices = price_data['close']
        
        # RSI
        rsi = TechnicalIndicators.calculate_rsi(close_prices, self._parameters['rsi_period'])
        
        # Moving averages
        mas = TechnicalIndicators.calculate_moving_averages(
            close_prices, 
            [self._parameters['ma_short'], self._parameters['ma_long']]
        )
        
        # Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger_bands(
            close_prices, 
            self._parameters['bb_period'], 
            self._parameters['bb_std']
        )
        
        # Generate signals based on latest data
        latest_idx = price_data.index[-1]
        latest_close = close_prices.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        latest_ma_short = mas[f"ma_{self._parameters['ma_short']}"].iloc[-1]
        latest_ma_long = mas[f"ma_{self._parameters['ma_long']}"].iloc[-1]
        latest_bb_upper = bb['upper'].iloc[-1]
        latest_bb_lower = bb['lower'].iloc[-1]
        
        # Signal logic
        signal_type = "HOLD"
        confidence = 0.5
        strength = "WEAK"
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if latest_rsi < self._parameters['rsi_oversold']:
            buy_signals += 1
        elif latest_rsi > self._parameters['rsi_overbought']:
            sell_signals += 1
        
        # Moving average crossover
        if latest_ma_short > latest_ma_long:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Bollinger Bands
        if latest_close < latest_bb_lower:
            buy_signals += 1
        elif latest_close > latest_bb_upper:
            sell_signals += 1
        
        # Determine final signal
        if buy_signals >= 2:
            signal_type = "BUY"
            confidence = min(0.9, 0.5 + (buy_signals * 0.2))
            strength = "STRONG" if buy_signals >= 3 else "MODERATE"
        elif sell_signals >= 2:
            signal_type = "SELL"
            confidence = min(0.9, 0.5 + (sell_signals * 0.2))
            strength = "STRONG" if sell_signals >= 3 else "MODERATE"
        
        return [{
            'symbol': symbol,
            'date': price_data['date'].iloc[-1],
            'signal_type': signal_type,
            'confidence': round(confidence, 3),
            'strength': strength,
            'price': latest_close,
            'indicators': {
                'rsi': round(latest_rsi, 2),
                'ma_short': round(latest_ma_short, 2),
                'ma_long': round(latest_ma_long, 2),
                'bb_upper': round(latest_bb_upper, 2),
                'bb_lower': round(latest_bb_lower, 2)
            },
            'strategy': self._name,
            'parameters': self._parameters
        }]

# model/signals/strategies/fundamental_strategy.py
from .base_strategy import BaseSignalStrategy
from ..components.fundamental_indicators import FundamentalIndicators

class FundamentalAnalysisStrategy(BaseSignalStrategy):
    """Fundamental analysis-based signal strategy"""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'pe_threshold': 15,
            'pb_threshold': 1.5,
            'revenue_growth_threshold': 0.1,
            'market_cap_revenue_threshold': 5
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            name="fundamental_analysis",
            description="Fundamental analysis strategy using financial ratios and growth metrics",
            parameters=default_params
        )
    
    def _create_config(self) -> SignalConfig:
        return SignalConfig(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            required_data=['stock_price', 'revenue', 'market_cap'],
            output_fields=['signal_type', 'confidence', 'strength', 'fundamentals']
        )
    
    def get_required_resources(self) -> List[str]:
        return ['stock_price', 'revenue', 'market_cap']
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        required = ['stock_price', 'revenue', 'market_cap']
        return all(resource in data and not data[resource].empty for resource in required)
    
    def _generate_signals_for_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate fundamental analysis signals"""
        price_data = data['stock_price']
        revenue_data = data['revenue']
        market_cap_data = data['market_cap']
        
        # Get latest values
        latest_price = price_data['close'].iloc[-1]
        latest_revenue = revenue_data['revenue'].iloc[-1]
        latest_market_cap = market_cap_data['market_cap'].iloc[-1]
        
        # Calculate fundamental ratios
        revenue_growth = FundamentalIndicators.calculate_revenue_growth(revenue_data).iloc[-1]
        market_cap_ratio = FundamentalIndicators.calculate_market_cap_ratio(latest_market_cap, latest_revenue)
        
        # Signal logic
        signal_type = "HOLD"
        confidence = 0.5
        strength = "WEAK"
        buy_signals = 0
        sell_signals = 0
        
        # Revenue growth
        if revenue_growth > self._parameters['revenue_growth_threshold']:
            buy_signals += 1
        elif revenue_growth < 0:
            sell_signals += 1
        
        # Market cap to revenue ratio
        if market_cap_ratio and market_cap_ratio < self._parameters['market_cap_revenue_threshold']:
            buy_signals += 1
        elif market_cap_ratio and market_cap_ratio > self._parameters['market_cap_revenue_threshold'] * 2:
            sell_signals += 1
        
        # Determine final signal
        if buy_signals >= 1:
            signal_type = "BUY"
            confidence = min(0.9, 0.5 + (buy_signals * 0.3))
            strength = "STRONG" if buy_signals >= 2 else "MODERATE"
        elif sell_signals >= 1:
            signal_type = "SELL"
            confidence = min(0.9, 0.5 + (sell_signals * 0.3))
            strength = "STRONG" if sell_signals >= 2 else "MODERATE"
        
        return [{
            'symbol': symbol,
            'date': price_data['date'].iloc[-1],
            'signal_type': signal_type,
            'confidence': round(confidence, 3),
            'strength': strength,
            'price': latest_price,
            'fundamentals': {
                'revenue_growth': round(revenue_growth, 3),
                'market_cap_revenue_ratio': round(market_cap_ratio, 2) if market_cap_ratio else None,
                'revenue': latest_revenue,
                'market_cap': latest_market_cap
            },
            'strategy': self._name,
            'parameters': self._parameters
        }]
```

#### **Composite Signal Generator**
```python
# model/signals/composite_generator.py
from typing import Dict, List, Any, Optional
import pandas as pd
from .interfaces.signal_generator_interface import SignalGeneratorInterface, SignalConfig
from .strategies.base_strategy import BaseSignalStrategy

class CompositeSignalGenerator(SignalGeneratorInterface):
    """Combines multiple signal strategies with weighting"""
    
    def __init__(self, strategies: List[BaseSignalStrategy], weights: Optional[Dict[str, float]] = None):
        self.strategies = strategies
        self.weights = weights or {strategy.get_name(): 1.0/len(strategies) for strategy in strategies}
        
        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def get_name(self) -> str:
        return "composite_generator"
    
    def get_config(self) -> SignalConfig:
        return SignalConfig(
            name=self.get_name(),
            description="Composite signal generator combining multiple strategies",
            parameters={'strategies': [s.get_name() for s in self.strategies], 'weights': self.weights},
            required_data=list(set().union(*[s.get_required_resources() for s in self.strategies])),
            output_fields=['signal_type', 'confidence', 'strength', 'composite_score', 'strategy_contributions']
        )
    
    def get_required_resources(self) -> List[str]:
        required = set()
        for strategy in self.strategies:
            required.update(strategy.get_required_resources())
        return list(required)
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        return all(strategy.validate_data(data) for strategy in self.strategies)
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], 
                        symbols: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Generate composite signals"""
        all_signals = []
        
        for symbol in symbols:
            # Generate signals from each strategy
            strategy_signals = {}
            for strategy in self.strategies:
                try:
                    signals = strategy._generate_signals_for_symbol(symbol, data)
                    if signals:
                        strategy_signals[strategy.get_name()] = signals[0]
                except Exception as e:
                    print(f"Error in strategy {strategy.get_name()} for {symbol}: {str(e)}")
                    continue
            
            if not strategy_signals:
                continue
            
            # Combine signals using weighted voting
            composite_signal = self._combine_signals(symbol, strategy_signals, data)
            if composite_signal:
                all_signals.append(composite_signal)
        
        return all_signals
    
    def _combine_signals(self, symbol: str, strategy_signals: Dict[str, Dict[str, Any]], 
                        data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Combine signals from multiple strategies"""
        if not strategy_signals:
            return None
        
        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_confidence = 0.0
        strategy_contributions = {}
        
        for strategy_name, signal in strategy_signals.items():
            weight = self.weights.get(strategy_name, 0.0)
            confidence = signal.get('confidence', 0.5)
            weighted_confidence = weight * confidence
            
            signal_type = signal.get('signal_type', 'HOLD')
            if signal_type == 'BUY':
                buy_score += weighted_confidence
            elif signal_type == 'SELL':
                sell_score += weighted_confidence
            else:
                hold_score += weighted_confidence
            
            total_confidence += confidence
            strategy_contributions[strategy_name] = {
                'signal_type': signal_type,
                'confidence': confidence,
                'weight': weight,
                'contribution': weighted_confidence
            }
        
        # Determine composite signal
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score == buy_score and buy_score > 0:
            signal_type = 'BUY'
            composite_confidence = buy_score
        elif max_score == sell_score and sell_score > 0:
            signal_type = 'SELL'
            composite_confidence = sell_score
        else:
            signal_type = 'HOLD'
            composite_confidence = hold_score
        
        # Get latest price
        price_data = data.get('stock_price')
        latest_price = price_data['close'].iloc[-1] if price_data is not None else 0.0
        
        return {
            'symbol': symbol,
            'date': price_data['date'].iloc[-1] if price_data is not None else pd.Timestamp.now(),
            'signal_type': signal_type,
            'confidence': round(composite_confidence, 3),
            'strength': 'STRONG' if composite_confidence > 0.7 else 'MODERATE' if composite_confidence > 0.5 else 'WEAK',
            'price': latest_price,
            'composite_score': round(composite_confidence, 3),
            'strategy_contributions': strategy_contributions,
            'strategy': self.get_name(),
            'parameters': {'weights': self.weights}
                 }
```

---

## 📈 **Backtesting Abstraction**

### **🎯 Modular Backtesting Framework**

The new backtesting system will support multiple execution engines, risk models, and performance analytics through a modular architecture:

#### **Backtesting Interface**
```python
# model/backtesting/interfaces/backtest_engine_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_data: List[str]
    output_metrics: List[str]

@dataclass
class TradeRecord:
    """Record of a single trade"""
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: int
    price: float
    timestamp: pd.Timestamp
    commission: float
    slippage: float
    signal_confidence: float
    metadata: Dict[str, Any]

@dataclass
class PositionRecord:
    """Record of current position"""
    symbol: str
    quantity: int
    avg_price: float
    entry_time: pd.Timestamp
    current_value: float
    unrealized_pnl: float

class BacktestEngineInterface(ABC):
    """Abstract interface for backtesting engines"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return engine name"""
        pass
    
    @abstractmethod
    def get_config(self) -> BacktestConfig:
        """Return engine configuration"""
        pass
    
    @abstractmethod
    def run_backtest(self, signals: List[Dict[str, Any]], 
                    market_data: Dict[str, pd.DataFrame],
                    initial_capital: float, **kwargs) -> Dict[str, Any]:
        """Run backtest simulation"""
        pass
    
    @abstractmethod
    def validate_signals(self, signals: List[Dict[str, Any]]) -> bool:
        """Validate signal format"""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Return required data resources"""
        pass
```

#### **Execution Engine Components**
```python
# model/backtesting/engines/execution_engine.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from ..interfaces.backtest_engine_interface import TradeRecord, PositionRecord

class ExecutionEngine(ABC):
    """Abstract execution engine for trade execution"""
    
    @abstractmethod
    def execute_trade(self, signal: Dict[str, Any], 
                     current_positions: Dict[str, PositionRecord],
                     available_capital: float,
                     market_data: pd.DataFrame) -> Optional[TradeRecord]:
        """Execute a trade based on signal"""
        pass
    
    @abstractmethod
    def calculate_execution_price(self, signal_price: float, 
                                signal_type: str, volume: float) -> Tuple[float, Dict[str, float]]:
        """Calculate execution price with costs"""
        pass
    
    @abstractmethod
    def validate_trade(self, signal: Dict[str, Any], 
                      current_positions: Dict[str, PositionRecord],
                      available_capital: float) -> bool:
        """Validate if trade can be executed"""
        pass

class StandardExecutionEngine(ExecutionEngine):
    """Standard execution engine with realistic costs"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.commission_rate = parameters.get('commission_rate', 0.001)
        self.slippage_rate = parameters.get('slippage_rate', 0.0005)
        self.max_position_size = parameters.get('max_position_size', 0.1)
        self.min_trade_value = parameters.get('min_trade_value', 1000)
    
    def execute_trade(self, signal: Dict[str, Any], 
                     current_positions: Dict[str, PositionRecord],
                     available_capital: float,
                     market_data: pd.DataFrame) -> Optional[TradeRecord]:
        """Execute a trade"""
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        signal_price = signal['price']
        confidence = signal.get('confidence', 0.5)
        
        # Skip HOLD signals
        if signal_type == 'HOLD':
            return None
        
        # Validate trade
        if not self.validate_trade(signal, current_positions, available_capital):
            return None
        
        # Calculate execution price
        execution_price, cost_breakdown = self.calculate_execution_price(
            signal_price, signal_type, signal.get('volume', 0)
        )
        
        # Calculate position size
        max_position_value = available_capital * self.max_position_size * confidence
        position_value = min(max_position_value, available_capital * 0.95)  # Keep 5% cash
        
        if signal_type == 'BUY':
            # Calculate shares to buy
            shares = int(position_value / execution_price)
            if shares <= 0:
                return None
            
            total_cost = shares * execution_price
            commission = total_cost * self.commission_rate
            
            return TradeRecord(
                symbol=symbol,
                action='BUY',
                quantity=shares,
                price=execution_price,
                timestamp=pd.Timestamp.now(),
                commission=commission,
                slippage=cost_breakdown['slippage'],
                signal_confidence=confidence,
                metadata={'cost_breakdown': cost_breakdown}
            )
        
        elif signal_type == 'SELL':
            # Check if we have position to sell
            if symbol not in current_positions:
                return None
            
            position = current_positions[symbol]
            shares = position.quantity
            
            total_proceeds = shares * execution_price
            commission = total_proceeds * self.commission_rate
            
            return TradeRecord(
                symbol=symbol,
                action='SELL',
                quantity=shares,
                price=execution_price,
                timestamp=pd.Timestamp.now(),
                commission=commission,
                slippage=cost_breakdown['slippage'],
                signal_confidence=confidence,
                metadata={'cost_breakdown': cost_breakdown}
            )
    
    def calculate_execution_price(self, signal_price: float, 
                                signal_type: str, volume: float) -> Tuple[float, Dict[str, float]]:
        """Calculate execution price with slippage and costs"""
        slippage = signal_price * self.slippage_rate
        
        if signal_type == 'BUY':
            execution_price = signal_price + slippage
        else:  # SELL
            execution_price = signal_price - slippage
        
        cost_breakdown = {
            'signal_price': signal_price,
            'slippage': slippage,
            'execution_price': execution_price
        }
        
        return execution_price, cost_breakdown
    
    def validate_trade(self, signal: Dict[str, Any], 
                      current_positions: Dict[str, PositionRecord],
                      available_capital: float) -> bool:
        """Validate trade execution"""
        symbol = signal['symbol']
        signal_type = signal['signal_type']
        
        if signal_type == 'BUY':
            # Check if we have enough capital
            return available_capital >= self.min_trade_value
        elif signal_type == 'SELL':
            # Check if we have position to sell
            return symbol in current_positions and current_positions[symbol].quantity > 0
        
        return False

# model/backtesting/engines/high_frequency_engine.py
class HighFrequencyExecutionEngine(ExecutionEngine):
    """High-frequency trading execution engine"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.commission_rate = parameters.get('commission_rate', 0.0005)  # Lower for HFT
        self.slippage_rate = parameters.get('slippage_rate', 0.0001)      # Lower slippage
        self.max_position_size = parameters.get('max_position_size', 0.05)  # Smaller positions
        self.execution_delay_ms = parameters.get('execution_delay_ms', 1)    # 1ms delay
        self.min_trade_value = parameters.get('min_trade_value', 100)
    
    def execute_trade(self, signal: Dict[str, Any], 
                     current_positions: Dict[str, PositionRecord],
                     available_capital: float,
                     market_data: pd.DataFrame) -> Optional[TradeRecord]:
        """Execute HFT trade with minimal latency"""
        # Similar to standard engine but with HFT-specific parameters
        # Implementation would include latency simulation and market microstructure
        pass
    
    def calculate_execution_price(self, signal_price: float, 
                                signal_type: str, volume: float) -> Tuple[float, Dict[str, float]]:
        """Calculate HFT execution price"""
        # HFT-specific price calculation with market impact
        pass
    
    def validate_trade(self, signal: Dict[str, Any], 
                      current_positions: Dict[str, PositionRecord],
                      available_capital: float) -> bool:
        """Validate HFT trade"""
        # HFT-specific validation (e.g., position limits, risk checks)
        pass
```

#### **Risk Management Components**
```python
# model/backtesting/risk/risk_manager.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from ..interfaces.backtest_engine_interface import PositionRecord, TradeRecord

class RiskManager(ABC):
    """Abstract risk manager"""
    
    @abstractmethod
    def validate_trade(self, trade: TradeRecord, 
                      current_positions: Dict[str, PositionRecord],
                      portfolio_value: float) -> bool:
        """Validate trade against risk limits"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], 
                              available_capital: float,
                              current_positions: Dict[str, PositionRecord]) -> float:
        """Calculate position size based on risk parameters"""
        pass
    
    @abstractmethod
    def get_risk_metrics(self, positions: Dict[str, PositionRecord],
                        portfolio_value: float) -> Dict[str, float]:
        """Calculate current risk metrics"""
        pass

class StandardRiskManager(RiskManager):
    """Standard risk management with position and portfolio limits"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.max_position_size = parameters.get('max_position_size', 0.1)  # 10% per position
        self.max_portfolio_risk = parameters.get('max_portfolio_risk', 0.02)  # 2% max risk
        self.max_drawdown = parameters.get('max_drawdown', 0.15)  # 15% max drawdown
        self.stop_loss = parameters.get('stop_loss', 0.05)  # 5% stop loss
    
    def validate_trade(self, trade: TradeRecord, 
                      current_positions: Dict[str, PositionRecord],
                      portfolio_value: float) -> bool:
        """Validate trade against risk limits"""
        symbol = trade.symbol
        
        # Check position size limit
        position_value = trade.quantity * trade.price
        if position_value / portfolio_value > self.max_position_size:
            return False
        
        # Check portfolio concentration
        total_position_value = sum(pos.current_value for pos in current_positions.values())
        if (total_position_value + position_value) / portfolio_value > 0.8:  # 80% max invested
            return False
        
        return True
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              available_capital: float,
                              current_positions: Dict[str, PositionRecord]) -> float:
        """Calculate position size based on risk parameters"""
        confidence = signal.get('confidence', 0.5)
        signal_price = signal.get('price', 0)
        
        if signal_price <= 0:
            return 0
        
        # Base position size
        base_size = available_capital * self.max_position_size
        
        # Adjust for confidence
        adjusted_size = base_size * confidence
        
        # Adjust for volatility (if available)
        volatility = signal.get('volatility', 0.2)
        volatility_adjustment = 1 / (1 + volatility)
        adjusted_size *= volatility_adjustment
        
        return adjusted_size
    
    def get_risk_metrics(self, positions: Dict[str, PositionRecord],
                        portfolio_value: float) -> Dict[str, float]:
        """Calculate current risk metrics"""
        if not positions:
            return {'total_risk': 0, 'concentration': 0, 'largest_position': 0}
        
        # Calculate concentration
        position_values = [pos.current_value for pos in positions.values()]
        total_position_value = sum(position_values)
        concentration = total_position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Largest position
        largest_position = max(position_values) / portfolio_value if portfolio_value > 0 else 0
        
        # Portfolio risk (simplified)
        total_risk = concentration * 0.5  # Simplified risk calculation
        
        return {
            'total_risk': total_risk,
            'concentration': concentration,
            'largest_position': largest_position,
            'position_count': len(positions)
        }

# model/backtesting/risk/var_risk_manager.py
class VaRRiskManager(RiskManager):
    """Value at Risk (VaR) based risk manager"""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.var_confidence = parameters.get('var_confidence', 0.95)  # 95% VaR
        self.max_var = parameters.get('max_var', 0.02)  # 2% max VaR
        self.lookback_period = parameters.get('lookback_period', 252)  # 1 year
    
    def validate_trade(self, trade: TradeRecord, 
                      current_positions: Dict[str, PositionRecord],
                      portfolio_value: float) -> bool:
        """Validate trade using VaR"""
        # Calculate portfolio VaR after trade
        # Implementation would include historical simulation or Monte Carlo
        pass
    
    def calculate_position_size(self, signal: Dict[str, Any], 
                              available_capital: float,
                              current_positions: Dict[str, PositionRecord]) -> float:
        """Calculate position size based on VaR"""
        # VaR-based position sizing
        # Implementation would include volatility and correlation analysis
        pass
    
    def get_risk_metrics(self, positions: Dict[str, PositionRecord],
                        portfolio_value: float) -> Dict[str, float]:
        """Calculate VaR-based risk metrics"""
        # Calculate portfolio VaR, expected shortfall, etc.
        pass
```

#### **Performance Analytics Components**
```python
# model/backtesting/analytics/performance_analyzer.py
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from ..interfaces.backtest_engine_interface import TradeRecord, PositionRecord

class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_returns(self, portfolio_values: List[float], 
                         initial_capital: float) -> Dict[str, float]:
        """Calculate return metrics"""
        if not portfolio_values:
            return {}
        
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculate daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Annualized return (assuming daily data)
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        
        # Volatility
        volatility = np.std(daily_returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value
        }
    
    def analyze_trades(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        """Analyze trade performance"""
        if not trades:
            return {}
        
        # Separate buy and sell trades
        buy_trades = [t for t in trades if t.action == 'BUY']
        sell_trades = [t for t in trades if t.action == 'SELL']
        
        # Calculate trade statistics
        total_trades = len(trades)
        profitable_trades = len([t for t in sell_trades if t.metadata.get('pnl', 0) > 0])
        win_rate = profitable_trades / len(sell_trades) if sell_trades else 0
        
        # Calculate P&L statistics
        pnls = [t.metadata.get('pnl', 0) for t in sell_trades]
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate costs
        total_commission = sum(t.commission for t in trades)
        total_slippage = sum(t.slippage for t in trades)
        
        return {
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'profitable_trades': profitable_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_costs': total_commission + total_slippage
        }
    
    def calculate_risk_metrics(self, portfolio_values: List[float]) -> Dict[str, float]:
        """Calculate risk metrics"""
        if len(portfolio_values) < 2:
            return {}
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # VaR (95% confidence)
        var_95 = np.percentile(returns, 5)
        
        # Expected shortfall (conditional VaR)
        es_95 = np.mean(returns[returns <= var_95])
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'expected_shortfall_95': es_95,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio
        }
    
    def generate_report(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        portfolio_values = backtest_results.get('portfolio_values', [])
        trades = backtest_results.get('trades', [])
        initial_capital = backtest_results.get('initial_capital', 0)
        
        returns_metrics = self.calculate_returns(portfolio_values, initial_capital)
        trade_metrics = self.analyze_trades(trades)
        risk_metrics = self.calculate_risk_metrics(portfolio_values)
        
        return {
            'summary': {
                'initial_capital': initial_capital,
                'final_value': returns_metrics.get('final_value', initial_capital),
                'total_return': returns_metrics.get('total_return', 0),
                'annualized_return': returns_metrics.get('annualized_return', 0),
                'sharpe_ratio': returns_metrics.get('sharpe_ratio', 0),
                'max_drawdown': returns_metrics.get('max_drawdown', 0)
            },
            'returns': returns_metrics,
            'trades': trade_metrics,
            'risk': risk_metrics,
            'metadata': backtest_results.get('metadata', {})
        }
```

#### **Base Backtesting Engine**
```python
# model/backtesting/engines/base_backtest_engine.py
from typing import Dict, List, Any, Optional
import pandas as pd
from ..interfaces.backtest_engine_interface import BacktestEngineInterface, BacktestConfig, TradeRecord, PositionRecord
from ..engines.execution_engine import ExecutionEngine
from ..risk.risk_manager import RiskManager
from ..analytics.performance_analyzer import PerformanceAnalyzer

class BaseBacktestEngine(BacktestEngineInterface):
    """Base backtesting engine with modular components"""
    
    def __init__(self, name: str, description: str, 
                 execution_engine: ExecutionEngine,
                 risk_manager: RiskManager,
                 parameters: Dict[str, Any]):
        self._name = name
        self._description = description
        self._execution_engine = execution_engine
        self._risk_manager = risk_manager
        self._parameters = parameters
        self._analyzer = PerformanceAnalyzer()
        self._config = self._create_config()
    
    def get_name(self) -> str:
        return self._name
    
    def get_config(self) -> BacktestConfig:
        return self._config
    
    def _create_config(self) -> BacktestConfig:
        return BacktestConfig(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            required_data=['stock_price'],
            output_metrics=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        )
    
    def get_required_data(self) -> List[str]:
        return ['stock_price']
    
    def validate_signals(self, signals: List[Dict[str, Any]]) -> bool:
        required_fields = ['symbol', 'signal_type', 'price', 'date']
        return all(all(field in signal for field in required_fields) for signal in signals)
    
    def run_backtest(self, signals: List[Dict[str, Any]], 
                    market_data: Dict[str, pd.DataFrame],
                    initial_capital: float, **kwargs) -> Dict[str, Any]:
        """Run backtest simulation"""
        
        # Validate inputs
        if not self.validate_signals(signals):
            raise ValueError("Invalid signal format")
        
        # Initialize portfolio
        current_capital = initial_capital
        positions: Dict[str, PositionRecord] = {}
        trades: List[TradeRecord] = []
        portfolio_values = [initial_capital]
        
        # Sort signals by date
        signals_sorted = sorted(signals, key=lambda x: x['date'])
        
        # Process signals
        for signal in signals_sorted:
            # Execute trade
            trade = self._execution_engine.execute_trade(
                signal, positions, current_capital, market_data['stock_price']
            )
            
            if trade:
                # Validate against risk limits
                if self._risk_manager.validate_trade(trade, positions, current_capital):
                    # Execute trade
                    trades.append(trade)
                    
                    # Update portfolio
                    if trade.action == 'BUY':
                        # Update cash
                        total_cost = trade.quantity * trade.price + trade.commission
                        current_capital -= total_cost
                        
                        # Update position
                        if trade.symbol in positions:
                            # Add to existing position
                            pos = positions[trade.symbol]
                            new_quantity = pos.quantity + trade.quantity
                            new_avg_price = ((pos.quantity * pos.avg_price) + (trade.quantity * trade.price)) / new_quantity
                            positions[trade.symbol] = PositionRecord(
                                symbol=trade.symbol,
                                quantity=new_quantity,
                                avg_price=new_avg_price,
                                entry_time=pos.entry_time,
                                current_value=new_quantity * trade.price,
                                unrealized_pnl=(trade.price - new_avg_price) * new_quantity
                            )
                        else:
                            # New position
                            positions[trade.symbol] = PositionRecord(
                                symbol=trade.symbol,
                                quantity=trade.quantity,
                                avg_price=trade.price,
                                entry_time=trade.timestamp,
                                current_value=trade.quantity * trade.price,
                                unrealized_pnl=0
                            )
                    
                    elif trade.action == 'SELL':
                        # Calculate P&L
                        position = positions[trade.symbol]
                        pnl = (trade.price - position.avg_price) * trade.quantity - trade.commission
                        
                        # Update cash
                        total_proceeds = trade.quantity * trade.price - trade.commission
                        current_capital += total_proceeds
                        
                        # Remove position
                        del positions[trade.symbol]
                        
                        # Add P&L to trade record
                        trade.metadata['pnl'] = pnl
            
            # Calculate portfolio value
            portfolio_value = current_capital
            for pos in positions.values():
                portfolio_value += pos.current_value
            
            portfolio_values.append(portfolio_value)
        
        # Generate performance report
        results = {
            'initial_capital': initial_capital,
            'final_value': portfolio_values[-1],
            'portfolio_values': portfolio_values,
            'trades': trades,
            'positions': positions,
            'metadata': {
                'engine': self._name,
                'parameters': self._parameters,
                'signal_count': len(signals),
                'trade_count': len(trades)
            }
        }
        
        # Add performance analysis
        performance_report = self._analyzer.generate_report(results)
        results['performance'] = performance_report
        
                 return results
```

---

## 🎓 **Training & Model Management System**

### **🧠 Model Training Framework**

The training system will support on-demand model training for advanced signal generation strategies, with separate training and inference phases:

#### **Training Interface**
```python
# model/training/interfaces/training_interface.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    name: str
    description: str
    model_type: str  # 'ml_model', 'statistical_model', 'custom_model'
    parameters: Dict[str, Any]
    required_data: List[str]
    training_window: Optional[int] = None  # Days of data for training
    validation_window: Optional[int] = None  # Days of data for validation
    retrain_frequency: Optional[str] = None  # 'daily', 'weekly', 'monthly', 'on_demand'

@dataclass
class ModelMetadata:
    """Metadata for trained models"""
    model_id: str
    name: str
    version: str
    training_date: datetime
    training_config: TrainingConfig
    performance_metrics: Dict[str, float]
    data_sources: List[str]
    symbols_trained: List[str]
    model_path: str
    is_active: bool = True

class TrainingInterface(ABC):
    """Abstract interface for model training"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return trainer name"""
        pass
    
    @abstractmethod
    def get_config(self) -> TrainingConfig:
        """Return training configuration"""
        pass
    
    @abstractmethod
    def train_model(self, training_data: Dict[str, pd.DataFrame], 
                   symbols: List[str], **kwargs) -> ModelMetadata:
        """Train a new model"""
        pass
    
    @abstractmethod
    def validate_training_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate training data format"""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """Return required data resources for training"""
        pass
    
    @abstractmethod
    def save_model(self, model, metadata: ModelMetadata) -> str:
        """Save trained model"""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load trained model"""
        pass
```

#### **Model Training Orchestrator**
```python
# model/training/training_orchestrator.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import uuid
import json
import os
from .interfaces.training_interface import TrainingInterface, ModelMetadata, TrainingConfig
from ..data.universal_fetcher import UniversalDataFetcher

class TrainingOrchestrator:
    """Orchestrates model training across different strategies"""
    
    def __init__(self, data_fetcher: UniversalDataFetcher, model_storage_path: str = "models/"):
        self.data_fetcher = data_fetcher
        self.model_storage_path = model_storage_path
        self.trainers: Dict[str, TrainingInterface] = {}
        self.model_registry: Dict[str, ModelMetadata] = {}
        
        # Create model storage directory
        os.makedirs(model_storage_path, exist_ok=True)
        self._load_model_registry()
    
    def register_trainer(self, trainer: TrainingInterface):
        """Register a training strategy"""
        self.trainers[trainer.get_name()] = trainer
    
    def train_model(self, trainer_name: str, symbols: List[str], 
                   start_date: str, end_date: str, **kwargs) -> ModelMetadata:
        """Train a model using specified trainer"""
        
        if trainer_name not in self.trainers:
            raise ValueError(f"Unknown trainer: {trainer_name}")
        
        trainer = self.trainers[trainer_name]
        config = trainer.get_config()
        
        # Fetch training data
        training_data = self._fetch_training_data(trainer, symbols, start_date, end_date)
        
        # Validate training data
        if not trainer.validate_training_data(training_data):
            raise ValueError(f"Invalid training data for {trainer_name}")
        
        # Train model
        model_metadata = trainer.train_model(training_data, symbols, **kwargs)
        
        # Save model
        model_path = trainer.save_model(model_metadata)
        model_metadata.model_path = model_path
        
        # Register model
        self._register_model(model_metadata)
        
        return model_metadata
    
    def get_trained_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Get a trained model by ID"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.model_registry[model_id]
        trainer_name = metadata.training_config.name
        
        if trainer_name not in self.trainers:
            raise ValueError(f"Trainer {trainer_name} not available")
        
        trainer = self.trainers[trainer_name]
        return trainer.load_model(model_id)
    
    def list_models(self, active_only: bool = True) -> List[ModelMetadata]:
        """List available models"""
        models = list(self.model_registry.values())
        if active_only:
            models = [m for m in models if m.is_active]
        return sorted(models, key=lambda x: x.training_date, reverse=True)
    
    def retrain_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Retrain an existing model"""
        if model_id not in self.model_registry:
            raise ValueError(f"Model {model_id} not found")
        
        original_metadata = self.model_registry[model_id]
        
        # Calculate new training window
        end_date = datetime.now().strftime('%Y-%m-%d')
        training_window = original_metadata.training_config.training_window or 252  # Default 1 year
        start_date = (datetime.now() - timedelta(days=training_window)).strftime('%Y-%m-%d')
        
        # Retrain with new data
        new_metadata = self.train_model(
            original_metadata.training_config.name,
            original_metadata.symbols_trained,
            start_date,
            end_date,
            **kwargs
        )
        
        # Deactivate old model
        original_metadata.is_active = False
        self._save_model_registry()
        
        return new_metadata
    
    def _fetch_training_data(self, trainer: TrainingInterface, symbols: List[str], 
                           start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch data required for training"""
        required_resources = trainer.get_required_data()
        training_data = {}
        
        for symbol in symbols:
            symbol_data = {}
            for resource in required_resources:
                data, metadata = self.data_fetcher.fetch_resource(
                    resource, symbol, start_date, end_date
                )
                if not data.empty:
                    symbol_data[resource] = data
            
            if symbol_data:
                training_data[symbol] = symbol_data
        
        return training_data
    
    def _register_model(self, metadata: ModelMetadata):
        """Register a new model"""
        self.model_registry[metadata.model_id] = metadata
        self._save_model_registry()
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_path = os.path.join(self.model_storage_path, "model_registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
                for model_id, metadata_dict in registry_data.items():
                    # Convert back to ModelMetadata object
                    metadata = ModelMetadata(**metadata_dict)
                    self.model_registry[model_id] = metadata
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_path = os.path.join(self.model_storage_path, "model_registry.json")
        registry_data = {
            model_id: {
                'model_id': metadata.model_id,
                'name': metadata.name,
                'version': metadata.version,
                'training_date': metadata.training_date.isoformat(),
                'training_config': metadata.training_config.__dict__,
                'performance_metrics': metadata.performance_metrics,
                'data_sources': metadata.data_sources,
                'symbols_trained': metadata.symbols_trained,
                'model_path': metadata.model_path,
                'is_active': metadata.is_active
            }
            for model_id, metadata in self.model_registry.items()
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
```

#### **ML-Based Training Strategies**
```python
# model/training/strategies/ml_trainer.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import uuid
from datetime import datetime
from ..interfaces.training_interface import TrainingInterface, TrainingConfig, ModelMetadata

class MLSignalTrainer(TrainingInterface):
    """Machine learning-based signal trainer"""
    
    def __init__(self, model_type: str = 'random_forest', parameters: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.parameters = parameters or {}
        
        # Model type mappings
        self.model_classes = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression
        }
        
        if model_type not in self.model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_name(self) -> str:
        return f"ml_trainer_{self.model_type}"
    
    def get_config(self) -> TrainingConfig:
        return TrainingConfig(
            name=self.get_name(),
            description=f"Machine learning signal trainer using {self.model_type}",
            model_type="ml_model",
            parameters=self.parameters,
            required_data=['stock_price', 'revenue', 'market_cap'],  # Can be customized
            training_window=252,  # 1 year of data
            validation_window=63,  # 3 months for validation
            retrain_frequency="monthly"
        )
    
    def get_required_data(self) -> List[str]:
        return ['stock_price', 'revenue', 'market_cap']
    
    def validate_training_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        required = self.get_required_data()
        return all(resource in data for resource in required)
    
    def train_model(self, training_data: Dict[str, pd.DataFrame], 
                   symbols: List[str], **kwargs) -> ModelMetadata:
        """Train ML model for signal generation"""
        
        # Prepare features and labels
        features, labels = self._prepare_training_data(training_data, symbols)
        
        if len(features) == 0:
            raise ValueError("No valid training data available")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Initialize model
        model_class = self.model_classes[self.model_type]
        model = model_class(**self.parameters)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        performance_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, features, labels, cv=5)
        performance_metrics['cv_mean'] = cv_scores.mean()
        performance_metrics['cv_std'] = cv_scores.std()
        
        # Create model metadata
        model_id = str(uuid.uuid4())
        metadata = ModelMetadata(
            model_id=model_id,
            name=f"ML_Signal_Model_{self.model_type}",
            version="1.0",
            training_date=datetime.now(),
            training_config=self.get_config(),
            performance_metrics=performance_metrics,
            data_sources=list(training_data.keys()),
            symbols_trained=symbols,
            model_path="",  # Will be set by orchestrator
            is_active=True
        )
        
        # Store model with metadata
        self._model = model
        self._metadata = metadata
        
        return metadata
    
    def save_model(self, metadata: ModelMetadata) -> str:
        """Save trained model to disk"""
        model_path = f"models/{metadata.model_id}.joblib"
        
        # Save model and metadata
        model_data = {
            'model': self._model,
            'metadata': metadata,
            'feature_names': self._feature_names if hasattr(self, '_feature_names') else None
        }
        
        joblib.dump(model_data, model_path)
        return model_path
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load trained model from disk"""
        model_path = f"models/{model_id}.joblib"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        return model_data['model'], model_data['metadata']
    
    def _prepare_training_data(self, training_data: Dict[str, pd.DataFrame], 
                             symbols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training"""
        all_features = []
        all_labels = []
        
        for symbol in symbols:
            if symbol not in training_data:
                continue
            
            symbol_data = training_data[symbol]
            
            # Prepare features
            features = self._extract_features(symbol_data)
            labels = self._extract_labels(symbol_data)
            
            if not features.empty and not labels.empty:
                all_features.append(features)
                all_labels.append(labels)
        
        if not all_features:
            return pd.DataFrame(), pd.Series()
        
        # Combine all symbols
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        # Store feature names for later use
        self._feature_names = combined_features.columns.tolist()
        
        return combined_features, combined_labels
    
    def _extract_features(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract features from symbol data"""
        features = []
        
        # Technical features from price data
        if 'stock_price' in symbol_data:
            price_data = symbol_data['stock_price']
            
            # Price-based features
            features.extend([
                price_data['close'].pct_change(),
                price_data['volume'].pct_change(),
                price_data['close'].rolling(5).mean(),
                price_data['close'].rolling(20).mean(),
                price_data['close'].rolling(50).mean(),
                price_data['volume'].rolling(20).mean(),
                (price_data['high'] - price_data['low']) / price_data['close'],  # Volatility
            ])
        
        # Fundamental features
        if 'revenue' in symbol_data:
            revenue_data = symbol_data['revenue']
            features.extend([
                revenue_data['revenue'].pct_change(),
                revenue_data['revenue'].rolling(4).mean(),  # Annual average
            ])
        
        if 'market_cap' in symbol_data:
            market_cap_data = symbol_data['market_cap']
            features.extend([
                market_cap_data['market_cap'].pct_change(),
                market_cap_data['market_cap'].rolling(20).mean(),
            ])
        
        if not features:
            return pd.DataFrame()
        
        # Combine features
        feature_df = pd.concat(features, axis=1)
        feature_df.columns = [f'feature_{i}' for i in range(len(feature_df.columns))]
        
        # Remove NaN values
        feature_df = feature_df.dropna()
        
        return feature_df
    
    def _extract_labels(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Extract labels for training (future returns)"""
        if 'stock_price' not in symbol_data:
            return pd.Series()
        
        price_data = symbol_data['stock_price']
        
        # Calculate future returns (5-day forward returns)
        future_returns = price_data['close'].shift(-5) / price_data['close'] - 1
        
        # Create binary labels (1 for positive return, 0 for negative)
        labels = (future_returns > 0).astype(int)
        
        # Remove NaN values
        labels = labels.dropna()
        
        return labels
```

#### **Statistical Model Training**
```python
# model/training/strategies/statistical_trainer.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from ..interfaces.training_interface import TrainingInterface, TrainingConfig, ModelMetadata

class StatisticalModelTrainer(TrainingInterface):
    """Statistical model trainer for signal generation"""
    
    def __init__(self, model_type: str = 'mean_reversion', parameters: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.parameters = parameters or {}
        
        self.model_types = {
            'mean_reversion': self._train_mean_reversion_model,
            'momentum': self._train_momentum_model,
            'volatility_regime': self._train_volatility_regime_model
        }
    
    def get_name(self) -> str:
        return f"statistical_trainer_{self.model_type}"
    
    def get_config(self) -> TrainingConfig:
        return TrainingConfig(
            name=self.get_name(),
            description=f"Statistical model trainer using {self.model_type}",
            model_type="statistical_model",
            parameters=self.parameters,
            required_data=['stock_price'],
            training_window=252,
            validation_window=63,
            retrain_frequency="weekly"
        )
    
    def get_required_data(self) -> List[str]:
        return ['stock_price']
    
    def validate_training_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        return 'stock_price' in data and not data['stock_price'].empty
    
    def train_model(self, training_data: Dict[str, pd.DataFrame], 
                   symbols: List[str], **kwargs) -> ModelMetadata:
        """Train statistical model"""
        
        # Prepare training data
        prepared_data = self._prepare_training_data(training_data, symbols)
        
        if not prepared_data:
            raise ValueError("No valid training data available")
        
        # Train model based on type
        if self.model_type not in self.model_types:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        train_method = self.model_types[self.model_type]
        model_params, performance_metrics = train_method(prepared_data)
        
        # Create model metadata
        model_id = str(uuid.uuid4())
        metadata = ModelMetadata(
            model_id=model_id,
            name=f"Statistical_Model_{self.model_type}",
            version="1.0",
            training_date=datetime.now(),
            training_config=self.get_config(),
            performance_metrics=performance_metrics,
            data_sources=list(training_data.keys()),
            symbols_trained=symbols,
            model_path="",
            is_active=True
        )
        
        # Store model
        self._model_params = model_params
        self._metadata = metadata
        
        return metadata
    
    def _train_mean_reversion_model(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train mean reversion model"""
        returns = data['returns'].dropna()
        
        # Calculate mean and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Calculate z-score thresholds
        z_score_upper = self.parameters.get('z_score_upper', 2.0)
        z_score_lower = self.parameters.get('z_score_lower', -2.0)
        
        # Generate signals for backtesting
        signals = []
        for z_score in np.arange(z_score_lower, z_score_upper + 0.1, 0.1):
            # Buy when z-score is low (oversold)
            buy_signals = returns < (mean_return + z_score * std_return)
            # Sell when z-score is high (overbought)
            sell_signals = returns > (mean_return + z_score * std_return)
            
            # Calculate performance
            buy_returns = returns[buy_signals].shift(-1)
            sell_returns = -returns[sell_signals].shift(-1)
            
            total_return = buy_returns.sum() + sell_returns.sum()
            signals.append((z_score, total_return))
        
        # Find optimal z-score
        optimal_z_score = max(signals, key=lambda x: x[1])[0]
        
        model_params = {
            'mean_return': mean_return,
            'std_return': std_return,
            'z_score_upper': optimal_z_score,
            'z_score_lower': -optimal_z_score,
            'model_type': 'mean_reversion'
        }
        
        performance_metrics = {
            'optimal_z_score': optimal_z_score,
            'mean_return': mean_return,
            'std_return': std_return,
            'signal_count': len(signals)
        }
        
        return model_params, performance_metrics
    
    def _train_momentum_model(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train momentum model"""
        returns = data['returns'].dropna()
        
        # Test different lookback periods
        lookback_periods = [5, 10, 20, 50, 100]
        momentum_signals = []
        
        for period in lookback_periods:
            # Calculate momentum
            momentum = returns.rolling(period).sum()
            
            # Generate signals
            buy_signals = momentum > 0
            sell_signals = momentum < 0
            
            # Calculate performance
            buy_returns = returns[buy_signals].shift(-1)
            sell_returns = -returns[sell_signals].shift(-1)
            
            total_return = buy_returns.sum() + sell_returns.sum()
            momentum_signals.append((period, total_return))
        
        # Find optimal lookback period
        optimal_period = max(momentum_signals, key=lambda x: x[1])[0]
        
        model_params = {
            'lookback_period': optimal_period,
            'model_type': 'momentum'
        }
        
        performance_metrics = {
            'optimal_lookback': optimal_period,
            'momentum_signals': len(momentum_signals)
        }
        
        return model_params, performance_metrics
    
    def _train_volatility_regime_model(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Train volatility regime model"""
        returns = data['returns'].dropna()
        
        # Calculate rolling volatility
        volatility_window = self.parameters.get('volatility_window', 20)
        rolling_vol = returns.rolling(volatility_window).std()
        
        # Identify volatility regimes
        vol_median = rolling_vol.median()
        high_vol_regime = rolling_vol > vol_median
        low_vol_regime = rolling_vol <= vol_median
        
        # Calculate returns in different regimes
        high_vol_returns = returns[high_vol_regime]
        low_vol_returns = returns[low_vol_regime]
        
        model_params = {
            'volatility_window': volatility_window,
            'volatility_threshold': vol_median,
            'high_vol_mean': high_vol_returns.mean(),
            'low_vol_mean': low_vol_returns.mean(),
            'model_type': 'volatility_regime'
        }
        
        performance_metrics = {
            'volatility_threshold': vol_median,
            'high_vol_periods': high_vol_regime.sum(),
            'low_vol_periods': low_vol_regime.sum()
        }
        
        return model_params, performance_metrics
```

#### **Enhanced Signal Generation with Models**
```python
# model/signals/strategies/ml_signal_strategy.py
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .base_strategy import BaseSignalStrategy
from ..interfaces.signal_generator_interface import SignalConfig

class MLSignalStrategy(BaseSignalStrategy):
    """Machine learning-based signal strategy"""
    
    def __init__(self, model_id: str, training_orchestrator, parameters: Optional[Dict[str, Any]] = None):
        self.model_id = model_id
        self.training_orchestrator = training_orchestrator
        self.model = None
        self.model_metadata = None
        
        # Load model
        self._load_model()
        
        default_params = {
            'confidence_threshold': 0.6,
            'prediction_horizon': 5
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(
            name="ml_signal_strategy",
            description=f"ML-based signal strategy using model {model_id}",
            parameters=default_params
        )
    
    def _load_model(self):
        """Load trained model"""
        try:
            self.model, self.model_metadata = self.training_orchestrator.get_trained_model(self.model_id)
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_id}: {str(e)}")
    
    def _create_config(self) -> SignalConfig:
        return SignalConfig(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            required_data=self.model_metadata.training_config.required_data,
            output_fields=['signal_type', 'confidence', 'prediction_probability', 'model_features']
        )
    
    def get_required_resources(self) -> List[str]:
        return self.model_metadata.training_config.required_data
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        required = self.get_required_resources()
        return all(resource in data and not data[resource].empty for resource in required)
    
    def _generate_signals_for_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate ML-based signals"""
        
        # Prepare features
        features = self._prepare_features(data)
        
        if features.empty:
            return []
        
        # Get latest feature vector
        latest_features = features.iloc[-1:].values
        
        # Make prediction
        prediction = self.model.predict(latest_features)[0]
        prediction_proba = self.model.predict_proba(latest_features)[0]
        
        # Determine signal type
        if prediction == 1:  # Positive return predicted
            signal_type = "BUY"
            confidence = prediction_proba[1]  # Probability of positive return
        else:
            signal_type = "SELL"
            confidence = prediction_proba[0]  # Probability of negative return
        
        # Apply confidence threshold
        if confidence < self._parameters['confidence_threshold']:
            signal_type = "HOLD"
            confidence = 0.5
        
        # Get latest price
        price_data = data['stock_price']
        latest_price = price_data['close'].iloc[-1]
        
        return [{
            'symbol': symbol,
            'date': price_data['date'].iloc[-1],
            'signal_type': signal_type,
            'confidence': round(confidence, 3),
            'strength': 'STRONG' if confidence > 0.8 else 'MODERATE' if confidence > 0.6 else 'WEAK',
            'price': latest_price,
            'prediction_probability': prediction_proba.tolist(),
            'model_features': features.columns.tolist(),
            'model_id': self.model_id,
            'strategy': self._name,
            'parameters': self._parameters
        }]
    
    def _prepare_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare features for ML model"""
        features = []
        
        # Technical features from price data
        if 'stock_price' in data:
            price_data = data['stock_price']
            
            features.extend([
                price_data['close'].pct_change(),
                price_data['volume'].pct_change(),
                price_data['close'].rolling(5).mean(),
                price_data['close'].rolling(20).mean(),
                price_data['close'].rolling(50).mean(),
                price_data['volume'].rolling(20).mean(),
                (price_data['high'] - price_data['low']) / price_data['close'],
            ])
        
        # Fundamental features
        if 'revenue' in data:
            revenue_data = data['revenue']
            features.extend([
                revenue_data['revenue'].pct_change(),
                revenue_data['revenue'].rolling(4).mean(),
            ])
        
        if 'market_cap' in data:
            market_cap_data = data['market_cap']
            features.extend([
                market_cap_data['market_cap'].pct_change(),
                market_cap_data['market_cap'].rolling(20).mean(),
            ])
        
        if not features:
            return pd.DataFrame()
        
        # Combine features
        feature_df = pd.concat(features, axis=1)
        feature_df.columns = [f'feature_{i}' for i in range(len(feature_df.columns))]
        
        # Remove NaN values
        feature_df = feature_df.dropna()
        
        return feature_df
```

#### **Training Pipeline Integration**
```python
# model/pipeline/training_pipeline.py
from typing import Dict, List, Any, Optional
import pandas as pd
from ..training.training_orchestrator import TrainingOrchestrator
from ..data.universal_fetcher import UniversalDataFetcher

class TrainingPipeline:
    """Pipeline for model training and management"""
    
    def __init__(self, data_fetcher: UniversalDataFetcher, 
                 training_orchestrator: TrainingOrchestrator):
        self.data_fetcher = data_fetcher
        self.training_orchestrator = training_orchestrator
    
    def train_ml_model(self, symbols: List[str], 
                      start_date: str, end_date: str,
                      model_type: str = 'random_forest',
                      **kwargs) -> ModelMetadata:
        """Train ML model for signal generation"""
        
        # Register ML trainer
        from ..training.strategies.ml_trainer import MLSignalTrainer
        trainer = MLSignalTrainer(model_type=model_type, parameters=kwargs)
        self.training_orchestrator.register_trainer(trainer)
        
        # Train model
        metadata = self.training_orchestrator.train_model(
            trainer.get_name(), symbols, start_date, end_date, **kwargs
        )
        
        return metadata
    
    def train_statistical_model(self, symbols: List[str],
                              start_date: str, end_date: str,
                              model_type: str = 'mean_reversion',
                              **kwargs) -> ModelMetadata:
        """Train statistical model for signal generation"""
        
        # Register statistical trainer
        from ..training.strategies.statistical_trainer import StatisticalModelTrainer
        trainer = StatisticalModelTrainer(model_type=model_type, parameters=kwargs)
        self.training_orchestrator.register_trainer(trainer)
        
        # Train model
        metadata = self.training_orchestrator.train_model(
            trainer.get_name(), symbols, start_date, end_date, **kwargs
        )
        
        return metadata
    
    def create_ml_signal_strategy(self, model_id: str) -> MLSignalStrategy:
        """Create ML signal strategy from trained model"""
        return MLSignalStrategy(model_id, self.training_orchestrator)
    
    def retrain_model(self, model_id: str, **kwargs) -> ModelMetadata:
        """Retrain existing model with new data"""
        return self.training_orchestrator.retrain_model(model_id, **kwargs)
    
    def list_available_models(self, active_only: bool = True) -> List[ModelMetadata]:
        """List all available trained models"""
                 return self.training_orchestrator.list_models(active_only)
```

---

## 🏗️ **Component Registry System**

### **🎯 Central Component Management**

The Component Registry System provides a centralized way to discover, register, and manage all system components dynamically:

#### **Component Registry Interface**
```python
# model/registry/component_registry.py
from typing import Dict, List, Any, Optional, Type, Union
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class ComponentMetadata:
    """Metadata for registered components"""
    name: str
    type: str
    version: str
    description: str
    author: str
    created_date: datetime
    last_updated: datetime
    dependencies: List[str]
    configuration_schema: Dict[str, Any]
    is_active: bool = True

class ComponentRegistry:
    """Central registry for all system components"""
    
    def __init__(self, registry_path: str = "config/component_registry.json"):
        self.registry_path = registry_path
        self.components: Dict[str, Dict[str, Any]] = {}
        self.component_metadata: Dict[str, ComponentMetadata] = {}
        self._load_registry()
    
    def register_component(self, component_type: str, name: str, 
                          component_class: Type, metadata: ComponentMetadata):
        """Register a new component"""
        
        if component_type not in self.components:
            self.components[component_type] = {}
        
        # Validate component
        if not self._validate_component(component_class, metadata):
            raise ValueError(f"Invalid component: {name}")
        
        # Register component
        self.components[component_type][name] = {
            'class': component_class,
            'metadata': metadata,
            'instance': None
        }
        
        self.component_metadata[f"{component_type}:{name}"] = metadata
        self._save_registry()
        
        print(f"✅ Registered {component_type}:{name}")
    
    def get_component(self, component_type: str, name: str, 
                     create_instance: bool = True, **kwargs):
        """Get a component by type and name"""
        
        if component_type not in self.components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        if name not in self.components[component_type]:
            raise ValueError(f"Unknown component: {component_type}:{name}")
        
        component_info = self.components[component_type][name]
        
        # Create instance if requested and not exists
        if create_instance and component_info['instance'] is None:
            component_info['instance'] = component_info['class'](**kwargs)
        
        return component_info['instance'] if create_instance else component_info['class']
    
    def list_components(self, component_type: str = None) -> List[Dict[str, Any]]:
        """List all components or components of a specific type"""
        
        if component_type:
            if component_type not in self.components:
                return []
            
            return [
                {
                    'name': name,
                    'type': component_type,
                    'metadata': asdict(info['metadata'])
                }
                for name, info in self.components[component_type].items()
            ]
        else:
            all_components = []
            for comp_type, components in self.components.items():
                for name, info in components.items():
                    all_components.append({
                        'name': name,
                        'type': comp_type,
                        'metadata': asdict(info['metadata'])
                    })
            return all_components
    
    def validate_component(self, component_type: str, name: str) -> bool:
        """Validate component configuration"""
        
        if component_type not in self.components:
            return False
        
        if name not in self.components[component_type]:
            return False
        
        component_info = self.components[component_type][name]
        metadata = component_info['metadata']
        
        # Check if component is active
        if not metadata.is_active:
            return False
        
        # Check dependencies
        for dependency in metadata.dependencies:
            if not self._check_dependency(dependency):
                return False
        
        return True
    
    def update_component(self, component_type: str, name: str, 
                        updates: Dict[str, Any]):
        """Update component metadata"""
        
        if component_type not in self.components or name not in self.components[component_type]:
            raise ValueError(f"Component not found: {component_type}:{name}")
        
        component_info = self.components[component_type][name]
        metadata = component_info['metadata']
        
        # Update metadata fields
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.last_updated = datetime.now()
        self._save_registry()
    
    def deactivate_component(self, component_type: str, name: str):
        """Deactivate a component"""
        self.update_component(component_type, name, {'is_active': False})
    
    def activate_component(self, component_type: str, name: str):
        """Activate a component"""
        self.update_component(component_type, name, {'is_active': True})
    
    def remove_component(self, component_type: str, name: str):
        """Remove a component from registry"""
        
        if component_type in self.components and name in self.components[component_type]:
            del self.components[component_type][name]
            
            metadata_key = f"{component_type}:{name}"
            if metadata_key in self.component_metadata:
                del self.component_metadata[metadata_key]
            
            self._save_registry()
    
    def _validate_component(self, component_class: Type, metadata: ComponentMetadata) -> bool:
        """Validate component class and metadata"""
        
        # Check if class has required methods (basic validation)
        required_methods = self._get_required_methods(metadata.type)
        
        for method in required_methods:
            if not hasattr(component_class, method):
                print(f"❌ Component {metadata.name} missing required method: {method}")
                return False
        
        return True
    
    def _get_required_methods(self, component_type: str) -> List[str]:
        """Get required methods for component type"""
        
        method_requirements = {
            'data_source': ['get_name', 'get_supported_resources', 'fetch_data'],
            'signal_generator': ['get_name', 'get_config', 'generate_signals'],
            'backtest_engine': ['get_name', 'get_config', 'run_backtest'],
            'training_strategy': ['get_name', 'get_config', 'train_model'],
            'execution_engine': ['execute_trade', 'calculate_execution_price'],
            'risk_manager': ['validate_trade', 'calculate_position_size'],
            'performance_analyzer': ['calculate_returns', 'analyze_trades']
        }
        
        return method_requirements.get(component_type, [])
    
    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is available"""
        
        # Parse dependency (format: "type:name")
        if ':' in dependency:
            dep_type, dep_name = dependency.split(':', 1)
            return self.validate_component(dep_type, dep_name)
        
        return True  # Assume satisfied if no specific component required
    
    def _load_registry(self):
        """Load component registry from disk"""
        
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                    
                    # Load components
                    self.components = registry_data.get('components', {})
                    
                    # Load metadata
                    metadata_data = registry_data.get('metadata', {})
                    for key, meta_dict in metadata_data.items():
                        # Convert datetime strings back to datetime objects
                        meta_dict['created_date'] = datetime.fromisoformat(meta_dict['created_date'])
                        meta_dict['last_updated'] = datetime.fromisoformat(meta_dict['last_updated'])
                        self.component_metadata[key] = ComponentMetadata(**meta_dict)
                        
            except Exception as e:
                print(f"⚠️ Warning: Could not load component registry: {e}")
                self.components = {}
                self.component_metadata = {}
    
    def _save_registry(self):
        """Save component registry to disk"""
        
        # Prepare data for serialization
        registry_data = {
            'components': self.components,
            'metadata': {
                key: asdict(metadata) 
                for key, metadata in self.component_metadata.items()
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        # Save to file
        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
```

#### **Component Registration Examples**
```python
# model/registry/register_components.py
from datetime import datetime
from .component_registry import ComponentRegistry, ComponentMetadata
from ..data.sources.yfinance_source import YFinanceDataSource
from ..signals.strategies.technical_strategy import TechnicalAnalysisStrategy
from ..backtesting.engines.base_backtest_engine import BaseBacktestEngine

def register_default_components(registry: ComponentRegistry):
    """Register default system components"""
    
    # Register YFinance Data Source
    yfinance_metadata = ComponentMetadata(
        name="yfinance",
        type="data_source",
        version="1.0.0",
        description="Yahoo Finance data source with multi-resource support",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "api_key": {"type": "string", "required": False},
            "rate_limits": {"type": "object", "required": False}
        }
    )
    
    registry.register_component(
        "data_source", "yfinance", YFinanceDataSource, yfinance_metadata
    )
    
    # Register Technical Analysis Strategy
    technical_metadata = ComponentMetadata(
        name="technical_analysis",
        type="signal_generator",
        version="1.0.0",
        description="Technical analysis-based signal generation",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=[],
        configuration_schema={
            "rsi_period": {"type": "integer", "default": 14},
            "ma_short": {"type": "integer", "default": 20},
            "ma_long": {"type": "integer", "default": 50}
        }
    )
    
    registry.register_component(
        "signal_generator", "technical_analysis", TechnicalAnalysisStrategy, technical_metadata
    )
    
    # Register Standard Backtest Engine
    backtest_metadata = ComponentMetadata(
        name="standard_engine",
        type="backtest_engine",
        version="1.0.0",
        description="Standard backtesting engine with modular components",
        author="BreadthFlow Team",
        created_date=datetime.now(),
        last_updated=datetime.now(),
        dependencies=["execution_engine", "risk_manager"],
        configuration_schema={
            "initial_capital": {"type": "float", "default": 100000},
            "commission_rate": {"type": "float", "default": 0.001}
        }
    )
    
    registry.register_component(
        "backtest_engine", "standard_engine", BaseBacktestEngine, backtest_metadata
    )
```

---

## ⚙️ **Configuration Management**

### **🎯 Centralized Configuration System**

The Configuration Management System provides a centralized way to manage all system configurations, component settings, and environment variables:

#### **Configuration Manager Interface**
```python
# model/config/configuration_manager.py
from typing import Dict, List, Any, Optional, Union
import yaml
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConfigSchema:
    """Schema definition for configuration validation"""
    name: str
    type: str
    required: bool = False
    default: Any = None
    description: str = ""
    validation_rules: Dict[str, Any] = None

class ConfigurationManager:
    """Manages system configuration and component settings"""
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = Path(config_path)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.schemas: Dict[str, Dict[str, ConfigSchema]] = {}
        self._load_configurations()
    
    def load_config(self, config_name: str, config_type: str = "yaml") -> Dict[str, Any]:
        """Load configuration from file"""
        
        config_file = self.config_path / f"{config_name}.{config_type}"
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                if config_type == "yaml":
                    config_data = yaml.safe_load(f)
                elif config_type == "json":
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")
            
            # Validate configuration
            if config_name in self.schemas:
                self._validate_config(config_data, self.schemas[config_name])
            
            self.configs[config_name] = config_data
            logger.info(f"✅ Loaded configuration: {config_name}")
            return config_data
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration {config_name}: {e}")
            return {}
    
    def save_config(self, config_name: str, config_data: Dict[str, Any], 
                   config_type: str = "yaml"):
        """Save configuration to file"""
        
        # Ensure config directory exists
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        config_file = self.config_path / f"{config_name}.{config_type}"
        
        try:
            with open(config_file, 'w') as f:
                if config_type == "yaml":
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif config_type == "json":
                    json.dump(config_data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")
            
            self.configs[config_name] = config_data
            logger.info(f"✅ Saved configuration: {config_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save configuration {config_name}: {e}")
    
    def get_config(self, config_name: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        
        if config_name not in self.configs:
            self.load_config(config_name)
        
        if config_name not in self.configs:
            return default
        
        config_data = self.configs[config_name]
        
        if key is None:
            return config_data
        
        # Support nested keys (e.g., "database.host")
        keys = key.split('.')
        value = config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, config_name: str, key: str, value: Any):
        """Set configuration value"""
        
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        # Support nested keys
        keys = key.split('.')
        config_data = self.configs[config_name]
        
        for k in keys[:-1]:
            if k not in config_data:
                config_data[k] = {}
            config_data = config_data[k]
        
        config_data[keys[-1]] = value
    
    def validate_config(self, config_data: Dict[str, Any], schema: Dict[str, ConfigSchema]) -> bool:
        """Validate configuration against schema"""
        
        for field_name, field_schema in schema.items():
            if field_schema.required and field_name not in config_data:
                logger.error(f"❌ Required field missing: {field_name}")
                return False
            
            if field_name in config_data:
                value = config_data[field_name]
                
                # Type validation
                if not self._validate_type(value, field_schema.type):
                    logger.error(f"❌ Invalid type for {field_name}: expected {field_schema.type}")
                    return False
                
                # Custom validation rules
                if field_schema.validation_rules:
                    if not self._validate_rules(value, field_schema.validation_rules):
                        logger.error(f"❌ Validation failed for {field_name}")
                        return False
        
        return True
    
    def get_component_config(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """Get configuration for specific component"""
        
        # Try component-specific config first
        component_config = self.get_config(f"{component_type}_{component_name}")
        if component_config:
            return component_config
        
        # Fall back to component type config
        type_config = self.get_config(f"{component_type}_default")
        if type_config:
            return type_config
        
        # Fall back to global config
        return self.get_config("global", f"components.{component_type}", {})
    
    def register_schema(self, config_name: str, schema: Dict[str, ConfigSchema]):
        """Register configuration schema for validation"""
        self.schemas[config_name] = schema
    
    def _validate_config(self, config_data: Dict[str, Any], schema: Dict[str, ConfigSchema]):
        """Internal validation method"""
        if not self.validate_config(config_data, schema):
            raise ValueError(f"Configuration validation failed for {config_name}")
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        
        type_mapping = {
            'string': str,
            'integer': int,
            'float': (int, float),
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_class = type_mapping.get(expected_type)
        if expected_class is None:
            return True  # Unknown type, skip validation
        
        return isinstance(value, expected_class)
    
    def _validate_rules(self, value: Any, rules: Dict[str, Any]) -> bool:
        """Validate against custom rules"""
        
        for rule, rule_value in rules.items():
            if rule == 'min' and value < rule_value:
                return False
            elif rule == 'max' and value > rule_value:
                return False
            elif rule == 'min_length' and len(value) < rule_value:
                return False
            elif rule == 'max_length' and len(value) > rule_value:
                return False
            elif rule == 'pattern' and not re.match(rule_value, str(value)):
                return False
        
        return True
    
    def _load_configurations(self):
        """Load all configuration files on startup"""
        
        if not self.config_path.exists():
            return
        
        # Load all yaml and json files
        for config_file in self.config_path.glob("*.yaml"):
            config_name = config_file.stem
            self.load_config(config_name, "yaml")
        
        for config_file in self.config_path.glob("*.json"):
            config_name = config_file.stem
            self.load_config(config_name, "json")
```

#### **Configuration Schemas**
```python
# model/config/schemas.py
from .configuration_manager import ConfigSchema

# Global system configuration schema
GLOBAL_CONFIG_SCHEMA = {
    'database': ConfigSchema(
        name='database',
        type='dict',
        required=True,
        description='Database configuration'
    ),
    'logging': ConfigSchema(
        name='logging',
        type='dict',
        required=False,
        default={'level': 'INFO'},
        description='Logging configuration'
    ),
    'components': ConfigSchema(
        name='components',
        type='dict',
        required=False,
        default={},
        description='Component-specific configurations'
    )
}

# Data source configuration schema
DATA_SOURCE_CONFIG_SCHEMA = {
    'api_key': ConfigSchema(
        name='api_key',
        type='string',
        required=False,
        description='API key for data source'
    ),
    'rate_limits': ConfigSchema(
        name='rate_limits',
        type='dict',
        required=False,
        default={'requests_per_minute': 60},
        description='Rate limiting configuration'
    ),
    'timeout': ConfigSchema(
        name='timeout',
        type='integer',
        required=False,
        default=30,
        validation_rules={'min': 1, 'max': 300},
        description='Request timeout in seconds'
    )
}

# Signal generator configuration schema
SIGNAL_GENERATOR_CONFIG_SCHEMA = {
    'parameters': ConfigSchema(
        name='parameters',
        type='dict',
        required=True,
        description='Signal generation parameters'
    ),
    'required_data': ConfigSchema(
        name='required_data',
        type='list',
        required=True,
        description='Required data resources'
    ),
    'output_format': ConfigSchema(
        name='output_format',
        type='string',
        required=False,
        default='standard',
        description='Output format for signals'
    )
}
```

#### **Configuration Files**
```yaml
# config/global.yaml
database:
  host: localhost
  port: 5432
  name: breadthflow
  user: pipeline
  password: pipeline123

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/breadthflow.log

components:
  data_fetcher:
    default_timeout: 30
    max_retries: 3
  
  signal_generator:
    default_confidence_threshold: 0.6
    max_signals_per_symbol: 10
  
  backtest_engine:
    default_commission_rate: 0.001
    default_slippage_rate: 0.0005

# config/data_source_yfinance.yaml
api_key: null  # Not required for yfinance
rate_limits:
  requests_per_minute: 60
  requests_per_hour: 2000
timeout: 30
retry_attempts: 3

# config/signal_generator_technical_analysis.yaml
parameters:
  rsi_period: 14
  rsi_oversold: 30
  rsi_overbought: 70
  ma_short: 20
  ma_long: 50
  bb_period: 20
  bb_std: 2

required_data:
  - stock_price

output_format: standard
confidence_threshold: 0.6

# config/backtest_engine_standard.yaml
initial_capital: 100000
commission_rate: 0.001
slippage_rate: 0.0005
max_position_size: 0.1
stop_loss: 0.05
take_profit: 0.1

execution_engine:
  type: standard
  parameters:
    min_trade_value: 1000
    max_trade_value: 10000

risk_manager:
  type: standard
  parameters:
    max_drawdown: 0.15
    max_portfolio_risk: 0.02
```

---

## 🚨 **Error Handling & Logging**

### **🎯 Centralized Error Management**

The Error Handling & Logging system provides comprehensive error tracking, logging, and recovery mechanisms:

#### **Error Handler Interface**
```python
# model/logging/error_handler.py
from typing import Dict, List, Any, Optional, Callable
import logging
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import os
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: datetime
    component: str
    operation: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self, log_level: str = "INFO", max_errors: int = 1000):
        self.log_level = log_level
        self.max_errors = max_errors
        self.error_counts = defaultdict(int)
        self.error_records = deque(maxlen=max_errors)
        self.error_thresholds = {
            'LOW': 100,
            'MEDIUM': 50,
            'HIGH': 20,
            'CRITICAL': 5
        }
        self.rollback_threshold = 0.05  # 5% error rate triggers rollback
        self._setup_logging()
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    component: str = "unknown", operation: str = "unknown"):
        """Handle and log errors"""
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            severity=self._determine_severity(error, context)
        )
        
        # Add to records
        self.error_records.append(error_record)
        
        # Update error counts
        error_key = f"{component}:{operation}"
        self.error_counts[error_key] += 1
        
        # Log error
        self._log_error(error_record)
        
        # Check if rollback is needed
        if self.should_rollback(component, operation):
            self._trigger_rollback(component, operation)
        
        return error_record
    
    def get_error_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get error summary for monitoring"""
        
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        recent_errors = [
            error for error in self.error_records 
            if error.timestamp >= cutoff_time
        ]
        
        # Group by component
        component_errors = defaultdict(list)
        for error in recent_errors:
            component_errors[error.component].append(error)
        
        # Calculate error rates
        error_rates = {}
        for component, errors in component_errors.items():
            total_operations = self._get_operation_count(component, time_window)
            error_rate = len(errors) / max(total_operations, 1)
            error_rates[component] = error_rate
        
        return {
            'total_errors': len(recent_errors),
            'error_rates': dict(error_rates),
            'component_breakdown': {
                component: len(errors) for component, errors in component_errors.items()
            },
            'severity_breakdown': {
                severity: len([e for e in recent_errors if e.severity == severity])
                for severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            },
            'time_window': str(time_window)
        }
    
    def should_rollback(self, component: str, operation: str) -> bool:
        """Determine if rollback is needed"""
        
        error_key = f"{component}:{operation}"
        error_count = self.error_counts[error_key]
        
        # Get recent error rate
        recent_errors = [
            error for error in self.error_records 
            if error.component == component and error.operation == operation
            and error.timestamp >= datetime.now() - timedelta(minutes=5)
        ]
        
        if len(recent_errors) == 0:
            return False
        
        # Calculate error rate
        total_operations = self._get_operation_count(component, timedelta(minutes=5))
        error_rate = len(recent_errors) / max(total_operations, 1)
        
        return error_rate > self.rollback_threshold
    
    def resolve_error(self, error_record: ErrorRecord, resolution_notes: str = ""):
        """Mark an error as resolved"""
        error_record.resolved = True
        error_record.resolution_time = datetime.now()
        
        # Update context with resolution notes
        error_record.context['resolution_notes'] = resolution_notes
        
        logger.info(f"✅ Error resolved: {error_record.component}:{error_record.operation}")
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status for a specific component"""
        
        recent_errors = [
            error for error in self.error_records 
            if error.component == component
            and error.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        total_operations = self._get_operation_count(component, timedelta(hours=1))
        error_rate = len(recent_errors) / max(total_operations, 1)
        
        # Determine health status
        if error_rate == 0:
            health_status = "HEALTHY"
        elif error_rate < 0.01:
            health_status = "WARNING"
        elif error_rate < 0.05:
            health_status = "DEGRADED"
        else:
            health_status = "CRITICAL"
        
        return {
            'component': component,
            'health_status': health_status,
            'error_rate': error_rate,
            'total_errors': len(recent_errors),
            'total_operations': total_operations,
            'last_error': recent_errors[-1] if recent_errors else None
        }
    
    def _determine_severity(self, error: Exception, context: Dict[str, Any]) -> str:
        """Determine error severity"""
        
        # Check error type
        critical_errors = ['ConnectionError', 'TimeoutError', 'AuthenticationError']
        high_errors = ['ValidationError', 'DataError', 'ConfigurationError']
        medium_errors = ['RateLimitError', 'TemporaryError']
        
        error_type = type(error).__name__
        
        if error_type in critical_errors:
            return 'CRITICAL'
        elif error_type in high_errors:
            return 'HIGH'
        elif error_type in medium_errors:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        
        log_message = f"Error in {error_record.component}:{error_record.operation} - {error_record.error_message}"
        
        if error_record.severity == 'CRITICAL':
            logger.critical(log_message, extra={'error_record': asdict(error_record)})
        elif error_record.severity == 'HIGH':
            logger.error(log_message, extra={'error_record': asdict(error_record)})
        elif error_record.severity == 'MEDIUM':
            logger.warning(log_message, extra={'error_record': asdict(error_record)})
        else:
            logger.info(log_message, extra={'error_record': asdict(error_record)})
    
    def _trigger_rollback(self, component: str, operation: str):
        """Trigger rollback for component"""
        
        logger.warning(f"🚨 Triggering rollback for {component}:{operation}")
        
        # Notify monitoring system
        self._notify_monitoring(component, operation, "ROLLBACK_TRIGGERED")
        
        # Could integrate with migration system here
        # migrator.rollback_component(component)
    
    def _notify_monitoring(self, component: str, operation: str, event: str):
        """Notify monitoring system of events"""
        
        # This could send alerts, update dashboards, etc.
        notification = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'operation': operation,
            'event': event,
            'error_rate': self.error_counts[f"{component}:{operation}"]
        }
        
        # Send to monitoring system (implementation depends on monitoring setup)
        logger.info(f"📊 Monitoring notification: {notification}")
    
    def _get_operation_count(self, component: str, time_window: timedelta) -> int:
        """Get total operation count for component in time window"""
        
        # This would typically come from metrics/monitoring system
        # For now, return a reasonable estimate
        return 100  # Placeholder
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/error_handler.log'),
                logging.StreamHandler()
            ]
        )
```

#### **Enhanced Logging System**
```python
# model/logging/enhanced_logger.py
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
from contextlib import contextmanager
import time

class EnhancedLogger:
    """Enhanced logging with structured data and performance tracking"""
    
    def __init__(self, name: str, component: str = "unknown"):
        self.logger = logging.getLogger(name)
        self.component = component
        self.performance_metrics = {}
    
    def log_operation(self, operation: str, data: Dict[str, Any] = None, 
                     level: str = "INFO"):
        """Log operation with structured data"""
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'component': self.component,
            'operation': operation,
            'data': data or {}
        }
        
        log_message = f"{operation} - {json.dumps(log_data)}"
        
        if level == "DEBUG":
            self.logger.debug(log_message)
        elif level == "INFO":
            self.logger.info(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "ERROR":
            self.logger.error(log_message)
        elif level == "CRITICAL":
            self.logger.critical(log_message)
    
    @contextmanager
    def log_performance(self, operation: str):
        """Context manager for logging operation performance"""
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log performance metrics
            self.log_operation(
                f"{operation}_performance",
                {
                    'duration_seconds': duration,
                    'memory_delta_mb': memory_delta,
                    'success': success
                }
            )
            
            # Store for monitoring
            self.performance_metrics[operation] = {
                'duration': duration,
                'memory_delta': memory_delta,
                'success': success,
                'timestamp': datetime.now()
            }
    
    def log_data_quality(self, data_source: str, quality_metrics: Dict[str, Any]):
        """Log data quality metrics"""
        
        self.log_operation(
            "data_quality_check",
            {
                'data_source': data_source,
                'quality_metrics': quality_metrics
            }
        )
    
    def log_component_health(self, health_status: Dict[str, Any]):
        """Log component health status"""
        
        self.log_operation(
            "component_health",
            health_status
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available
```

#### **Error Recovery Strategies**
```python
# model/logging/error_recovery.py
from typing import Callable, Dict, Any, Optional
import time
from functools import wraps

class ErrorRecovery:
    """Error recovery and retry mechanisms"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def retry_on_error(self, retryable_errors: list = None):
        """Decorator for retrying operations on specific errors"""
        
        if retryable_errors is None:
            retryable_errors = ['ConnectionError', 'TimeoutError', 'TemporaryError']
        
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        # Check if error is retryable
                        if type(e).__name__ not in retryable_errors:
                            raise e
                        
                        # If this is the last attempt, raise the exception
                        if attempt == self.max_retries:
                            raise e
                        
                        # Calculate backoff delay
                        delay = (self.backoff_factor ** attempt)
                        time.sleep(delay)
                
                raise last_exception
            
            return wrapper
        return decorator
    
    def circuit_breaker(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Circuit breaker pattern for preventing cascade failures"""
        
        def decorator(func: Callable):
            failure_count = 0
            last_failure_time = None
            circuit_open = False
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal failure_count, last_failure_time, circuit_open
                
                # Check if circuit is open
                if circuit_open:
                    if time.time() - last_failure_time > recovery_timeout:
                        circuit_open = False
                        failure_count = 0
                    else:
                        raise Exception("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    # Reset failure count on success
                    failure_count = 0
                    return result
                except Exception as e:
                    failure_count += 1
                    last_failure_time = time.time()
                    
                    # Open circuit if threshold reached
                    if failure_count >= failure_threshold:
                        circuit_open = True
                    
                    raise e
            
            return wrapper
        return decorator
    
    def fallback_strategy(self, fallback_func: Callable):
        """Fallback strategy for when primary operation fails"""
        
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Log the primary failure
                    logging.warning(f"Primary operation failed, using fallback: {e}")
                    
                    # Use fallback
                    return fallback_func(*args, **kwargs)
            
            return wrapper
        return decorator
```

---

## 🔍 **Data Validation & Quality**

### **🎯 Data Quality Assurance**

The Data Validation & Quality system ensures data integrity and consistency across all components:

#### **Data Validator Interface**
```python
# model/validation/data_validator.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationRule:
    """Definition of a validation rule"""
    name: str
    rule_type: str  # 'range', 'format', 'completeness', 'consistency'
    parameters: Dict[str, Any]
    severity: str  # 'ERROR', 'WARNING', 'INFO'

@dataclass
class ValidationResult:
    """Result of data validation"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    timestamp: datetime

class DataValidator:
    """Validates data quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {}
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.98,
            'consistency': 0.90
        }
    
    def validate_data_quality(self, data: pd.DataFrame, resource_name: str) -> ValidationResult:
        """Validate data quality for a resource"""
        
        errors = []
        warnings = []
        metrics = {}
        
        # Completeness check
        completeness = self._check_completeness(data)
        metrics['completeness'] = completeness
        
        if completeness < self.quality_thresholds['completeness']:
            warnings.append(f"Low completeness: {completeness:.2%}")
        
        # Data type validation
        type_errors = self._validate_data_types(data, resource_name)
        errors.extend(type_errors)
        
        # Range validation
        range_errors = self._validate_ranges(data, resource_name)
        errors.extend(range_errors)
        
        # Duplicate check
        duplicate_count = data.duplicated().sum()
        metrics['duplicates'] = duplicate_count
        
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate records")
        
        # Missing values analysis
        missing_analysis = self._analyze_missing_values(data)
        metrics['missing_values'] = missing_analysis
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def check_data_consistency(self, data: Dict[str, pd.DataFrame]) -> ValidationResult:
        """Check consistency across multiple data sources"""
        
        errors = []
        warnings = []
        metrics = {}
        
        if len(data) < 2:
            return ValidationResult(True, [], [], {}, datetime.now())
        
        # Check for overlapping symbols
        symbols_by_source = {}
        for source, df in data.items():
            if 'symbol' in df.columns:
                symbols_by_source[source] = set(df['symbol'].unique())
        
        # Find common symbols
        if symbols_by_source:
            common_symbols = set.intersection(*symbols_by_source.values())
            metrics['common_symbols'] = len(common_symbols)
            
            if len(common_symbols) == 0:
                warnings.append("No common symbols found across data sources")
        
        # Check date range consistency
        date_ranges = {}
        for source, df in data.items():
            if 'date' in df.columns:
                date_ranges[source] = (df['date'].min(), df['date'].max())
        
        if len(date_ranges) > 1:
            # Check for overlapping date ranges
            overlap_issues = self._check_date_overlap(date_ranges)
            warnings.extend(overlap_issues)
        
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            timestamp=datetime.now()
        )
    
    def generate_quality_report(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_quality_score': 0.0,
            'recommendations': []
        }
        
        total_score = 0.0
        source_count = 0
        
        for source_name, source_data in data.items():
            validation_result = self.validate_data_quality(source_data, source_name)
            
            report['sources'][source_name] = {
                'passed': validation_result.passed,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'metrics': validation_result.metrics
            }
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(validation_result)
            report['sources'][source_name]['quality_score'] = quality_score
            
            total_score += quality_score
            source_count += 1
        
        if source_count > 0:
            report['overall_quality_score'] = total_score / source_count
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _check_completeness(self, data: pd.DataFrame) -> float:
        """Check data completeness"""
        total_cells = data.size
        non_null_cells = data.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    def _validate_data_types(self, data: pd.DataFrame, resource_name: str) -> List[str]:
        """Validate data types"""
        errors = []
        
        # Define expected types for different resources
        expected_types = {
            'stock_price': {
                'date': 'datetime64[ns]',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'int64'
            }
        }
        
        if resource_name in expected_types:
            for column, expected_type in expected_types[resource_name].items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if actual_type != expected_type:
                        errors.append(f"Column {column}: expected {expected_type}, got {actual_type}")
        
        return errors
    
    def _validate_ranges(self, data: pd.DataFrame, resource_name: str) -> List[str]:
        """Validate data ranges"""
        errors = []
        
        if resource_name == 'stock_price':
            # Check for negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_prices = (data[col] < 0).sum()
                    if negative_prices > 0:
                        errors.append(f"Found {negative_prices} negative values in {col}")
            
            # Check for zero volumes
            if 'volume' in data.columns:
                zero_volumes = (data['volume'] == 0).sum()
                if zero_volumes > 0:
                    errors.append(f"Found {zero_volumes} zero volume records")
        
        return errors
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        return {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist()
        }
    
    def _check_date_overlap(self, date_ranges: Dict[str, Tuple]) -> List[str]:
        """Check for date range overlap issues"""
        warnings = []
        
        ranges = list(date_ranges.values())
        for i, (start1, end1) in enumerate(ranges):
            for j, (start2, end2) in enumerate(ranges[i+1:], i+1):
                if start1 > end2 or start2 > end1:
                    warnings.append(f"No date overlap between sources {i} and {j}")
        
        return warnings
    
    def _calculate_quality_score(self, validation_result: ValidationResult) -> float:
        """Calculate quality score from validation result"""
        base_score = 1.0
        
        # Deduct for errors
        base_score -= len(validation_result.errors) * 0.1
        
        # Deduct for warnings
        base_score -= len(validation_result.warnings) * 0.05
        
        # Consider completeness
        completeness = validation_result.metrics.get('completeness', 1.0)
        base_score *= completeness
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality report"""
        recommendations = []
        
        for source_name, source_report in report['sources'].items():
            quality_score = source_report.get('quality_score', 0.0)
            
            if quality_score < 0.8:
                recommendations.append(f"Improve data quality for {source_name}")
            
            if source_report['warnings']:
                recommendations.append(f"Address warnings in {source_name}")
        
        if report['overall_quality_score'] < 0.9:
            recommendations.append("Overall data quality needs improvement")
        
        return recommendations
```

---

## 📊 **Performance Monitoring**

### **🎯 System Performance Tracking**

The Performance Monitoring system tracks system performance, identifies bottlenecks, and provides optimization insights:

#### **Performance Monitor Interface**
```python
# model/monitoring/performance_monitor.py
from typing import Dict, List, Any, Optional
import time
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class PerformanceMetric:
    """Performance metric record"""
    timestamp: datetime
    component: str
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    success: bool
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """Monitors system performance and bottlenecks"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics = deque(maxlen=max_metrics)
        self.thresholds = {
            'duration_warning': 5.0,  # seconds
            'duration_critical': 30.0,  # seconds
            'memory_warning': 80.0,  # percentage
            'memory_critical': 95.0,  # percentage
            'cpu_warning': 80.0,  # percentage
            'cpu_critical': 95.0  # percentage
        }
        self._monitoring_active = False
        self._monitor_thread = None
    
    def track_execution_time(self, component: str, operation: str, duration: float):
        """Track execution time for operations"""
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            duration=duration,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            success=True,
            metadata={}
        )
        
        self.metrics.append(metric)
        self._check_thresholds(metric)
    
    def monitor_memory_usage(self, component: str, memory_usage: float):
        """Monitor memory usage"""
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=component,
            operation='memory_check',
            duration=0.0,
            memory_usage=memory_usage,
            cpu_usage=self._get_cpu_usage(),
            success=True,
            metadata={'memory_usage_mb': memory_usage}
        )
        
        self.metrics.append(metric)
        self._check_thresholds(metric)
    
    def generate_performance_report(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Generate performance report"""
        
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        recent_metrics = [
            metric for metric in self.metrics 
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'message': 'No performance data available'}
        
        # Calculate statistics
        durations = [m.duration for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics]
        cpu_usage = [m.cpu_usage for m in recent_metrics]
        
        report = {
            'time_window': str(time_window),
            'total_operations': len(recent_metrics),
            'success_rate': sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
            'duration_stats': {
                'mean': np.mean(durations),
                'median': np.median(durations),
                'max': np.max(durations),
                'min': np.min(durations),
                'std': np.std(durations)
            },
            'memory_stats': {
                'mean': np.mean(memory_usage),
                'max': np.max(memory_usage),
                'min': np.min(memory_usage)
            },
            'cpu_stats': {
                'mean': np.mean(cpu_usage),
                'max': np.max(cpu_usage),
                'min': np.min(cpu_usage)
            },
            'component_breakdown': self._get_component_breakdown(recent_metrics),
            'bottlenecks': self.detect_bottlenecks()
        }
        
        return report
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        
        bottlenecks = []
        
        # Group by component and operation
        component_ops = defaultdict(list)
        for metric in self.metrics:
            key = f"{metric.component}:{metric.operation}"
            component_ops[key].append(metric)
        
        # Analyze each component-operation pair
        for key, metrics in component_ops.items():
            if len(metrics) < 5:  # Need minimum data points
                continue
            
            durations = [m.duration for m in metrics]
            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            
            # Check for slow operations
            if avg_duration > self.thresholds['duration_warning']:
                bottlenecks.append({
                    'type': 'slow_operation',
                    'component': key,
                    'avg_duration': avg_duration,
                    'max_duration': max_duration,
                    'threshold': self.thresholds['duration_warning']
                })
            
            # Check for memory issues
            memory_usage = [m.memory_usage for m in metrics]
            avg_memory = np.mean(memory_usage)
            if avg_memory > self.thresholds['memory_warning']:
                bottlenecks.append({
                    'type': 'high_memory_usage',
                    'component': key,
                    'avg_memory_usage': avg_memory,
                    'threshold': self.thresholds['memory_warning']
                })
        
        return bottlenecks
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds"""
        
        if metric.duration > self.thresholds['duration_critical']:
            self._alert_performance_issue('CRITICAL_DURATION', metric)
        elif metric.duration > self.thresholds['duration_warning']:
            self._alert_performance_issue('WARNING_DURATION', metric)
        
        if metric.memory_usage > self.thresholds['memory_critical']:
            self._alert_performance_issue('CRITICAL_MEMORY', metric)
        elif metric.memory_usage > self.thresholds['memory_warning']:
            self._alert_performance_issue('WARNING_MEMORY', metric)
    
    def _alert_performance_issue(self, issue_type: str, metric: PerformanceMetric):
        """Alert on performance issues"""
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'issue_type': issue_type,
            'component': metric.component,
            'operation': metric.operation,
            'metric': metric
        }
        
        # Log alert
        logging.warning(f"Performance alert: {alert}")
        
        # Could send to monitoring system, dashboard, etc.
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent()
        except:
            return 0.0
    
    def _monitor_loop(self):
        """Continuous monitoring loop"""
        
        while self._monitoring_active:
            try:
                # Monitor system resources
                memory_usage = self._get_memory_usage()
                cpu_usage = self._get_cpu_usage()
                
                # Create system metric
                metric = PerformanceMetric(
                    timestamp=datetime.now(),
                    component='system',
                    operation='resource_monitor',
                    duration=0.0,
                    memory_usage=memory_usage,
                    cpu_usage=cpu_usage,
                    success=True,
                    metadata={}
                )
                
                self.metrics.append(metric)
                self._check_thresholds(metric)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _get_component_breakdown(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Get performance breakdown by component"""
        
        component_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'max_duration': 0.0,
            'success_count': 0
        })
        
        for metric in metrics:
            stats = component_stats[metric.component]
            stats['count'] += 1
            stats['total_duration'] += metric.duration
            stats['max_duration'] = max(stats['max_duration'], metric.duration)
            if metric.success:
                stats['success_count'] += 1
        
        # Calculate averages
        for component, stats in component_stats.items():
            if stats['count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['count']
                stats['success_rate'] = stats['success_count'] / stats['count']
        
        return dict(component_stats)
```

---

## 🔐 **Security & Authentication**

### **🎯 Security Framework**

The Security & Authentication system provides API key management, access control, and audit logging:

#### **Authentication Manager**
```python
# model/security/auth_manager.py
from typing import Dict, List, Any, Optional
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

@dataclass
class APIKey:
    """API key information"""
    key_id: str
    key_hash: str
    user_id: str
    permissions: List[str]
    created_date: datetime
    expires_date: Optional[datetime]
    is_active: bool = True

@dataclass
class AccessLog:
    """Access log entry"""
    timestamp: datetime
    user_id: str
    api_key_id: str
    component: str
    operation: str
    success: bool
    ip_address: str
    user_agent: str

class AuthManager:
    """Manages authentication and authorization"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.api_keys: Dict[str, APIKey] = {}
        self.access_logs: List[AccessLog] = []
        self.permission_mappings = {
            'data_fetch': ['read_data'],
            'signal_generate': ['read_data', 'generate_signals'],
            'backtest_run': ['read_data', 'run_backtests'],
            'training_train': ['read_data', 'train_models'],
            'admin': ['*']  # All permissions
        }
    
    def create_api_key(self, user_id: str, permissions: List[str], 
                      expires_in_days: int = 365) -> str:
        """Create a new API key"""
        
        key_id = secrets.token_hex(16)
        api_key = secrets.token_hex(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        expires_date = datetime.now() + timedelta(days=expires_in_days)
        
        api_key_info = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            permissions=permissions,
            created_date=datetime.now(),
            expires_date=expires_date
        )
        
        self.api_keys[key_id] = api_key_info
        
        return api_key
    
    def validate_api_key(self, api_key: str, component: str) -> bool:
        """Validate API key for component access"""
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key_id, key_info in self.api_keys.items():
            if key_info.key_hash == key_hash and key_info.is_active:
                # Check if key is expired
                if key_info.expires_date and datetime.now() > key_info.expires_date:
                    return False
                
                # Check permissions
                return self._check_permissions(key_info.permissions, component)
        
        return False
    
    def check_permissions(self, user: str, operation: str) -> bool:
        """Check user permissions for operations"""
        
        # Find user's API keys
        user_keys = [
            key_info for key_info in self.api_keys.values()
            if key_info.user_id == user and key_info.is_active
        ]
        
        if not user_keys:
            return False
        
        # Check if any key has required permissions
        for key_info in user_keys:
            if self._check_permissions(key_info.permissions, operation):
                return True
        
        return False
    
    def log_access(self, user: str, component: str, operation: str, 
                  success: bool, ip_address: str = "", user_agent: str = ""):
        """Log access attempts"""
        
        access_log = AccessLog(
            timestamp=datetime.now(),
            user_id=user,
            api_key_id="",  # Could be enhanced to track specific key
            component=component,
            operation=operation,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.access_logs.append(access_log)
        
        if not success:
            logging.warning(f"Failed access attempt: {user} -> {component}:{operation}")
    
    def get_access_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get access summary for monitoring"""
        
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        recent_logs = [
            log for log in self.access_logs 
            if log.timestamp >= cutoff_time
        ]
        
        total_attempts = len(recent_logs)
        successful_attempts = sum(1 for log in recent_logs if log.success)
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': successful_attempts / total_attempts if total_attempts > 0 else 0,
            'failed_attempts': total_attempts - successful_attempts,
            'time_window': str(time_window)
        }
    
    def _check_permissions(self, user_permissions: List[str], required_operation: str) -> bool:
        """Check if user permissions allow required operation"""
        
        # Check for admin permission
        if '*' in user_permissions:
            return True
        
        # Check for specific permissions
        if required_operation in user_permissions:
            return True
        
        # Check permission mappings
        if required_operation in self.permission_mappings:
            required_permissions = self.permission_mappings[required_operation]
            return any(perm in user_permissions for perm in required_permissions)
        
        return False
```

---

## 🔄 **Migration Strategy & Backward Compatibility**

### **📊 Conversion Complexity Analysis**

#### **Current System State**
The current system has **moderate complexity** with these key components:

1. **Data Fetching**: `TimeframeAgnosticFetcher` (already abstracted)
2. **Signal Generation**: `TimeframeAgnosticSignalGenerator` (hardcoded logic)
3. **Backtesting**: `TimeframeAgnosticBacktestEngine` (monolithic design)
4. **Pipeline Runner**: Sequential command execution
5. **CLI Interface**: Direct command mapping

#### **Migration Difficulty Assessment**

| Component | Current State | Target State | Difficulty | Risk Level |
|-----------|---------------|--------------|------------|------------|
| **Data Fetching** | ✅ Already abstracted | ✅ Enhanced with resources | 🟢 **Low** | 🟢 **Low** |
| **Signal Generation** | 🔴 Hardcoded | ✅ Modular strategies | 🟡 **Medium** | 🟡 **Medium** |
| **Backtesting** | 🔴 Monolithic | ✅ Modular engines | 🟡 **Medium** | 🟡 **Medium** |
| **Training System** | ❌ Not exists | ✅ New system | 🟢 **Low** | 🟢 **Low** |
| **Pipeline Runner** | 🔴 Sequential | ✅ Component-based | 🟠 **High** | 🟠 **High** |
| **CLI Interface** | 🔴 Direct mapping | ✅ Registry-based | 🟠 **High** | 🟠 **High** |

**Overall Difficulty**: 🟡 **Medium** (6-8 weeks for full migration)

### **🛡️ Backward Compatibility Strategy**

#### **Phase 1: Adapter Layer (Week 1-2)**
Create adapter classes that wrap new components to maintain existing interfaces:

```python
# model/adapters/legacy_adapters.py
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from ..data.universal_fetcher import UniversalDataFetcher
from ..signals.strategies.technical_strategy import TechnicalAnalysisStrategy
from ..backtesting.engines.base_backtest_engine import BaseBacktestEngine

class LegacyDataFetcherAdapter:
    """Adapter to maintain backward compatibility with TimeframeAgnosticFetcher"""
    
    def __init__(self, universal_fetcher: UniversalDataFetcher):
        self.universal_fetcher = universal_fetcher
    
    def fetch_data(self, symbol: str, timeframe: str = '1day', 
                   start_date: str = None, end_date: str = None,
                   data_source: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Maintain existing interface"""
        return self.universal_fetcher.fetch_resource(
            resource_name="stock_price",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            timeframe=timeframe
        )
    
    def get_supported_timeframes(self) -> List[str]:
        """Maintain existing interface"""
        return ['1min', '5min', '15min', '1hour', '1day']
    
    def get_data_sources(self) -> List[str]:
        """Maintain existing interface"""
        return self.universal_fetcher.get_data_sources()

class LegacySignalGeneratorAdapter:
    """Adapter to maintain backward compatibility with TimeframeAgnosticSignalGenerator"""
    
    def __init__(self, component_registry):
        self.registry = component_registry
        self.default_strategy = TechnicalAnalysisStrategy()
    
    def generate_signals(self, data: pd.DataFrame, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Maintain existing interface"""
        # Convert single DataFrame to new format
        data_dict = {'stock_price': data}
        
        # Use default technical analysis strategy
        return self.default_strategy.generate_signals(data_dict, symbols or [])
    
    def get_timeframe_parameters(self, timeframe: str) -> Dict[str, Any]:
        """Maintain existing interface"""
        return self.default_strategy.parameters

class LegacyBacktestEngineAdapter:
    """Adapter to maintain backward compatibility with TimeframeAgnosticBacktestEngine"""
    
    def __init__(self, component_registry):
        self.registry = component_registry
        self.default_engine = BaseBacktestEngine(
            name="legacy_engine",
            description="Legacy backtest engine adapter",
            execution_engine=StandardExecutionEngine({}),
            risk_manager=StandardRiskManager({}),
            parameters={}
        )
    
    def run_backtest(self, signals: List[Dict[str, Any]], 
                    market_data: pd.DataFrame,
                    initial_capital: float, **kwargs) -> Dict[str, Any]:
        """Maintain existing interface"""
        # Convert single DataFrame to new format
        data_dict = {'stock_price': market_data}
        
        return self.default_engine.run_backtest(signals, data_dict, initial_capital, **kwargs)
```

#### **Phase 2: Gradual Component Replacement (Week 3-6)**
Replace components one by one while maintaining adapters:

```python
# model/migration/component_migrator.py
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ComponentMigrator:
    """Handles gradual migration from legacy to new components"""
    
    def __init__(self, component_registry):
        self.registry = component_registry
        self.migration_config = self._load_migration_config()
    
    def get_data_fetcher(self, use_new_system: bool = False):
        """Get appropriate data fetcher based on migration config"""
        if use_new_system and self.migration_config.get('data_fetcher_migrated', False):
            return self.registry.get_component('data_fetcher', 'universal_fetcher')
        else:
            # Return legacy adapter
            universal_fetcher = self.registry.get_component('data_fetcher', 'universal_fetcher')
            return LegacyDataFetcherAdapter(universal_fetcher)
    
    def get_signal_generator(self, use_new_system: bool = False):
        """Get appropriate signal generator based on migration config"""
        if use_new_system and self.migration_config.get('signal_generator_migrated', False):
            return self.registry.get_component('signal_generator', 'technical_analysis')
        else:
            # Return legacy adapter
            return LegacySignalGeneratorAdapter(self.registry)
    
    def get_backtest_engine(self, use_new_system: bool = False):
        """Get appropriate backtest engine based on migration config"""
        if use_new_system and self.migration_config.get('backtest_engine_migrated', False):
            return self.registry.get_component('backtest_engine', 'standard_engine')
        else:
            # Return legacy adapter
            return LegacyBacktestEngineAdapter(self.registry)
    
    def migrate_component(self, component_name: str):
        """Mark a component as migrated"""
        self.migration_config[f'{component_name}_migrated'] = True
        self._save_migration_config()
        logger.info(f"Component {component_name} marked as migrated")
    
    def _load_migration_config(self) -> Dict[str, Any]:
        """Load migration configuration"""
        # Load from file or database
        return {
            'data_fetcher_migrated': False,
            'signal_generator_migrated': False,
            'backtest_engine_migrated': False,
            'training_system_enabled': False
        }
    
    def _save_migration_config(self):
        """Save migration configuration"""
        # Save to file or database
        pass
```

#### **Phase 3: Configuration-Driven Migration (Week 7-8)**
Use configuration to control which system to use:

```python
# config/migration_config.yaml
migration:
  # Component migration status
  components:
    data_fetcher:
      migrated: true
      new_component: "universal_fetcher"
      legacy_adapter: "legacy_data_fetcher"
    
    signal_generator:
      migrated: false
      new_component: "technical_analysis"
      legacy_adapter: "legacy_signal_generator"
    
    backtest_engine:
      migrated: false
      new_component: "standard_engine"
      legacy_adapter: "legacy_backtest_engine"
  
  # Feature flags
  features:
    training_system: false
    component_registry: true
    resource_based_fetching: true
  
  # Rollback settings
  rollback:
    enabled: true
    automatic_rollback_on_error: true
    rollback_threshold: 0.05  # 5% error rate triggers rollback
```

### **🔧 Migration Implementation Plan**

#### **Week 1-2: Foundation & Adapters**
```bash
# 1. Create adapter layer
mkdir -p model/adapters
touch model/adapters/__init__.py
touch model/adapters/legacy_adapters.py

# 2. Create migration utilities
mkdir -p model/migration
touch model/migration/__init__.py
touch model/migration/component_migrator.py

# 3. Add configuration
mkdir -p config
touch config/migration_config.yaml

# 4. Update imports in existing files
# - Add adapter imports
# - Maintain existing function signatures
# - Add migration config loading
```

#### **Week 3-4: Data Fetching Migration**
```bash
# 1. Implement Universal Data Fetcher
mkdir -p model/data
touch model/data/universal_fetcher.py
touch model/data/resources.py
touch model/data/predefined_resources.py

# 2. Create YFinance adapter
mkdir -p model/data/sources
touch model/data/sources/yfinance_source.py

# 3. Test with legacy adapter
python -m pytest tests/test_data_fetching_migration.py

# 4. Enable new system gradually
# - Test with subset of symbols
# - Monitor performance
# - Rollback if issues
```

#### **Week 5-6: Signal Generation & Training Page Migration**
```bash
# 1. Implement signal strategies
mkdir -p model/signals/strategies
touch model/signals/strategies/technical_strategy.py
touch model/signals/strategies/fundamental_strategy.py

# 2. Create signal components
mkdir -p model/signals/components
touch model/signals/components/technical_indicators.py

# 3. Create training page
mkdir -p cli/training_dashboard
touch cli/training_dashboard/__init__.py
touch cli/training_dashboard/training_page.py
touch cli/training_dashboard/training_handlers.py

# 4. Test signal generation
python -m pytest tests/test_signal_generation_migration.py

# 5. Test training page
python -m pytest tests/test_training_page.py

# 6. Compare results with legacy system
python scripts/compare_signal_results.py
```

#### **Week 7-8: Backtesting & Integration**
```bash
# 1. Implement backtesting engines
mkdir -p model/backtesting/engines
touch model/backtesting/engines/base_backtest_engine.py

# 2. Create component registry
mkdir -p model/registry
touch model/registry/component_registry.py

# 3. Update pipeline runner
# - Add component registry integration
# - Maintain existing CLI interface
# - Add migration config support

# 4. Final integration testing
python -m pytest tests/test_full_migration.py
```

### **🛡️ Backward Compatibility Guarantees**

#### **1. API Compatibility**
```python
# Legacy code continues to work
from model.timeframe_agnostic_fetcher import TimeframeAgnosticFetcher
from model.timeframe_agnostic_signals import TimeframeAgnosticSignalGenerator
from model.timeframe_agnostic_backtest import TimeframeAgnosticBacktestEngine

# These classes now use adapters internally
fetcher = TimeframeAgnosticFetcher()  # Uses UniversalDataFetcher via adapter
generator = TimeframeAgnosticSignalGenerator()  # Uses new strategies via adapter
backtest = TimeframeAgnosticBacktestEngine()  # Uses new engines via adapter
```

#### **2. Dashboard & CLI Compatibility**
```bash
# Existing dashboard commands continue to work
# Web Dashboard: http://localhost:8083/commands
# All existing command buttons work unchanged:
# - Data Summary
# - Data Fetch (with timeframe selection)
# - Signal Generation (with timeframe selection)
# - Signal Summary
# - Backtest Run (with timeframe selection)
# - Pipeline Management

# Existing CLI commands continue to work
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT --timeframe 1day
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py signals generate --symbols AAPL,MSFT --timeframe 1day
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py backtest run --symbols AAPL,MSFT --timeframe 1day

# New dashboard features are additive
# - Component Registry page
# - Training Management page
# - Advanced Configuration page
# - Model Management page

# New CLI commands are additive
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py components list
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py training train --model-type random_forest
```

#### **3. Data Format Compatibility**
```python
# Existing data formats are preserved
# DataFrame structure remains the same
# Signal format remains the same
# Backtest results format remains the same
```

#### **4. Configuration Compatibility**
```yaml
# Existing config files continue to work
# New config options are additive
# Migration config controls which system to use
```

### **📊 Migration Testing Strategy**

#### **1. Unit Testing**
```python
# tests/test_migration_compatibility.py
def test_legacy_data_fetching_compatibility():
    """Test that legacy data fetching interface still works"""
    legacy_fetcher = TimeframeAgnosticFetcher()
    data, metadata = legacy_fetcher.fetch_data('AAPL', '1day')
    
    # Verify data format is identical
    assert 'Date' in data.columns
    assert 'Close' in data.columns
    assert len(data) > 0

def test_legacy_signal_generation_compatibility():
    """Test that legacy signal generation interface still works"""
    legacy_generator = TimeframeAgnosticSignalGenerator()
    signals = legacy_generator.generate_signals(sample_data, ['AAPL'])
    
    # Verify signal format is identical
    assert len(signals) > 0
    assert 'symbol' in signals[0]
    assert 'signal_type' in signals[0]
```

#### **2. Integration Testing**
```python
# tests/test_migration_integration.py
def test_full_pipeline_migration():
    """Test that full pipeline works with both old and new systems"""
    # Test with legacy system
    legacy_results = run_legacy_pipeline()
    
    # Test with new system
    new_results = run_new_pipeline()
    
    # Compare results (should be similar)
    assert abs(legacy_results['total_return'] - new_results['total_return']) < 0.01
```

#### **3. Performance Testing**
```python
# tests/test_migration_performance.py
def test_performance_comparison():
    """Test that new system performs at least as well as legacy"""
    legacy_time = measure_legacy_performance()
    new_time = measure_new_performance()
    
    # New system should not be significantly slower
    assert new_time <= legacy_time * 1.2  # Allow 20% overhead
```

### **🚨 Risk Mitigation**

#### **1. Gradual Rollout**
- **Phase 1**: 10% of traffic uses new system
- **Phase 2**: 50% of traffic uses new system
- **Phase 3**: 100% of traffic uses new system
- **Rollback**: Immediate rollback if issues detected

#### **2. Monitoring & Alerting**
```python
# monitoring/migration_monitor.py
class MigrationMonitor:
    def __init__(self):
        self.error_rates = {}
        self.performance_metrics = {}
    
    def check_migration_health(self):
        """Monitor migration health and trigger rollback if needed"""
        for component, metrics in self.performance_metrics.items():
            if metrics['error_rate'] > 0.05:  # 5% error rate
                self.trigger_rollback(component)
    
    def trigger_rollback(self, component):
        """Rollback specific component to legacy system"""
        logger.warning(f"Rolling back {component} to legacy system")
        # Implementation
```

#### **3. Data Validation**
```python
# validation/migration_validator.py
class MigrationValidator:
    def validate_data_consistency(self, legacy_data, new_data):
        """Validate that new system produces consistent data"""
        # Compare data formats
        # Compare data ranges
        # Compare statistical properties
        pass
    
    def validate_signal_consistency(self, legacy_signals, new_signals):
        """Validate that new system produces consistent signals"""
        # Compare signal counts
        # Compare signal types
        # Compare confidence scores
        pass
```

### **📈 Migration Success Metrics**

#### **1. Functional Metrics**
- ✅ All existing tests pass
- ✅ All existing CLI commands work
- ✅ All existing API endpoints work
- ✅ Data formats remain consistent

#### **2. Performance Metrics**
- ✅ No significant performance degradation
- ✅ Memory usage remains reasonable
- ✅ Response times remain acceptable

#### **3. Quality Metrics**
- ✅ Error rates remain low (< 1%)
- ✅ Signal quality remains consistent
- ✅ Backtest results remain comparable

### **🎯 Migration Timeline Summary**

| Week | Phase | Focus | Risk Level | Deliverables |
|------|-------|-------|------------|--------------|
| **1-2** | Foundation | Adapters & Migration Tools | 🟢 Low | Adapter layer, migration config |
| **3-4** | Data Fetching | Universal Data Fetcher | 🟢 Low | Enhanced data fetching |
| **5-6** | Signal Generation & Training | Modular Strategies + Training Page | 🟡 Medium | Flexible signal generation + Training interface |
| **7-8** | Integration | Full System Integration | 🟠 High | Complete migration with training page |

**Total Timeline**: 8 weeks for full migration with backward compatibility

**Risk Level**: 🟡 **Medium** - Well-managed with gradual rollout and rollback capabilities

### **🌐 Dashboard Migration Strategy**

#### **Current Dashboard Architecture**
```
Web Dashboard (http://localhost:8083)
    ↓
HTTP API (http://localhost:8081)
    ↓
Spark Command Server
    ↓
kibana_enhanced_bf.py CLI
    ↓
Current Components (TimeframeAgnosticFetcher, etc.)
```

#### **Target Dashboard Architecture**
```
Web Dashboard (http://localhost:8083)
    ↓
HTTP API (http://localhost:8081)
    ↓
Spark Command Server
    ↓
kibana_enhanced_bf.py CLI (Enhanced)
    ↓
Component Registry
    ↓
New Modular Components
```

#### **Dashboard Migration Approach**

##### **Phase 1: Enhanced CLI (Week 3-4)**
```python
# cli/kibana_enhanced_bf.py (Enhanced)
@cli.group()
def components():
    """Component management commands"""
    pass

@components.command()
def list():
    """List available components"""
    # Use component registry to list available components
    registry = ComponentRegistry()
    components = registry.list_components()
    # Display in dashboard-friendly format

@cli.group()
def training():
    """Model training commands"""
    pass

@training.command()
@click.option('--model-type', default='random_forest', help='Model type')
@click.option('--symbols', help='Comma-separated symbols')
def train(model_type, symbols):
    """Train a new model"""
    # Use training pipeline to train models
    training_pipeline = TrainingPipeline(data_fetcher, training_orchestrator)
    metadata = training_pipeline.train_ml_model(symbols.split(','), model_type=model_type)
    # Return results for dashboard display
```

##### **Phase 2: Dashboard UI Enhancement (Week 5-6)**
```html
<!-- New dashboard pages -->
<!-- Component Registry Page -->
<div class="component-registry">
    <h2>Component Registry</h2>
    <div class="component-list">
        <!-- List available components -->
    </div>
    <div class="component-config">
        <!-- Configure components -->
    </div>
</div>

<!-- Training Management Page (Separate Page) -->
<div class="training-dashboard">
    <h2>🎓 Model Training & Management</h2>
    
    <!-- Model Training Section -->
    <div class="training-section">
        <h3>🎯 Model Training</h3>
        
        <!-- Data Configuration -->
        <div class="config-card">
            <h4>📊 Data Configuration</h4>
            <div class="form-grid">
                <div class="form-group">
                    <label>Symbols:</label>
                    <input type="text" id="training-symbols" placeholder="AAPL,MSFT,GOOGL,TSLA">
                </div>
                <div class="form-group">
                    <label>Timeframe:</label>
                    <select id="training-timeframe">
                        <option value="1day">Daily</option>
                        <option value="1hour">Hourly</option>
                        <option value="15min">15 Minutes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Date Range:</label>
                    <input type="date" id="start-date">
                    <input type="date" id="end-date">
                </div>
                <div class="form-group">
                    <label>Data Sources:</label>
                    <div class="checkbox-group">
                        <input type="checkbox" id="ohlcv-data" checked> OHLCV Data
                        <input type="checkbox" id="fundamental-data"> Fundamental Data
                        <input type="checkbox" id="sentiment-data"> Sentiment Data
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Model Configuration -->
        <div class="config-card">
            <h4>🤖 Model Configuration</h4>
            <div class="form-grid">
                <div class="form-group">
                    <label>Model Type:</label>
                    <select id="model-type">
                        <option value="random_forest">Random Forest</option>
                        <option value="gradient_boosting">Gradient Boosting</option>
                        <option value="neural_network">Neural Network</option>
                        <option value="lstm">LSTM</option>
                        <option value="ensemble">Ensemble</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Training Strategy:</label>
                    <select id="training-strategy">
                        <option value="classification">Classification (Buy/Sell/Hold)</option>
                        <option value="regression">Regression (Price Prediction)</option>
                        <option value="ranking">Ranking (Symbol Ranking)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Feature Engineering:</label>
                    <div class="checkbox-group">
                        <input type="checkbox" id="technical-features" checked> Technical Indicators
                        <input type="checkbox" id="fundamental-features"> Fundamental Ratios
                        <input type="checkbox" id="sentiment-features"> Sentiment Features
                        <input type="checkbox" id="market-features"> Market Features
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Training Controls -->
        <div class="training-controls">
            <button id="start-training" class="btn-primary">🚀 Start Training</button>
            <button id="stop-training" class="btn-secondary" disabled>⏹️ Stop Training</button>
            <button id="save-config" class="btn-outline">💾 Save Configuration</button>
        </div>
        
        <!-- Training Progress -->
        <div class="progress-section" style="display: none;">
            <h4>📈 Training Progress</h4>
            <div class="progress-bar">
                <div class="progress-fill" id="training-progress"></div>
            </div>
            <div class="progress-details">
                <span id="current-epoch">Epoch: 0/100</span>
                <span id="current-loss">Loss: 0.000</span>
                <span id="current-accuracy">Accuracy: 0.00%</span>
            </div>
        </div>
    </div>
    
    <!-- Model Management Section -->
    <div class="management-section">
        <h3>📊 Model Management</h3>
        
        <!-- Model Registry Table -->
        <div class="model-registry">
            <h4>📋 Model Registry</h4>
            <table class="model-table">
                <thead>
                    <tr>
                        <th>Model Name</th>
                        <th>Type</th>
                        <th>Created</th>
                        <th>Performance</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="model-table-body">
                    <!-- Dynamically populated -->
                </tbody>
            </table>
        </div>
        
        <!-- Model Comparison -->
        <div class="model-comparison">
            <h4>📊 Model Comparison</h4>
            <div class="comparison-charts">
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="confusion-matrix"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Model Deployment -->
        <div class="deployment-section">
            <h4>🚀 Model Deployment</h4>
            <div class="deployment-controls">
                <select id="deploy-model">
                    <option value="">Select Model to Deploy</option>
                </select>
                <button id="deploy-btn" class="btn-success">Deploy to Production</button>
                <button id="test-deploy-btn" class="btn-warning">Test Deployment</button>
            </div>
        </div>
    </div>
    
    <!-- Advanced Configuration Section -->
    <div class="advanced-section">
        <h3>🔧 Advanced Configuration</h3>
        
        <!-- Hyperparameter Tuning -->
        <div class="config-card">
            <h4>🎛️ Hyperparameter Tuning</h4>
            <div class="form-grid">
                <div class="form-group">
                    <label>Optimization Method:</label>
                    <select id="optimization-method">
                        <option value="grid_search">Grid Search</option>
                        <option value="random_search">Random Search</option>
                        <option value="bayesian">Bayesian Optimization</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Cross-Validation Folds:</label>
                    <input type="number" id="cv-folds" value="5" min="2" max="10">
                </div>
                <div class="form-group">
                    <label>Max Trials:</label>
                    <input type="number" id="max-trials" value="50" min="10" max="200">
                </div>
            </div>
        </div>
        
        <!-- Feature Engineering -->
        <div class="config-card">
            <h4>🔬 Feature Engineering</h4>
            <div class="feature-config">
                <div class="feature-group">
                    <h5>Technical Indicators</h5>
                    <div class="checkbox-grid">
                        <input type="checkbox" id="rsi" checked> RSI
                        <input type="checkbox" id="macd" checked> MACD
                        <input type="checkbox" id="bollinger" checked> Bollinger Bands
                        <input type="checkbox" id="moving_averages" checked> Moving Averages
                        <input type="checkbox" id="volume_indicators"> Volume Indicators
                        <input type="checkbox" id="momentum_indicators"> Momentum Indicators
                    </div>
                </div>
                <div class="feature-group">
                    <h5>Fundamental Features</h5>
                    <div class="checkbox-grid">
                        <input type="checkbox" id="pe_ratio"> P/E Ratio
                        <input type="checkbox" id="pb_ratio"> P/B Ratio
                        <input type="checkbox" id="revenue_growth"> Revenue Growth
                        <input type="checkbox" id="earnings_growth"> Earnings Growth
                        <input type="checkbox" id="debt_ratio"> Debt Ratio
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Training Analytics Section -->
    <div class="analytics-section">
        <h3>📈 Training Analytics</h3>
        
        <!-- Training History -->
        <div class="history-chart">
            <h4>📊 Training History</h4>
            <canvas id="training-history-chart"></canvas>
        </div>
        
        <!-- Performance Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h5>Overall Performance</h5>
                <div class="metric-value" id="overall-accuracy">0.00%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <h5>Model Drift</h5>
                <div class="metric-value" id="drift-score">0.00</div>
                <div class="metric-label">Drift Score</div>
            </div>
            <div class="metric-card">
                <h5>Training Time</h5>
                <div class="metric-value" id="avg-training-time">0s</div>
                <div class="metric-label">Average Time</div>
            </div>
        </div>
        
        <!-- A/B Testing -->
        <div class="ab-testing">
            <h4>🧪 A/B Testing</h4>
            <div class="ab-controls">
                <button id="start-ab-test" class="btn-primary">Start A/B Test</button>
                <button id="view-ab-results" class="btn-secondary">View Results</button>
            </div>
        </div>
    </div>
</div>

<!-- Advanced Configuration Page -->
<div class="advanced-config">
    <h2>Advanced Configuration</h2>
    <div class="migration-controls">
        <!-- Migration status and controls -->
    </div>
</div>
```

##### **Phase 3: Backward Compatibility (Week 7-8)**
```python
# cli/kibana_enhanced_bf.py (Backward Compatible)
@data.command()
@click.option('--symbols', help='Comma-separated symbols')
@click.option('--timeframe', default='1day', help='Data timeframe')
def fetch(symbols, timeframe):
    """Fetch data (backward compatible)"""
    # Use migration config to determine which system to use
    migrator = ComponentMigrator(component_registry)
    data_fetcher = migrator.get_data_fetcher(use_new_system=False)  # Start with legacy
    
    # Execute with existing interface
    data, metadata = data_fetcher.fetch_data(symbols, timeframe)
    # Return same format as before
```

#### **Dashboard Migration Benefits**

##### **✅ Seamless User Experience**
- **No UI Changes**: Existing dashboard pages work unchanged
- **Progressive Enhancement**: New features added gradually
- **Familiar Interface**: Users don't need to learn new UI

##### **✅ Backward Compatibility**
- **Existing Commands**: All current dashboard buttons work
- **Same Results**: Data formats and outputs remain identical
- **No Breaking Changes**: Users can continue using system as before

##### **✅ Enhanced Capabilities**
- **Component Management**: Visual component registry
- **Separate Training Page**: Dedicated training interface with advanced features
- **Advanced Configuration**: Migration controls and monitoring

##### **✅ Risk Mitigation**
- **Gradual Rollout**: Enable new features one by one
- **A/B Testing**: Test new components with subset of users
- **Easy Rollback**: Disable new features if issues arise

### **🎓 Separate Training Page Strategy**

#### **Why a Separate Training Page?**

##### **1. Different Complexity Level**
- **Current Commands**: Simple, single-purpose operations (fetch data, generate signals, run backtest)
- **Training System**: Complex, multi-step workflows with multiple components and configurations

##### **2. Different User Workflow**
- **Current Commands**: Execute → Get Results → Done
- **Training System**: Configure → Train → Evaluate → Deploy → Monitor → Iterate

##### **3. Different UI Requirements**
- **Current Commands**: Simple forms with basic parameters
- **Training System**: Complex forms, progress tracking, model comparison, performance charts, configuration management

##### **4. Different Frequency of Use**
- **Current Commands**: Used frequently for daily operations
- **Training System**: Used occasionally for model development and optimization

#### **Training Page Architecture**

##### **New Page: http://localhost:8083/training**

```
Training Dashboard
├── 🎯 Model Training
│   ├── Training Configuration
│   ├── Data Selection
│   ├── Model Selection
│   ├── Training Progress
│   └── Results & Evaluation
├── 📊 Model Management
│   ├── Model Registry
│   ├── Performance Comparison
│   ├── Model Deployment
│   └── Model Retirement
├── 🔧 Advanced Configuration
│   ├── Feature Engineering
│   ├── Hyperparameter Tuning
│   ├── Cross-Validation
│   └── Model Ensembles
└── 📈 Training Analytics
    ├── Training History
    ├── Performance Trends
    ├── Model Drift Detection
    └── A/B Testing Results
```

#### **Training Page Benefits**

##### **✅ Better User Experience**
- **Focused Interface**: Dedicated space for complex training workflows
- **Progressive Disclosure**: Show advanced options only when needed
- **Better Organization**: Logical grouping of related features

##### **✅ Scalability**
- **Room for Growth**: Can add more training features without cluttering main page
- **Component Reuse**: Training components can be used across different workflows
- **Modular Design**: Each section can be developed independently

##### **✅ Performance**
- **Lazy Loading**: Training page loads only when needed
- **Reduced Complexity**: Main command page stays simple and fast
- **Better Caching**: Training-specific resources cached separately

##### **✅ Development Benefits**
- **Clear Separation**: Different teams can work on different pages
- **Easier Testing**: Training features can be tested independently
- **Better Maintenance**: Changes to training don't affect main commands

#### **Integration with Existing System**

##### **Navigation**
```html
<!-- Add to existing navigation -->
<button class="nav-btn" onclick="window.location.href='/training'">🎓 Training</button>
```

##### **Cross-Page Integration**
- **From Training Page**: Deploy models to be used in signal generation
- **From Commands Page**: Quick access to training page for model management
- **From Pipeline Page**: Use trained models in automated pipelines

##### **Data Flow**
```
Training Page → Train Model → Register in Component Registry → Use in Commands Page
```

#### **Training Page Implementation Plan**

##### **Phase 1: Basic Training Page (Week 3-4)**
- Create new training page route at `/training`
- Basic model training interface
- Simple model management
- Integration with existing navigation

##### **Phase 2: Advanced Features (Week 5-6)**
- Hyperparameter tuning interface
- Feature engineering configuration
- Model comparison and visualization
- Training progress tracking

##### **Phase 3: Analytics & Integration (Week 7-8)**
- Training analytics dashboard
- A/B testing interface
- Model drift detection
- Full integration with component registry

#### **Training Page Features**

##### **🎯 Model Training Section**
- **Data Configuration**: Symbol selection, timeframe, date range, data sources
- **Model Configuration**: Model type, training strategy, feature engineering
- **Training Controls**: Start/stop training, save configurations
- **Progress Tracking**: Real-time training progress with metrics

##### **📊 Model Management Section**
- **Model Registry**: Table of all trained models with performance metrics
- **Model Comparison**: Side-by-side comparison with charts
- **Model Deployment**: Deploy models to production or testing
- **Model Retirement**: Archive or delete old models

##### **🔧 Advanced Configuration Section**
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Feature Engineering**: Technical indicators, fundamental features, sentiment features
- **Cross-Validation**: K-fold cross-validation configuration
- **Model Ensembles**: Combine multiple models for better performance

##### **📈 Training Analytics Section**
- **Training History**: Chart showing training performance over time
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Model Drift Detection**: Monitor model performance degradation
- **A/B Testing**: Compare different models or configurations

#### **User Experience Flow**

##### **1. Training a Model**
1. Navigate to http://localhost:8083/training
2. Configure data sources and symbols
3. Select model type and training strategy
4. Configure feature engineering options
5. Start training and monitor progress
6. Review results and performance metrics

##### **2. Managing Models**
1. View all trained models in the registry
2. Compare model performance
3. Deploy best-performing models
4. Monitor deployed models for drift
5. Retire underperforming models

##### **3. Advanced Configuration**
1. Tune hyperparameters for better performance
2. Engineer new features
3. Set up cross-validation
4. Create model ensembles
5. Run A/B tests

#### **Backend Integration**

##### **Enhanced CLI Commands**
```python
# cli/kibana_enhanced_bf.py (Enhanced)
@cli.group()
def training():
    """Model training commands with dual logging."""
    pass

@training.command()
@click.option('--model-type', default='random_forest', help='Model type')
@click.option('--symbols', help='Comma-separated symbols')
@click.option('--timeframe', default='1day', help='Data timeframe')
@click.option('--strategy', default='classification', help='Training strategy')
def train(model_type, symbols, timeframe, strategy):
    """Train a new model with dual logging."""
    # Implementation with progress tracking

@training.command()
def list_models():
    """List all trained models."""
    # Return model registry

@training.command()
@click.option('--model-name', help='Model name to deploy')
def deploy(model_name):
    """Deploy a trained model."""
    # Deploy model to component registry
```

##### **HTTP API Integration**
```python
# cli/spark_command_server.py (Enhanced)
def execute_command(self, command, parameters):
    commands = {
        # Existing commands...
        
        # New training commands
        'training_train': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'training', 'train',
                          '--model-type', parameters.get("model_type", "random_forest"),
                          '--symbols', parameters.get("symbols", ""),
                          '--timeframe', parameters.get("timeframe", "1day"),
                          '--strategy', parameters.get("strategy", "classification")],
        'training_list': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'training', 'list'],
        'training_deploy': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'training', 'deploy',
                           '--model-name', parameters.get("model_name", "")],
        'training_compare': ['python3', '/opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py', 'training', 'compare',
                            '--models', parameters.get("models", "")],
    }
```

---

*This document will be expanded section by section with detailed implementation plans.*
