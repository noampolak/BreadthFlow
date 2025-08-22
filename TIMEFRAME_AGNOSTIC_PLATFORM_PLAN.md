# Timeframe-Agnostic Platform Transformation Plan

## 🎯 Goal
Transform BreadthFlow from daily-only to support multiple timeframes (daily, hourly, minute) while maintaining backward compatibility.

## 📊 Current State Analysis

### Current Limitations:
1. **Fixed Daily Data**: `yfinance` fetches daily OHLCV only
2. **Hardcoded Intervals**: Pipeline assumes daily processing
3. **Single Timeframe**: All analysis assumes daily bars
4. **Date-Based Logic**: Signal generation tied to daily dates
5. **Storage Structure**: MinIO folders assume daily granularity

## 🏗️ Phase 1: Data Layer Transformation

### 1.1 Multi-Timeframe Data Sources
**⚠️ RULE: Add new sources, don't override existing**

**New Data Sources to Add:**
```python
# New data fetchers (ADD to existing):
- yfinance_intraday.py     # Intraday data (1min, 5min, 15min, 1hour)
- alpha_vantage_intraday.py # Alternative intraday source
- polygon_intraday.py      # High-quality intraday data
- custom_data_adapter.py   # Framework for new data sources
```

**Data Source Interface:**
```python
class TimeframeAgnosticDataSource:
    def fetch_data(self, symbol, timeframe, start_date, end_date):
        # timeframe: '1min', '5min', '15min', '1hour', '1day'
        pass
```

### 1.2 Storage Structure Enhancement
**⚠️ RULE: Add new structure, maintain existing**

**New MinIO Structure:**
```
breadthflow/
├── ohlcv/
│   ├── daily/           # Existing structure (unchanged)
│   │   ├── AAPL/
│   │   └── MSFT/
│   ├── hourly/          # NEW
│   │   ├── AAPL/
│   │   └── MSFT/
│   ├── minute/          # NEW
│   │   ├── AAPL/
│   │   └── MSFT/
│   └── metadata/        # NEW - timeframe configuration
├── trading_signals/
│   ├── daily/           # Existing (unchanged)
│   ├── hourly/          # NEW
│   └── minute/          # NEW
└── analytics/
    ├── daily/           # Existing (unchanged)
    ├── hourly/          # NEW
    └── minute/          # NEW
```

## 🏗️ Phase 2: Core Logic Transformation

### 2.1 Timeframe-Agnostic Signal Generator
**⚠️ RULE: Add new generator, maintain existing**

**New Signal Generator:**
```python
class TimeframeAgnosticSignalGenerator:
    def __init__(self, timeframe='1day'):
        self.timeframe = timeframe
        self.parameters = self.get_timeframe_parameters(timeframe)
    
    def get_timeframe_parameters(self, timeframe):
        # Different parameters for different timeframes
        return {
            '1day': {'ma_short': 20, 'ma_long': 50, 'rsi_period': 14},
            '1hour': {'ma_short': 12, 'ma_long': 24, 'rsi_period': 14},
            '15min': {'ma_short': 8, 'ma_long': 16, 'rsi_period': 14},
            '1min': {'ma_short': 5, 'ma_long': 10, 'rsi_period': 14}
        }
    
    def generate_signals(self, data, symbols):
        # Timeframe-aware signal generation
        pass
```

### 2.2 Timeframe-Agnostic Backtesting
**⚠️ RULE: Add new engine, maintain existing**

**New Backtesting Engine:**
```python
class TimeframeAgnosticBacktestEngine:
    def __init__(self, timeframe='1day'):
        self.timeframe = timeframe
        self.trading_hours = self.get_trading_hours(timeframe)
    
    def get_trading_hours(self, timeframe):
        # Different trading hours for different timeframes
        return {
            '1day': {'start': '09:30', 'end': '16:00', 'timezone': 'US/Eastern'},
            '1hour': {'start': '09:30', 'end': '16:00', 'timezone': 'US/Eastern'},
            '15min': {'start': '09:30', 'end': '16:00', 'timezone': 'US/Eastern'},
            '1min': {'start': '09:30', 'end': '16:00', 'timezone': 'US/Eastern'}
        }
```

## 🏗️ Phase 3: CLI and API Transformation

### 3.1 Enhanced CLI Commands
**⚠️ RULE: Add timeframe options, don't override existing**

**New CLI Options:**
```python
@data.command()
@click.option('--timeframe', default='1day', 
              help='Timeframe: 1min, 5min, 15min, 1hour, 1day')
@click.option('--data-source', default='yfinance',
              help='Data source: yfinance, alpha_vantage, polygon')
def fetch(timeframe, data_source, symbols, start_date, end_date):
    """Fetch data with specified timeframe."""
```

### 3.2 Enhanced Pipeline Commands
**⚠️ RULE: Add timeframe-aware pipeline, maintain existing**

**New Pipeline Options:**
```python
@pipeline.command()
@click.option('--timeframe', default='1day',
              help='Pipeline timeframe')
@click.option('--interval', default=86400,  # 1 day in seconds
              help='Interval between runs (seconds)')
def start(timeframe, interval, mode, symbols):
    """Start timeframe-aware pipeline."""
```

## 🏗️ Phase 4: Dashboard Transformation

### 4.1 Timeframe Selection UI
**⚠️ RULE: Add new UI elements, don't override existing**

**New Dashboard Features:**
- **Timeframe Selector**: Dropdown for 1min, 5min, 15min, 1hour, 1day
- **Data Source Selector**: Choose data provider
- **Timeframe-Specific Metrics**: Different KPIs for different timeframes
- **Multi-Timeframe Comparison**: Compare signals across timeframes

### 4.2 Enhanced Monitoring
**⚠️ RULE: Add new monitoring, maintain existing**

**New Monitoring Features:**
- **Timeframe-Specific Alerts**: Different alert rules per timeframe
- **Performance Comparison**: Compare daily vs intraday performance
- **Resource Usage Tracking**: Monitor data volume per timeframe

## 🏗️ Phase 5: Configuration and Metadata

### 5.1 Timeframe Configuration
**⚠️ RULE: Add new config, maintain existing**

**New Configuration File: `config/timeframes.yaml`**
```yaml
timeframes:
  1day:
    data_source: yfinance
    interval_seconds: 86400
    trading_hours: "09:30-16:00"
    parameters:
      ma_short: 20
      ma_long: 50
      rsi_period: 14
  
  1hour:
    data_source: yfinance_intraday
    interval_seconds: 3600
    trading_hours: "09:30-16:00"
    parameters:
      ma_short: 12
      ma_long: 24
      rsi_period: 14
  
  15min:
    data_source: yfinance_intraday
    interval_seconds: 900
    trading_hours: "09:30-16:00"
    parameters:
      ma_short: 8
      ma_long: 16
      rsi_period: 14
```

### 5.2 Database Schema Enhancement
**⚠️ RULE: Add new tables, maintain existing**

**New Database Tables:**
```sql
-- Timeframe configuration
CREATE TABLE timeframe_configs (
    timeframe VARCHAR(20) PRIMARY KEY,
    data_source VARCHAR(50) NOT NULL,
    interval_seconds INTEGER NOT NULL,
    trading_hours VARCHAR(20),
    parameters JSONB
);

-- Enhanced pipeline runs with timeframe
ALTER TABLE pipeline_runs ADD COLUMN timeframe VARCHAR(20) DEFAULT '1day';

-- Timeframe-specific signals
CREATE TABLE signals_timeframe (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    signal_type VARCHAR(10) NOT NULL,
    confidence REAL,
    create_time TIMESTAMP DEFAULT NOW()
);
```

## 📋 Implementation Strategy

### Backward Compatibility Strategy:
1. **Default to Daily**: All existing commands default to '1day' timeframe
2. **Gradual Migration**: Add timeframe options without breaking existing functionality
3. **Data Migration**: Keep existing daily data structure unchanged
4. **API Compatibility**: Maintain existing API endpoints with default behavior

### Testing Strategy:
1. **Unit Tests**: Test each timeframe independently
2. **Integration Tests**: Test multi-timeframe workflows
3. **Performance Tests**: Test data volume and processing speed
4. **Backward Compatibility Tests**: Ensure existing functionality works

## 📋 Risk Assessment

### High Risk Areas:
1. **Data Volume**: Intraday data is much larger than daily
2. **Processing Time**: More data = longer processing
3. **Storage Costs**: Significantly more storage required
4. **API Limits**: Data providers have rate limits
5. **Complexity**: More complex logic and configuration

### Mitigation Strategies:
1. **Incremental Rollout**: Start with hourly, then add minute data
2. **Resource Monitoring**: Track storage and processing usage
3. **Rate Limiting**: Implement proper API rate limiting
4. **Data Retention**: Implement data retention policies
5. **Fallback Mechanisms**: Fall back to daily if intraday fails

## 📋 Success Criteria

1. **Backward Compatibility**: All existing daily functionality works unchanged
2. **Multi-Timeframe Support**: Successfully process 1min, 5min, 15min, 1hour, 1day
3. **Performance**: Intraday processing completes within reasonable time
4. **Data Quality**: Intraday signals are as reliable as daily signals
5. **User Experience**: Dashboard clearly shows timeframe options and results

## 📋 File Structure Changes

### Files to ADD (New):
```
model/timeframe_agnostic_fetcher.py    # Multi-timeframe data fetching
model/timeframe_agnostic_signals.py    # Timeframe-aware signal generation
model/timeframe_agnostic_backtest.py   # Timeframe-aware backtesting
config/timeframes.yaml                 # Timeframe configuration
```

### Files to MODIFY (Add Only):
```
cli/kibana_enhanced_bf.py              # ADD timeframe options to commands
cli/spark_command_server.py            # ADD timeframe endpoints
cli/postgres_dashboard.py              # ADD timeframe UI elements
```

## 📋 Implementation Rules

### ⚠️ CRITICAL RULES:
1. **NO OVERRIDING**: Never override existing code without explicit approval
2. **ADDITION ONLY**: Only add new functions, classes, and endpoints
3. **BACKWARD COMPATIBILITY**: All existing functionality must continue to work
4. **GRADUAL MIGRATION**: Move functionality gradually, not all at once
5. **TESTING REQUIRED**: Each addition must be tested before proceeding

### Approval Process:
1. **Plan Review**: You review and approve this plan
2. **Phase-by-Phase**: Implement one phase at a time
3. **Testing**: Test each phase before moving to next
4. **Approval**: Get your approval before each phase

## Questions for Your Approval:

1. **Priority Timeframes**: Which timeframes should we implement first? (1hour, 15min, 1min?)
2. **Data Sources**: Which data sources should we prioritize? (yfinance intraday, Alpha Vantage, Polygon?)
3. **Storage Strategy**: How should we handle the increased data volume?
4. **Rollout Strategy**: Should we implement all timeframes at once or incrementally?
5. **Resource Limits**: What are the acceptable processing times and storage limits?

---

**Note**: This plan follows the established pattern of adding new functionality without overriding existing code, ensuring backward compatibility throughout the transformation process.
