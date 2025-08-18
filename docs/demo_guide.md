# Breadth/Thrust Signals POC - Demo Guide

## 🎯 Overview

This guide demonstrates the complete Breadth/Thrust Signals system, a quantitative trading platform that uses market breadth indicators to generate trading signals. The system combines multiple technical indicators to create composite signals for market timing and trend analysis.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Poetry (Python package manager)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd BreadthFlow

# Install dependencies
poetry install

# Setup environment
cp env.example .env
# Edit .env with your configuration
```

### 3. Start Infrastructure

```bash
# Start all services (Spark, Kafka, MinIO, Elasticsearch, Kibana)
poetry run bf infra start

# Check health
poetry run bf infra health
```

## 🎬 Running the Demo

### Option 1: Full Demo (Recommended)

```bash
# Run complete end-to-end demo
poetry run bf demo

# This will:
# 1. Check infrastructure health
# 2. Fetch sample market data
# 3. Generate breadth/thrust signals
# 4. Run backtesting analysis
# 5. Display comprehensive results
```

### Option 2: Quick Demo

```bash
# Run quick demo with limited data
poetry run bf demo --quick

# This uses:
# - 3 symbols (AAPL, MSFT, GOOGL)
# - 6 months of data
# - $50,000 initial capital
```

### Option 3: Skip Infrastructure Checks

```bash
# If infrastructure is already running
poetry run bf demo --skip-infra
```

## 📊 Demo Walkthrough

### Step 1: Data Summary
The demo starts by checking what data is available in the system:
```bash
poetry run bf data summary
```

### Step 2: Fetch Market Data
Downloads historical OHLCV data for selected symbols:
```bash
poetry run bf data fetch \
  --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Step 3: Generate Signals
Computes breadth indicators and generates trading signals:
```bash
poetry run bf signals generate \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols AAPL,MSFT,GOOGL,AMZN,TSLA
```

### Step 4: Signal Summary
Shows statistics about generated signals:
```bash
poetry run bf signals summary
```

### Step 5: Run Backtest
Simulates trading based on signals with realistic constraints:
```bash
poetry run bf backtest run \
  --from-date 2024-01-01 \
  --to-date 2024-12-31 \
  --symbols AAPL,MSFT,GOOGL,AMZN,TSLA \
  --initial-capital 100000 \
  --save-results
```

### Step 6: Analyze Results
Provides comprehensive performance analysis:
```bash
poetry run bf backtest analyze
```

## 🔧 Individual Commands

### Data Management

```bash
# Fetch data for specific symbols
poetry run bf data fetch --symbols AAPL,MSFT,GOOGL

# Replay data to Kafka for real-time simulation
poetry run bf data replay --speed 60 --symbols AAPL,MSFT

# Get data summary
poetry run bf data summary
```

### Signal Generation

```bash
# Generate signals for a specific date
poetry run bf signals generate --date 2024-06-01

# Generate signals for a period
poetry run bf signals generate \
  --start-date 2024-01-01 \
  --end-date 2024-06-30

# Force recalculation of all features
poetry run bf signals generate --force-recalculate

# Get signal summary
poetry run bf signals summary
```

### Backtesting

```bash
# Run backtest with custom parameters
poetry run bf backtest run \
  --from-date 2024-01-01 \
  --to-date 2024-12-31 \
  --initial-capital 50000 \
  --position-size 0.05 \
  --max-positions 5 \
  --commission-rate 0.002 \
  --save-results

# Analyze specific backtest results
poetry run bf backtest analyze \
  --results-path backtests/out/results_2024-01-01_2024-12-31

# Analyze with date filters
poetry run bf backtest analyze \
  --start-date 2024-06-01 \
  --end-date 2024-12-31
```

### Infrastructure Management

```bash
# Start infrastructure
poetry run bf infra start

# Check health
poetry run bf infra health

# View logs
poetry run bf infra logs --follow

# Stop infrastructure
poetry run bf infra stop

# Restart infrastructure
poetry run bf infra restart
```

## 🌐 Web Interfaces

After starting the infrastructure, access these web interfaces:

- **Spark UI**: http://localhost:8080
  - Monitor Spark jobs and performance
  - View application logs and metrics

- **MinIO Console**: http://localhost:9001
  - Username: `minioadmin`
  - Password: `minioadmin`
  - Browse Delta Lake data and files

- **Kibana**: http://localhost:5601
  - Create dashboards for signal monitoring
  - Search and analyze trading signals

- **Elasticsearch**: http://localhost:9200
  - REST API for signal data
  - Health monitoring and cluster status

## 📈 Understanding the Results

### Signal Types

The system generates several types of signals:

1. **Buy Signals**: Bullish signals with strong confidence
2. **Sell Signals**: Bearish signals with strong confidence
3. **Strong Buy/Sell**: High-confidence extreme signals
4. **Hold Signals**: Neutral or low-confidence periods

### Performance Metrics

Key metrics to understand:

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Largest peak-to-trough decline
- **Hit Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses

### Breadth Indicators

The system uses these market breadth indicators:

1. **Advance/Decline (A/D)**: Ratio of advancing to declining stocks
2. **Moving Averages**: MA20/MA50 crossovers and momentum
3. **McClellan Oscillator**: Smoothed A/D momentum indicator
4. **Zweig Breadth Thrust (ZBT)**: Rare but powerful momentum signal

## 🔍 Troubleshooting

### Common Issues

1. **Infrastructure not starting**
   ```bash
   # Check Docker is running
   docker --version
   
   # Check ports are available
   lsof -i :8080,9000,9200,5601
   
   # Restart infrastructure
   poetry run bf infra restart
   ```

2. **Data fetch failures**
   ```bash
   # Check internet connection
   curl -I https://finance.yahoo.com
   
   # Try with fewer symbols
   poetry run bf data fetch --symbols AAPL,MSFT
   
   # Check Delta Lake storage
   poetry run bf data summary
   ```

3. **Signal generation errors**
   ```bash
   # Ensure data exists
   poetry run bf data summary
   
   # Regenerate with force flag
   poetry run bf signals generate --force-recalculate
   
   # Check logs
   poetry run bf infra logs
   ```

4. **Backtest failures**
   ```bash
   # Ensure signals exist
   poetry run bf signals summary
   
   # Try with smaller date range
   poetry run bf backtest run \
     --from-date 2024-06-01 \
     --to-date 2024-12-31
   ```

### Performance Optimization

1. **For large datasets**
   - Use `--quick` flag for initial testing
   - Reduce symbol count for faster processing
   - Use shorter date ranges

2. **Memory issues**
   - Increase Spark memory settings in `infra/docker-compose.yml`
   - Reduce parallel workers in data fetch
   - Use smaller position sizes

3. **Storage optimization**
   - Clean up old data: `poetry run bf dev clean`
   - Optimize Delta tables periodically
   - Monitor MinIO storage usage

## 🎯 Advanced Usage

### Custom Configuration

Edit `.env` file to customize:
- Indicator parameters
- Signal thresholds
- Backtest settings
- Infrastructure configuration

### Extending the System

1. **Add new indicators**
   - Create new feature calculator in `features/`
   - Update signal scoring weights
   - Add to CLI commands

2. **Custom backtest strategies**
   - Modify `BacktestConfig` parameters
   - Implement custom position sizing
   - Add new risk management rules

3. **Real-time deployment**
   - Use data replay for simulation
   - Implement real-time signal generation
   - Add alerting and notifications

## 📚 Next Steps

1. **Explore the codebase**
   - Review feature implementations
   - Understand signal generation logic
   - Study backtesting framework

2. **Experiment with parameters**
   - Try different indicator weights
   - Test various backtest configurations
   - Analyze different time periods

3. **Build custom dashboards**
   - Use Kibana for signal monitoring
   - Create performance visualizations
   - Set up alerting systems

4. **Production deployment**
   - Scale infrastructure for larger datasets
   - Implement real-time data feeds
   - Add monitoring and alerting

## 🤝 Support

For issues and questions:
- Check the troubleshooting section above
- Review logs: `poetry run bf infra logs`
- Check infrastructure health: `poetry run bf infra health`
- Consult the technical documentation

---

**Happy Trading! 📈**
