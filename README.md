# 🚀 BreadthFlow - Advanced Financial Pipeline

> A production-ready quantitative trading signal system with **modular abstraction architecture**, **workflow management**, and **timeframe-agnostic capabilities**. Built on PySpark, Kafka, PostgreSQL, MinIO, and Elasticsearch with a modern web dashboard, streaming capabilities, and multi-timeframe analytics (1min, 5min, 15min, 1hour, 1day).

## 🚫 **CRITICAL RULE: NO MOCK DATA**

**NEVER use mock data, fake data, or placeholder data in production code or API endpoints.**

- **No fake financial data** (fake stock prices, fake trading signals, fake metrics)
- **No fake timestamps** (fake dates, fake creation times)
- **No fake IDs** (fake model IDs, fake run IDs)
- **Return proper error responses** (501 Not Implemented) instead of fake data
- **Use TODO comments** to indicate functionality needs implementation

---

## 🆕 **NEW: Modular Abstraction System**

BreadthFlow now features a **complete modular abstraction system** that allows you to:
- **Interchangeable Components** - Swap data sources, signal strategies, and backtesting engines
- **Workflow Management** - Complex multi-step process orchestration
- **Real-time Monitoring** - System health and performance tracking
- **Enhanced Data Fetching** - Multiple data sources and resource types
- **Advanced Signal Generation** - Technical, fundamental, and sentiment analysis
- **Comprehensive Backtesting** - Multiple engines with risk management
- **Dashboard Integration** - Seamless connection to new abstraction system

## 🎯 **What This System Does**

BreadthFlow analyzes market breadth signals across 100+ stocks to generate trading signals using advanced technical indicators. The system fetches real-time financial data across **multiple timeframes** (1min, 5min, 15min, 1hour, 1day), processes it through distributed computing, and provides comprehensive monitoring and analytics with timeframe-specific optimizations.

### **🆕 Enhanced Capabilities with Modular Abstraction:**

#### **Data Fetching**
- **Multiple Data Sources**: YFinance, Alpha Vantage, Polygon, custom sources
- **Multiple Resource Types**: Stock prices, fundamentals (revenue, market cap), sentiment data
- **Quality Validation**: Data completeness and accuracy checks
- **Rate Limiting**: Respectful API usage with retry logic

#### **Signal Generation**
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX, ATR
- **Fundamental Analysis**: P/E, P/B, ROE, Revenue Growth, Debt-to-Equity ratios
- **Sentiment Analysis**: News sentiment, social media sentiment, analyst ratings
- **Multi-Strategy**: Composite signals with consensus filtering

#### **Backtesting**
- **Multiple Engines**: Standard backtesting, High-Frequency Trading (HFT) simulation
- **Risk Management**: Position limits, VaR (Value at Risk), stress testing
- **Performance Analysis**: Sharpe ratio, Sortino ratio, maximum drawdown, win rate
- **Execution Simulation**: Realistic trade execution with slippage and commissions

#### **Workflow Management**
- **Complex Orchestration**: Multi-step process management with dependencies
- **Parallel Execution**: Independent workflow steps run concurrently
- **Error Recovery**: Automatic retry and fallback mechanisms
- **Real-time Monitoring**: Live system health and performance tracking

---

## ⚡ **Quick Start (5 Minutes)**

### **Prerequisites**
- **Docker & Docker Compose** (required)
- **Python 3.9+** (optional for local development)
- **8GB RAM minimum** (16GB recommended)

### **1. Clone & Start**
```bash
git clone <repository-url>
cd BreadthFlow

# Complete startup with all timeframe features
./scripts/start_infrastructure.sh
```

### **2. Verify Infrastructure**
```bash
# Check all services are running
./scripts/check_status.sh

# Should show: spark-master, spark-worker-1, spark-worker-2, postgres, kafka, kafdrop, minio, elasticsearch, kibana, dashboard
```

### **3. Run Your First Pipeline**
```bash
# Option 1: Web Dashboard (Recommended)
# Go to http://localhost:8083 → Click "Commands" → Select "Demo Flow" → Execute commands

# Option 2: Scripts
# Run complete demo (recommended for first time)
./scripts/run_demo.sh

# Option 3: CLI Commands (Legacy System)
# Get data summary
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

# Fetch real market data (multiple timeframes supported)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT --start-date 2024-08-15 --end-date 2024-08-16 --timeframe 1day --data-source yfinance

# Fetch intraday data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --timeframe 1hour --data-source yfinance

# 🆕 Option 4: New Abstraction System CLI
# Install dependencies first
pip install pandas numpy scikit-learn yfinance

# Use new CLI with workflow manager
python cli/bf_abstracted.py demo

# Test individual commands
python cli/bf_abstracted.py data fetch --symbols AAPL,MSFT --timeframe 1day
python cli/bf_abstracted.py signals generate --symbols AAPL,MSFT --timeframe 1day
python cli/bf_abstracted.py backtest run --symbols AAPL,MSFT --timeframe 1day
python cli/bf_abstracted.py pipeline start --mode demo
```

### **4. Access Monitoring & UIs**
- **🎯 Real-time Dashboard**: http://localhost:8083 (Pipeline monitoring, Infrastructure overview & **Commands execution**)
- **🎮 Pipeline Management**: http://localhost:8083/pipeline (Automated batch processing control)
- **📊 Trading Signals**: http://localhost:8083/trading (Real-time signal monitoring with export)
- **🏗️ Infrastructure Status**: http://localhost:8083/infrastructure (System health monitoring)
- **📊 Kibana Analytics**: http://localhost:5601 (Advanced log analysis)
- **🎨 Kafka UI (Kafdrop)**: http://localhost:9002 (Streaming data & message monitoring)
- **🗄️ MinIO Data Storage**: http://localhost:9001 (minioadmin/minioadmin)
- **⚡ Spark Cluster**: http://localhost:8080 (Processing status)
- **🔧 Spark Command API**: http://localhost:8081 (HTTP API for command execution)

### **5. Execute Commands via Web Interface**
- **🚀 Quick Flows**: Demo, Small, Medium, Full pipeline configurations
- **📊 Data Commands**: Data summary, market data fetching (with timeframe selection)
- **🎯 Signal Commands**: Signal generation, signal summary (timeframe-aware)
- **🔄 Backtesting**: Run backtesting simulations (timeframe-optimized)
- **🎮 Pipeline Management**: Automated batch processing with continuous execution
- **🎨 Kafka Commands**: Kafka demo, real integration testing
- **⚡ HTTP API**: Clean communication between dashboard and Spark container
- **⏰ Timeframe Support**: 1min, 5min, 15min, 1hour, 1day with optimized parameters

### **🆕 6. New Abstraction System Features**
- **🔄 Workflow Management**: Complex multi-step process orchestration
- **📊 System Monitoring**: Real-time health checks and performance metrics
- **🎯 Enhanced Data Fetching**: Multiple sources and resource types
- **⚡ Advanced Signal Generation**: Technical, fundamental, and sentiment analysis
- **📈 Comprehensive Backtesting**: Multiple engines with risk management
- **🔧 Dashboard Integration**: Seamless connection to new abstraction system

### **🧪 7. Testing the New System**

#### **Local Testing**
```bash
# Test the new abstraction system
cd cli
python test_dashboard_integration_minimal.py

# Test the new CLI
python bf_abstracted.py demo

# Test dashboard connector
python dashboard_connector.py

# Test individual commands
python bf_abstracted.py data fetch --symbols AAPL,MSFT --timeframe 1day
python bf_abstracted.py signals generate --symbols AAPL,MSFT --timeframe 1day
python bf_abstracted.py backtest run --symbols AAPL,MSFT --timeframe 1day
python bf_abstracted.py pipeline start --mode demo
```

#### **🐳 Docker Testing (Recommended)**
```bash
# Test the new abstraction system in Docker
cd cli
python test_docker_integration.py

# Test individual commands in Docker container
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py health
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py demo
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py data fetch --symbols AAPL,MSFT --timeframe 1day

# Test via HTTP API
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "health", "parameters": {}}' \
  http://localhost:8081/execute
```

---

## 🏗️ **Infrastructure Overview**

### **🐳 Docker Services (10 Containers)**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Spark Master** | 8080 | Distributed processing coordinator | http://localhost:8080 |
| **Spark Worker 1** | 8081 | Processing node | http://localhost:8081 |
| **Spark Worker 2** | 8082 | Processing node | http://localhost:8082 |
| **PostgreSQL** | 5432 | Pipeline metadata & run tracking | Database only |
| **Kafka** | 9092/9094 | Streaming platform & message broker | http://localhost:9002 |
| **Kafdrop** | 9002 | Kafka management & monitoring UI | http://localhost:9002 |
| **MinIO** | 9000/9001 | S3-compatible data storage | http://localhost:9001 |
| **Elasticsearch** | 9200 | Search and analytics engine | http://localhost:9200 |
| **Kibana** | 5601 | Data visualization & dashboards | http://localhost:5601 |
| **Web Dashboard** | 8083 | Real-time pipeline monitoring | http://localhost:8083 |
| **Spark Command API** | 8081 | HTTP API for command execution | http://localhost:8081 |

### **📁 Timeframe-Agnostic Data Flow Architecture**
```
Yahoo Finance API → TimeframeAgnosticFetcher → TimeframeEnhancedStorage
                                   ↓
             MinIO Storage (ohlcv/daily/, ohlcv/hourly/, ohlcv/minute/)
                                   ↓
                     Kafka ← Streaming Data & Real-time Events
                                   ↓
        TimeframeAgnosticSignalGenerator → Trading Signals (by timeframe)
                                   ↓
                     PostgreSQL ← Pipeline Metadata → Web Dashboard
                                   ↓
                    Elasticsearch Logs → Kibana Analytics
                                   ↓
                    HTTP API ← Command Server → Spark Container
```

### **🆕 Modular Abstraction Architecture**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Dashboard Integration Layer                  │
├─────────────────────────────────────────────────────────────────┤
│  cli/dashboard_integration.py  │  cli/dashboard_connector.py    │
│  cli/bf_abstracted.py          │  cli/test_new_cli_minimal.py   │
├─────────────────────────────────────────────────────────────────┤
│                    Orchestration Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  model/orchestration/          │  Workflow Management           │
│  ├── pipeline_orchestrator.py  │  System Monitoring             │
│  ├── workflow_manager.py       │  Error Recovery                │
│  └── system_monitor.py         │  Performance Tracking          │
├─────────────────────────────────────────────────────────────────┤
│                    Core Components                              │
├─────────────────────────────────────────────────────────────────┤
│  model/registry/               │  Component Registry            │
│  model/config/                 │  Configuration Management      │
│  model/logging/                │  Enhanced Logging              │
│  model/data/                   │  Universal Data Fetching       │
│  model/signals/                │  Multi-Strategy Signals        │
│  model/backtesting/            │  Comprehensive Backtesting     │
└─────────────────────────────────────────────────────────────────┘
```

### **⚠️ Important: Timeframe-Aware File Naming Requirements**

**Signal Generation requires specific file naming patterns in MinIO based on timeframe:**

- **Daily Pattern**: `ohlcv/daily/{SYMBOL}/{SYMBOL}_{START_DATE}_{END_DATE}.parquet`
- **Hourly Pattern**: `ohlcv/hourly/{SYMBOL}/{SYMBOL}_{START_DATE}_{END_DATE}_1H.parquet`
- **Minute Pattern**: `ohlcv/minute/{SYMBOL}/{SYMBOL}_{START_DATE}_{END_DATE}_{TIMEFRAME}.parquet`

**Examples:**
- Daily: `ohlcv/daily/AAPL/AAPL_2024-01-01_2024-12-31.parquet`
- Hourly: `ohlcv/hourly/AAPL/AAPL_2024-01-01_2024-12-31_1H.parquet`
- 5-minute: `ohlcv/minute/AAPL/AAPL_2024-01-01_2024-12-31_5M.parquet`

**Critical Requirements:**
- Date ranges must exactly match between data fetching and signal generation
- Timeframe must be consistent across fetch, signal generation, and backtesting
- Files are organized in timeframe-specific folders for optimal performance

**Legacy Support:** The system maintains backward compatibility with old `ohlcv/{SYMBOL}/` structure for daily data.

**Solution**: Always specify the same `--timeframe` parameter for data fetch, signal generation, and backtesting.

---

## 🔧 **Command Execution Architecture**

### **🎯 How Commands Work**
The system uses a clean HTTP-based architecture for command execution:

```
Dashboard Container ←→ HTTP API ←→ Spark Command Server ←→ CLI Scripts
```

### **📡 Command Flow**
1. **Dashboard Commands**: User clicks command in web interface
2. **HTTP Request**: Dashboard sends POST to `http://spark-master:8081/execute`
3. **Command Execution**: Spark Command Server runs CLI script in Spark container
4. **Response**: Results returned via HTTP to dashboard
5. **Display**: Real-time output shown in dashboard

### **🔌 Available Commands**
- **Data Commands**: `data_summary`, `data_fetch` with `--timeframe` and `--data-source` parameters (run in Spark container)
- **Signal Commands**: `signal_generate`, `signal_summary` with timeframe-aware processing (run in Spark container)
- **Backtesting**: `backtest_run` with timeframe-optimized parameters (run in Spark container)
- **Kafka Commands**: `kafka_demo`, `kafka_real_test` (run in dashboard container)

### **🆕 New Abstraction System Commands**
- **Enhanced Data Fetching**: Multiple sources and resource types
- **Advanced Signal Generation**: Technical, fundamental, and sentiment analysis
- **Comprehensive Backtesting**: Multiple engines with risk management
- **Workflow Management**: Complex multi-step process orchestration
- **System Monitoring**: Real-time health checks and performance metrics

### **⚡ API Endpoints**
- **Health Check**: `GET http://localhost:8081/health`
- **Execute Command**: `POST http://localhost:8081/execute`
  ```json
  {
    "command": "data_fetch",
    "parameters": {
      "symbols": "AAPL,MSFT",
      "start_date": "2024-08-15",
      "end_date": "2024-08-16",
      "timeframe": "1hour",
      "data_source": "yfinance"
    }
  }
  ```

---

## 🚀 **Startup Process - Important!**

### **⚠️ Why Not Just `docker-compose up`?**

The BreadthFlow platform requires **additional setup steps** beyond just starting Docker containers:

1. **Database Schema**: The timeframe-agnostic features require 5 new database tables and views
2. **MinIO Bucket**: The `breadthflow` bucket must be created for data storage
3. **Kafka Topics**: Financial data streaming topics need to be created
4. **Kibana Dashboards**: Monitoring dashboards need initialization
5. **Spark Command Server**: HTTP API server needs to be started

### **✅ Correct Startup Process**

**Use the provided startup script:**
```bash
./scripts/start_infrastructure.sh
```

**This script automatically handles:**
- ✅ Docker container startup
- ✅ Database schema application (`timeframe_schema.sql`)
- ✅ MinIO bucket creation (`breadthflow`)
- ✅ Kafka topic creation (market-data, trading-signals, etc.)
- ✅ Kibana dashboard initialization
- ✅ Spark Command Server startup
- ✅ Health verification

### **❌ What NOT to do:**
```bash
# Don't just run this - it won't set up timeframe features
docker-compose up -d

# Don't run this - it won't create the database schema
docker-compose -f infra/docker-compose.yml up -d
```

---

## 🚀 **Startup Scripts**

### **📋 Available Scripts**
| Script | Purpose | Usage |
|--------|---------|-------|
| **`start_infrastructure.sh`** | Complete startup with Kafka topics & Kibana setup | `./scripts/start_infrastructure.sh` |
| **`run_demo.sh`** | Full pipeline demonstration | `./scripts/run_demo.sh` |
| **`kafka_demo.sh`** | Kafka streaming demonstration | `./scripts/kafka_demo.sh` |
| **`real_kafka_integration_test.sh`** | Real Kafka integration testing | `./scripts/real_kafka_integration_test.sh` |
| **`check_status.sh`** | Health check for all services | `./scripts/check_status.sh` |
| **`stop_infrastructure.sh`** | Safely stop all services | `./scripts/stop_infrastructure.sh` |
| **`restart_infrastructure.sh`** | Restart all services | `./scripts/restart_infrastructure.sh` |

### **🔧 New Components**
| Component | Purpose | Location |
|-----------|---------|----------|
| **Spark Command Server** | HTTP API for executing CLI commands in Spark container | `cli/spark_command_server.py` |
| **Dashboard Commands** | Web interface for executing all pipeline commands | `http://localhost:8083/commands` |
| **HTTP API Bridge** | Clean communication between dashboard and Spark | Port 8081 |

### **🎯 What Each Script Does**

**`start_infrastructure.sh`** - Complete Setup:
- ✅ Starts all 11 Docker containers (including Command Server)
- ✅ Creates Kafka topics (market-data, trading-signals, pipeline-events, backtest-results)
- ✅ Initializes Kibana dashboards
- ✅ Starts Spark Command Server on port 8081
- ✅ **Applies timeframe database schema** (5 new tables/views for multi-timeframe support)
- ✅ **Creates MinIO breadthflow bucket** for timeframe-organized data storage
- ✅ Generates initial data and logs
- ✅ Verifies all services are healthy

**`run_demo.sh`** - Full Pipeline Demo:
- ✅ Data fetching from Yahoo Finance
- ✅ Signal generation with technical analysis
- ✅ Backtesting simulation
- ✅ Results summary and monitoring

**`kafka_demo.sh`** - Kafka Streaming Demo:
- ✅ Demonstrates Kafka topics and message flow
- ✅ Sends sample market data, trading signals, and pipeline events
- ✅ Shows real-time streaming capabilities
- ✅ Explains Kafka use cases in financial pipeline

**`check_status.sh`** - Health Monitoring:
- ✅ Container status verification
- ✅ Service health checks (PostgreSQL, Elasticsearch, Kibana, MinIO, Kafka, Kafdrop)
- ✅ Spark Command Server health check (port 8081)
- ✅ Kafka topics listing
- ✅ Pipeline data summary

### **💡 Quick Commands**
```bash
# First time setup (includes timeframe database schema and MinIO bucket)
./scripts/start_infrastructure.sh

# Option 1: Web Dashboard (Recommended)
# Go to http://localhost:8083 → Click "Commands" → Execute commands

# Option 2: Direct API Calls
# Test command server health
curl http://localhost:8081/health

# Execute data summary via API
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "data_summary", "parameters": {}}' \
  http://localhost:8081/execute

# Option 3: Scripts
# Run complete demo
./scripts/run_demo.sh

# Try Kafka streaming demo
./scripts/kafka_demo.sh

# Test real Kafka integration
./scripts/real_kafka_integration_test.sh

# Check everything is working
./scripts/check_status.sh

# Stop when done
./scripts/stop_infrastructure.sh

# To restart with all features
./scripts/start_infrastructure.sh
```

---

## 🛠️ **Core CLI Tools**

### **🎯 Primary CLI (Recommended)**
```bash
# Kibana-Enhanced CLI with dual logging (PostgreSQL + Elasticsearch)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py COMMAND
```

**Available Commands:**
```bash
# Data Management (Timeframe-Aware)
data summary                    # Overview of stored data across all timeframes
data fetch --symbols AAPL,MSFT --start-date 2024-08-15 --end-date 2024-08-16 --timeframe 1day --data-source yfinance
data fetch --symbols AAPL --timeframe 1hour --data-source yfinance  # Intraday data
data fetch --symbol-list demo_small --timeframe 15min               # 15-minute bars

# Signal Generation (Multi-Timeframe)
signals generate --symbols AAPL --timeframe 1day     # Daily signals
signals generate --symbols AAPL --timeframe 1hour    # Hourly signals
signals summary                                        # Signal overview

# Backtesting (Timeframe-Optimized)
backtest run --symbols AAPL --timeframe 1day         # Daily strategy backtest
backtest run --symbols AAPL --timeframe 1hour        # Intraday strategy backtest

# Monitoring & Demos
demo --quick                    # 2-symbol demo
demo                           # Full 4-symbol demo
setup-kibana                   # Initialize Kibana integration
```

### **⚡ Alternative CLIs**
```bash
# Feature-Rich CLI with complete pipeline
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_minio.py COMMAND

# Available commands: data, analytics, symbols, signals, backtest, replay, demo
```

> **Note**: Legacy and redundant CLI files have been removed for cleaner codebase. The above two CLIs provide complete functionality.

---

## 🆕 **Recent Dashboard Improvements** (Updated 2025-01-24)

### **✅ Fixed Empty Dashboard Pages**
The BreadthFlow dashboard has been significantly improved with comprehensive content for all pages:

#### **📊 Main Dashboard** (http://localhost:8083)
- **✅ Stats Cards**: Total Pipeline Runs, Success Rate, Last 24h Runs, Average Duration
- **✅ Recent Pipeline Runs Table**: Command, Status, Duration, Start Time, Actions
- **✅ Quick Actions**: Direct navigation to Pipeline, Trading Signals, Commands, Training
- **✅ Auto-refresh**: Updates every 30 seconds with live data
- **✅ Pagination**: Navigate through pipeline run history

#### **🏗️ Infrastructure Status** (http://localhost:8083/infrastructure)
- **✅ System Health Overview**: Database, Pipeline, API, Uptime status indicators
- **✅ Service Status Table**: Dashboard Server, PostgreSQL, API Endpoints, Static Files
- **✅ System Resources**: CPU, Memory, Disk usage, Network status
- **✅ Real-time Monitoring**: Auto-refreshes every 30 seconds with live status checks
- **✅ Color-coded Status**: Green (healthy), Yellow (warning), Red (error)

#### **🔧 Technical Improvements**
- **✅ Direct HTML Rendering**: Replaced template-based rendering for reliable content display
- **✅ JavaScript Integration**: Added comprehensive client-side functionality
- **✅ API Integration**: Connected to backend APIs for real-time data
- **✅ Error Handling**: Graceful handling of database connection failures
- **✅ Responsive Design**: Works on desktop and mobile devices

#### **🎯 All Dashboard Pages Now Working**
- **📊 Main Dashboard**: ✅ Full content with stats and recent runs
- **🏗️ Infrastructure Status**: ✅ System health monitoring
- **📈 Trading Signals**: ✅ Signal monitoring with export (already working)
- **🎮 Pipeline Management**: ✅ Pipeline control (already working)
- **⚡ Commands**: ✅ Command execution (already working)
- **🎓 Training**: ✅ Model training (already working)
- **⚙️ Parameters**: ✅ Configuration management (already working)

### **🚀 Quick Test**
```bash
# Start the dashboard
./scripts/start_infrastructure.sh

# Test all pages
curl -s http://localhost:8083/ | grep -A 5 "stats"           # Main dashboard
curl -s http://localhost:8083/infrastructure | grep -A 5 "stats"  # Infrastructure
curl -s http://localhost:8083/trading | head -10             # Trading signals
curl -s http://localhost:8083/pipeline | head -10            # Pipeline management
```

---

## 🎮 **Pipeline Management - Automated Batch Processing**

### **🚀 Overview**
The Enhanced Batch Processing system transforms manual command execution into automated, continuous pipeline execution with comprehensive monitoring and dashboard control.

### **✨ Key Features**
- **🔄 Continuous Execution**: Automated pipeline runs at configurable intervals
- **🎛️ Web Control**: Start/stop/monitor pipelines from the dashboard
- **📊 Real-time Monitoring**: Live status updates, metrics, and run history
- **⚙️ Flexible Configuration**: Multiple modes, intervals, and timeframes
- **📋 Run History**: Complete tracking of all pipeline executions
- **❌ Error Handling**: Graceful error handling and recovery
- **🔍 Detailed Logging**: Comprehensive logs for troubleshooting

### **🎯 Access Pipeline Management**
- **Web Interface**: http://localhost:8083/pipeline
- **API Endpoints**: http://localhost:8081/execute (pipeline commands)

### **📋 Pipeline Commands**
```bash
# Start continuous pipeline (CLI)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline start --mode demo --interval 5m --timeframe 1day

# Stop pipeline
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline stop

# Check status
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline status

# View logs
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline logs --lines 20

# Run specific number of cycles
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline run --mode demo --cycles 3 --timeframe 1hour
```

### **🎛️ Pipeline Configuration Options**

#### **📊 Pipeline Modes**
| Mode | Description | Symbols | Best For |
|------|-------------|---------|----------|
| **demo** | Basic demo (3 symbols) | AAPL, MSFT, GOOGL | Testing, development |
| **demo_small** | Small demo | 3 symbols | Quick testing |
| **tech_leaders** | Technology leaders | 8 major tech stocks | Production testing |
| **all_symbols** | All available symbols | 100+ symbols | Full production |
| **custom** | User-defined symbols | Custom list | Specific analysis |

#### **⏰ Execution Intervals**
| Interval | Best For | Use Case |
|----------|----------|----------|
| **1m** | Ultra-high frequency | Real-time trading |
| **5m** | High frequency | Intraday updates |
| **15m** | Medium frequency | Regular monitoring |
| **1h** | Hourly updates | Standard operation |
| **6h** | Twice daily | Conservative approach |
| **1d** | Daily execution | Traditional trading |

#### **📈 Timeframe Support**
| Timeframe | Data Granularity | Parameters Optimized |
|-----------|------------------|---------------------|
| **1min** | 1-minute bars | Ultra-high frequency trading |
| **5min** | 5-minute bars | High frequency trading |
| **15min** | 15-minute bars | Medium frequency trading |
| **1hour** | 1-hour bars | Intraday swing trading |
| **1day** | Daily bars | Traditional daily trading |

### **🎯 Pipeline Execution Flow**
1. **Data Fetch**: Retrieves latest market data for configured symbols and timeframe
2. **Signal Generation**: Analyzes data and generates trading signals using timeframe-optimized parameters
3. **Backtesting**: Runs performance simulation with timeframe-specific costs and logic
4. **Logging**: Records all activities to PostgreSQL and Elasticsearch
5. **Wait**: Pauses for configured interval before next cycle

### **📊 Monitoring & Metrics**
- **Pipeline Status**: Real-time status (stopped/running/starting/stopping)
- **Success Rate**: Percentage of successful pipeline runs
- **Run History**: Complete log of all pipeline executions
- **Performance Metrics**: Average duration, symbols processed, signals generated
- **Error Analysis**: Error rates, common failures, troubleshooting information
- **Uptime Tracking**: Total runtime and availability statistics

### **🔄 Web Dashboard Integration**
The Pipeline Management page (http://localhost:8083/pipeline) provides:
- **🎛️ Control Panel**: Start/stop buttons with real-time status
- **⚙️ Configuration Forms**: Mode, interval, timeframe, and symbol selection
- **📊 Live Metrics**: Real-time pipeline statistics and performance data
- **📋 Run History Table**: Clickable table of recent pipeline executions
- **🔄 Auto-refresh**: Updates every 30 seconds automatically
- **💬 Status Messages**: Real-time feedback on pipeline operations

### **🚀 Quick Start Examples**

#### **Demo Pipeline (Recommended)**
```bash
# Start demo pipeline via dashboard
# 1. Go to http://localhost:8083/pipeline
# 2. Select Mode: "Demo"
# 3. Set Interval: "5 Minutes"
# 4. Choose Timeframe: "1day"
# 5. Click "Start Pipeline"

# Or via CLI
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline start --mode demo --interval 5m --timeframe 1day
```

#### **Intraday Trading Pipeline**
```bash
# Hourly intraday pipeline
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline start --mode tech_leaders --interval 1h --timeframe 1hour
```

#### **Custom Symbols Pipeline**
```bash
# Custom symbol list with specific timeframe
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py pipeline start --mode custom --symbols AAPL,TSLA,NVDA --interval 15m --timeframe 5min
```

### **⚡ Advanced Features**
- **🔄 Background Execution**: Pipelines run continuously in background threads
- **🛑 Graceful Shutdown**: Clean stop with final statistics
- **📊 Metadata Tracking**: Complete run metadata stored in PostgreSQL
- **🔍 Error Recovery**: Automatic retry logic and fallback mechanisms
- **📈 Performance Optimization**: Timeframe-specific parameter optimization
- **💾 State Persistence**: Pipeline state maintained across restarts
- **🌐 HTTP API**: Full programmatic control via REST API

### **🔧 Pipeline Architecture**
```
Web Dashboard → HTTP API → Spark Command Server → Pipeline Runner
                                    ↓
                      Execute: data fetch → signals generate → backtest run
                                    ↓
                      Log Results → PostgreSQL + Elasticsearch
                                    ↓
                      Wait for Interval → Repeat Cycle
```

---

## 🎮 **Pipeline Management Dashboard - Real-Time Control**

### **🚀 Overview**
The Pipeline Management Dashboard provides **real-time control** over pipeline execution with a modern web interface. This feature allows you to start, stop, and monitor pipelines directly from the browser with comprehensive status tracking and run history.

### **✨ Key Features**
- **🎛️ Real-Time Control**: Start/stop pipelines with one-click buttons
- **📊 Live Status Monitoring**: Real-time pipeline status and metrics
- **📋 Run History**: Complete table of all pipeline runs from the last 2 days
- **🔒 Smart State Management**: Only one pipeline can run at a time
- **⚡ Dynamic UI**: Buttons automatically enable/disable based on pipeline state
- **🔄 Auto-Refresh**: Dashboard updates automatically every 30 seconds
- **📈 Duration Tracking**: Real-time calculation of pipeline run durations
- **❌ Error Handling**: Comprehensive error tracking and display

### **🎯 Access Pipeline Management**
- **Web Interface**: http://localhost:8083/pipeline
- **API Endpoints**: 
  - `POST /api/pipeline/start` - Start a new pipeline
  - `POST /api/pipeline/stop` - Stop the current pipeline
  - `GET /api/pipeline/runs` - Get recent pipeline runs
  - `GET /api/pipeline/status` - Get current pipeline status
  - `GET /api/pipeline/running-status` - Check if pipeline is running

### **🎮 Dashboard Interface**

#### **🎛️ Control Panel**
- **🚀 Start Pipeline Button**: Creates and starts a new pipeline
  - **Disabled when**: A pipeline is already running
  - **Enabled when**: No pipeline is currently running
- **🛑 Stop Pipeline Button**: Stops the currently running pipeline
  - **Disabled when**: No pipeline is running or current pipeline is stopped
  - **Enabled when**: A pipeline is currently running
- **🔄 Refresh Button**: Manually refresh the dashboard data

#### **⚙️ Configuration Form**
- **Mode Selection**: Demo, Small, Medium, Full pipeline configurations
- **Interval Selection**: 1m, 5m, 15m, 1h, 6h, 1d execution intervals
- **Timeframe Selection**: 1min, 5min, 15min, 1hour, 1day data timeframes
- **Symbols Input**: Custom symbol list (optional)
- **Data Source**: yfinance, alpha_vantage, etc.

#### **📊 Live Metrics Display**
- **Pipeline State**: Running, Stopped, Starting, Stopping
- **Total Runs**: Number of pipeline executions
- **Success Rate**: Percentage of successful runs
- **Average Duration**: Mean execution time
- **Last Run Time**: Timestamp of most recent execution
- **Uptime**: Current pipeline runtime (if running)

#### **📋 Recent Pipeline Runs Table**
- **Run ID**: Unique identifier for each pipeline execution
- **Command**: Pipeline command executed
- **Status**: Running, Stopped, Completed, Failed
- **Start Time**: When the pipeline started
- **End Time**: When the pipeline ended (if completed)
- **Duration**: Total execution time in seconds
- **Error Message**: Any error details (if failed)
- **Configuration**: Pipeline parameters used

### **🔄 Pipeline State Management**

#### **🚀 Starting a Pipeline**
1. **Click "Start Pipeline"** button
2. **System checks**: Ensures no other pipeline is running
3. **Creates new record**: Adds pipeline run to database with "running" status
4. **Executes pipeline**: Starts the actual data processing
5. **Updates UI**: Start button becomes disabled, stop button becomes enabled
6. **Shows in table**: New pipeline appears in Recent Pipeline Runs with "running" status

#### **🛑 Stopping a Pipeline**
1. **Click "Stop Pipeline"** button
2. **System identifies**: Finds the currently running pipeline
3. **Stops execution**: Terminates the pipeline process
4. **Updates status**: Changes status from "running" to "stopped"
5. **Records end time**: Updates database with completion timestamp
6. **Updates UI**: Stop button becomes disabled, start button becomes enabled
7. **Shows in table**: Pipeline appears with "stopped" status and duration

#### **🔄 Multiple Pipeline Runs**
- **First start**: Creates pipeline with "running" status
- **Stop**: Changes status to "stopped", remains in table
- **Second start**: Creates NEW pipeline with "running" status
- **Result**: Table shows both pipelines - old (stopped) and new (running)

### **📊 API Endpoints**

#### **Start Pipeline**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"mode": "demo", "interval": "5m", "timeframe": "1day", "symbols": "AAPL,MSFT", "data_source": "yfinance"}' \
  http://localhost:8083/api/pipeline/start
```

**Response:**
```json
{
  "success": true,
  "message": "Pipeline started successfully with ID: pipeline_20250824_115327_10709148",
  "pipeline_id": "pipeline_20250824_115327_10709148",
  "config": {
    "mode": "demo",
    "interval": "5m",
    "timeframe": "1day",
    "symbols": "AAPL,MSFT",
    "data_source": "yfinance"
  }
}
```

#### **Stop Pipeline**
```bash
curl -X POST -H "Content-Type: application/json" \
  http://localhost:8083/api/pipeline/stop
```

**Response:**
```json
{
  "success": true,
  "message": "Pipeline pipeline_20250824_115327_10709148 stopped successfully",
  "pipeline_id": "pipeline_20250824_115327_10709148"
}
```

#### **Get Pipeline Runs**
```bash
curl -s http://localhost:8083/api/pipeline/runs
```

**Response:**
```json
{
  "success": true,
  "runs": [
    {
      "run_id": "pipeline_20250824_115819_c682d6d5",
      "command": "spark_streaming_start_demo",
      "status": "running",
      "start_time": "2025-08-24 11:58:19",
      "end_time": null,
      "duration_seconds": 118,
      "error_message": null,
      "config": {}
    },
    {
      "run_id": "pipeline_20250824_115327_10709148",
      "command": "spark_streaming_start_demo",
      "status": "stopped",
      "start_time": "2025-08-24 11:53:27",
      "end_time": "2025-08-24 11:58:01",
      "duration_seconds": 273,
      "error_message": null,
      "config": {}
    }
  ]
}
```

#### **Get Pipeline Status**
```bash
curl -s http://localhost:8083/api/pipeline/status
```

**Response:**
```json
{
  "state": "running",
  "total_runs": 5,
  "successful_runs": 4,
  "failed_runs": 0,
  "stopped_runs": 1,
  "uptime_seconds": 118,
  "last_run_time": "2025-08-24 11:58:19"
}
```

### **🎯 Usage Examples**

#### **Quick Start Demo**
1. **Open Dashboard**: http://localhost:8083/pipeline
2. **Select Configuration**:
   - Mode: "Demo"
   - Interval: "5 Minutes"
   - Timeframe: "1day"
   - Symbols: "AAPL,MSFT"
3. **Click "Start Pipeline"**
4. **Monitor**: Watch the pipeline run in real-time
5. **Stop**: Click "Stop Pipeline" when done

#### **Production Pipeline**
1. **Configure for Production**:
   - Mode: "Tech Leaders"
   - Interval: "1 Hour"
   - Timeframe: "1hour"
   - Symbols: Leave empty (uses predefined list)
2. **Start Pipeline**: Click start button
3. **Monitor**: Check metrics and run history
4. **Manage**: Stop and restart as needed

#### **Custom Configuration**
1. **Set Custom Parameters**:
   - Mode: "Custom"
   - Interval: "15 Minutes"
   - Timeframe: "5min"
   - Symbols: "AAPL,TSLA,NVDA,GOOGL"
2. **Start**: Execute with custom settings
3. **Track**: Monitor performance and results

### **🔧 Technical Architecture**

#### **Frontend Components**
- **Pipeline Dashboard**: `cli/pipeline_dashboard.py` - HTML/CSS/JavaScript UI
- **Real-time Updates**: JavaScript polling for status changes
- **Dynamic Buttons**: State-based enable/disable logic
- **Responsive Design**: Works on desktop and mobile

#### **Backend Components**
- **Pipeline Controller**: `cli/pipeline_controller.py` - Core business logic
- **Database Integration**: PostgreSQL for pipeline metadata
- **Spark Integration**: HTTP API calls to Spark command server
- **State Management**: Real-time pipeline state tracking

#### **Data Flow**
```
User Action → Dashboard → Pipeline Controller → Database
                                    ↓
                      Spark Command Server → Pipeline Execution
                                    ↓
                      Status Updates → Database → Dashboard UI
```

### **🛠️ Development & Customization**

#### **Adding New Pipeline Modes**
1. **Update Pipeline Controller**: Add new mode logic in `pipeline_controller.py`
2. **Update Dashboard**: Add mode option in `pipeline_dashboard.py`
3. **Test**: Verify start/stop functionality
4. **Deploy**: Rebuild dashboard container

#### **Customizing UI**
1. **Modify HTML**: Update `pipeline_dashboard.py` HTML generation
2. **Add CSS**: Customize styling for buttons and tables
3. **Enhance JavaScript**: Add new interactive features
4. **Test**: Verify all functionality works

#### **Extending API**
1. **Add Endpoints**: Create new API routes in `postgres_dashboard.py`
2. **Update Controller**: Add corresponding methods in `pipeline_controller.py`
3. **Test API**: Verify endpoints work correctly
4. **Update Documentation**: Document new features

### **🔍 Monitoring & Troubleshooting**

#### **Dashboard Not Loading**
```bash
# Check dashboard container
docker logs breadthflow-dashboard

# Verify API endpoints
curl http://localhost:8083/api/pipeline/status

# Check database connection
docker exec breadthflow-postgres psql -U pipeline -d breadthflow -c "SELECT COUNT(*) FROM pipeline_runs;"
```

#### **Pipeline Won't Start**
```bash
# Check if pipeline is already running
curl http://localhost:8083/api/pipeline/running-status

# Check Spark command server
curl http://localhost:8081/health

# View pipeline logs
docker logs spark-master
```

#### **Pipeline Won't Stop**
```bash
# Check running pipeline ID
curl http://localhost:8083/api/pipeline/status

# Force stop via database
docker exec breadthflow-postgres psql -U pipeline -d breadthflow -c "UPDATE pipeline_runs SET status = 'stopped', end_time = NOW() WHERE status = 'running';"
```

#### **No Pipeline Runs Showing**
```bash
# Check database for pipeline runs
docker exec breadthflow-postgres psql -U pipeline -d breadthflow -c "SELECT * FROM pipeline_runs WHERE command LIKE '%spark_streaming%' OR command LIKE '%pipeline%' ORDER BY start_time DESC LIMIT 5;"

# Rebuild dashboard if needed
docker-compose build dashboard && docker-compose up -d dashboard
```

### **📈 Performance Characteristics**
- **Response Time**: <1 second for API calls
- **UI Updates**: 30-second auto-refresh interval
- **Database Queries**: Optimized for recent pipeline runs
- **Concurrent Users**: Supports multiple dashboard users
- **Memory Usage**: Minimal overhead for dashboard operations
- **Scalability**: Can handle hundreds of pipeline runs

---

## 📊 **Monitoring & Analytics**

### **🎯 Real-time Web Dashboard** (Primary)
1. **Open Dashboard**: http://localhost:8083
2. **Main Features**:
   - **Pipeline Metrics**: Total runs, success rate, average duration
   - **Live Updates**: Auto-refreshing statistics every 30 seconds
   - **Recent Runs**: Detailed view of latest pipeline executions
   - **Infrastructure Overview**: Interactive D3.js architecture diagram
   - **📈 Trading Signals Dashboard**: Real-time signal monitoring with 4-day history
   - **📊 Data Export**: CSV and JSON export functionality for signals

**Dashboard Pages:**
- **📊 Main Dashboard**: Live pipeline metrics, stats cards, recent runs table, and quick actions
- **🎮 Pipeline Management**: Real-time pipeline control with start/stop buttons
- **📈 Trading Signals**: Comprehensive signal monitoring with export capabilities
- **🏗️ Infrastructure Status**: System health monitoring with service status and resource metrics
- **⚡ Commands**: Quick command execution interface
- **🎓 Training**: Model training and management interface
- **⚙️ Parameters**: System configuration and parameter management

### **🔍 Kibana Analytics** (Advanced)
1. **Open Kibana**: http://localhost:5601
2. **Go to Discover**: Click "Discover" in left menu
3. **Select Index**: Choose "breadthflow-logs*"
4. **View Data**: See all pipeline logs with filtering/searching

**Pre-built Dashboards:**
- **🚀 BreadthFlow Working Dashboard**: Real-time pipeline monitoring
- **🔍 Index Pattern**: `breadthflow-logs*` for custom visualizations

### **📈 Trading Signals Dashboard** (Enhanced)
1. **Access**: http://localhost:8083/signals
2. **Key Features**:
   - **📊 4-Day Signal History**: Reads all signal files from the last 4 days
   - **⏰ Chronological Ordering**: Signals ordered by timestamp (newest first)
   - **🕒 Create Time Tracking**: Each signal includes creation timestamp
   - **📊 Multi-Timeframe Support**: Displays signals from all timeframes (1min, 5min, 15min, 1hour, 1day)
   - **📥 Export Functionality**: Download signals in CSV or JSON format
   - **🔍 Real-time Updates**: Auto-refreshing signal data every 30 seconds

**Signal Export Options:**
- **CSV Export**: `GET /api/signals/export?format=csv` - Download as CSV file
- **JSON Export**: `GET /api/signals/export?format=json` - Download as JSON file
- **Fields Included**: Symbol, Signal Type, Confidence, Strength, Date, Timeframe, Create Time

### **📈 What You Can Monitor**
- **Pipeline Success Rates**: Track successful vs failed runs
- **Performance Trends**: Monitor execution durations over time
- **Symbol Processing**: Success/failure by individual stocks
- **Error Analysis**: Detailed error logs and patterns
- **Real-time Activity**: Live updates as pipelines execute
- **PostgreSQL Metrics**: Pipeline run history with metadata
- **Trading Signals**: Real-time signal monitoring with 4-day history
- **Signal Performance**: Track signal generation across multiple timeframes
- **Export Analytics**: Monitor data export usage and patterns

### **🔍 Useful Kibana Searches**
```bash
# Find errors
level:ERROR

# Track specific symbols
metadata.symbol:AAPL

# Find slow operations
duration:>10

# Recent activity
@timestamp:>=now-1h

# Signal generation activity
metadata.command:signal_generate

# Export operations
metadata.command:export
```

---

## 💾 **Data Storage**

### **📦 MinIO (S3-Compatible Storage) - Timeframe-Enhanced**
- **Access**: http://localhost:9001 (minioadmin/minioadmin)
- **Timeframe-Aware Structure**:
  ```
  breadthflow/
  ├── ohlcv/
  │   ├── daily/
  │   │   ├── AAPL/
  │   │   │   └── AAPL_2024-01-01_2024-12-31.parquet
  │   │   ├── MSFT/
  │   │   └── GOOGL/
  │   ├── hourly/
  │   │   ├── AAPL/
  │   │   │   └── AAPL_2024-01-01_2024-12-31_1H.parquet
  │   │   └── MSFT/
  │   └── minute/
  │       ├── AAPL/
  │       │   └── AAPL_2024-01-01_2024-12-31_5M.parquet
  │       └── MSFT/
  ├── trading_signals/
  │   ├── signals_YYYYMMDD_HHMMSS.parquet (timeframe-aware)
  │   ├── signals_YYYYMMDD_HHMMSS_1day.parquet (daily signals)
  │   ├── signals_YYYYMMDD_HHMMSS_1hour.parquet (hourly signals)
  │   └── signals_YYYYMMDD_HHMMSS_15min.parquet (intraday signals)
  └── analytics/
      └── processed_results.parquet
  ```

**Signal File Organization:**
- **4-Day History**: Dashboard reads all signal files from the last 4 days
- **Chronological Ordering**: Files processed by timestamp (newest first)
- **Multi-Timeframe Support**: Separate files for each timeframe
- **Create Time Tracking**: Each signal includes creation timestamp

### **🗃️ Database Storage**

**PostgreSQL (Primary Pipeline Metadata) - Timeframe-Enhanced**
- **Purpose**: Stores pipeline run history for web dashboard with timeframe tracking
- **Connection**: postgresql://pipeline:pipeline123@postgres:5432/breadthflow
- **Contains**: Run status, durations, timestamps, error messages, timeframe metadata
- **New Tables**: `timeframe_configs`, `timeframe_data_summary`, `signals_metadata`, `backtest_results`
- **Enhanced Views**: `timeframe_performance_stats`, `signals_summary_by_timeframe`, `data_availability_by_timeframe`

**Elasticsearch (Advanced Logs & Analytics)**
- **Index**: `breadthflow-logs`
- **Contains**: All pipeline execution logs with metadata
- **Query API**: http://localhost:9200/breadthflow-logs/_search

### **🚀 Kafka (Streaming Platform)**
- **Access**: http://localhost:9002 (Kafdrop UI)
- **Purpose**: Real-time data streaming and message processing
- **Features**:
  - **Topic Management**: Create, delete, and monitor topics
  - **Message Browsing**: View and search through messages
  - **Consumer Monitoring**: Track consumer groups and lag
  - **Real-time Streaming**: Live data pipeline capabilities
- **Use Cases**:
  - Historical data replay for backtesting
  - Real-time market data streaming
  - Event-driven pipeline processing
  - Message queuing for distributed processing

---

## 🎮 **Usage Examples**

### **📊 Multi-Timeframe Data Analysis Workflow**
```bash
# 1. Check current data across all timeframes
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

# 2. Fetch data for different timeframes
# Daily data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT --timeframe 1day --start-date 2024-01-01 --end-date 2024-12-31

# Hourly intraday data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --timeframe 1hour --start-date 2024-08-20 --end-date 2024-08-21

# 3. Generate signals for specific timeframes
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py signals generate --symbols AAPL --timeframe 1day
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py signals generate --symbols AAPL --timeframe 1hour

# 4. Run timeframe-optimized backtesting
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py backtest run --symbols AAPL --timeframe 1day
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py backtest run --symbols AAPL --timeframe 1hour

# 5. Monitor signals in real-time
# Trading Signals Dashboard: http://localhost:8083/signals (4-day history, multi-timeframe)
# Web Dashboard: http://localhost:8083 (timeframe-aware interface)
# Kibana: http://localhost:5601 → Discover → breadthflow-logs*

# 6. Export signal data
# CSV Export: curl "http://localhost:8083/api/signals/export?format=csv" > signals.csv
# JSON Export: curl "http://localhost:8083/api/signals/export?format=json" > signals.json

# 7. Check timeframe-organized data storage
# MinIO: http://localhost:9001 → Browse ohlcv/daily/, ohlcv/hourly/, ohlcv/minute/

# 8. Monitor streaming data
# Kafka UI: http://localhost:9002 → View topics and messages

### **🔧 Development Workflow**
```bash
# 1. Start infrastructure (includes timeframe setup)
./scripts/start_infrastructure.sh

# 2. Develop and test
# Option A: Web Dashboard (Recommended)
# Go to http://localhost:8083 → Click "Commands" → Execute commands

# Option B: Direct API Testing
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "data_summary", "parameters": {}}' \
  http://localhost:8081/execute

# Option C: Scripts
./scripts/run_demo.sh

# 3. Monitor in real-time
# Dashboard: http://localhost:8083
# Trading Signals: http://localhost:8083/signals (4-day history)
# Kibana: http://localhost:5601
# Kafka UI: http://localhost:9002
# Spark UI: http://localhost:8080
# Command API: http://localhost:8081

# 4. Export and analyze data
# CSV Export: curl "http://localhost:8083/api/signals/export?format=csv" > signals.csv
# JSON Export: curl "http://localhost:8083/api/signals/export?format=json" > signals.json

# 5. Stop when done
./scripts/stop_infrastructure.sh

# 6. Restart with all features
./scripts/start_infrastructure.sh
```

---

## 🔧 **Configuration & Customization**

### **🐳 Docker Configuration**
- **File**: `infra/docker-compose.yml`
- **Customize**: Ports, memory allocation, worker count
- **Volumes**: Code is mounted read-only for security

### **📦 Package Management**
- **File**: `infra/Dockerfile.spark`
- **Installed**: yfinance, pandas, numpy, spark, delta-lake, boto3, pyarrow, psycopg2-binary
- **Rebuild**: `docker-compose build --no-cache`

### **⚙️ Spark Configuration**
- **Master**: Local mode with all cores
- **Workers**: 2 workers with 2GB RAM each
- **Packages**: Delta Lake, Kafka integration included

---

## 📁 **Project Structure**

```
BreadthFlow/
├── cli/                               # 🎮 Command-line interfaces
│   ├── kibana_enhanced_bf.py          # Primary CLI with PostgreSQL + Elasticsearch logging (timeframe-aware)
│   ├── bf_minio.py                   # Feature-rich CLI with complete pipeline
│   ├── postgres_dashboard.py         # Web dashboard backend (PostgreSQL) with timeframe UI
│   ├── pipeline_controller.py        # Pipeline management business logic
│   ├── pipeline_dashboard.py         # Pipeline management UI components
│   ├── spark_command_server.py       # HTTP API server for command execution (timeframe support)
│   └── elasticsearch_logger.py       # Elasticsearch integration
├── infra/                             # 🐳 Infrastructure setup
│   ├── docker-compose.yml            # Service orchestration (11 services)
│   ├── Dockerfile.spark              # Spark container with all dependencies
│   └── Dockerfile.dashboard          # Web dashboard container
├── ingestion/                         # 📥 Data fetching and processing
│   ├── data_fetcher.py               # PySpark-based data fetching
│   └── replay.py                     # Historical data replay
├── features/                          # 🧮 Feature engineering
│   ├── common/                       # Shared utilities
│   ├── ad_features.py                # Advance/Decline indicators
│   └── ma_features.py                # Moving average features
├── model/                             # 🎯 Signal generation (timeframe-agnostic)
│   ├── scoring.py                    # Composite scoring
│   ├── signal_generator.py           # Signal logic
│   ├── timeframe_agnostic_fetcher.py # Multi-timeframe data fetching interface
│   ├── timeframe_enhanced_storage.py # Timeframe-organized MinIO storage
│   ├── timeframe_agnostic_signals.py # Timeframe-adaptive signal generation
│   ├── timeframe_agnostic_backtest.py # Timeframe-optimized backtesting
│   └── timeframe_config.py           # Centralized timeframe configuration
├── backtests/                         # 📈 Performance analysis
│   ├── engine.py                     # Backtesting engine
│   └── metrics.py                    # Performance metrics
├── sql/                               # 🗃️ Database schemas
│   └── timeframe_schema.sql          # PostgreSQL timeframe enhancements
├── docs/                              # 📚 Documentation
│   ├── monitoring_guide.md           # Complete monitoring setup
│   ├── kibana_dashboard_guide.md     # Kibana customization
│   └── infrastructure_guide.md       # Infrastructure details
├── TIMEFRAME_AGNOSTIC_PLATFORM_PLAN.md # 📋 Transformation planning document
└── data/                              # 📂 Sample data and configs
    └── symbols.json                   # Predefined symbol lists
```

---

## 🎯 **New Timeframe Features** (v2.0)

### **⏰ Multi-Timeframe Support**
BreadthFlow now supports **5 different timeframes** with optimized parameters for each:

| Timeframe | Best For | Parameters | Commission Rate | Max Position |
|-----------|----------|------------|-----------------|--------------|
| **1day** | Traditional daily trading | MA: 20/50, RSI: 14 | 0.1% | 10% |
| **1hour** | Intraday swing trading | MA: 12/24, RSI: 14 | 0.15% | 8% |
| **15min** | Medium frequency trading | MA: 8/16, RSI: 14 | 0.2% | 6% |
| **5min** | High frequency trading | MA: 6/12, RSI: 10 | 0.25% | 5% |
| **1min** | Ultra-high frequency | MA: 5/10, RSI: 8 | 0.3% | 3% |

### **🔄 Key Improvements**
- **📁 Smart Storage**: Automatic timeframe-based file organization (`ohlcv/daily/`, `ohlcv/hourly/`, `ohlcv/minute/`)
- **⚙️ Auto-Configuration**: Timeframe-specific parameters loaded from database configurations
- **📊 Enhanced Dashboard**: Timeframe selection dropdowns in all command interfaces
- **🎯 Backward Compatibility**: Legacy daily data structure still supported
- **🗃️ Database Enhancement**: 5 new tables/views for timeframe tracking and analytics

### **🚀 Quick Timeframe Examples**
```bash
# Daily traditional trading
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --timeframe 1day

# Hourly intraday trading
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --timeframe 1hour

# Generate signals for specific timeframe
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py signals generate --symbols AAPL --timeframe 1hour

# Run timeframe-optimized backtesting
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py backtest run --symbols AAPL --timeframe 1day
```

### **💾 New Database Schema**
The platform now includes enhanced PostgreSQL schema:
- `timeframe_configs` - Configuration parameters for each timeframe
- `timeframe_data_summary` - Data availability tracking by timeframe
- `signals_metadata` - Signal generation metadata with timeframe context
- `backtest_results` - Backtesting results with timeframe-specific metrics
- Performance views: `timeframe_performance_stats`, `signals_summary_by_timeframe`

---

## 📈 **Enhanced Signal Monitoring & Export**

### **🎯 Trading Signals Dashboard Features**
The Trading Signals Dashboard (http://localhost:8083/signals) provides comprehensive signal monitoring with advanced features:

#### **📊 4-Day Signal History**
- **Comprehensive Coverage**: Reads all signal files from the last 4 days
- **Chronological Ordering**: Signals ordered by timestamp (newest first)
- **Multi-Timeframe Support**: Displays signals from all timeframes (1min, 5min, 15min, 1hour, 1day)
- **Create Time Tracking**: Each signal includes creation timestamp for precise timing analysis

#### **📥 Data Export Functionality**
- **CSV Export**: Download signals as CSV file with all metadata
  ```bash
  curl "http://localhost:8083/api/signals/export?format=csv" > signals.csv
  ```
- **JSON Export**: Download signals as JSON file with structured data
  ```bash
  curl "http://localhost:8083/api/signals/export?format=json" > signals.json
  ```
- **Export Fields**: Symbol, Signal Type, Confidence, Strength, Date, Timeframe, Create Time

#### **🔍 Signal Organization**
- **Timeframe Grouping**: Signals organized by timeframe for easy analysis
- **Real-time Updates**: Auto-refreshing signal data every 30 seconds
- **Error Handling**: Graceful handling of missing or corrupted signal files
- **Performance Optimized**: Efficient file reading with chronological processing

### **🎯 API Endpoints**
```bash
# Get latest signals (4-day history)
GET /api/signals/latest

# Export signals as CSV
GET /api/signals/export?format=csv

# Export signals as JSON
GET /api/signals/export?format=json

# Get pipeline parameters
GET /api/parameters
```

### **📊 Signal Data Structure**
```json
{
  "signals": [
    {
      "symbol": "AAPL",
      "signal_type": "BUY",
      "confidence": 0.85,
      "strength": "STRONG",
      "date": "2024-08-24",
      "timeframe": "1day",
      "create_time": "2024-08-24T10:30:00Z"
    }
  ]
}
```

---

## 🚫 **CRITICAL RULE: NO MOCK DATA**

**NEVER use mock data, fake data, or placeholder data in production code or API endpoints.**

### What This Means:
- **No fake user data** (fake names, emails, etc.)
- **No fake financial data** (fake stock prices, fake trading signals)
- **No fake metrics** (fake accuracy scores, fake performance numbers)
- **No fake timestamps** (fake dates, fake creation times)
- **No fake IDs** (fake model IDs, fake run IDs)

### What To Do Instead:
- **Return proper error responses** (501 Not Implemented, 404 Not Found)
- **Use TODO comments** to indicate functionality needs implementation
- **Implement actual database queries** when possible
- **Use real data sources** when available
- **Return empty arrays/objects** when no real data exists
- **Log when functionality is not yet implemented**

### Examples of Correct Implementation:
```python
# ✅ CORRECT - No mock data
def get_training_models(self):
    """Get list of trained models"""
    # TODO: Implement actual model database query
    self.send_response(501)
    self.send_header('Content-type', 'application/json')
    self.end_headers()
    self.wfile.write(json.dumps({
        "error": "Training models functionality not yet implemented",
        "message": "This endpoint will query the actual model database when implemented"
    }).encode())

# ❌ WRONG - Mock data
def get_training_models(self):
    models = [
        {"id": "fake_001", "name": "Fake Model", "accuracy": "87.5"}
    ]
    return models
```

### Why This Rule Exists:
- **Prevents confusion** about what's real vs. fake
- **Maintains data integrity** in the system
- **Forces proper implementation** instead of shortcuts
- **Builds trust** with users and stakeholders
- **Avoids production issues** from fake data

---

## 🚨 **Troubleshooting**

### **🔧 Common Issues**

#### **Services Won't Start**
```bash
# Check Docker is running
docker --version

# Check ports are available
lsof -i :8080,9000,9200,5601

# Use the correct startup script (not just docker-compose)
./scripts/start_infrastructure.sh

# If you used docker-compose directly, restart with proper setup
./scripts/stop_infrastructure.sh
./scripts/start_infrastructure.sh
```

#### **Dashboard Not Updating**
```bash
# Check web dashboard
curl http://localhost:8083/api/summary

# Run test pipeline with monitoring
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

# Check PostgreSQL connection
docker exec breadthflow-postgres psql -U pipeline -d breadthflow -c "SELECT COUNT(*) FROM pipeline_runs;"
```

#### **No Data in Kibana**
```bash
# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Re-setup Kibana
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py setup-kibana

# Generate test data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_minio.py demo
```

#### **Timeframe Features Not Working**
```bash
# Check if database schema was applied
docker exec breadthflow-postgres psql -U pipeline -d breadthflow -c "SELECT COUNT(*) FROM timeframe_configs;"

# If 0 results, apply schema manually
docker exec -i breadthflow-postgres psql -U pipeline -d breadthflow < sql/timeframe_schema.sql

# Check if MinIO bucket exists
docker exec minio mc ls minio/breadthflow

# If bucket doesn't exist, create it
docker exec minio mc mb minio/breadthflow
```

#### **Data Fetch Failures**
```bash
# Test with single symbol (specify timeframe)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --start-date 2024-08-15 --end-date 2024-08-15 --timeframe 1day --data-source yfinance

# Check data summary
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_minio.py data summary
```

### **📊 Performance Optimization**
- **Memory**: Increase worker memory in docker-compose.yml
- **Parallelism**: Use `--parallel N` flag for data fetching
- **Time Ranges**: Fetch smaller date ranges for faster processing
- **Symbol Lists**: Start with demo_small, expand gradually

---

## 🎯 **Key Features**

### **✅ Production-Ready**
- **Containerized**: Complete Docker setup with service orchestration
- **Scalable**: Multi-worker Spark cluster with horizontal scaling
- **Monitored**: Comprehensive logging and analytics with Kibana
- **Reliable**: Health checks, auto-restart, and fallback mechanisms
- **Web Interface**: Complete command execution through dashboard
- **API-First**: HTTP API for programmatic command execution
- **Clean Architecture**: Separation of concerns with dedicated command server
- **Pipeline Management**: Real-time pipeline control with start/stop functionality
- **Signal Monitoring**: 4-day signal history with multi-timeframe support
- **Data Export**: CSV and JSON export functionality for signals

### **📊 Financial Data Processing - Timeframe-Agnostic**
- **Multi-Timeframe Fetching**: Yahoo Finance integration for 1min, 5min, 15min, 1hour, 1day data
- **Timeframe-Organized Storage**: Automatic organization by timeframe in MinIO S3 storage (`ohlcv/daily/`, `ohlcv/hourly/`, `ohlcv/minute/`)
- **Intelligent Data Source Selection**: `TimeframeAgnosticDataSource` with `YFinanceIntradaySource` implementation
- **Enhanced Storage Management**: `TimeframeEnhancedStorage` for optimal file organization and retrieval
- **Progress Tracking**: Real-time progress updates during processing with timeframe context
- **Error Handling**: Detailed error logging and recovery mechanisms with timeframe awareness

### **🔍 Advanced Monitoring - Timeframe-Enhanced**
- **Dual Logging**: PostgreSQL for pipeline metadata + Elasticsearch for detailed analytics (both timeframe-aware)
- **Real-time Dashboard**: Live web interface with auto-refreshing metrics and timeframe selection
- **Interactive Architecture**: D3.js visualization of system components with timeframe capabilities
- **Performance Metrics**: Duration tracking, success rates, error analysis per timeframe
- **Timeframe Analytics**: Database views for `timeframe_performance_stats`, `signals_summary_by_timeframe`
- **Search & Filter**: Powerful query capabilities across all pipeline data with timeframe filtering
- **Web Commands**: Execute all pipeline commands directly from dashboard with timeframe selection
- **Quick Flows**: Pre-configured pipeline flows (Demo/Small/Medium/Full) with timeframe options
- **HTTP API**: Programmatic access to all pipeline commands with timeframe parameters
- **Command Server**: Dedicated HTTP server for Spark command execution with timeframe support
- **Trading Signals Dashboard**: 4-day signal history with chronological ordering and create time tracking
- **Multi-Timeframe Signal Display**: Support for all timeframes (1min, 5min, 15min, 1hour, 1day)
- **Data Export**: CSV and JSON export functionality with comprehensive signal metadata

### **⚡ Developer Experience - Timeframe-Enhanced**
- **Streamlined CLIs**: Two primary CLIs with complete timeframe functionality (legacy files removed)
- **Web Interface**: Execute all commands through intuitive dashboard with timeframe selection dropdowns
- **HTTP API**: Direct programmatic access to all commands with timeframe parameters
- **Immediate Feedback**: Real-time progress and status updates with timeframe context
- **Easy Debugging**: Detailed logs with unique run IDs and timeframe tracking
- **Extensible Architecture**: Modular timeframe-agnostic components (`TimeframeAgnosticFetcher`, `TimeframeEnhancedStorage`, etc.)
- **Clean Codebase**: 25+ redundant files removed for maintainability
- **Container Communication**: Elegant HTTP-based inter-container communication with timeframe support
- **Flexible Configuration**: `TimeframeConfigManager` for centralized timeframe-specific parameters

### **🆕 New Abstraction System - Developer Experience**
- **Modular Architecture**: Complete abstraction system with interchangeable components
- **Workflow Management**: Complex multi-step process orchestration with dependency resolution
- **System Monitoring**: Real-time health checks and performance metrics
- **Enhanced Logging**: Structured logging with performance tracking
- **Error Recovery**: Comprehensive error handling and retry mechanisms
- **Dashboard Integration**: Seamless connection to new abstraction system
- **Testing Framework**: Comprehensive test suite for all components
- **Documentation**: Complete guides for implementation and testing

---

## 📈 **Performance Characteristics**

### **🎯 Tested Capabilities** (Updated 2025-01-24 - Dashboard Improvements)
- **Multi-Timeframe Processing**: Successfully processes 1day, 1hour data with automatic storage organization
- **Symbols**: Successfully processes 25+ symbols across multiple timeframes
- **Data Volume**: 251 records (daily), 1745 records (hourly) per symbol with optimized Parquet storage
- **Pipeline Runs**: Multiple runs tracked with timeframe metadata and >95% success rate
- **Processing Speed**: 1.3s (summary) to 28.9s (data fetch) per operation across all timeframes
- **Dashboard Updates**: Real-time metrics with PostgreSQL backend and timeframe selection UI
- **Web Commands**: Complete command execution through dashboard interface with timeframe dropdowns
- **HTTP API**: Command server with 100% uptime, <1s response times, and timeframe parameter support
- **Infrastructure**: 11 Docker containers running simultaneously (including enhanced database schema)
- **Timeframe Features**: Database schema with 5 new tables/views for timeframe tracking and analytics
- **Signal Generation**: Working across multiple timeframes with optimized parameters
- **Container Communication**: HTTP-based command execution with timeframe-aware parameters
- **Signal Monitoring**: 4-day signal history with chronological ordering and create time tracking
- **Multi-Timeframe Signal Display**: Support for all timeframes (1min, 5min, 15min, 1hour, 1day)
- **Data Export**: CSV and JSON export functionality with comprehensive signal metadata
- **Signal Performance**: 42+ signals displayed with multi-timeframe support and export capabilities
- **Dashboard Pages**: All 7 dashboard pages now fully functional with comprehensive content
- **System Health Monitoring**: Real-time infrastructure status with service health checks
- **UI/UX Improvements**: Direct HTML rendering, JavaScript integration, responsive design

### **⚡ Scaling Guidelines - Timeframe-Aware**
- **Small Scale**: 1-10 symbols, demo_small list, daily timeframe (fastest processing)
- **Medium Scale**: 10-50 symbols, demo_medium list, hourly/daily timeframes
- **Large Scale**: 50+ symbols, intraday timeframes (1hour, 15min), requires additional workers
- **High-Frequency**: 100+ symbols, minute-level timeframes (1min, 5min), dedicated infrastructure
- **Enterprise**: 500+ symbols, multi-timeframe concurrent processing, distributed cluster

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# 1. Clone repository
git clone <repository-url>
cd BreadthFlow

# 2. Start development environment
./scripts/start_infrastructure.sh

# 3. Make changes to CLI files
# Files are mounted as volumes, changes reflect immediately

# 4. Test changes
# Option A: Web Dashboard (Recommended)
# Go to http://localhost:8083 → Click "Commands" → Execute commands

# Option B: Direct API Testing
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "data_summary", "parameters": {}}' \
  http://localhost:8081/execute

# Option C: Scripts
./scripts/run_demo.sh

# Option D: CLI Commands
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/your_script.py

# 5. Submit pull request
```

### **Adding New Features**
1. **Create new CLI script** in `/cli` directory
2. **Add to docker volumes** in docker-compose.yml if needed
3. **Test thoroughly** with demo data
4. **Update documentation** and README
5. **Submit pull request** with examples

---

## 📚 **Documentation**

- **📊 Monitoring Guide**: `docs/monitoring_guide.md` - Complete monitoring setup
- **🎨 Kibana Dashboards**: `docs/kibana_dashboard_guide.md` - Custom visualization creation
- **🏗️ Infrastructure**: `docs/infrastructure_guide.md` - Detailed infrastructure setup
- **🎬 Demo Guide**: `docs/demo_guide.md` - Step-by-step demonstrations

### **🆕 New Abstraction System Documentation**
- **📋 Dashboard Connection Guide**: `DASHBOARD_CONNECTION_GUIDE.md` - Complete implementation guide
- **📊 Dashboard Connection Summary**: `DASHBOARD_CONNECTION_SUMMARY.md` - Implementation summary
- **📁 Implementation Status**: `IMPLEMENTATION_STATUS.md` - Detailed progress tracking
- **📋 Phase 5 Completion Summary**: `PHASE_5_COMPLETION_SUMMARY.md` - Final phase summary
- **🐳 Docker Testing Guide**: `DOCKER_TESTING_GUIDE.md` - Complete Docker integration testing guide

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🚀 Built with modern big data technologies for scalable financial analysis**

*For questions and support, check the documentation in `/docs` or open an issue.*