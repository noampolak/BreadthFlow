# 🚀 BreadthFlow - Advanced Financial Pipeline

> A production-ready quantitative trading signal system with **timeframe-agnostic capabilities**, real-time monitoring, built on PySpark, Kafka, PostgreSQL, MinIO, and Elasticsearch. Complete end-to-end financial data processing with modern web dashboard, streaming capabilities, and multi-timeframe analytics (1min, 5min, 15min, 1hour, 1day).

## 🎯 **What This System Does**

BreadthFlow analyzes market breadth signals across 100+ stocks to generate trading signals using advanced technical indicators. The system fetches real-time financial data across **multiple timeframes** (1min, 5min, 15min, 1hour, 1day), processes it through distributed computing, and provides comprehensive monitoring and analytics with timeframe-specific optimizations.

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

# Option 3: CLI Commands
# Get data summary
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

# Fetch real market data (multiple timeframes supported)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT --start-date 2024-08-15 --end-date 2024-08-16 --timeframe 1day --data-source yfinance

# Fetch intraday data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --timeframe 1hour --data-source yfinance
```

### **4. Access Monitoring & UIs**
- **🎯 Real-time Dashboard**: http://localhost:8083 (Pipeline monitoring, Infrastructure overview & **Commands execution**)
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
- **🎨 Kafka Commands**: Kafka demo, real integration testing
- **⚡ HTTP API**: Clean communication between dashboard and Spark container
- **⏰ Timeframe Support**: 1min, 5min, 15min, 1hour, 1day with optimized parameters

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

## 📊 **Monitoring & Analytics**

### **🎯 Real-time Web Dashboard** (Primary)
1. **Open Dashboard**: http://localhost:8083
2. **Main Features**:
   - **Pipeline Metrics**: Total runs, success rate, average duration
   - **Live Updates**: Auto-refreshing statistics every 30 seconds
   - **Recent Runs**: Detailed view of latest pipeline executions
   - **Infrastructure Overview**: Interactive D3.js architecture diagram

**Dashboard Pages:**
- **📊 Main Dashboard**: Live pipeline metrics and run history
- **🏗️ Infrastructure**: System architecture visualization with 8 services

### **🔍 Kibana Analytics** (Advanced)
1. **Open Kibana**: http://localhost:5601
2. **Go to Discover**: Click "Discover" in left menu
3. **Select Index**: Choose "breadthflow-logs*"
4. **View Data**: See all pipeline logs with filtering/searching

**Pre-built Dashboards:**
- **🚀 BreadthFlow Working Dashboard**: Real-time pipeline monitoring
- **🔍 Index Pattern**: `breadthflow-logs*` for custom visualizations

### **📈 What You Can Monitor**
- **Pipeline Success Rates**: Track successful vs failed runs
- **Performance Trends**: Monitor execution durations over time
- **Symbol Processing**: Success/failure by individual stocks
- **Error Analysis**: Detailed error logs and patterns
- **Real-time Activity**: Live updates as pipelines execute
- **PostgreSQL Metrics**: Pipeline run history with metadata

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
  │   └── signals_YYYYMMDD_HHMMSS.parquet (timeframe-aware)
  └── analytics/
      └── processed_results.parquet
  ```

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

# 5. Monitor in real-time
# Web Dashboard: http://localhost:8083 (timeframe-aware interface)
# Kibana: http://localhost:5601 → Discover → breadthflow-logs*

# 6. Check timeframe-organized data storage
# MinIO: http://localhost:9001 → Browse ohlcv/daily/, ohlcv/hourly/, ohlcv/minute/

# 7. Monitor streaming data
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
# Kibana: http://localhost:5601
# Kafka UI: http://localhost:9002
# Spark UI: http://localhost:8080
# Command API: http://localhost:8081

# 4. Stop when done
./scripts/stop_infrastructure.sh

# 5. Restart with all features
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

---

## 📈 **Performance Characteristics**

### **🎯 Tested Capabilities** (Updated 2025-08-21 - Timeframe-Agnostic)
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

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🚀 Built with modern big data technologies for scalable financial analysis**

*For questions and support, check the documentation in `/docs` or open an issue.*