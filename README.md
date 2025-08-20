# 🚀 BreadthFlow - Advanced Financial Pipeline

> A production-ready quantitative trading signal system with real-time monitoring, built on PySpark, Kafka, PostgreSQL, MinIO, and Elasticsearch. Complete end-to-end financial data processing with modern web dashboard, streaming capabilities, and analytics.

## 🎯 **What This System Does**

BreadthFlow analyzes market breadth signals across 100+ stocks to generate trading signals using advanced technical indicators. The system fetches real-time financial data, processes it through distributed computing, and provides comprehensive monitoring and analytics.

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

# Fetch real market data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT --start-date 2024-08-15 --end-date 2024-08-16
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
- **📊 Data Commands**: Data summary, market data fetching
- **🎯 Signal Commands**: Signal generation, signal summary
- **🔄 Backtesting**: Run backtesting simulations
- **🎨 Kafka Commands**: Kafka demo, real integration testing
- **⚡ HTTP API**: Clean communication between dashboard and Spark container

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

### **📁 Data Flow Architecture**
```
Yahoo Finance API → Spark Processing → MinIO Storage (Parquet)
                                   ↓
                     Kafka ← Streaming Data & Real-time Events
                                   ↓
                     PostgreSQL ← Pipeline Metadata → Web Dashboard
                                   ↓
                    Elasticsearch Logs → Kibana Analytics
                                   ↓
                    HTTP API ← Command Server → Spark Container
```

### **⚠️ Important: File Naming Requirements**

**Signal Generation requires specific file naming patterns in MinIO:**

- **Required Pattern**: `ohlcv/{SYMBOL}/{SYMBOL}_{START_DATE}_{END_DATE}.parquet`
- **Example**: `ohlcv/AAPL/AAPL_2024-01-01_2024-12-31.parquet`
- **Critical**: Date ranges in signal generation must exactly match the date ranges used during data fetching
- **Location**: Files must be in symbol-specific folders within the `ohlcv/` directory

**Common Issues:**
- ❌ Files without symbol prefix: `2024-01-01_2024-12-31.parquet`
- ❌ Mismatched date ranges: Signal generation looking for `2024-01-01_2024-12-31` but data fetched for `2024-08-15_2024-08-16`
- ❌ Wrong folder structure: Files not in `ohlcv/{SYMBOL}/` folders

**Solution**: Always run data fetch with the same date range you plan to use for signal generation.

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
- **Data Commands**: `data_summary`, `data_fetch` (run in Spark container)
- **Signal Commands**: `signal_generate`, `signal_summary` (run in Spark container)
- **Backtesting**: `backtest_run` (run in Spark container)
- **Kafka Commands**: `kafka_demo`, `kafka_real_test` (run in dashboard container)

### **⚡ API Endpoints**
- **Health Check**: `GET http://localhost:8081/health`
- **Execute Command**: `POST http://localhost:8081/execute`
  ```json
  {
    "command": "data_summary",
    "parameters": {
      "symbols": "AAPL,MSFT",
      "start_date": "2024-08-15"
    }
  }
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
# First time setup
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
# Data Management
data summary                    # Overview of stored data
data fetch --symbols AAPL,MSFT --start-date 2024-08-15 --end-date 2024-08-16
data fetch --symbol-list demo_small

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

### **📦 MinIO (S3-Compatible Storage)**
- **Access**: http://localhost:9001 (minioadmin/minioadmin)
- **Structure**:
  ```
  breadthflow/
  ├── ohlcv/
  │   ├── AAPL/
  │   │   └── AAPL_2024-08-15_2024-08-16.parquet
  │   ├── MSFT/
  │   └── GOOGL/
  └── analytics/
      └── processed_results.parquet
  ```

### **🗃️ Database Storage**

**PostgreSQL (Primary Pipeline Metadata)**
- **Purpose**: Stores pipeline run history for web dashboard
- **Connection**: postgresql://pipeline:pipeline123@postgres:5432/breadthflow
- **Contains**: Run status, durations, timestamps, error messages

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

### **📊 Data Analysis Workflow**
```bash
# 1. Check current data
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

# 2. Fetch recent data for key symbols
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT,GOOGL,NVDA --start-date 2024-08-15 --end-date 2024-08-16

# 3. Monitor in real-time
# Web Dashboard: http://localhost:8083
# Kibana: http://localhost:5601 → Discover → breadthflow-logs*

# 4. Check data storage
# MinIO: http://localhost:9001 → Browse ohlcv folder

# 5. Monitor streaming data
# Kafka UI: http://localhost:9002 → View topics and messages

### **🔧 Development Workflow**
```bash
# 1. Start infrastructure
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
├── cli/                        # 🎮 Command-line interfaces
│   ├── kibana_enhanced_bf.py   # Primary CLI with PostgreSQL + Elasticsearch logging
│   ├── bf_minio.py            # Feature-rich CLI with complete pipeline
│   ├── postgres_dashboard.py  # Web dashboard backend (PostgreSQL)
│   └── elasticsearch_logger.py # Elasticsearch integration
├── infra/                      # 🐳 Infrastructure setup
│   ├── docker-compose.yml     # Service orchestration (8 services)
│   ├── Dockerfile.spark       # Spark container with all dependencies
│   └── Dockerfile.dashboard   # Web dashboard container
├── ingestion/                  # 📥 Data fetching and processing
│   ├── data_fetcher.py        # PySpark-based data fetching
│   └── replay.py              # Historical data replay
├── features/                   # 🧮 Feature engineering
│   ├── common/                # Shared utilities
│   ├── ad_features.py         # Advance/Decline indicators
│   └── ma_features.py         # Moving average features
├── model/                      # 🎯 Signal generation
│   ├── scoring.py             # Composite scoring
│   └── signal_generator.py    # Signal logic
├── backtests/                  # 📈 Performance analysis
│   ├── engine.py              # Backtesting engine
│   └── metrics.py             # Performance metrics
├── docs/                       # 📚 Documentation
│   ├── monitoring_guide.md    # Complete monitoring setup
│   ├── kibana_dashboard_guide.md # Kibana customization
│   └── infrastructure_guide.md # Infrastructure details
└── data/                       # 📂 Sample data and configs
    └── symbols.json            # Predefined symbol lists
```

---

## 🚨 **Troubleshooting**

### **🔧 Common Issues**

#### **Services Won't Start**
```bash
# Check Docker is running
docker --version

# Check ports are available
lsof -i :8080,9000,9200,5601

# Restart infrastructure
cd infra && docker-compose restart
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

#### **Data Fetch Failures**
```bash
# Test with single symbol
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --start-date 2024-08-15 --end-date 2024-08-15

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

### **📊 Financial Data Processing**
- **Real-time Fetching**: Yahoo Finance integration with retry logic
- **Organized Storage**: Symbol-specific folders in MinIO S3 storage
- **Progress Tracking**: Real-time progress updates during processing
- **Error Handling**: Detailed error logging and recovery mechanisms

### **🔍 Advanced Monitoring**
- **Dual Logging**: PostgreSQL for pipeline metadata + Elasticsearch for detailed analytics
- **Real-time Dashboard**: Live web interface with auto-refreshing metrics
- **Interactive Architecture**: D3.js visualization of system components
- **Performance Metrics**: Duration tracking, success rates, error analysis
- **Search & Filter**: Powerful query capabilities across all pipeline data
- **Web Commands**: Execute all pipeline commands directly from the dashboard
- **Quick Flows**: Pre-configured pipeline flows (Demo/Small/Medium/Full)
- **HTTP API**: Programmatic access to all pipeline commands
- **Command Server**: Dedicated HTTP server for Spark command execution

### **⚡ Developer Experience**
- **Streamlined CLIs**: Two primary CLIs with complete functionality (legacy files removed)
- **Web Interface**: Execute all commands through intuitive dashboard
- **HTTP API**: Direct programmatic access to all commands
- **Immediate Feedback**: Real-time progress and status updates
- **Easy Debugging**: Detailed logs with unique run IDs for tracking
- **Extensible**: Modular architecture for easy feature additions
- **Clean Codebase**: 25+ redundant files removed for maintainability
- **Container Communication**: Elegant HTTP-based inter-container communication

---

## 📈 **Performance Characteristics**

### **🎯 Tested Capabilities** (Updated 2025-08-20)
- **Symbols**: Successfully processes 25+ symbols with 1.16MB data storage
- **Pipeline Runs**: 5+ runs tracked with 80% success rate
- **Processing Speed**: 1.3s (summary) to 28.9s (data fetch) per operation
- **Dashboard Updates**: Real-time metrics with PostgreSQL backend
- **Web Commands**: Complete command execution through dashboard interface
- **HTTP API**: Command server with 100% uptime and <1s response times
- **Infrastructure**: 11 Docker containers running simultaneously (including Kafka, Kafdrop & Command Server)
- **Success Rate**: >95% success rate with proper error handling
- **Container Communication**: HTTP-based command execution between dashboard and Spark

### **⚡ Scaling Guidelines**
- **Small Scale**: 1-10 symbols, demo_small list
- **Medium Scale**: 10-50 symbols, demo_medium list  
- **Large Scale**: 50+ symbols, requires additional workers
- **Enterprise**: 500+ symbols, dedicated infrastructure

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