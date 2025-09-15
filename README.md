# 🚀 BreadthFlow - Advanced Financial Pipeline with Complete ML Training System

> A production-ready quantitative trading signal system with **complete ML pipeline**, **automated model training**, **model serving**, and **A/B testing capabilities**. Built on PySpark, Kafka, PostgreSQL, MinIO, Elasticsearch, MLflow, Seldon Core, and Grafana with a modern web dashboard, streaming capabilities, multi-timeframe analytics, and comprehensive machine learning infrastructure.

## 🚀 **Quick Start (5 Minutes)**

- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running in minutes
- **[ML Pipeline Guide](docs/ML_PIPELINE_GUIDE.md)** - Complete ML workflow
- **[Infrastructure Guide](docs/INFRASTRUCTURE_GUIDE.md)** - Services and architecture

## 📚 **Documentation**

- **[API Reference](docs/API_REFERENCE.md)** - All API endpoints
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and fixes

---

## 🎯 **What This System Does**

BreadthFlow analyzes market breadth signals across 100+ stocks to generate trading signals using advanced technical indicators. The system fetches real-time financial data across **multiple timeframes** (1min, 5min, 15min, 1hour, 1day), processes it through distributed computing, and provides comprehensive monitoring and analytics with timeframe-specific optimizations.

## 🆕 **Complete ML Training Pipeline System**

BreadthFlow now features a **complete machine learning training pipeline** that includes:
- **🤖 Automated Model Training** - Auto-sklearn, TPOT, H2O AutoML integration
- **🔧 Feature Engineering** - Automated feature generation with Featuretools, Tsfresh, Feature-engine
- **📊 Experiment Tracking** - MLflow for model versioning and experiment management
- **🚀 Model Serving** - Seldon Core for production model deployment and A/B testing
- **📈 Monitoring & Visualization** - Grafana dashboards with Prometheus metrics
- **🔄 Data Orchestration** - Apache Airflow for workflow management
- **⚡ Real-time Processing** - Apache Spark for distributed data processing
- **📦 Model Registry** - Complete model lifecycle management
- **🎯 Ready-to-Use Examples** - Complete Jupyter notebook with multi-source analysis
- **✅ Fully Tested & Working** - All services connected and operational

---

## 🏗️ **Architecture Overview**

### **🐳 Docker Services (20+ Containers)**

#### **⚡ Core Data Processing**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Spark Master** | 8080 | Distributed processing coordinator | http://localhost:8080 |
| **PostgreSQL** | 5432 | Pipeline metadata & run tracking | Database only |
| **Kafka** | 9092/9094 | Streaming platform & message broker | http://localhost:9002 |
| **MinIO** | 9000/9001 | S3-compatible data storage | http://localhost:9001 |
| **Elasticsearch** | 9200 | Search and analytics engine | http://localhost:9200 |
| **Kibana** | 5601 | Data visualization & dashboards | http://localhost:5601 |
| **Web Dashboard** | 8083 | Real-time pipeline monitoring | http://localhost:8083 |

#### **🤖 ML Training & Experimentation**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **MLflow** | 5001 | Experiment tracking & model registry | http://localhost:5001 |
| **Jupyter Lab** | 8888 | Interactive development environment | http://localhost:8888 |
| **AutoML API** | 8004 | Automated model training | http://localhost:8004 |
| **Model Training API** | 8003 | Model training & hyperparameter optimization | http://localhost:8003 |
| **Feature Engineering API** | 8002 | Automated feature generation | http://localhost:8002 |
| **Data Pipeline API** | 8001 | Data ingestion & validation | http://localhost:8001 |

#### **🚀 Model Serving & Production**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Seldon Core** | 8084 | Model serving & A/B testing platform | http://localhost:8084 |
| **Model Serving** | 8005 | Production model endpoints | http://localhost:8005 |
| **Model Registry** | 8006 | Model lifecycle management | http://localhost:8006 |

#### **📈 Monitoring & Visualization**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Grafana** | 3001 | System & model performance dashboards | http://localhost:3001 |
| **Prometheus** | 9090 | Metrics collection & alerting | http://localhost:9090 |

---

## 🎉 **Complete ML Pipeline Implementation Status**

### **✅ All Phases Completed Successfully!**

| Phase | Status | Description | Services |
|-------|--------|-------------|----------|
| **Phase 1** | ✅ **COMPLETED** | Foundation & Data Pipeline | MinIO, Spark, PostgreSQL, Redis |
| **Phase 1.5** | ✅ **COMPLETED** | Data Orchestration with Airflow | Apache Airflow, Data Pipeline API |
| **Phase 2** | ✅ **COMPLETED** | Feature Engineering | Feature Engineering API, Technical Indicators |
| **Phase 2.5** | ✅ **COMPLETED** | Automated Feature Engineering | Featuretools, Tsfresh, Feature-engine |
| **Phase 3** | ✅ **COMPLETED** | Model Training & Experimentation | MLflow, Jupyter, Model Training API |
| **Phase 3.5** | ✅ **COMPLETED** | AutoML Integration | Auto-sklearn, TPOT, H2O AutoML |
| **Phase 4** | ✅ **COMPLETED** | Monitoring & Visualization | Grafana, Prometheus, Elasticsearch, Kibana |
| **Phase 4.5** | ✅ **COMPLETED** | Model Serving & A/B Testing | Seldon Core, Model Registry, Model Serving |

### **🚀 Production-Ready ML Platform**

**The complete ML training pipeline is now operational and ready for production use!**

- **🤖 20+ Docker Services** running simultaneously
- **📊 Complete ML Workflow** from data ingestion to model serving
- **🔧 Automated Feature Engineering** with multiple open-source tools
- **📈 Comprehensive Monitoring** with Grafana dashboards and Prometheus metrics
- **🚀 Production Model Serving** with Seldon Core and A/B testing capabilities
- **⚡ One-Line Testing** for new trading ideas with automated ML pipeline
- **✅ All Issues Fixed** - Service connectivity, config paths, and circular references resolved
- **🎯 Ready-to-Use Examples** - Complete Jupyter notebook with multi-source analysis

---

## 🎯 **Key Features**

### **✅ Production-Ready**
- **Containerized**: Complete Docker setup with service orchestration
- **Scalable**: Multi-worker Spark cluster with horizontal scaling
- **Monitored**: Comprehensive logging and analytics with Kibana
- **Reliable**: Health checks, auto-restart, and fallback mechanisms
- **Web Interface**: Complete command execution through dashboard
- **API-First**: HTTP API for programmatic command execution

### **🤖 Complete ML Pipeline**
- **Automated Model Training**: Auto-sklearn, TPOT, H2O AutoML integration
- **Feature Engineering**: Automated feature generation with Featuretools, Tsfresh, Feature-engine
- **Experiment Tracking**: MLflow for model versioning and experiment management
- **Model Serving**: Seldon Core for production model deployment and A/B testing
- **Monitoring**: Grafana dashboards with Prometheus metrics
- **Data Orchestration**: Apache Airflow for workflow management

### **📊 Financial Data Processing**
- **Multi-Timeframe Fetching**: Yahoo Finance integration for 1min, 5min, 15min, 1hour, 1day data
- **Timeframe-Organized Storage**: Automatic organization by timeframe in MinIO S3 storage
- **Intelligent Data Source Selection**: Multiple data sources with automatic selection
- **Enhanced Storage Management**: Optimized file organization and retrieval
- **Progress Tracking**: Real-time progress updates during processing

---

## 🚀 **Getting Started**

### **1. Quick Start**
```bash
# Start all ML services
docker-compose -f docker-compose.ml.yml up -d

# Check services
docker-compose -f docker-compose.ml.yml ps
```

### **2. Access Services**
- **MLflow**: http://localhost:5001 (Experiment tracking)
- **Jupyter Lab**: http://localhost:8888 (Interactive development)
- **Grafana**: http://localhost:3001 (Monitoring dashboards)
- **Web Dashboard**: http://localhost:8083 (Pipeline monitoring)

### **3. Train Your First Model**

#### **Option A: Use the Ready-to-Use Jupyter Notebook (Recommended)**
```bash
# Open Jupyter Lab
open http://localhost:8888
# Token: breadthflow123

# Navigate to: notebooks/multi_source_training_example.ipynb
# Run all cells for complete end-to-end workflow
```

#### **Option B: Use API Commands**
```bash
# Generate features
curl -X POST http://localhost:8002/features/generate \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day"}'

# Train model
curl -X POST http://localhost:8003/train-models \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day", "algorithm": "random_forest"}'
```

---

## 📚 **Documentation**

- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running in 5 minutes
- **[ML Pipeline Guide](docs/ML_PIPELINE_GUIDE.md)** - Complete ML workflow instructions
- **[Infrastructure Guide](docs/INFRASTRUCTURE_GUIDE.md)** - Services and architecture details
- **[API Reference](docs/API_REFERENCE.md)** - All API endpoints and usage examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

---

## 🎯 **The Bottom Line**

**For beginners**: Use the [Quick Start Guide](docs/QUICK_START.md) - it's the most straightforward
**For developers**: Use the [ML Pipeline Guide](docs/ML_PIPELINE_GUIDE.md) - it's interactive and flexible  
**For production**: Use the [Infrastructure Guide](docs/INFRASTRUCTURE_GUIDE.md) - it's the most reliable
**For learning**: Use the [API Reference](docs/API_REFERENCE.md) - it shows the full process

The key is that you now have **practical, copy-paste examples** for every step! 🚀

---

## 🆘 **Getting Help**

- **Documentation**: Check the guides in `/docs`
- **Logs**: Use `docker-compose logs <service>`
- **Community**: Stack Overflow, GitHub issues
- **Monitoring**: Check Grafana dashboards for system health

---

**🚀 Built with modern big data technologies for scalable financial analysis**

*For questions and support, check the documentation in `/docs` or open an issue.*
