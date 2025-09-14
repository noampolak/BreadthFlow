# 🏗️ BreadthFlow Infrastructure Guide

> Complete guide to the infrastructure, services, and architecture

## 🎯 **Overview**

BreadthFlow runs on 20+ Docker services providing complete ML pipeline capabilities.

---

## 🐳 **Docker Services**

### **⚡ Core Data Processing**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Spark Master** | 8080 | Distributed processing | http://localhost:8080 |
| **PostgreSQL** | 5432 | Pipeline metadata | Database only |
| **Kafka** | 9092/9094 | Streaming platform | http://localhost:9002 |
| **MinIO** | 9000/9001 | S3-compatible storage | http://localhost:9001 |
| **Elasticsearch** | 9200 | Search engine | http://localhost:9200 |
| **Kibana** | 5601 | Data visualization | http://localhost:5601 |
| **Web Dashboard** | 8083 | Pipeline monitoring | http://localhost:8083 |

### **🤖 ML Training & Experimentation**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **MLflow** | 5001 | Experiment tracking | http://localhost:5001 |
| **Jupyter Lab** | 8888 | Interactive development | http://localhost:8888 |
| **AutoML API** | 8004 | Automated training | http://localhost:8004 |
| **Model Training API** | 8003 | Model training | http://localhost:8003 |
| **Feature Engineering API** | 8002 | Feature generation | http://localhost:8002 |
| **Data Pipeline API** | 8001 | Data ingestion | http://localhost:8001 |

### **🚀 Model Serving & Production**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Seldon Core** | 8084 | Model serving | http://localhost:8084 |
| **Model Serving** | 8005 | Production endpoints | http://localhost:8005 |
| **Model Registry** | 8006 | Model lifecycle | http://localhost:8006 |

### **📈 Monitoring & Visualization**
| Service | Port | Purpose | UI Access |
|---------|------|---------|-----------|
| **Grafana** | 3001 | Performance dashboards | http://localhost:3001 |
| **Prometheus** | 9090 | Metrics collection | http://localhost:9090 |

---

## 📁 **ML Pipeline Architecture**

```
Data Sources → Apache Airflow → MinIO Storage
     ↓
Data Pipeline → Feature Engineering → PostgreSQL
     ↓
Model Training → MLflow → Model Registry
     ↓
Seldon Core → Production Deployment
     ↓
Grafana ← Prometheus ← Monitoring
```

---

## 📊 **Data Storage**

- **MinIO**: S3-compatible object storage (http://localhost:9001)
- **PostgreSQL**: Pipeline metadata and run tracking
- **Elasticsearch**: Advanced logs and analytics
- **Kafka**: Real-time data streaming

---

## 📚 **Next Steps**

- **Quick Start**: See [Quick Start Guide](QUICK_START.md)
- **ML Pipeline**: See [ML Pipeline Guide](ML_PIPELINE_GUIDE.md)
- **API Reference**: See [API Reference](API_REFERENCE.md)
- **Troubleshooting**: See [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*This guide provides infrastructure details. For quick setup, see the Quick Start Guide!*