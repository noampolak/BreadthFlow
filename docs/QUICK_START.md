# ⚡ BreadthFlow Quick Start Guide

> Get up and running with the complete ML pipeline in 5 minutes!

## 🎯 **What You'll Get**
- **Complete ML Pipeline** - From data ingestion to model serving
- **20+ Docker Services** - All running simultaneously
- **Interactive Development** - Jupyter Lab for experimentation
- **Production Ready** - Seldon Core for model serving and A/B testing

---

## 🚀 **Quick Start (5 Minutes)**

### **Step 1: Start the ML Platform**
```bash
# Start all ML services
docker-compose -f docker-compose.ml.yml up -d

# Check everything is running
docker-compose -f docker-compose.ml.yml ps
```

### **Step 2: Test the Services**
```bash
# Test each service
curl http://localhost:8001/health  # Data Pipeline
curl http://localhost:8002/health  # Feature Engineering  
curl http://localhost:8003/health  # Model Training
curl http://localhost:8004/health  # AutoML
curl http://localhost:8005/health  # Model Serving
curl http://localhost:8006/health  # Model Registry
```

### **Step 3: Train Your First Model**

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
# Generate features first
curl -X POST http://localhost:8002/features/generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "timeframe": "1day",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'

# Train the model
curl -X POST http://localhost:8003/train-models \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "timeframe": "1day",
    "algorithm": "random_forest"
  }'
```

---

## 🎯 **Access Your Services**

### **🤖 ML Training & Experimentation**
- **MLflow UI**: http://localhost:5001 (Experiment tracking, model registry)
- **Jupyter Lab**: http://localhost:8888 (Interactive development, token: `breadthflow123`)
- **AutoML API**: http://localhost:8004 (Automated model training)
- **Model Training API**: http://localhost:8003 (Model training and hyperparameter optimization)
- **Feature Engineering API**: http://localhost:8002 (Automated feature generation)

### **🚀 Model Serving & Production**
- **Seldon Core**: http://localhost:8084 (Model serving and A/B testing platform)
- **Model Serving**: http://localhost:8005 (Production model endpoints)
- **Model Registry**: http://localhost:8006 (Model lifecycle management)

### **📈 Monitoring & Analytics**
- **Grafana**: http://localhost:3001 (System and model performance dashboards, admin/admin)
- **Prometheus**: http://localhost:9090 (Metrics collection and alerting)
- **Kibana**: http://localhost:5601 (Advanced log analysis and visualization)
- **Elasticsearch**: http://localhost:9200 (Search and analytics engine)

### **⚡ Data & Processing**
- **Kafka UI (Kafdrop)**: http://localhost:9002 (Streaming data & message monitoring)
- **MinIO Data Storage**: http://localhost:9001 (minioadmin/minioadmin)
- **Spark Cluster**: http://localhost:8080 (Processing status)
- **Spark Command API**: http://localhost:8081 (HTTP API for command execution)

---

## 🎯 **Quick Reference Commands**

| What You Want to Do | Command |
|---------------------|---------|
| **Start ML Platform** | `docker-compose -f docker-compose.ml.yml up -d` |
| **Check Services** | `docker-compose -f docker-compose.ml.yml ps` |
| **Train Model** | `curl -X POST http://localhost:8003/train -H "Content-Type: application/json" -d '{"symbols": ["AAPL"], "timeframe": "1day"}'` |
| **View Experiments** | Open http://localhost:5001 |
| **Monitor Performance** | Open http://localhost:3001 (admin/admin) |
| **Check Model Registry** | Open http://localhost:8006 |

---

## 🔧 **Troubleshooting**

### **Services Not Running**
```bash
# Check what's running
docker-compose -f docker-compose.ml.yml ps

# Restart specific service
docker-compose -f docker-compose.ml.yml restart model-training
```

### **API Calls Failing**
```bash
# Check service logs
docker-compose -f docker-compose.ml.yml logs model-training

# Test basic connectivity
curl http://localhost:8003/health
```

### **No Data Available**
```bash
# Check if data exists
curl http://localhost:8001/data/summary

# Fetch some data first
curl -X POST http://localhost:8001/data/fetch \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day"}'
```

---

## 📚 **Next Steps**

- **Complete ML Workflow**: See [ML Pipeline Guide](ML_PIPELINE_GUIDE.md)
- **Service Details**: See [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md)
- **API Reference**: See [API Reference](API_REFERENCE.md)
- **Troubleshooting**: See [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*This quick start gets you up and running in minutes. For detailed workflows and advanced usage, check the other guides!*
