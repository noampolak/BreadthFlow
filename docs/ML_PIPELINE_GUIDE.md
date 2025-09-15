# 🤖 BreadthFlow ML Pipeline Guide

> Complete guide to using the ML pipeline for training, checking, and deploying models

## 🎯 **Overview**

The BreadthFlow ML Pipeline provides a complete machine learning workflow from data ingestion to production model serving. This guide shows you how to use all the components effectively.

---

## 🚀 **How to Actually Use the ML Pipeline - Step by Step**

### **🎯 Method 1: Ready-to-Use Jupyter Notebook (Recommended for Everyone)**

#### **Step 1: Open the Complete Example**
```bash
# Open Jupyter Lab
open http://localhost:8888
# Token: breadthflow123

# Navigate to: notebooks/multi_source_training_example.ipynb
# This notebook contains a complete end-to-end workflow
```

#### **What the Notebook Includes:**
- ✅ **Service Connectivity Testing** - Verifies all ML services are working
- ✅ **Configuration Loading** - Loads experiment settings from YAML
- ✅ **Sample Data Generation** - Creates realistic market data for demonstration
- ✅ **Feature Engineering** - Uses generic, reusable feature modules
- ✅ **Model Training** - Trains multiple algorithms using the ML pipeline
- ✅ **MLflow Integration** - Tracks experiments and models
- ✅ **Seldon Deployment** - Deploys models for production serving
- ✅ **Complete Workflow** - From data to deployed model in one notebook

### **🎯 Method 2: API-Based Workflow (For Advanced Users)**

#### **Step 1: Start the ML Platform**
```bash
# Start all ML services
docker-compose -f docker-compose.ml.yml up -d

# Check everything is running
docker-compose -f docker-compose.ml.yml ps
```

#### **Step 2: Test the Services**
```bash
# Test each service
curl http://localhost:8001/health  # Data Pipeline
curl http://localhost:8002/health  # Feature Engineering  
curl http://localhost:8003/health  # Model Training
curl http://localhost:8004/health  # AutoML
curl http://localhost:8005/health  # Model Serving
curl http://localhost:8006/health  # Model Registry
```

#### **Step 3: Train Your First Model**
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
curl -X POST http://localhost:8003/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL"],
    "timeframe": "1day",
    "algorithm": "random_forest"
  }'
```

### **🎯 Method 2: Using Jupyter Notebooks (Interactive Development)**

#### **Step 1: Access Jupyter Lab**
- Go to http://localhost:8888
- Use token: `breadthflow123`

#### **Step 2: Create a New Notebook**
```python
# Cell 1: Import libraries
import requests
import json
import pandas as pd

# Cell 2: Check services
services = {
    "data_pipeline": "http://localhost:8001/health",
    "feature_engineering": "http://localhost:8002/health",
    "model_training": "http://localhost:8003/health",
    "automl": "http://localhost:8004/health",
    "model_serving": "http://localhost:8005/health",
    "model_registry": "http://localhost:8006/health"
}

for name, url in services.items():
    try:
        response = requests.get(url)
        print(f"✅ {name}: {response.status_code}")
    except:
        print(f"❌ {name}: Not running")

# Cell 3: Train a model
def train_model(symbols, timeframe, algorithm="random_forest"):
    """Train a model using the ML pipeline API"""
    
    # Generate features first
    features_response = requests.post('http://localhost:8002/features/generate', 
        json={
            "symbols": symbols,
            "timeframe": timeframe,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
    )
    print(f"Features: {features_response.json()}")
    
    # Train the model
    train_response = requests.post('http://localhost:8003/train', 
        json={
            "symbols": symbols,
            "timeframe": timeframe,
            "algorithm": algorithm
        }
    )
    print(f"Training: {train_response.json()}")
    
    return train_response.json()

# Cell 4: Use it
results = train_model(["AAPL", "MSFT"], "1day", "random_forest")
print(results)
```

### **🎯 Method 3: Using the Existing CLI (Most Reliable)**

#### **Step 1: Use Existing Commands**
```bash
# Run the demo (this works right now)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py demo

# Or run specific steps
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL --timeframe 1day
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py signals generate --symbols AAPL --timeframe 1day
```

### **🎯 Method 4: Complete Workflow Example**

#### **Step 1: Data Pipeline**
```bash
# Fetch data
curl -X POST http://localhost:8001/data/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "timeframe": "1day",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

#### **Step 2: Feature Engineering**
```bash
# Generate features
curl -X POST http://localhost:8002/features/generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "timeframe": "1day"
  }'
```

#### **Step 3: Model Training**
```bash
# Train model
curl -X POST http://localhost:8003/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "timeframe": "1day",
    "algorithm": "random_forest"
  }'
```

#### **Step 4: Check Results**
- **MLflow**: http://localhost:5001 (View experiments)
- **Model Registry**: http://localhost:8006 (Check model versions)
- **Grafana**: http://localhost:3001 (Monitor performance)

#### **Step 5: Deploy Model**
```bash
# Deploy to production
curl -X POST http://localhost:8006/models/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "breadthflow-model",
    "version": "latest"
  }'
```

---

## 🤖 **Automated Model Training (One-Line Approach)**

### **The Ultimate Simple Workflow**
```python
# Test new trading ideas with automated ML pipeline
from model.automl.automl_manager import AutoMLTrainingManager

# Initialize the AutoML training manager
training_manager = AutoMLTrainingManager()

# Test your new idea in ONE line
results = training_manager.train_new_idea({
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "strategy": "momentum",  # Pre-built strategy template
    "timeframe": "1day",
    "auto_deploy": True  # Deploy if performance > threshold
})

# That's it! Everything else is automatic:
# ✅ Airflow fetches data
# ✅ Featuretools engineers features  
# ✅ AutoML trains multiple models
# ✅ MLflow tracks experiments
# ✅ Seldon deploys best model
```

---

## 🔧 **Feature Engineering**

### **Automated Feature Engineering**
```bash
# Test feature engineering service
curl http://localhost:8002/health

# Generate features for specific symbols
curl -X POST http://localhost:8002/features/generate \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"], "timeframe": "1day"}'
```

### **Available Feature Types**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, CCI, ADX, ATR
- **Time-based Features**: Cyclical, calendar, and seasonal feature generation
- **Market Microstructure**: Volume, price-volume, and order flow features
- **Automated Features**: Featuretools, Tsfresh, Feature-engine integration

---

## 📊 **Model Training**

### **Training APIs**
```bash
# Test model training service
curl http://localhost:8003/health

# Train a new model
curl -X POST http://localhost:8003/train \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day", "algorithm": "random_forest"}'
```

### **Available Algorithms**
- **Random Forest**: `random_forest`
- **XGBoost**: `xgboost`
- **LightGBM**: `lightgbm`
- **SVM**: `svm`
- **Logistic Regression**: `logistic_regression`
- **AutoML**: `automl` (uses Auto-sklearn, TPOT, H2O)

---

## 🚀 **Model Serving & A/B Testing**

### **Model Serving**
```bash
# Test model serving
curl http://localhost:8005/health

# Test model registry
curl http://localhost:8006/health

# Create A/B test
curl -X POST http://localhost:8006/ab-tests \
  -H "Content-Type: application/json" \
  -d '{"model_name": "breadthflow-model", "model_a_version": "v1", "model_b_version": "v2", "traffic_split": 0.5}'
```

### **Production Deployment**
- **Seldon Core**: Production-grade model serving platform
- **A/B Testing**: Traffic splitting and model comparison
- **Model Versioning**: Complete model lifecycle management
- **Health Monitoring**: Real-time model performance tracking

---

## 📈 **Monitoring & Visualization**

### **Real-time Monitoring**
```bash
# Test Grafana dashboards
curl http://localhost:3001/api/health

# Test Prometheus metrics
curl http://localhost:9090/api/v1/status/config

# Test MLflow experiment tracking
curl http://localhost:5001/health
```

### **Available Dashboards**
- **Grafana**: System and model performance dashboards
- **MLflow**: Experiment tracking and model registry
- **Prometheus**: Metrics collection and alerting
- **Kibana**: Advanced log analysis and visualization

---

## 🎯 **The Bottom Line**

**For beginners**: Use Method 1 (Quick Start) - it's the most straightforward
**For developers**: Use Method 2 (Jupyter) - it's interactive and flexible  
**For production**: Use Method 3 (CLI) - it's the most reliable
**For learning**: Use Method 4 (Complete Workflow) - it shows the full process

The key is that you now have **practical, copy-paste examples** for every step! 🚀

---

## 📚 **Next Steps**

- **Quick Start**: See [Quick Start Guide](QUICK_START.md)
- **Infrastructure**: See [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md)
- **API Reference**: See [API Reference](API_REFERENCE.md)
- **Troubleshooting**: See [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*This guide provides comprehensive instructions for using the ML pipeline. For quick setup, see the Quick Start Guide!*
