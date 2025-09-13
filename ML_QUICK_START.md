# 🚀 ML Training Pipeline - Quick Start Guide

## 🎯 **Quick Setup (5 minutes)**

### **1. Start ML Services**
```bash
# Start all ML services (including new enhanced services)
docker-compose -f docker-compose.ml.yml up -d

# Check status
docker-compose -f docker-compose.ml.yml ps
```

### **2. Access Services**
| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **Jupyter** | http://localhost:8888 | Token: `breadthflow123` | Interactive development |
| **MLflow** | http://localhost:5001 | No auth | Experiment tracking |
| **Grafana** | http://localhost:3001 | admin/admin | Monitoring dashboards |
| **Prometheus** | http://localhost:9090 | No auth | Metrics collection |
| **Spark UI** | http://localhost:8080 | No auth | Data processing |
| **MinIO** | http://localhost:9001 | admin/password123 | Object storage |
| **Kibana** | http://localhost:5601 | No auth | Log analysis |
| **Elasticsearch** | http://localhost:9200 | No auth | Search & analytics |
| **Airflow** | http://localhost:8081 | admin/admin | Data orchestration |
| **Data Pipeline** | http://localhost:8001 | No auth | Data ingestion API |
| **Feature Engineering** | http://localhost:8002 | No auth | Automated feature engineering |
| **Model Training** | http://localhost:8003 | No auth | Model training API |
| **AutoML** | http://localhost:8004 | No auth | Automated model training |

### **3. Test Setup**
```bash
# Test MLflow
curl http://localhost:5001

# Test Grafana
curl http://localhost:3001/api/health

# Test Prometheus
curl http://localhost:9090/api/v1/status/config

# Test Spark
curl http://localhost:8080

# Test MinIO
curl http://localhost:9000/minio/health/live

# Test Elasticsearch
curl http://localhost:9200

# Test Kibana
curl http://localhost:5601

# Test Airflow
curl http://localhost:8081/health

# Test Data Pipeline
curl http://localhost:8001/health

# Test Feature Engineering
curl http://localhost:8002/health

# Test Model Training
curl http://localhost:8003/health

# Test AutoML
curl http://localhost:8004/health
```

## 🚀 **Ultra-Simple New Idea Testing (After Full Implementation)**

### **The One-Line Workflow**
```python
# After all phases are complete, testing a new idea becomes this simple:

from model.training.automl_training_manager import AutoMLTrainingManager

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

### **What Happens Automatically**
1. **Airflow** fetches data for new symbols
2. **Featuretools** engineers features automatically
3. **AutoML** trains multiple models in parallel
4. **MLflow** tracks all experiments
5. **Seldon** compares and ranks models
6. **Seldon** deploys best model if performance threshold met

---

## 📊 **Phase 1: Data Pipeline (Week 1)**

### **Start with Basic Services**
```bash
# Start only essential services
docker-compose -f docker-compose.ml.yml up -d postgres redis minio spark-master spark-worker
```

### **Create Data Directory**
```bash
mkdir -p data/{raw,processed,features,models}
mkdir -p notebooks
mkdir -p mlflow
mkdir -p pipelines
mkdir -p deployment
```

### **Test Data Ingestion**
```python
# In Jupyter notebook
import pandas as pd
import requests

# Test API connection
response = requests.get('http://localhost:8005/api/dashboard/summary')
print(response.json())

# Test new data pipeline
response = requests.get('http://localhost:8001/health')
print("Data Pipeline:", response.json())
```

## 🔧 **Phase 2: Feature Engineering (Week 2)**

### **Install Python Dependencies**
```bash
# Create requirements.txt for ML
cat > ml-requirements.txt << EOF
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
mlflow==2.5.0
feast==0.34.0
great-expectations==0.17.0
plotly==5.15.0
jupyter==1.0.0
optuna==3.3.0
xgboost==1.7.6
lightgbm==4.0.0
EOF
```

### **Start Feature Engineering Services**
```bash
# Add MLflow and Jupyter
docker-compose -f docker-compose.ml.yml up -d mlflow jupyter

# Add new data pipeline service
docker-compose -f docker-compose.ml.yml up -d data-pipeline
```

### **Test Open Source Feature Engineering**
```python
# In Jupyter notebook - test open source feature engineering
import featuretools as ft
import tsfresh
from feature_engine import selection
import pandas as pd

# Load sample data
df = pd.read_csv('data/processed/sample_data.csv')

# Test Featuretools
es = ft.EntitySet(id="trading_data")
es = es.add_dataframe(df, index="timestamp")
featuretools_features, feature_defs = ft.dfs(entityset=es, target_dataframe_name="trading_data")

# Test Tsfresh
tsfresh_features = tsfresh.extract_features(df, column_id="symbol", column_sort="timestamp")

# Test Feature-engine
selector = selection.SelectKBestFeatures(k=50)
selected_features = selector.fit_transform(featuretools_features, df['target'])

print(f"Featuretools features: {len(featuretools_features.columns)}")
print(f"Tsfresh features: {len(tsfresh_features.columns)}")
print(f"Selected features: {len(selected_features.columns)}")
```

## 🤖 **Phase 3: Model Training (Week 3)**

### **Create MLflow Experiment**
```python
# In Jupyter notebook
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
mlflow.create_experiment("breadthflow_training")
mlflow.set_experiment("breadthflow_training")
```

### **Test AutoML Integration**
```python
# Test the AutoML training manager
from model.training.automl_training_manager import AutoMLTrainingManager
import autosklearn.classification
import tpot
import h2o
from h2o.automl import H2OAutoML

# Initialize AutoML training manager
training_manager = AutoMLTrainingManager()

# Test with a simple configuration
config = {
    "symbols": ["AAPL"],
    "strategy": "momentum",
    "timeframe": "1day",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "auto_deploy": False  # Don't deploy for testing
}

# Run training with AutoML (this will take a few minutes)
results = training_manager.train_new_idea(config)

print(f"AutoML training completed!")
print(f"Best model accuracy: {results.best_accuracy}")
print(f"Models trained: {results.models_trained}")
print(f"Best algorithm: {results.best_algorithm}")
print(f"Backtest Sharpe: {results.sharpe_ratio}")
```

### **Test Traditional Training (Fallback)**
```python
# Simple training example (if smart training not ready)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
# df = pd.read_csv('data/processed/training_data.csv')

# Start MLflow run
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    # model.fit(X_train, y_train)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log metrics
    # mlflow.log_metric("accuracy", accuracy)
```

## 📊 **Phase 4: Monitoring (Week 4)**

### **Start Monitoring Services**
```bash
# Add monitoring stack
docker-compose -f docker-compose.ml.yml up -d grafana prometheus elasticsearch kibana

# Add model deployment service
docker-compose -f docker-compose.ml.yml up -d model-deployment
```

### **Configure Grafana**
1. Go to http://localhost:3000
2. Login with admin/admin
3. Add Prometheus data source: http://prometheus:9090
4. Import ML dashboard

### **Test Seldon Core Deployment**
```python
# Test Seldon Core model serving
from seldon_core import SeldonClient
import mlflow
import mlflow.sklearn

# Load model from MLflow
model_uri = "models:/breadthflow-trading/Production"
model = mlflow.sklearn.load_model(model_uri)

# Deploy with Seldon Core
seldon_client = SeldonClient()
deployment_result = seldon_client.deploy(
    name="breadthflow-trading",
    model=model,
    replicas=3
)

print(f"Seldon deployment status: {deployment_result.status}")
print(f"Endpoint: {deployment_result.endpoint}")
print(f"Replicas: {deployment_result.replicas}")
```

### **Test A/B Testing with Seldon**
```python
# Test A/B testing between models with Seldon
ab_test_result = seldon_client.create_ab_test(
    name="breadthflow-ab-test",
    model_a="breadthflow-model-v1",
    model_b="breadthflow-model-v2",
    traffic_split=0.5,  # 50/50 split
    duration_hours=24
)

print(f"A/B test started: {ab_test_result.test_id}")
print(f"Monitor at: http://localhost:8003/ab-tests/{ab_test_result.test_id}")
```

## 🛠️ **Development Workflow**

### **Daily Development**
```bash
# Start development environment
docker-compose -f docker-compose.ml.yml up -d

# Work in Jupyter
# http://localhost:8888

# Track experiments in MLflow
# http://localhost:5000

# Monitor in Grafana
# http://localhost:3000

# Manage data orchestration
# http://localhost:8081

# Manage feature engineering
# http://localhost:8001

# Manage AutoML
# http://localhost:8002

# Manage model serving
# http://localhost:8003
```

### **Testing New Ideas (After Full Implementation)**
```python
# The ultimate simple workflow for testing new ideas:

# 1. Start everything (30 seconds)
# docker-compose -f docker-compose.ml.yml up -d

# 2. Open Jupyter (http://localhost:8888)
# 3. Run this ONE cell:

from model.training.automl_training_manager import AutoMLTrainingManager

training_manager = AutoMLTrainingManager()

# Test your new idea
results = training_manager.train_new_idea({
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "strategy": "momentum",  # or "mean_reversion", "ensemble"
    "timeframe": "1day",
    "auto_deploy": True
})

# 4. Check results in MLflow: http://localhost:5000
# 5. Monitor in Grafana: http://localhost:3000
# 6. Deploy if good: http://localhost:8003

# That's it! From idea to production in minutes.
```

### **Data Pipeline Development**
```python
# In Jupyter notebook
import pandas as pd
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("BreadthFlowML") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# Read data
df = spark.read.parquet("data/raw/ohlcv/")

# Process data
processed_df = df.filter(df.volume > 0)

# Save processed data
processed_df.write.mode("overwrite").parquet("data/processed/")
```

## 🚨 **Troubleshooting**

### **Common Issues**

**Port Conflicts**
```bash
# Check what's using ports
lsof -i :5000
lsof -i :8888

# Stop conflicting services
docker stop $(docker ps -q)
```

**Memory Issues**
```bash
# Check Docker resources
docker system df
docker system prune

# Increase Docker memory limit in Docker Desktop
```

**Service Not Starting**
```bash
# Check logs
docker-compose -f docker-compose.ml.yml logs mlflow
docker-compose -f docker-compose.ml.yml logs spark-master

# Restart specific service
docker-compose -f docker-compose.ml.yml restart mlflow
```

### **Performance Optimization**

**For Large Datasets**
```bash
# Increase Spark memory
# Edit docker-compose.ml.yml
environment:
  - SPARK_WORKER_MEMORY=4G
  - SPARK_WORKER_CORES=4
```

**For GPU Training**
```bash
# Add GPU support to MLflow service
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📚 **Next Steps**

1. **Week 1**: Set up basic data pipeline
2. **Week 2.5**: Integrate Apache Airflow for data orchestration
3. **Week 2**: Implement feature engineering
4. **Week 3.5**: Integrate Featuretools and Tsfresh for automated feature engineering
5. **Week 3**: Create model training pipeline
6. **Week 5.5**: Integrate AutoML tools (auto-sklearn, TPOT, H2O)
7. **Week 4**: Add monitoring and visualization
8. **Week 7.5**: Integrate Seldon Core for model serving and A/B testing

## 🎯 **The Ultimate Goal**

After all phases are complete, testing a new trading idea becomes:

```python
# ONE LINE to test any new idea
results = training_manager.train_new_idea({
    "symbols": ["YOUR_SYMBOLS"],
    "strategy": "YOUR_STRATEGY", 
    "auto_deploy": True
})
```

**Time from idea to production: <15 minutes**
**Number of clicks: <3 clicks**
**Everything automated: Airflow → Featuretools → AutoML → MLflow → Seldon**

## 🆘 **Getting Help**

- **Documentation**: Check the main implementation plan
- **Logs**: Use `docker-compose logs <service>`
- **Community**: Stack Overflow, GitHub issues
- **Monitoring**: Check Grafana dashboards for system health

---

*This quick start guide gets you up and running in minutes. Follow the phases for a complete implementation.*
