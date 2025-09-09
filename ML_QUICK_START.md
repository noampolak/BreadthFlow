# 🚀 ML Training Pipeline - Quick Start Guide

## 🎯 **Quick Setup (5 minutes)**

### **1. Start ML Services**
```bash
# Start all ML services
docker-compose -f docker-compose.ml.yml up -d

# Check status
docker-compose -f docker-compose.ml.yml ps
```

### **2. Access Services**
| Service | URL | Credentials |
|---------|-----|-------------|
| **Jupyter** | http://localhost:8888 | Token: `breadthflow123` |
| **MLflow** | http://localhost:5000 | No auth |
| **Grafana** | http://localhost:3000 | admin/admin |
| **Spark UI** | http://localhost:8080 | No auth |
| **MinIO** | http://localhost:9001 | admin/password123 |
| **Kibana** | http://localhost:5601 | No auth |

### **3. Test Setup**
```bash
# Test MLflow
curl http://localhost:5000

# Test Spark
curl http://localhost:8080

# Test MinIO
curl http://localhost:9000/minio/health/live
```

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
```

### **Test Data Ingestion**
```python
# In Jupyter notebook
import pandas as pd
import requests

# Test API connection
response = requests.get('http://localhost:8005/api/dashboard/summary')
print(response.json())
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
EOF
```

### **Start Feature Engineering Services**
```bash
# Add MLflow and Jupyter
docker-compose -f docker-compose.ml.yml up -d mlflow jupyter
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

### **Test Model Training**
```python
# Simple training example
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
```

### **Configure Grafana**
1. Go to http://localhost:3000
2. Login with admin/admin
3. Add Prometheus data source: http://prometheus:9090
4. Import ML dashboard

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
2. **Week 2**: Implement feature engineering
3. **Week 3**: Create model training pipeline
4. **Week 4**: Add monitoring and visualization

## 🆘 **Getting Help**

- **Documentation**: Check the main implementation plan
- **Logs**: Use `docker-compose logs <service>`
- **Community**: Stack Overflow, GitHub issues
- **Monitoring**: Check Grafana dashboards for system health

---

*This quick start guide gets you up and running in minutes. Follow the phases for a complete implementation.*
