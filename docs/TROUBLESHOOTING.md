# 🔧 BreadthFlow Troubleshooting Guide

> Common issues and solutions for the ML pipeline

## 🎯 **Quick Diagnostics**

### **Check All Services**
```bash
# Check what's running
docker-compose -f docker-compose.ml.yml ps

# Check service health
curl http://localhost:8001/health  # Data Pipeline
curl http://localhost:8002/health  # Feature Engineering  
curl http://localhost:8003/health  # Model Training
curl http://localhost:8004/health  # AutoML
curl http://localhost:8005/health  # Model Serving
curl http://localhost:8006/health  # Model Registry
```

---

## 🚨 **Common Issues**

### **Services Not Starting**

#### **Problem**: Services won't start or keep crashing
```bash
# Check Docker is running
docker --version

# Check ports are available
lsof -i :8001,8002,8003,8004,8005,8006

# Check Docker resources
docker system df
docker system prune
```

#### **Solution**:
```bash
# Restart Docker Desktop
# Or restart specific service
docker-compose -f docker-compose.ml.yml restart model-training

# Check service logs
docker-compose -f docker-compose.ml.yml logs model-training
```

### **API Calls Failing**

#### **Problem**: API endpoints return connection refused
```bash
# Test basic connectivity
curl http://localhost:8003/health

# Check if service is running
docker-compose -f docker-compose.ml.yml ps | grep model-training
```

#### **Solution**:
```bash
# Restart the service
docker-compose -f docker-compose.ml.yml restart model-training

# Wait for service to start
sleep 30

# Test again
curl http://localhost:8003/health
```

### **No Data Available**

#### **Problem**: Data fetching returns empty results
```bash
# Check if data exists
curl http://localhost:8001/data/summary

# Check MinIO storage
curl http://localhost:9001/minio/health/live
```

#### **Solution**:
```bash
# Fetch some data first
curl -X POST http://localhost:8001/data/fetch \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day"}'

# Check data summary again
curl http://localhost:8001/data/summary
```

### **Model Training Fails**

#### **Problem**: Model training returns errors
```bash
# Check training service logs
docker-compose -f docker-compose.ml.yml logs model-training

# Check if features exist
curl http://localhost:8002/features/summary
```

#### **Solution**:
```bash
# Generate features first
curl -X POST http://localhost:8002/features/generate \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day"}'

# Try training again
curl -X POST http://localhost:8003/train \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"], "timeframe": "1day", "algorithm": "random_forest"}'
```

### **MLflow Not Accessible**

#### **Problem**: Can't access MLflow UI at http://localhost:5001
```bash
# Check MLflow service
docker-compose -f docker-compose.ml.yml ps | grep mlflow

# Check MLflow logs
docker-compose -f docker-compose.ml.yml logs mlflow
```

#### **Solution**:
```bash
# Restart MLflow service
docker-compose -f docker-compose.ml.yml restart mlflow

# Wait for service to start
sleep 30

# Test MLflow
curl http://localhost:5001/health
```

### **Jupyter Lab Not Accessible**

#### **Problem**: Can't access Jupyter Lab at http://localhost:8888
```bash
# Check Jupyter service
docker-compose -f docker-compose.ml.yml ps | grep jupyter

# Check Jupyter logs
docker-compose -f docker-compose.ml.yml logs jupyter
```

#### **Solution**:
```bash
# Restart Jupyter service
docker-compose -f docker-compose.ml.yml restart jupyter

# Wait for service to start
sleep 30

# Test Jupyter
curl http://localhost:8888
```

---

## 🔍 **Advanced Troubleshooting**

### **Check Service Logs**
```bash
# Check specific service logs
docker-compose -f docker-compose.ml.yml logs model-training
docker-compose -f docker-compose.ml.yml logs mlflow
docker-compose -f docker-compose.ml.yml logs jupyter

# Follow logs in real-time
docker-compose -f docker-compose.ml.yml logs -f model-training
```

### **Check Resource Usage**
```bash
# Check Docker resource usage
docker stats

# Check disk space
df -h

# Check memory usage
free -h
```

### **Reset Services**
```bash
# Stop all services
docker-compose -f docker-compose.ml.yml down

# Remove volumes (WARNING: This will delete data)
docker-compose -f docker-compose.ml.yml down -v

# Start services again
docker-compose -f docker-compose.ml.yml up -d
```

### **Check Network Connectivity**
```bash
# Test internal network
docker exec model-training curl http://mlflow:5001/health
docker exec model-training curl http://postgres:5432

# Test external connectivity
docker exec model-training curl http://google.com
```

---

## 🎯 **Service-Specific Issues**

### **Data Pipeline Issues**
```bash
# Check data pipeline logs
docker-compose -f docker-compose.ml.yml logs data-pipeline

# Test data pipeline API
curl http://localhost:8001/health
curl http://localhost:8001/data/summary
```

### **Feature Engineering Issues**
```bash
# Check feature engineering logs
docker-compose -f docker-compose.ml.yml logs feature-engineering

# Test feature engineering API
curl http://localhost:8002/health
curl http://localhost:8002/features/summary
```

### **Model Training Issues**
```bash
# Check model training logs
docker-compose -f docker-compose.ml.yml logs model-training

# Test model training API
curl http://localhost:8003/health
curl http://localhost:8003/algorithms
```

### **AutoML Issues**
```bash
# Check AutoML logs
docker-compose -f docker-compose.ml.yml logs automl

# Test AutoML API
curl http://localhost:8004/health
```

### **Model Serving Issues**
```bash
# Check model serving logs
docker-compose -f docker-compose.ml.yml logs seldon-deployment

# Test model serving API
curl http://localhost:8005/health
```

### **Model Registry Issues**
```bash
# Check model registry logs
docker-compose -f docker-compose.ml.yml logs model-registry

# Test model registry API
curl http://localhost:8006/health
```

---

## 📊 **Performance Issues**

### **Slow API Responses**
```bash
# Check service resource usage
docker stats

# Check for memory leaks
docker-compose -f docker-compose.ml.yml logs | grep -i memory

# Restart services
docker-compose -f docker-compose.ml.yml restart
```

### **High Memory Usage**
```bash
# Check memory usage
free -h
docker stats

# Increase Docker memory limit in Docker Desktop
# Or restart services to free memory
docker-compose -f docker-compose.ml.yml restart
```

### **Disk Space Issues**
```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a

# Remove unused volumes
docker volume prune
```

---

## 🆘 **Getting Help**

### **Check Logs First**
```bash
# Get recent logs
docker-compose -f docker-compose.ml.yml logs --tail=100

# Get logs for specific service
docker-compose -f docker-compose.ml.yml logs --tail=100 model-training
```

### **Common Error Messages**

#### **"Connection refused"**
- Service is not running
- Port is not accessible
- Network issue

#### **"No data available"**
- Data not fetched yet
- MinIO storage issue
- Database connection problem

#### **"Model training failed"**
- Features not generated
- Insufficient data
- Algorithm not supported

#### **"Service unhealthy"**
- Service crashed
- Resource constraints
- Configuration error

---

## 📚 **Next Steps**

- **Quick Start**: See [Quick Start Guide](QUICK_START.md)
- **ML Pipeline**: See [ML Pipeline Guide](ML_PIPELINE_GUIDE.md)
- **Infrastructure**: See [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md)
- **API Reference**: See [API Reference](API_REFERENCE.md)

---

*This guide helps resolve common issues. For quick setup, see the Quick Start Guide!*
