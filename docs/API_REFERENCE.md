# 📚 BreadthFlow API Reference

> Complete reference for all API endpoints and usage examples

## 🎯 **Overview**

BreadthFlow provides comprehensive APIs for data pipeline, feature engineering, model training, and model serving. All APIs follow REST conventions and return JSON responses.

---

## 🔗 **Base URLs**

| Service | Base URL | Purpose |
|---------|----------|---------|
| **Data Pipeline** | http://localhost:8001 | Data ingestion and validation |
| **Feature Engineering** | http://localhost:8002 | Automated feature generation |
| **Model Training** | http://localhost:8003 | Model training and hyperparameter optimization |
| **AutoML** | http://localhost:8004 | Automated model training |
| **Model Serving** | http://localhost:8005 | Production model endpoints |
| **Model Registry** | http://localhost:8006 | Model lifecycle management |

---

## 📥 **Data Pipeline API (Port 8001)**

### **Health Check**
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Data Pipeline API is running"
}
```

### **Data Summary**
```bash
GET /data/summary
```

**Response:**
```json
{
  "total_symbols": 150,
  "total_records": 50000,
  "timeframes": ["1day", "1hour", "15min"],
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### **Fetch Data**
```bash
POST /data/fetch
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT"],
  "timeframe": "1day",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Data fetched successfully",
  "records_processed": 500,
  "symbols": ["AAPL", "MSFT"]
}
```

---

## 🔧 **Feature Engineering API (Port 8002)**

### **Health Check**
```bash
GET /health
```

### **Generate Features**
```bash
POST /features/generate
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT"],
  "timeframe": "1day",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Features generated successfully",
  "features_count": 150,
  "feature_types": ["technical", "time_based", "microstructure"]
}
```

### **Get Feature Summary**
```bash
GET /features/summary
```

**Response:**
```json
{
  "total_features": 150,
  "feature_categories": {
    "technical_indicators": 50,
    "time_features": 30,
    "microstructure": 20,
    "automated": 50
  }
}
```

---

## 🤖 **Model Training API (Port 8003)**

### **Health Check**
```bash
GET /health
```

### **Train Model**
```bash
POST /train
Content-Type: application/json

{
  "symbols": ["AAPL"],
  "timeframe": "1day",
  "algorithm": "random_forest",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model trained successfully",
  "model_id": "model_20240115_103000",
  "algorithm": "random_forest",
  "accuracy": 0.85,
  "mlflow_run_id": "abc123def456"
}
```

### **Get Training Status**
```bash
GET /train/status/{model_id}
```

**Response:**
```json
{
  "model_id": "model_20240115_103000",
  "status": "completed",
  "progress": 100,
  "accuracy": 0.85,
  "training_time": "00:05:30"
}
```

### **Available Algorithms**
```bash
GET /algorithms
```

**Response:**
```json
{
  "algorithms": [
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm",
    "logistic_regression",
    "automl"
  ]
}
```

---

## 🚀 **AutoML API (Port 8004)**

### **Health Check**
```bash
GET /health
```

### **Run AutoML**
```bash
POST /automl/run
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT"],
  "timeframe": "1day",
  "max_models": 10,
  "time_limit": 3600
}
```

**Response:**
```json
{
  "status": "success",
  "message": "AutoML training started",
  "job_id": "automl_20240115_103000",
  "estimated_time": "00:30:00"
}
```

### **Get AutoML Results**
```bash
GET /automl/results/{job_id}
```

**Response:**
```json
{
  "job_id": "automl_20240115_103000",
  "status": "completed",
  "best_model": "xgboost",
  "best_accuracy": 0.87,
  "models_tested": 10,
  "total_time": "00:25:30"
}
```

---

## 🚀 **Model Serving API (Port 8005)**

### **Health Check**
```bash
GET /health
```

### **Predict**
```bash
POST /predict
Content-Type: application/json

{
  "model_name": "breadthflow-model",
  "features": {
    "symbol": "AAPL",
    "rsi": 65.5,
    "macd": 2.3,
    "volume": 1000000
  }
}
```

**Response:**
```json
{
  "prediction": "BUY",
  "confidence": 0.85,
  "probability": {
    "BUY": 0.85,
    "SELL": 0.10,
    "HOLD": 0.05
  }
}
```

### **Get Model Info**
```bash
GET /models/{model_name}
```

**Response:**
```json
{
  "model_name": "breadthflow-model",
  "version": "v1.2.0",
  "status": "active",
  "accuracy": 0.85,
  "last_updated": "2024-01-15T10:30:00Z"
}
```

---

## 📦 **Model Registry API (Port 8006)**

### **Health Check**
```bash
GET /health
```

### **Register Model**
```bash
POST /models/register
Content-Type: application/json

{
  "model_name": "breadthflow-model",
  "version": "v1.2.0",
  "description": "Updated model with better features",
  "tags": {
    "algorithm": "xgboost",
    "timeframe": "1day"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model registered successfully",
  "model_id": "breadthflow-model-v1.2.0"
}
```

### **Get Model Versions**
```bash
GET /models/{model_name}/versions
```

**Response:**
```json
{
  "model_name": "breadthflow-model",
  "versions": [
    {
      "version": "v1.2.0",
      "stage": "Production",
      "accuracy": 0.85,
      "created_at": "2024-01-15T10:30:00Z"
    },
    {
      "version": "v1.1.0",
      "stage": "Staging",
      "accuracy": 0.82,
      "created_at": "2024-01-10T15:20:00Z"
    }
  ]
}
```

### **Deploy Model**
```bash
POST /models/deploy
Content-Type: application/json

{
  "model_name": "breadthflow-model",
  "version": "v1.2.0",
  "stage": "Production"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model deployed to production",
  "endpoint": "http://localhost:8005/predict"
}
```

### **Create A/B Test**
```bash
POST /ab-tests
Content-Type: application/json

{
  "model_name": "breadthflow-model",
  "model_a_version": "v1.1.0",
  "model_b_version": "v1.2.0",
  "traffic_split": 0.5,
  "duration_hours": 24
}
```

**Response:**
```json
{
  "status": "success",
  "message": "A/B test created successfully",
  "test_id": "ab_test_20240115_103000",
  "traffic_split": 0.5
}
```

---

## 🔧 **Common Response Formats**

### **Success Response**
```json
{
  "status": "success",
  "message": "Operation completed successfully",
  "data": { ... }
}
```

### **Error Response**
```json
{
  "status": "error",
  "message": "Error description",
  "error_code": "VALIDATION_ERROR",
  "details": { ... }
}
```

### **Health Check Response**
```json
{
  "status": "healthy",
  "message": "Service is running",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 🎯 **Quick Reference**

| What You Want to Do | Endpoint | Method |
|---------------------|----------|--------|
| **Check Service Health** | `/health` | GET |
| **Fetch Data** | `/data/fetch` | POST |
| **Generate Features** | `/features/generate` | POST |
| **Train Model** | `/train` | POST |
| **Run AutoML** | `/automl/run` | POST |
| **Make Prediction** | `/predict` | POST |
| **Register Model** | `/models/register` | POST |
| **Deploy Model** | `/models/deploy` | POST |
| **Create A/B Test** | `/ab-tests` | POST |

---

## 📚 **Next Steps**

- **Quick Start**: See [Quick Start Guide](QUICK_START.md)
- **ML Pipeline**: See [ML Pipeline Guide](ML_PIPELINE_GUIDE.md)
- **Infrastructure**: See [Infrastructure Guide](INFRASTRUCTURE_GUIDE.md)
- **Troubleshooting**: See [Troubleshooting Guide](TROUBLESHOOTING.md)

---

*This reference provides complete API documentation. For quick setup, see the Quick Start Guide!*
