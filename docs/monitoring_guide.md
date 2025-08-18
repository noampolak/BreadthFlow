# 📊 BreadthFlow Monitoring & Visibility Guide

Complete guide to monitoring your BreadthFlow data pipelines with multiple UI solutions for maximum visibility.

## 🎯 **Monitoring Solutions Overview**

Your BreadthFlow system now has **3 levels of monitoring** for complete pipeline visibility:

### 1. **🚀 Real-time Web Dashboard** (Primary)
- **URL**: http://localhost:8081
- **Purpose**: Real-time pipeline monitoring and progress tracking
- **Best for**: Active development, debugging, immediate feedback

### 2. **📊 Kibana Dashboards** (Long-term)
- **URL**: http://localhost:5601
- **Purpose**: Historical analysis and advanced visualizations  
- **Best for**: Trend analysis, production monitoring, alerts

### 3. **🖥️ Terminal CLI** (Direct)
- **Purpose**: Direct command execution with structured logging
- **Best for**: Scripting, automation, quick operations

---

## 🚀 **Real-time Web Dashboard**

### **Quick Start**
```bash
# Start the dashboard (runs in background)
docker exec -d spark-master python3 /opt/bitnami/spark/jobs/cli/web_dashboard.py --port 8081 --host 0.0.0.0

# Access dashboard
open http://localhost:8081
```

### **Features**
- ✅ **Real-time updates** every 30 seconds
- ✅ **Pipeline progress tracking** with live logs
- ✅ **System health monitoring** (Spark, MinIO, Kafka, etc.)
- ✅ **Success rate metrics** and performance stats
- ✅ **Interactive log viewer** for debugging
- ✅ **Responsive design** for mobile/desktop

### **Dashboard Sections**

#### 📈 **Statistics Cards**
- **Total Runs**: Lifetime pipeline executions
- **Success Rate**: Percentage of successful runs
- **Last 24h**: Recent activity count
- **Avg Duration**: Performance metrics

#### 📋 **Recent Pipeline Runs**
- Click any run to view detailed logs
- Status indicators: ✅ Completed, 🔄 Running, ❌ Failed
- Duration and timestamp information

#### 🏥 **System Status**
- Real-time health checks for all services
- Green/Red indicators for service availability

#### 📝 **Live Logs**
- Terminal-style log viewer
- Color-coded log levels (INFO/WARN/ERROR)
- Auto-scroll to latest entries

---

## 📊 **Kibana Dashboards**

### **Setup**
```bash
# Setup Kibana dashboards (one-time)
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/setup_kibana_dashboards.py

# Access Kibana
open http://localhost:5601
```

### **Available Dashboards**

#### 🔄 **BreadthFlow Pipeline Monitoring**
- Pipeline runs over time (trend charts)
- Success rate gauges
- Recent pipeline logs table
- Auto-refresh every 30 seconds

#### 🏥 **BreadthFlow System Health**
- Service health status metrics
- Pipeline duration trends
- Infrastructure monitoring
- 7-day historical view

### **Kibana Features**
- ✅ **Advanced filtering** and search
- ✅ **Custom time ranges** and drill-downs
- ✅ **Alerting** capabilities (set thresholds)
- ✅ **Data export** (CSV, PDF reports)
- ✅ **Persistent dashboards** with sharing

---

## 🖥️ **Enhanced CLI with Logging**

### **Quick Start**
```bash
# Use enhanced CLI with dashboard integration
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py --help

# Run commands with automatic logging
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py data summary
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py demo --quick
```

### **Features**
- ✅ **Structured logging** to dashboard database
- ✅ **Progress tracking** with percentage completion
- ✅ **Metadata collection** (symbols, timing, results)
- ✅ **Error tracking** with detailed stack traces
- ✅ **Unique run IDs** for tracking individual executions

---

## 🎯 **Usage Scenarios**

### **During Development** 
```bash
# 1. Start dashboard
docker exec -d spark-master python3 /opt/bitnami/spark/jobs/cli/web_dashboard.py --port 8081 --host 0.0.0.0

# 2. Open in browser
open http://localhost:8081

# 3. Run pipeline commands and watch progress
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/enhanced_bf_minio.py data fetch --symbols AAPL,MSFT
```

### **For Production Monitoring**
```bash
# 1. Setup Kibana dashboards
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/setup_kibana_dashboards.py

# 2. Configure alerts in Kibana
open http://localhost:5601

# 3. Set up automated pipeline runs with monitoring
```

### **For Debugging Issues**
1. **Check real-time dashboard** for immediate status
2. **View detailed logs** in the log viewer
3. **Use Kibana** for historical pattern analysis
4. **Check system health** across all services

---

## 📱 **Access URLs**

| Service | URL | Purpose |
|---------|-----|---------|
| **Web Dashboard** | http://localhost:8081 | Real-time monitoring |
| **Kibana** | http://localhost:5601 | Advanced analytics |
| **Spark UI** | http://localhost:8080 | Spark job monitoring |
| **MinIO Console** | http://localhost:9001 | Data storage management |
| **Elasticsearch** | http://localhost:9200 | Search API |

---

## 🔧 **Configuration**

### **Dashboard Settings**
```python
# In cli/web_dashboard.py
DEFAULT_PORT = 8081
AUTO_REFRESH_INTERVAL = 30000  # 30 seconds
DATABASE_PATH = "dashboard.db"
```

### **Logging Configuration**
```python
# In cli/enhanced_bf_minio.py  
LOG_LEVEL = logging.INFO
LOG_FILE = "breadthflow.log"
```

---

## 🚨 **Troubleshooting**

### **Dashboard Not Loading**
```bash
# Check if dashboard is running
docker exec spark-master ps aux | grep web_dashboard

# Restart dashboard
docker exec -d spark-master python3 /opt/bitnami/spark/jobs/cli/web_dashboard.py --port 8081 --host 0.0.0.0
```

### **No Data in Kibana**
```bash
# Check Elasticsearch is running
curl http://localhost:9200/_cluster/health

# Re-setup dashboards
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/setup_kibana_dashboards.py
```

### **Services Not Responding**
```bash
# Check all services
docker-compose ps

# Restart infrastructure
docker-compose restart
```

---

## 🎉 **Best Practices**

### **For Development**
1. **Always have the web dashboard open** during development
2. **Use enhanced CLI** instead of basic commands
3. **Check logs immediately** if something fails
4. **Monitor system health** before running large operations

### **For Production**
1. **Set up Kibana alerts** for critical failures
2. **Monitor success rates** and performance trends
3. **Regular health checks** of all services
4. **Archive old logs** to prevent database growth

### **For Debugging**
1. **Start with real-time dashboard** for immediate context
2. **Use unique run IDs** to track specific executions
3. **Check system status** before investigating pipeline issues
4. **Use Kibana search** for pattern analysis across time

---

## 📈 **Metrics to Monitor**

### **Pipeline Health**
- ✅ Success rate (target: >95%)
- ✅ Average duration (watch for degradation)
- ✅ Failed runs (investigate patterns)
- ✅ Recent activity (ensure regular execution)

### **System Health**
- ✅ Service availability (all green)
- ✅ Resource usage (CPU, memory)
- ✅ Storage growth (MinIO space)
- ✅ Network connectivity (between services)

### **Data Quality**
- ✅ Symbol fetch success rate
- ✅ Data completeness (records per symbol)
- ✅ Update frequency (data freshness)
- ✅ Error patterns (specific symbols/dates)

---

🎉 **You now have enterprise-grade monitoring for your BreadthFlow pipelines!**

The combination of real-time web dashboard + Kibana analytics + enhanced CLI gives you complete visibility into every aspect of your data pipeline operations.
