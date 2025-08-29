# 🐳 Docker Testing Guide - BreadthFlow Abstraction System

This guide will help you test the new abstraction system in your Docker-based BreadthFlow infrastructure.

## 🚀 **Quick Start Testing**

### **Step 1: Verify Infrastructure is Running**

```bash
# Check if all containers are running
./scripts/check_status.sh

# Should show: spark-master, spark-worker-1, spark-worker-2, postgres, kafka, kafdrop, minio, elasticsearch, kibana, dashboard
```

### **Step 2: Test the New CLI Commands**

```bash
# Test the new abstraction system CLI
cd cli
python test_docker_integration.py
```

### **Step 3: Test Individual Commands**

```bash
# Test health check
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py health

# Test demo
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py demo

# Test data fetch
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py data fetch --symbols AAPL,MSFT --timeframe 1day

# Test signal generation
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py signals generate --symbols AAPL,MSFT --timeframe 1day

# Test backtesting
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py backtest run --symbols AAPL,MSFT --timeframe 1day --initial-capital 100000

# Test pipeline management
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py pipeline start --mode demo
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py pipeline status
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py pipeline stop
```

## 🎯 **Dashboard Integration Testing**

### **Step 1: Access the Dashboard**

1. Open your browser and go to: `http://localhost:8083`
2. Navigate to the "Commands" section
3. You should see the existing commands working

### **Step 2: Test New Commands via HTTP API**

The new abstraction system can be called via the existing HTTP API. Here's how to test it:

```bash
# Test health check via HTTP API
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "health", "parameters": {}}' \
  http://localhost:8081/execute

# Test data fetch via HTTP API
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "data_fetch", "parameters": {"symbols": "AAPL,MSFT", "timeframe": "1day"}}' \
  http://localhost:8081/execute

# Test signal generation via HTTP API
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "signals_generate", "parameters": {"symbols": "AAPL,MSFT", "timeframe": "1day"}}' \
  http://localhost:8081/execute

# Test backtesting via HTTP API
curl -X POST -H "Content-Type: application/json" \
  -d '{"command": "backtest_run", "parameters": {"symbols": "AAPL,MSFT", "timeframe": "1day", "initial_capital": 100000}}' \
  http://localhost:8081/execute
```

### **Step 3: Monitor Results**

1. **Dashboard**: Check the real-time results in the web interface
2. **Logs**: Monitor logs in the spark-master container
3. **Database**: Check pipeline runs in PostgreSQL
4. **Kibana**: View detailed logs and analytics

## 🔧 **Troubleshooting**

### **Common Issues and Solutions**

#### **1. Module Import Errors**

If you see `ModuleNotFoundError`:

```bash
# Check if the model directory is mounted correctly
docker exec spark-master ls -la /opt/bitnami/spark/jobs/model/

# If missing, restart the containers
./scripts/start_infrastructure.sh
```

#### **2. Dependencies Missing**

If you see missing package errors:

```bash
# Install dependencies in the spark-master container
docker exec spark-master pip3 install pandas numpy scikit-learn yfinance
```

#### **3. Database Connection Issues**

If you see PostgreSQL connection errors:

```bash
# Check if postgres container is running
docker ps | grep postgres

# Check database connectivity
docker exec spark-master python3 -c "
import psycopg2
conn = psycopg2.connect('postgresql://pipeline:pipeline123@breadthflow-postgres:5432/breadthflow')
print('Database connection successful')
conn.close()
"
```

#### **4. Permission Issues**

If you see permission errors:

```bash
# Check file permissions
docker exec spark-master ls -la /opt/bitnami/spark/jobs/cli/

# Fix permissions if needed
docker exec spark-master chmod +x /opt/bitnami/spark/jobs/cli/bf_abstracted_docker.py
```

## 📊 **Expected Results**

### **Successful Test Output**

When everything is working correctly, you should see:

```
🚀 Testing BreadthFlow Docker Integration
============================================================
🔍 Checking if containers are running...
✅ spark-master container is running

🧪 Testing: health
============================================================
✅ Command executed successfully
📤 Output:
📝 Pipeline Run: 12345678-1234-1234-1234-123456789abc | health | running
✅ System health check completed
🏥 Overall Health: HEALTHY
⏱️ Duration: 0.15s
📝 Pipeline Run: 12345678-1234-1234-1234-123456789abc | health | completed

🧪 Testing: demo
============================================================
✅ Command executed successfully
📤 Output:
🚀 Starting BreadthFlow Abstracted Demo...

📊 Step 1: Fetching Data...
✅ Data fetch completed successfully

🎯 Step 2: Generating Signals...
✅ Signal generation completed successfully

📈 Step 3: Running Backtest...
✅ Backtesting completed successfully

🏥 Step 4: Checking System Health...
✅ System health check completed

✅ Demo completed successfully!
📊 Data Fetch: ✅
🎯 Signal Generation: ✅
📈 Backtesting: ✅
🏥 System Health: ✅
⏱️ Total Duration: 5.23s
```

### **Database Logging**

Check the PostgreSQL database for pipeline run logs:

```bash
# Connect to the database
docker exec -it breadthflow-postgres psql -U pipeline -d breadthflow

# View recent pipeline runs
SELECT run_id, command, status, start_time, duration 
FROM pipeline_runs 
ORDER BY start_time DESC 
LIMIT 10;
```

## 🎮 **Dashboard Integration**

### **Current Status**

The new abstraction system is now integrated with your Docker infrastructure:

1. **✅ CLI Integration**: New CLI commands work in the spark-master container
2. **✅ HTTP API**: Commands can be called via the existing HTTP API
3. **✅ Database Logging**: All runs are logged to PostgreSQL
4. **✅ Error Handling**: Comprehensive error handling and recovery
5. **✅ Monitoring**: Real-time health checks and performance tracking

### **Next Steps for Full Dashboard Integration**

To fully integrate with your dashboard, you would need to:

1. **Add New Command Buttons**: Add buttons for the new abstraction commands
2. **Update Command Handler**: Modify the dashboard to call the new CLI
3. **Display Results**: Show the results from the new system
4. **Real-time Updates**: Display real-time status updates

### **Testing the Integration**

You can test the integration by:

1. **Using the HTTP API directly** (as shown above)
2. **Calling the CLI commands manually** in the container
3. **Monitoring the logs** for successful execution
4. **Checking the database** for run history

## 🎉 **Success Criteria**

The integration is successful when:

- ✅ All CLI commands execute without errors
- ✅ Database logging works correctly
- ✅ HTTP API calls return proper responses
- ✅ Health checks show system is healthy
- ✅ Demo runs complete successfully
- ✅ Pipeline management works correctly

## 📞 **Support**

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the logs: `docker logs spark-master`
3. Check the database for error messages
4. Verify all containers are running properly

The new abstraction system is now ready for use in your Docker environment! 🚀
