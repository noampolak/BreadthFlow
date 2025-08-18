# Infrastructure Guide - Docker Compose Setup

## 🏗️ Overview

The Breadth/Thrust Signals POC uses **Docker Compose** to run a complete big data infrastructure locally. This includes Apache Spark, Kafka, MinIO (S3-compatible storage), Elasticsearch, and Kibana.

## 🐳 What is Docker Compose?

**Docker Compose** is a tool that allows you to define and run multi-container Docker applications. Instead of running each service separately, you define all services in a single `docker-compose.yml` file and start everything with one command.

### **Why Docker Compose?**
- **Single Command**: Start all services with `docker-compose up`
- **Service Dependencies**: Automatically handles service startup order
- **Network Management**: Services can communicate with each other
- **Volume Management**: Persistent data storage across restarts
- **Environment Consistency**: Same setup on any machine

## 🏢 Infrastructure Services

### **1. Apache Spark Cluster**
```yaml
spark-master:    # Spark master node (coordinates the cluster)
spark-worker:    # Spark worker node (processes data)
```
- **Purpose**: Distributed data processing and analytics
- **Ports**: 
  - `8080`: Spark UI (web interface)
  - `7077`: Spark master port
- **What it does**: Runs PySpark jobs for data processing, feature calculation, and signal generation

### **2. Apache Kafka**
```yaml
kafka:           # Message streaming platform
```
- **Purpose**: Real-time data streaming and message queuing
- **Port**: `9092`: Kafka broker
- **What it does**: Handles real-time data replay and streaming between components

### **3. MinIO (S3-compatible Storage)**
```yaml
minio:           # Object storage (like AWS S3)
```
- **Purpose**: Data storage for Delta Lake tables
- **Ports**:
  - `9000`: S3 API
  - `9001`: MinIO Console (web interface)
- **Credentials**: `minioadmin` / `minioadmin`
- **What it does**: Stores all the data (OHLCV, features, signals, backtest results)

### **4. Elasticsearch**
```yaml
elasticsearch:   # Search and analytics engine
```
- **Purpose**: Real-time search and analytics
- **Ports**:
  - `9200`: REST API
  - `9300`: Node communication
- **What it does**: Stores and indexes trading signals for real-time search and monitoring

### **5. Kibana**
```yaml
kibana:          # Data visualization platform
```
- **Purpose**: Web interface for Elasticsearch
- **Port**: `5601`: Kibana web interface
- **What it does**: Creates dashboards and visualizations for monitoring signals

### **6. Zookeeper**
```yaml
zookeeper:       # Coordination service
```
- **Purpose**: Service coordination (used by Kafka)
- **Port**: `2181`: Zookeeper port
- **What it does**: Helps Kafka manage cluster coordination

## 🚀 How to Use the Infrastructure

### **Starting the Infrastructure**

#### **Option 1: Using CLI (Recommended)**
```bash
# Start all services
poetry run bf infra start

# Start with custom wait time (default: 30 seconds)
poetry run bf infra start --wait 60
```

#### **Option 2: Using Docker Compose Directly**
```bash
# Start all services in background
docker-compose -f infra/docker-compose.yml up -d

# Start and see logs
docker-compose -f infra/docker-compose.yml up

# Start specific services only
docker-compose -f infra/docker-compose.yml up -d spark-master spark-worker
```

### **Checking Infrastructure Status**

#### **Health Check**
```bash
# Check if all services are healthy
poetry run bf infra health

# Check status
poetry run bf infra status
```

#### **View Logs**
```bash
# View all logs
poetry run bf infra logs

# Follow logs in real-time
poetry run bf infra logs --follow

# View specific service logs
docker-compose -f infra/docker-compose.yml logs spark-master
```

### **Stopping the Infrastructure**

#### **Stop All Services**
```bash
# Using CLI
poetry run bf infra stop

# Using Docker Compose
docker-compose -f infra/docker-compose.yml down
```

#### **Restart Services**
```bash
# Restart all services
poetry run bf infra restart

# Restart specific service
docker-compose -f infra/docker-compose.yml restart spark-master
```

## 🌐 Web Interfaces

Once the infrastructure is running, you can access these web interfaces:

### **Spark UI** - http://localhost:8080
- **Purpose**: Monitor Spark jobs and performance
- **Features**:
  - View running applications
  - Monitor job progress
  - Check resource usage
  - View application logs

### **MinIO Console** - http://localhost:9001
- **Login**: `minioadmin` / `minioadmin`
- **Purpose**: Browse and manage data storage
- **Features**:
  - Browse Delta Lake tables
  - Upload/download files
  - Create buckets
  - Monitor storage usage

### **Kibana** - http://localhost:5601
- **Purpose**: Create dashboards and visualizations
- **Features**:
  - Create signal monitoring dashboards
  - Search trading signals
  - Create alerts
  - Visualize performance metrics

### **Elasticsearch** - http://localhost:9200
- **Purpose**: REST API for data access
- **Features**:
  - Health monitoring
  - Index management
  - Search API
  - Cluster status

## 📊 Data Flow

### **How Data Flows Through the System**

```
1. Data Fetching
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ Yahoo       │───▶│ PySpark     │───▶│ MinIO       │
   │ Finance     │    │ DataFetcher │    │ (Delta)     │
   └─────────────┘    └─────────────┘    └─────────────┘

2. Feature Processing
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ MinIO       │───▶│ PySpark     │───▶│ MinIO       │
   │ (OHLCV)     │    │ Features    │    │ (Features)  │
   └─────────────┘    └─────────────┘    └─────────────┘

3. Signal Generation
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ MinIO       │───▶│ PySpark     │───▶│ MinIO       │
   │ (Features)  │    │ SignalGen   │    │ (Signals)   │
   └─────────────┘    └─────────────┘    └─────────────┘

4. Real-time Streaming
   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
   │ MinIO       │───▶│ Kafka       │───▶│ Elasticsearch│
   │ (Signals)   │    │ (Streaming) │    │ (Search)    │
   └─────────────┘    └─────────────┘    └─────────────┘

5. Monitoring
   ┌─────────────┐    ┌─────────────┐
   │ Elasticsearch│───▶│ Kibana      │
   │ (Data)      │    │ (Dashboard) │
   └─────────────┘    └─────────────┘
```

## 🔧 Configuration

### **Environment Variables**
The infrastructure uses environment variables defined in `.env` file:

```bash
# Spark Configuration
SPARK_MASTER_URL=spark://spark-master:7077
SPARK_WORKER_MEMORY=1G
SPARK_WORKER_CORES=1

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Elasticsearch Configuration
ELASTICSEARCH_HOST=localhost:9200
```

### **Docker Compose Configuration**
Key configuration options in `infra/docker-compose.yml`:

```yaml
# Memory limits
environment:
  - "ES_JAVA_OPTS=-Xms512m -Xmx512m"  # Elasticsearch memory
  - SPARK_WORKER_MEMORY=1G            # Spark worker memory

# Port mappings
ports:
  - "8080:8080"  # Spark UI
  - "9000:9000"  # MinIO API
  - "5601:5601"  # Kibana

# Volume persistence
volumes:
  - spark-data:/opt/bitnami/spark
  - minio-data:/data
  - elasticsearch-data:/usr/share/elasticsearch/data
```

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Port Already in Use**
```bash
# Check what's using the port
lsof -i :8080

# Kill the process or change the port in docker-compose.yml
```

#### **2. Services Not Starting**
```bash
# Check Docker is running
docker --version

# Check Docker Compose
docker-compose --version

# View detailed logs
poetry run bf infra logs
```

#### **3. Memory Issues**
```bash
# Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory

# Or reduce service memory in docker-compose.yml
environment:
  - "ES_JAVA_OPTS=-Xms256m -Xmx256m"
```

#### **4. Data Persistence Issues**
```bash
# Check volumes
docker volume ls

# Remove volumes (WARNING: deletes all data)
docker-compose -f infra/docker-compose.yml down -v

# Recreate volumes
docker-compose -f infra/docker-compose.yml up -d
```

### **Performance Optimization**

#### **For Development**
```yaml
# Reduce memory usage
environment:
  - "ES_JAVA_OPTS=-Xms256m -Xmx256m"
  - SPARK_WORKER_MEMORY=512m
```

#### **For Production**
```yaml
# Increase memory and cores
environment:
  - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
  - SPARK_WORKER_MEMORY=4G
  - SPARK_WORKER_CORES=4
```

## 🔄 Development Workflow

### **Typical Development Session**

1. **Start Infrastructure**
   ```bash
   poetry run bf infra start
   ```

2. **Check Health**
   ```bash
   poetry run bf infra health
   ```

3. **Run Analysis**
   ```bash
   poetry run bf data fetch --symbol-list demo_small
   poetry run bf signals generate --symbol-list demo_small
   poetry run bf backtest run --symbol-list demo_small
   ```

4. **Monitor Progress**
   - Spark UI: http://localhost:8080
   - MinIO Console: http://localhost:9001
   - Kibana: http://localhost:5601

5. **Stop Infrastructure**
   ```bash
   poetry run bf infra stop
   ```

### **Data Persistence**
- **Data survives restarts**: All data is stored in Docker volumes
- **Clean slate**: Use `docker-compose down -v` to remove all data
- **Backup**: Copy volume data for backup/restore

## 🎯 Best Practices

### **Resource Management**
- **Start small**: Use demo lists for initial testing
- **Monitor usage**: Check Spark UI and system resources
- **Scale up**: Increase memory/cores for larger datasets

### **Development Tips**
- **Use CLI commands**: `poetry run bf infra *` for convenience
- **Check logs**: Monitor service logs for issues
- **Health checks**: Always verify services are healthy before running analysis

### **Production Considerations**
- **Security**: Change default passwords
- **Monitoring**: Set up proper monitoring and alerting
- **Backup**: Implement regular data backups
- **Scaling**: Consider multi-node Spark cluster for large datasets

## 📚 Next Steps

1. **Start with demo**: Use `poetry run bf infra start` to get started
2. **Explore interfaces**: Visit the web interfaces to understand the system
3. **Run analysis**: Use the CLI commands to process data
4. **Monitor performance**: Use Spark UI and Kibana for monitoring
5. **Customize**: Modify docker-compose.yml for your needs

---

**The infrastructure is designed to be simple to use but powerful enough for serious analysis! 🚀**
