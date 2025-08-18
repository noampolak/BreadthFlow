# Infrastructure Architecture Diagram

## 🔄 How CLI Commands Work with Docker Compose

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI Commands                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ poetry run bf   │  │ poetry run bf   │  │ poetry run bf   │ │
│  │ infra start     │  │ data fetch      │  │ signals generate│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                Docker Compose Commands                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ docker-compose -f infra/docker-compose.yml up -d           │ │
│  │ docker-compose -f infra/docker-compose.yml down            │ │
│  │ docker-compose -f infra/docker-compose.yml logs            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Containers                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ spark-master│  │ spark-worker│  │   kafka     │            │
│  │   :8080     │  │             │  │   :9092     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    minio    │  │elasticsearch│  │   kibana    │            │
│  │ :9000/9001  │  │   :9200     │  │   :5601     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Your Applications                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   PySpark       │  │   DataFetcher   │  │ SignalGenerator │ │
│  │   Jobs          │  │                 │  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Command Flow Examples

### **1. Starting Infrastructure**
```bash
poetry run bf infra start
```
**What happens:**
1. CLI runs `docker-compose -f infra/docker-compose.yml up -d`
2. Docker starts all 6 containers
3. CLI waits 30 seconds for services to start
4. CLI performs health checks
5. CLI shows service URLs

### **2. Fetching Data**
```bash
poetry run bf data fetch --symbol-list demo_small
```
**What happens:**
1. CLI creates PySpark session connecting to `spark://spark-master:7077`
2. PySpark job runs on spark-worker container
3. DataFetcher downloads data from Yahoo Finance
4. Data is stored in MinIO (Delta Lake format)
5. CLI shows results

### **3. Generating Signals**
```bash
poetry run bf signals generate --symbol-list demo_small
```
**What happens:**
1. CLI creates PySpark session
2. SignalGenerator reads data from MinIO
3. PySpark processes features and generates signals
4. Results stored in MinIO and Elasticsearch
5. CLI shows signal statistics

## 🌐 Network Communication

### **Container-to-Container Communication**
```
spark-master:7077  ←──  spark-worker
spark-master:7077  ←──  Your PySpark App
kafka:9092         ←──  Your Streaming App
elasticsearch:9200 ←──  Your Search App
minio:9000         ←──  Your Storage App
```

### **External Access (Your Computer)**
```
localhost:8080     →──  Spark UI
localhost:9001     →──  MinIO Console
localhost:5601     →──  Kibana
localhost:9200     →──  Elasticsearch API
localhost:9092     →──  Kafka
```

## 💾 Data Storage

### **Docker Volumes (Persistent Data)**
```
spark-data         ←──  Spark logs and checkpoints
minio-data         ←──  All your data (OHLCV, features, signals)
elasticsearch-data ←──  Search indices
kafka-data         ←──  Message logs
zookeeper-data     ←──  Configuration data
```

### **Data Flow**
```
Yahoo Finance → PySpark → MinIO (Delta Lake)
MinIO → PySpark → MinIO (Features)
MinIO → PySpark → MinIO + Elasticsearch (Signals)
MinIO → Kafka → Elasticsearch (Real-time)
```

## 🔧 Key Configuration Files

### **1. docker-compose.yml**
- Defines all services and their relationships
- Sets up networking between containers
- Configures volumes for data persistence
- Maps ports for external access

### **2. .env**
- Environment variables for your application
- Connection strings to services
- Configuration parameters

### **3. CLI Commands**
- `poetry run bf infra start` → `docker-compose up -d`
- `poetry run bf infra stop` → `docker-compose down`
- `poetry run bf infra logs` → `docker-compose logs`

## 🎯 Why This Architecture?

### **Benefits:**
1. **Isolation**: Each service runs in its own container
2. **Consistency**: Same setup on any machine
3. **Scalability**: Easy to add more workers or services
4. **Persistence**: Data survives container restarts
5. **Monitoring**: Web interfaces for each service

### **Development Workflow:**
1. **Start**: `poetry run bf infra start`
2. **Work**: Run your analysis commands
3. **Monitor**: Use web interfaces
4. **Stop**: `poetry run bf infra stop`

### **Production Ready:**
- Can be deployed to cloud platforms
- Supports horizontal scaling
- Includes monitoring and logging
- Data backup and recovery

---

**The beauty of this setup is that you get enterprise-grade infrastructure with a single command! 🚀**
