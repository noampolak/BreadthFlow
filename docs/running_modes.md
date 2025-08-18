# Running Modes - Batch vs Continuous vs Streaming

## 🎯 Overview

The Breadth/Thrust Signals system supports **three different running modes** to suit different use cases:

1. **🔄 Batch Mode** (Command-by-Command) - Run each step manually
2. **🔄 Pipeline Mode** (Continuous) - Run the full pipeline repeatedly
3. **🌊 Streaming Mode** (Real-time) - Continuous data flow with real-time processing

## 🔄 Mode 1: Batch Mode (Current Default)

### **What it is:**
Run each step of the pipeline manually, one command at a time.

### **How to use:**
```bash
# Step 1: Start infrastructure
poetry run bf infra start

# Step 2: Fetch data
poetry run bf data fetch --symbol-list demo_small

# Step 3: Generate signals
poetry run bf signals generate --symbol-list demo_small

# Step 4: Run backtest
poetry run bf backtest run --symbol-list demo_small

# Step 5: Stop infrastructure
poetry run bf infra stop
```

### **When to use:**
- ✅ **Development and testing**
- ✅ **One-time analysis**
- ✅ **Debugging specific steps**
- ✅ **Manual control over each step**

### **Pros:**
- Full control over each step
- Can inspect results between steps
- Easy to debug issues
- Can skip steps if needed

### **Cons:**
- Manual intervention required
- Not suitable for continuous operation
- Easy to forget steps

## 🔄 Mode 2: Pipeline Mode (Continuous)

### **What it is:**
Run the complete pipeline (fetch → generate → backtest) continuously in a loop until stopped.

### **How to use:**
```bash
# Basic continuous pipeline (runs every 5 minutes)
poetry run bf pipeline --symbol-list demo_small

# Custom interval (runs every 10 minutes)
poetry run bf pipeline --symbol-list demo_small --interval 600

# Auto-start infrastructure
poetry run bf pipeline --symbol-list demo_small --auto-start-infra

# Custom date range
poetry run bf pipeline --symbol-list demo_small --start-date 2024-01-01 --end-date 2024-12-31

# No interval (runs immediately after each completion)
poetry run bf pipeline --symbol-list demo_small --interval 0
```

### **What happens:**
1. **Starts infrastructure** (if `--auto-start-infra` is used)
2. **Loops continuously:**
   - Fetch latest data
   - Generate signals
   - Run backtest
   - Wait for interval
   - Repeat
3. **Stops when you press Ctrl+C**

### **Example Output:**
```
🔄 Starting Continuous Pipeline Mode
==================================================
📊 Symbol List: demo_small
⏰ Interval: 300 seconds (5.0 minutes)
📅 Date Range: 2024-01-01 to 2024-12-31
🏗️  Auto-start Infrastructure: False

🔄 Pipeline will run continuously until stopped
⏹️  Press Ctrl+C to stop the pipeline

✅ Infrastructure is healthy

🔄 Pipeline Run #1
⏰ Started at: 2024-12-19 14:30:00
----------------------------------------
📥 Step 1: Fetching data...
✅ Data fetched successfully
🎯 Step 2: Generating signals...
✅ Signals generated successfully
📈 Step 3: Running backtest...
✅ Backtest completed successfully
✅ Pipeline run #1 completed in 45.2s
📊 Total runs: 1, Avg duration: 45.2s
⏳ Waiting 300 seconds until next run...
🕐 Next run at: 2024-12-19 14:35:00
```

### **When to use:**
- ✅ **Continuous monitoring**
- ✅ **Regular analysis updates**
- ✅ **Automated trading signals**
- ✅ **Production deployment**

### **Pros:**
- Fully automated
- Runs continuously
- Configurable intervals
- Error recovery (continues after failures)
- Statistics tracking

### **Cons:**
- Less control over individual steps
- Resource intensive (runs continuously)
- May need monitoring for long runs

## 🌊 Mode 3: Streaming Mode (Real-time)

### **What it is:**
Continuously replay historical data at high speed to simulate real-time market data flow.

### **How to use:**
```bash
# Basic streaming (60x speed)
poetry run bf stream --symbol-list demo_small

# High-speed streaming (120x speed)
poetry run bf stream --symbol-list demo_small --speed 120

# Auto-start infrastructure
poetry run bf stream --symbol-list demo_small --auto-start-infra
```

### **What happens:**
1. **Starts infrastructure** (if `--auto-start-infra` is used)
2. **Starts data replay** at specified speed
3. **Data flows continuously** through the system
4. **Real-time processing** of signals and analysis
5. **Stops when you press Ctrl+C**

### **Example Output:**
```
🌊 Starting Streaming Mode
==================================================
📊 Symbol List: demo_small
⚡ Speed: 60x (real-time)
🏗️  Auto-start Infrastructure: False

🌊 Streaming will run continuously until stopped
⏹️  Press Ctrl+C to stop the stream

✅ Infrastructure is healthy
📡 Starting data replay...
✅ Data replay started
🌊 Streaming mode active - data is flowing through the system
📊 Check web interfaces for real-time monitoring:
📋 Service URLs:
  • Spark UI: http://localhost:8080
  • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
  • Kibana: http://localhost:5601
  • Elasticsearch: http://localhost:9200
  • Kafka: localhost:9092
```

### **When to use:**
- ✅ **Real-time system testing**
- ✅ **Streaming pipeline development**
- ✅ **Performance testing**
- ✅ **Simulating live trading**

### **Pros:**
- Real-time data flow
- High-speed processing
- Good for testing streaming components
- Web interfaces show live data

### **Cons:**
- Uses historical data (not live)
- High resource usage
- May overwhelm downstream systems

## 🎯 Comparison Table

| Feature | Batch Mode | Pipeline Mode | Streaming Mode |
|---------|------------|---------------|----------------|
| **Control** | Manual | Automated | Automated |
| **Continuity** | One-time | Continuous loop | Continuous flow |
| **Speed** | Normal | Normal | High-speed |
| **Resource Usage** | Low | Medium | High |
| **Use Case** | Development | Production | Testing |
| **Data Source** | Live/Historical | Live/Historical | Historical replay |
| **Monitoring** | Manual | CLI stats | Web interfaces |

## 🚀 Quick Start Examples

### **For Development:**
```bash
# Batch mode - step by step
poetry run bf infra start
poetry run bf data fetch --symbol-list demo_small
poetry run bf signals generate --symbol-list demo_small
poetry run bf backtest run --symbol-list demo_small
poetry run bf infra stop
```

### **For Continuous Monitoring:**
```bash
# Pipeline mode - runs every 5 minutes
poetry run bf pipeline --symbol-list demo_small --auto-start-infra
```

### **For Real-time Testing:**
```bash
# Streaming mode - high-speed data flow
poetry run bf stream --symbol-list demo_small --speed 120 --auto-start-infra
```

## 🔧 Configuration Options

### **Pipeline Mode Options:**
- `--symbol-list`: Which symbols to analyze
- `--interval`: Seconds between runs (default: 300)
- `--start-date`: Start date for analysis
- `--end-date`: End date for analysis
- `--auto-start-infra`: Automatically start/stop infrastructure

### **Streaming Mode Options:**
- `--symbol-list`: Which symbols to replay
- `--speed`: Replay speed multiplier (default: 60)
- `--auto-start-infra`: Automatically start/stop infrastructure

## 🎯 Best Practices

### **For Development:**
1. **Start with batch mode** to understand each step
2. **Use demo_small** for quick testing
3. **Check web interfaces** to monitor progress

### **For Production:**
1. **Use pipeline mode** for continuous operation
2. **Set appropriate intervals** (5-15 minutes)
3. **Monitor resource usage**
4. **Set up alerts** for failures

### **For Testing:**
1. **Use streaming mode** for real-time testing
2. **Start with lower speeds** (30x-60x)
3. **Monitor web interfaces** for live data
4. **Test error handling** by stopping/starting

## 🚨 Troubleshooting

### **Pipeline Mode Issues:**
```bash
# If pipeline fails, check logs
poetry run bf infra logs

# If infrastructure is down
poetry run bf infra start

# If you need to stop pipeline
# Press Ctrl+C
```

### **Streaming Mode Issues:**
```bash
# If streaming is too fast, reduce speed
poetry run bf stream --speed 30

# If data replay fails, check data exists
poetry run bf data summary

# If you need to stop streaming
# Press Ctrl+C
```

## 📚 Next Steps

1. **Try batch mode first** to understand the system
2. **Graduate to pipeline mode** for continuous operation
3. **Use streaming mode** for real-time testing
4. **Monitor performance** and adjust intervals/speeds
5. **Set up production monitoring** for long-running pipelines

---

**Choose the mode that fits your use case! 🚀**
