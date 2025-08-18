# 📊 Kibana Dashboard Creation Guide for BreadthFlow

Complete guide to creating powerful dashboards in Kibana for monitoring your BreadthFlow pipelines.

## 🎯 **What's Already Created**

✅ **Basic Dashboard**: "🚀 BreadthFlow Pipeline Monitor" - Shows recent pipeline logs  
✅ **Error Search**: "🚨 BreadthFlow Errors" - Filters for errors and warnings  
✅ **Index Pattern**: "breadthflow-logs*" - Connected to your pipeline data  

## 🔧 **How to Access**

1. **Open Kibana**: http://localhost:5601
2. **Go to Dashboard**: Click "☰" menu → "Dashboard" 
3. **Browse Dashboards**: Click "Browse dashboards"
4. **Find BreadthFlow**: Look for "🚀 BreadthFlow Pipeline Monitor"

---

## 🎨 **Create Custom Dashboards (Manual)**

### **Step 1: Start Creating**
1. Go to **Dashboard** → **Create new dashboard**
2. Click **"Create visualization"**
3. Choose your visualization type

### **Step 2: Configure Data Source**
- **Index pattern**: Select `breadthflow-logs*`
- **Time field**: `@timestamp`

---

## 📈 **Recommended Visualizations**

### **1. 📊 Pipeline Runs Timeline**
**Type**: Line Chart
```
X-axis: @timestamp (Date Histogram, interval: auto)
Y-axis: Count of documents
Filter: status:started
Split series: command.keyword
```

**Shows**: Number of pipeline runs over time, split by command type

### **2. 🎯 Success Rate Gauge**
**Type**: Metric
```
Metric: Count of documents where status:completed
Filter: status:completed OR status:failed
```

**Formula**: (completed / (completed + failed)) * 100

### **3. ⏱️ Duration Heatmap**
**Type**: Line Chart
```
X-axis: @timestamp
Y-axis: Average of duration
Filter: status:completed AND duration:*
Split series: command.keyword
```

**Shows**: Average pipeline duration trends

### **4. 🔍 Symbol Processing Status**
**Type**: Data Table
```
Rows: metadata.symbol.keyword (Top 20)
Metrics: Count
Filter: metadata.symbol:*
Split table: metadata.fetch_status.keyword
```

**Shows**: Which symbols are processed most and their success rates

### **5. 🚨 Error Frequency**
**Type**: Bar Chart
```
X-axis: level.keyword
Y-axis: Count
Filter: level:ERROR OR level:WARN
```

**Shows**: Distribution of error types

### **6. 📋 Recent Runs Table**
**Type**: Data Table
```
Columns: @timestamp, command.keyword, status.keyword, duration, run_id.keyword
Sort: @timestamp descending
Filter: status:started OR status:completed OR status:failed
```

**Shows**: Recent pipeline executions with details

---

## 🏗️ **Dashboard Layout Ideas**

### **📊 Executive Dashboard**
```
┌─────────────────┬─────────────────┐
│  Success Rate   │  Total Runs     │
│     Gauge       │    Metric       │
├─────────────────┴─────────────────┤
│        Pipeline Runs Timeline     │
│           (Last 24h)              │
├─────────────────┬─────────────────┤
│ Command Types   │  Recent Runs    │
│   Pie Chart     │    Table        │
└─────────────────┴─────────────────┘
```

### **🔍 Operations Dashboard**
```
┌─────────────────────────────────────┐
│     Duration Trends (Line Chart)   │
├─────────────────┬─────────────────┤
│ Symbol Success  │   Error Logs    │
│     Table       │     Table       │
├─────────────────┴─────────────────┤
│         Live Log Stream            │
│        (Auto-refresh)              │
└─────────────────────────────────────┘
```

### **🚨 Troubleshooting Dashboard**
```
┌─────────────────────────────────────┐
│        Error Timeline               │
├─────────────────┬─────────────────┤
│ Error Types     │ Failed Symbols  │
│  Bar Chart      │     Table       │
├─────────────────┴─────────────────┤
│         Detailed Error Logs        │
│         (with full messages)       │
└─────────────────────────────────────┘
```

---

## 🔧 **Advanced Configurations**

### **🔄 Auto-Refresh Setup**
1. Open your dashboard
2. Click **"Refresh"** button (top-right)
3. Select **"30 seconds"** or **"1 minute"**
4. **Save** the dashboard

### **📅 Time Range Presets**
Create quick time filters:
- **Last 1 hour**: `now-1h` to `now`
- **Today**: `now/d` to `now/d`
- **Last 24 hours**: `now-24h` to `now`
- **This week**: `now/w` to `now/w`

### **🎯 Useful Filters**
Add these as dashboard filters:

```bash
# Only show completed runs
status: "completed"

# Hide test runs  
NOT command: "*test*"

# Only show data fetch operations
command: "*fetch*"

# Only show errors and warnings
level: ("ERROR" OR "WARN")

# Specific symbols only
metadata.symbol: ("AAPL" OR "MSFT" OR "GOOGL")
```

### **🔍 Search Patterns**
Use these in the search bar:

```bash
# Find long-running operations
duration: >10

# Find failed symbol fetches
metadata.fetch_status: "failed"

# Find specific run
run_id: "your-run-id"

# Find patterns in messages
message: "*timeout*" OR message: "*connection*"
```

---

## 💡 **Pro Tips**

### **🎨 Visualization Tips**
1. **Use consistent colors**: Set color schemes across visualizations
2. **Add descriptions**: Help team members understand each chart
3. **Size appropriately**: Important metrics get bigger panels
4. **Group related**: Put similar visualizations near each other

### **⚡ Performance Tips**
1. **Limit time ranges**: Don't query months of data unnecessarily
2. **Use filters**: Pre-filter data to reduce query load
3. **Aggregate when possible**: Use averages/sums instead of raw data
4. **Cache queries**: Kibana caches similar queries automatically

### **🔧 Maintenance Tips**
1. **Export dashboards**: Backup your configurations
2. **Document filters**: Note what each filter does
3. **Regular cleanup**: Remove unused visualizations
4. **Monitor performance**: Watch for slow queries

---

## 📱 **Mobile-Friendly Dashboard**

For monitoring on mobile devices:

```
┌─────────────────┐
│   Success Rate  │
│     (Large)     │
├─────────────────┤
│  Recent Errors  │
│   (Compact)     │
├─────────────────┤
│  Current Status │
│   (Simple)      │
└─────────────────┘
```

---

## 🎯 **Quick Start Checklist**

- [ ] Open Kibana at http://localhost:5601
- [ ] Go to Dashboard → Browse dashboards  
- [ ] Open "🚀 BreadthFlow Pipeline Monitor"
- [ ] Set auto-refresh to 30 seconds
- [ ] Add time filter for "Last 24 hours"
- [ ] Create your first custom visualization
- [ ] Add it to a new dashboard
- [ ] Save and share with your team

---

## 🚀 **Sample KQL Queries**

Copy these into Kibana's search bar:

```bash
# Pipeline overview
status:started

# Successful operations only
status:completed AND duration:*

# Problem detection
level:ERROR OR (status:failed)

# Performance monitoring  
duration:>5 AND command:*fetch*

# Symbol-specific analysis
metadata.symbol:AAPL AND metadata.fetch_status:*

# Recent activity
@timestamp:>=now-1h AND status:(started OR completed)
```

---

🎉 **You now have everything needed to create powerful Kibana dashboards for monitoring your BreadthFlow pipelines!**

The combination of pre-built dashboards + this guide gives you both immediate value and the ability to customize for your specific needs.
