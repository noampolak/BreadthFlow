# Breadth/Thrust Signals POC - PRD & Technical Design

**Version:** 2.0  
**Date:** 2025-01-XX  
**Authors:** Noam (Engineering), Yossi (Research)  

---

## Executive Summary

### Project Overview
Real-time quantitative trading signal system using market breadth indicators. PySpark-based architecture with Kafka streaming, Delta Lake storage, and ML-ready design.

### Key Objectives
- **4-week POC** with production-ready architecture
- **Real-time signal generation** using market breadth analysis
- **Scalable PySpark pipeline** with Kafka and Delta Lake
- **Comprehensive backtesting** and performance analysis
- **ML-ready design** for future enhancements

### Success Criteria
- **Performance:** 100 symbols, <1 second latency, >1000 msg/sec
- **Accuracy:** >60% hit rate in backtesting
- **Scalability:** Support 1000+ symbols
- **Usability:** 10-minute demo for new users

---

## Technical Architecture

### High-Level Architecture
```
Data Sources → PySpark Fetching → Delta Lake → Kafka Streaming → 
Spark Jobs → Feature Tables → Signal Generation → Backtesting → CLI
```

### Technology Stack
- **Core:** Python 3.11, PySpark 3.5.x, Kafka 3.x, Delta Lake 3.x
- **Storage:** MinIO (S3-compatible), Elasticsearch
- **Infrastructure:** Docker Compose, Spark Cluster
- **Monitoring:** Kibana, Spark UI

### Component Architecture

#### 1. Data Ingestion Layer
```python
class DataFetcher:
    def fetch_parallel(self, symbols, start_date, end_date):
        # Spark UDF for concurrent fetching
        # Rate limiting and error handling
        # Fault tolerance and retry logic
```

#### 2. Streaming Processing Layer
```python
class StreamingProcessor:
    def process_ad_features(self):
        # Window-based aggregation
        # Exactly-once processing
        # Checkpointing and recovery
```

#### 3. Signal Generation Layer
```python
class SignalGenerator:
    def compute_score(self, features):
        # Z-score normalization
        # Weighted combination
        # Threshold application
```

---

## Data Architecture

### Data Flow
```
Raw OHLCV → Cleaned Data → Features → Signals → Analysis
```

### Key Schemas

#### Raw Data (CSV)
```python
{
    "ts": "2024-01-02T14:35:00Z",
    "symbol": "AAPL",
    "open": 181.23, "high": 181.80,
    "low": 180.95, "close": 181.50,
    "volume": 345678
}
```

#### Features (Delta Tables)
```python
# A/D Basics
{
    "date": "2024-01-02",
    "adv_count": 65, "dec_count": 35,
    "adv_ratio": 0.65,
    "up_vol": 1234567, "dn_vol": 567890,
    "up_dn_vol_ratio": 2.17
}

# MA Flags
{
    "date": "2024-01-02",
    "pct_above_ma20": 0.72,
    "pct_above_ma50": 0.58
}

# McClellan
{
    "date": "2024-01-02",
    "mcclellan_osc": 7.3,
    "summation": 1250.8
}
```

#### Signals (Delta Table)
```python
{
    "date": "2024-01-02",
    "score_raw": 1.25,
    "score_0_100": 67.3,
    "flags": "THRUST",
    "inputs_version": "1.0"
}
```

---

## Implementation Plan

### Phase 1: Infrastructure & Setup (Week 1)
- **T-01:** Define POC scope and success metrics
- **T-02:** Select 100 liquid tickers
- **T-03:** Docker Compose setup (Spark, Kafka, MinIO, Elasticsearch)
- **T-04:** Project structure and dependencies

### Phase 2: Data Ingestion & Storage (Week 1)
- **T-05:** PySpark concurrent data fetcher
- **T-06:** Delta Lake storage setup
- **T-07:** Data quality framework

### Phase 3: Streaming Pipeline (Week 2)
- **T-08:** Delta to Kafka replay job
- **T-09:** A/D features streaming job
- **T-10:** Moving average features job
- **T-11:** McClellan oscillator job
- **T-12:** ZBT detection job

### Phase 4: Signal Generation (Week 2)
- **T-13:** Composite scoring job
- **T-14:** ML-ready architecture

### Phase 5: Backtesting & Analysis (Week 3)
- **T-15:** Backtesting engine
- **T-16:** Threshold calibration

### Phase 6: Monitoring & Search (Week 3)
- **T-17:** Elasticsearch integration
- **T-18:** Kibana dashboards

### Phase 7: CLI & Interface (Week 3)
- **T-19:** Enhanced CLI
- **T-20:** Demo & documentation

### Phase 8: ML Enhancement (Stretch - Week 4)
- **S-01:** Spark ML pipeline
- **S-02:** Advanced analytics

---

## Technical Specifications

### Performance Requirements
- **Throughput:** 1000+ messages/second
- **Latency:** <1 second end-to-end
- **Scalability:** 1000+ symbols
- **Reliability:** 99.9% uptime

### Data Quality Rules
- **Completeness:** ≤2% missing data
- **Accuracy:** Valid price/volume ranges
- **Consistency:** Schema validation, chronological order

### Security Requirements
- **Encryption:** Data at rest and in transit
- **Access Control:** Role-based permissions
- **Audit Logging:** Complete audit trail

---

## Testing Strategy

### Unit Testing
```python
def test_z_score_calculation():
    data = [1, 2, 3, 4, 5]
    z_scores = calculate_z_scores(data)
    assert len(z_scores) == 5
    assert abs(z_scores.mean()) < 0.001
```

### Integration Testing
```python
def test_data_pipeline():
    # End-to-end pipeline test
    data = fetch_test_data()
    features = process_features(data)
    signals = generate_signals(features)
    assert 'score_0_100' in signals.columns
```

### Performance Testing
- **Load Testing:** 100 concurrent users
- **Stress Testing:** Maximum throughput
- **Scalability Testing:** 1000+ symbols

---

## Deployment & Operations

### Development Environment
```yaml
# docker-compose.yml
version: '3.9'
services:
  spark:
    image: bitnami/spark:3.5
    ports: ["8080:8080"]
  
  kafka:
    image: bitnami/kafka:3.6
    ports: ["9092:9092"]
  
  minio:
    image: minio/minio:latest
    ports: ["9000:9000", "9001:9001"]
  
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11
    ports: ["9200:9200"]
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11
    ports: ["5601:5601"]
```

### Monitoring & Alerting
```python
MONITORING_CONFIG = {
    'metrics': {
        'processing_latency': {'threshold': 1000, 'unit': 'ms'},
        'throughput': {'threshold': 1000, 'unit': 'msg/s'},
        'error_rate': {'threshold': 0.01, 'unit': 'percentage'}
    }
}
```

---

## Risk Assessment

### Technical Risks
- **High:** Data quality issues, performance bottlenecks
- **Medium:** API rate limits, infrastructure failures
- **Low:** Dependency compatibility issues

### Mitigation Strategies
- **Comprehensive Testing:** Unit, integration, performance
- **Monitoring & Alerting:** Real-time system monitoring
- **Fault Tolerance:** Automatic recovery from failures
- **Documentation:** Complete technical documentation

---

## Success Metrics

### Technical Metrics
- **Latency:** <1 second end-to-end processing
- **Throughput:** >1000 messages/second
- **Availability:** >99.9% system uptime
- **Data Quality:** <2% missing/invalid data

### Business Metrics
- **Sharpe Ratio:** >1.0 for signal strategy
- **Hit Rate:** >60% profitable trades
- **Maximum Drawdown:** <20% portfolio drawdown
- **Demo Time:** <10 minutes for new users

---

## Appendices

### Appendix A: Algorithm Details

#### Z-Score Normalization
```python
def calculate_z_score(series, window=252):
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z_score = (series - mean) / std
    return z_score.clip(-3, 3)  # Winsorize
```

#### Composite Score
```python
def calculate_composite_score(features, weights=None):
    if weights is None:
        weights = {feature: 1.0/len(features) for feature in features}
    
    z_scores = {feature: calculate_z_score(features[feature]) 
                for feature in features}
    
    return sum(weights[feature] * z_scores[feature] 
               for feature in features)
```

### Appendix B: Configuration

#### Environment Variables
```bash
KAFKA_BROKERS=localhost:9092
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=breadthflow
SPARK_MASTER=local[*]
ELASTICSEARCH_HOST=localhost:9200
```

#### Spark Configuration
```python
spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension
spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
spark.sql.adaptive.enabled=true
spark.sql.streaming.checkpointLocation=.checkpoints
```

### Appendix C: CLI Commands

#### Core Commands
```bash
# Data fetching
bf fetch --symbols AAPL,MSFT,GOOGL --start 2023-01-01 --end 2024-12-31

# Streaming replay
bf replay --speed 60 --duration 1month

# Signal generation
bf score --date 2024-06-01

# Search signals
bf search --query "score_0_100:>70"

# Backtesting
bf backtest --from 2024-01-01 --to 2024-12-31
```

### Appendix D: Performance Benchmarks

#### Baseline Performance
- **Data Fetching:** 100 symbols in 8 minutes
- **Streaming Processing:** 1000 msg/sec throughput
- **Signal Generation:** <1 second latency
- **Backtesting:** 2 years of data in 5 minutes

#### Scalability Targets
- **Symbols:** 100 → 1000 → 10000
- **Time Granularity:** Daily → 5min → 1min
- **Processing Speed:** 1x → 60x → Real-time

---

**Document Version:** 2.0  
**Last Updated:** 2025-01-XX  
**Next Review:** 2025-02-XX  
**Approved By:** Noam (Engineering), Yossi (Research)
