# Enhanced Batch Processing Plan

## Core Concept
Transform the current manual command execution into an automated, scheduled batch processing system that runs continuously until stopped, with comprehensive monitoring and dashboard integration.

## System Architecture

### Components
1. **Pipeline Runner**: Automated executor that triggers existing commands
2. **Scheduler**: Controls execution frequency and timing
3. **Enhanced Signal Storage**: Signals with `create_time` timestamps
4. **Pipeline Metadata Tracking**: Comprehensive run history and metrics
5. **Dashboard Integration**: Real-time monitoring and control

## Technical Implementation

### 1. New CLI Commands in `kibana_enhanced_bf.py`
```bash
# Start continuous pipeline
python kibana_enhanced_bf.py pipeline start --mode=demo --interval=5m

# Stop pipeline
python kibana_enhanced_bf.py pipeline stop

# Status check
python kibana_enhanced_bf.py pipeline status

# View logs
python kibana_enhanced_bf.py pipeline logs
```

### 2. Pipeline Runner
- **Triggers**: `data fetch`, `signals generate`, `backtest run` commands
- **Uses**: Existing Spark API (no changes to core commands)
- **Runs**: Continuously until stop order
- **Logs**: All activities to PostgreSQL/Elasticsearch

### 3. Enhanced Signal Storage
```python
# Signal format with create_time
{
    "symbol": "AAPL",
    "date": "2025-08-21",
    "signal_type": "BUY",
    "confidence": 0.85,
    "create_time": "2025-08-21T10:30:00Z",  # NEW
    "pipeline_run_id": "run_20250821_103000"  # NEW
}
```

### 4. Pipeline Metadata Tracking
```python
# Pipeline run record
{
    "run_id": "run_20250821_103000",
    "start_time": "2025-08-21T10:30:00Z",
    "end_time": "2025-08-21T10:35:00Z",
    "status": "completed",
    "mode": "demo",
    "symbols_processed": 5,
    "signals_generated": 3,
    "backtest_results": {...},
    "error_count": 0
}
```

## Dashboard Integration

### New Dashboard Page: "Pipeline Management"
**Location**: New tab/page in the dashboard alongside "Commands", "Trading Signals", etc.

### Dashboard Features
1. **Pipeline Control Panel**
   - Start/Stop pipeline buttons (execute via Spark command server)
   - Real-time status display
   - Configuration options (mode, interval) with form inputs
   - Current pipeline status indicator

2. **Pipeline Runs History**
   - Enhanced table showing all pipeline runs
   - Success/failure rates with visual indicators
   - Performance metrics (execution time, symbols processed)
   - Drill-down capability to see individual run details

3. **Pipeline Configuration**
   - Mode selection (demo, all_symbols, custom_symbols)
   - Interval setting (5min, 15min, 1hour, custom)
   - Symbol list configuration
   - Error handling preferences

4. **Real-time Monitoring**
   - Live pipeline status updates
   - Error alerts and notifications
   - Performance graphs and metrics
   - Log streaming capability

## Monitoring & Alerting

### Metrics Tracked
- Pipeline execution time
- Symbols processed per run
- Signals generated per run
- Error rates and types
- System resource usage

### Alerts
- Pipeline failures
- High error rates
- Performance degradation
- System resource issues

## Integration with Existing Commands

### Current Commands (No Changes)
- `kibana_enhanced_bf.py data fetch` - Data fetching
- `kibana_enhanced_bf.py signals generate` - Signal generation
- `kibana_enhanced_bf.py backtest run` - Backtesting

### New Pipeline Commands
- `python kibana_enhanced_bf.py pipeline start` - Start automated pipeline
- `python kibana_enhanced_bf.py pipeline stop` - Stop pipeline
- `python kibana_enhanced_bf.py pipeline status` - Check status
- `python kibana_enhanced_bf.py pipeline logs` - View logs

## Implementation Phases

### Phase 1: Core Pipeline Runner
1. Create `pipeline` command group in `kibana_enhanced_bf.py`
2. Implement basic start/stop functionality
3. Add pipeline metadata tracking
4. Test with demo mode

### Phase 2: Dashboard Integration
1. Add "Pipeline Management" page to dashboard
2. Implement pipeline control panel (start/stop buttons)
3. Add pipeline configuration forms (mode, interval, symbols)
4. Enhance pipeline runs table with drill-down capability
5. Add real-time status updates and monitoring

### Phase 3: Advanced Features
1. Add scheduling options and cron-like functionality
2. Implement error handling and retries
3. Add performance monitoring and alerting
4. Create log streaming and debugging tools

## File Structure

### New Files
```
model/pipeline_runner.py           # Core pipeline logic
model/pipeline_metadata.py         # Metadata tracking
```

### Modified Files
```
cli/kibana_enhanced_bf.py          # Add pipeline commands
cli/postgres_dashboard.py          # Add Pipeline Management page
cli/spark_command_server.py        # Add pipeline endpoints
infra/docker-compose.yml           # Add pipeline service (if needed)
README.md                          # Update with pipeline docs
```

## Success Criteria

1. **Automation**: Pipeline runs continuously without manual intervention
2. **Monitoring**: All activities visible in dashboard
3. **Reliability**: Graceful error handling and recovery
4. **Performance**: Efficient execution with minimal overhead
5. **Usability**: Simple start/stop controls

## Questions for Approval

1. **Execution Frequency**: How often should the pipeline run? (5min, 15min, 1hour?)
2. **Modes**: What modes should be supported? (demo, all_symbols, custom_symbols?)
3. **Error Handling**: How should failures be handled? (retry, skip, stop?)
4. **Resource Limits**: Any constraints on system resources?
5. **Scheduling**: Should it support cron-like scheduling or just intervals?

---

**Note**: This plan uses `kibana_enhanced_bf.py` as the primary CLI and integrates with the existing dashboard infrastructure. All pipeline management will be done through the dashboard UI, which communicates with the Spark command server to execute the actual pipeline commands.
