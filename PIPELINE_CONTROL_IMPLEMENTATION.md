# Pipeline Control Implementation

## Overview

This implementation provides a clean, modular approach to pipeline management with proper state tracking and user interface controls. The solution is split into three main components:

1. **Pipeline Controller** (`cli/pipeline_controller.py`) - Core business logic
2. **Pipeline Dashboard** (`cli/pipeline_dashboard.py`) - UI generation
3. **Updated Main Dashboard** (`cli/postgres_dashboard.py`) - Integration

## Key Features Implemented

### ✅ Pipeline State Management
- **Single Pipeline Rule**: Only one pipeline can run at a time
- **Proper State Tracking**: Running, stopped, failed states
- **Database Persistence**: All pipeline states stored in PostgreSQL
- **Unique Pipeline IDs**: Each pipeline gets a unique identifier

### ✅ Button State Management
- **Start Button**: Disabled when pipeline is running
- **Stop Button**: Disabled when no pipeline is running
- **Dynamic Updates**: Button states update automatically based on current status

### ✅ Pipeline Runs Table
- **Last 2 Days**: Shows pipelines from the last 2 days (not just 2 hours)
- **Individual Sessions**: Each pipeline start creates a separate row
- **Status Tracking**: Shows running, stopped, completed, failed statuses
- **Duration Calculation**: Accurate duration tracking for each pipeline

### ✅ User Experience
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Clear Feedback**: Success/error messages for all actions
- **Visual Indicators**: Color-coded status indicators
- **Responsive Design**: Modern, clean interface

## Architecture

### Pipeline Controller (`pipeline_controller.py`)

```python
class PipelineController:
    def is_pipeline_running(self) -> bool
    def get_running_pipeline_id(self) -> Optional[str]
    def start_pipeline(self, config: Dict[str, Any]) -> Dict[str, Any]
    def stop_pipeline(self) -> Dict[str, Any]
    def get_pipeline_status(self) -> Dict[str, Any]
    def get_recent_pipeline_runs(self, days: int = 2) -> List[Dict[str, Any]]
```

**Key Methods:**
- `start_pipeline()`: Creates new pipeline with unique ID, checks for running pipelines
- `stop_pipeline()`: Stops currently running pipeline, updates status
- `get_pipeline_status()`: Returns current state and metrics
- `get_recent_pipeline_runs()`: Returns pipeline history from last N days

### Pipeline Dashboard (`pipeline_dashboard.py`)

```python
class PipelineDashboard:
    def generate_html(self) -> str
```

**Features:**
- Clean, modern HTML/CSS interface
- JavaScript for real-time updates
- Button state management
- Responsive design

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pipeline/start` | POST | Start new pipeline |
| `/api/pipeline/stop` | POST | Stop running pipeline |
| `/api/pipeline/status` | GET | Get current status |
| `/api/pipeline/runs` | GET | Get recent runs |

## Database Schema

The implementation uses the existing `pipeline_runs` table with enhanced tracking:

```sql
CREATE TABLE pipeline_runs (
    run_id VARCHAR(255) PRIMARY KEY,
    command TEXT,
    status VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration FLOAT,
    error_message TEXT,
    metadata JSONB
);
```

**Key Fields:**
- `run_id`: Unique pipeline identifier
- `status`: running, stopped, completed, failed
- `metadata`: JSON configuration for each pipeline
- `duration`: Calculated runtime

## Usage Flow

### Starting a Pipeline
1. User clicks "Start Pipeline" button
2. Frontend sends configuration to `/api/pipeline/start`
3. Controller checks if pipeline is already running
4. If not running, creates new pipeline record with unique ID
5. Calls Spark command server to start actual pipeline
6. Updates UI with success/error message
7. Button states update automatically

### Stopping a Pipeline
1. User clicks "Stop Pipeline" button
2. Frontend sends request to `/api/pipeline/stop`
3. Controller finds currently running pipeline
4. Calls Spark command server to stop pipeline
5. Updates pipeline status to "stopped"
6. Updates UI with success/error message
7. Button states update automatically

### Viewing Pipeline History
1. Dashboard loads automatically
2. Fetches recent runs from `/api/pipeline/runs`
3. Displays table with last 2 days of pipeline activity
4. Shows individual pipeline sessions with status and duration
5. Auto-refreshes every 30 seconds

## Error Handling

### Database Connection Errors
- Graceful fallback to default states
- Clear error messages to user
- Logging for debugging

### Spark Server Errors
- Proper error propagation
- User-friendly error messages
- Status tracking for failed operations

### Concurrent Access
- Database-level state checking
- Prevents multiple pipelines from running
- Clear feedback when pipeline is already running

## Testing

Run the test script to verify functionality:

```bash
cd cli
python test_pipeline_controller.py
```

This will test:
- Pipeline state checking
- Status retrieval
- Recent runs fetching
- Start/stop operations (with proper error handling)

## Benefits

1. **Modular Design**: Clean separation of concerns
2. **Maintainable Code**: Easy to modify and extend
3. **Reliable State Management**: Proper database tracking
4. **User-Friendly Interface**: Clear feedback and controls
5. **Scalable Architecture**: Easy to add new features

## Future Enhancements

1. **Pipeline Configuration**: Save/load pipeline configurations
2. **Advanced Metrics**: More detailed performance tracking
3. **Logging Integration**: Real-time log viewing
4. **Alerting**: Notifications for pipeline failures
5. **Scheduling**: Automated pipeline scheduling

## Files Modified/Created

### New Files
- `cli/pipeline_controller.py` - Core pipeline management logic
- `cli/pipeline_dashboard.py` - Clean UI implementation
- `cli/test_pipeline_controller.py` - Test script
- `PIPELINE_CONTROL_IMPLEMENTATION.md` - This documentation

### Modified Files
- `cli/postgres_dashboard.py` - Updated to use new controller

The implementation provides a robust, user-friendly pipeline management system that meets all the specified requirements while maintaining clean, maintainable code.
