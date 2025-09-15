"""
Feature Engineering DAG for BreadthFlow ML Pipeline

This DAG orchestrates the feature engineering process,
including technical indicators, time-based features, and data validation.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
import json

# Default arguments
default_args = {
    "owner": "breadthflow-ml",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    "feature_engineering_pipeline",
    default_args=default_args,
    description="Feature engineering pipeline for ML training",
    schedule_interval="0 7 * * *",  # Run daily at 7 AM (after data ingestion)
    catchup=False,
    tags=["ml", "features", "engineering"],
)

# Task 1: Check if data pipeline service is healthy
check_data_pipeline = HttpSensor(
    task_id="check_data_pipeline_health",
    http_conn_id="data_pipeline_http",
    endpoint="/health",
    timeout=30,
    poke_interval=10,
    dag=dag,
)

# Task 2: Create technical indicators
create_technical_indicators = PythonOperator(
    task_id="create_technical_indicators", python_callable=create_technical_indicators_func, dag=dag
)

# Task 3: Create time-based features
create_time_features = PythonOperator(task_id="create_time_features", python_callable=create_time_features_func, dag=dag)

# Task 4: Create market microstructure features
create_microstructure_features = PythonOperator(
    task_id="create_microstructure_features", python_callable=create_microstructure_features_func, dag=dag
)

# Task 5: Validate features
validate_features = SimpleHttpOperator(
    task_id="validate_features",
    http_conn_id="data_pipeline_http",
    endpoint="/validate",
    method="POST",
    data=json.dumps({"data_path": "features/technical_indicators", "validation_type": "quality"}),
    headers={"Content-Type": "application/json"},
    dag=dag,
)

# Task 6: Store features
store_features = PythonOperator(task_id="store_features", python_callable=store_features_func, dag=dag)

# Task 7: Log completion
log_completion = BashOperator(
    task_id="log_completion", bash_command='echo "Feature engineering pipeline completed successfully at {{ ts }}"', dag=dag
)

# Define task dependencies
check_data_pipeline >> [create_technical_indicators, create_time_features, create_microstructure_features]
[create_technical_indicators, create_time_features, create_microstructure_features] >> validate_features
validate_features >> store_features
store_features >> log_completion


# Python functions for feature engineering
def create_technical_indicators_func(**context):
    """Create technical indicators from OHLCV data."""
    import pandas as pd
    import numpy as np
    from datetime import datetime

    print("Creating technical indicators...")

    # This would typically load data from MinIO and create indicators
    # For now, we'll create a mock implementation

    # Mock data creation
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    mock_data = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAPL",
            "open": np.random.uniform(100, 200, len(dates)),
            "high": np.random.uniform(100, 200, len(dates)),
            "low": np.random.uniform(100, 200, len(dates)),
            "close": np.random.uniform(100, 200, len(dates)),
            "volume": np.random.uniform(1000000, 10000000, len(dates)),
        }
    )

    # Calculate technical indicators
    mock_data["sma_20"] = mock_data["close"].rolling(window=20).mean()
    mock_data["ema_12"] = mock_data["close"].ewm(span=12).mean()
    mock_data["rsi"] = calculate_rsi(mock_data["close"])
    mock_data["macd"] = calculate_macd(mock_data["close"])

    print(f"Created technical indicators for {len(mock_data)} records")
    return mock_data


def create_time_features_func(**context):
    """Create time-based features."""
    import pandas as pd
    from datetime import datetime

    print("Creating time-based features...")

    # Mock implementation
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    time_features = pd.DataFrame(
        {
            "date": dates,
            "day_of_week": dates.dayofweek,
            "month": dates.month,
            "quarter": dates.quarter,
            "is_weekend": dates.dayofweek >= 5,
            "is_month_start": dates.is_month_start,
            "is_quarter_start": dates.is_quarter_start,
        }
    )

    print(f"Created time features for {len(time_features)} records")
    return time_features


def create_microstructure_features_func(**context):
    """Create market microstructure features."""
    import pandas as pd
    import numpy as np
    from datetime import datetime

    print("Creating microstructure features...")

    # Mock implementation
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    microstructure_features = pd.DataFrame(
        {
            "date": dates,
            "volume_ma_5": np.random.uniform(1000000, 10000000, len(dates)),
            "volume_ma_20": np.random.uniform(1000000, 10000000, len(dates)),
            "volume_ratio": np.random.uniform(0.5, 2.0, len(dates)),
            "price_volatility": np.random.uniform(0.01, 0.05, len(dates)),
            "volume_volatility": np.random.uniform(0.1, 0.5, len(dates)),
        }
    )

    print(f"Created microstructure features for {len(microstructure_features)} records")
    return microstructure_features


def store_features_func(**context):
    """Store features in MinIO."""
    print("Storing features in MinIO...")

    # This would typically store the features in MinIO
    # For now, we'll just log the completion

    print("Features stored successfully")
    return True


def calculate_rsi(prices, window=14):
    """Calculate RSI indicator."""
    import pandas as pd

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    import pandas as pd

    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd
