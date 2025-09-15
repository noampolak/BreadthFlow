"""
Data Ingestion DAG for BreadthFlow ML Pipeline

This DAG orchestrates the data ingestion process for ML training,
including data fetching, validation, and storage.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.http_sensor import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
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
    "data_ingestion_pipeline",
    default_args=default_args,
    description="Data ingestion pipeline for ML training",
    schedule_interval="0 6 * * *",  # Run daily at 6 AM
    catchup=False,
    tags=["ml", "data", "ingestion"],
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

# Task 2: Get available symbol lists
get_symbol_lists = SimpleHttpOperator(
    task_id="get_symbol_lists", http_conn_id="data_pipeline_http", endpoint="/symbol-lists", method="GET", dag=dag
)

# Task 3: Ingest data for demo_small symbol list
ingest_demo_small = SimpleHttpOperator(
    task_id="ingest_demo_small_data",
    http_conn_id="data_pipeline_http",
    endpoint="/ingest",
    method="POST",
    data=json.dumps(
        {
            "symbol_list": "demo_small",
            "start_date": "{{ ds }}",  # Use Airflow's execution date
            "end_date": "{{ ds }}",
            "validate_data": True,
        }
    ),
    headers={"Content-Type": "application/json"},
    dag=dag,
)

# Task 4: Ingest data for demo_medium symbol list
ingest_demo_medium = SimpleHttpOperator(
    task_id="ingest_demo_medium_data",
    http_conn_id="data_pipeline_http",
    endpoint="/ingest",
    method="POST",
    data=json.dumps({"symbol_list": "demo_medium", "start_date": "{{ ds }}", "end_date": "{{ ds }}", "validate_data": True}),
    headers={"Content-Type": "application/json"},
    dag=dag,
)

# Task 5: Validate ingested data
validate_data = SimpleHttpOperator(
    task_id="validate_ingested_data",
    http_conn_id="data_pipeline_http",
    endpoint="/validate",
    method="POST",
    data=json.dumps({"data_path": "ohlcv/demo_small", "validation_type": "quality"}),
    headers={"Content-Type": "application/json"},
    dag=dag,
)

# Task 6: Get storage summary
get_storage_summary = SimpleHttpOperator(
    task_id="get_storage_summary", http_conn_id="data_pipeline_http", endpoint="/storage/summary", method="GET", dag=dag
)

# Task 7: Log completion
log_completion = BashOperator(
    task_id="log_completion", bash_command='echo "Data ingestion pipeline completed successfully at {{ ts }}"', dag=dag
)

# Define task dependencies
check_data_pipeline >> get_symbol_lists
get_symbol_lists >> [ingest_demo_small, ingest_demo_medium]
[ingest_demo_small, ingest_demo_medium] >> validate_data
validate_data >> get_storage_summary
get_storage_summary >> log_completion
