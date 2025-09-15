"""
Configuration management for Breadth/Thrust Signals POC.

Provides centralized configuration management with environment variable support.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the application."""

    def __init__(self):
        self._config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables."""
        self._config = {
            # Infrastructure
            "kafka_brokers": os.getenv("KAFKA_BROKERS", "localhost:9092"),
            "minio_endpoint": os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            "minio_access_key": os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            "minio_secret_key": os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            "minio_bucket": os.getenv("MINIO_BUCKET", "breadthflow"),
            "spark_master": os.getenv("SPARK_MASTER", "spark://spark-master:7077"),
            "elasticsearch_host": os.getenv("ELASTICSEARCH_HOST", "localhost:9200"),
            # Data Storage
            "DELTA_OHLCV_PATH": os.getenv("DELTA_OHLCV_PATH", "s3a://breadthflow/ohlcv"),
            "DELTA_SIGNALS_PATH": os.getenv("DELTA_SIGNALS_PATH", "s3a://breadthflow/signals"),
            "DELTA_BACKTEST_PATH": os.getenv("DELTA_BACKTEST_PATH", "s3a://breadthflow/backtests"),
            # Application
            "app_name": os.getenv("APP_NAME", "breadth-thrust-signals"),
            "app_version": os.getenv("APP_VERSION", "1.0.0"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            # Data
            "default_symbols": os.getenv("DEFAULT_SYMBOLS", "AAPL,MSFT,GOOGL,SPY,QQQ").split(","),
            "default_start_date": os.getenv("DEFAULT_START_DATE", "2023-01-01"),
            "default_end_date": os.getenv("DEFAULT_END_DATE", "2024-12-31"),
            "default_replay_speed": int(os.getenv("DEFAULT_REPLAY_SPEED", "60")),
            # Features
            "ad_window_size": os.getenv("AD_WINDOW_SIZE", "1m"),
            "ma_20_period": int(os.getenv("MA_20_PERIOD", "20")),
            "ma_50_period": int(os.getenv("MA_50_PERIOD", "50")),
            "mcclellan_ema_19": int(os.getenv("MCCLELLAN_EMA_19", "19")),
            "mcclellan_ema_39": int(os.getenv("MCCLELLAN_EMA_39", "39")),
            "zbt_window": int(os.getenv("ZBT_WINDOW", "10")),
            "zbt_low_threshold": float(os.getenv("ZBT_LOW_THRESHOLD", "0.40")),
            "zbt_high_threshold": float(os.getenv("ZBT_HIGH_THRESHOLD", "0.615")),
            # Scoring
            "score_buy_threshold": int(os.getenv("SCORE_BUY_THRESHOLD", "60")),
            "score_sell_threshold": int(os.getenv("SCORE_SELL_THRESHOLD", "40")),
            "score_weights": self._parse_score_weights(),
            # Backtesting
            "backtest_holding_days": int(os.getenv("BACKTEST_HOLDING_DAYS", "5")),
            "backtest_transaction_cost_bps": int(os.getenv("BACKTEST_TRANSACTION_COST_BPS", "2")),
            "backtest_slippage_ticks": int(os.getenv("BACKTEST_SLIPPAGE_TICKS", "1")),
            "backtest_max_positions": int(os.getenv("BACKTEST_MAX_POSITIONS", "10")),
            # Performance
            "spark_executor_memory": os.getenv("SPARK_EXECUTOR_MEMORY", "1g"),
            "spark_executor_cores": int(os.getenv("SPARK_EXECUTOR_CORES", "1")),
            "spark_driver_memory": os.getenv("SPARK_DRIVER_MEMORY", "1g"),
            "kafka_batch_size": int(os.getenv("KAFKA_BATCH_SIZE", "16384")),
            "kafka_linger_ms": int(os.getenv("KAFKA_LINGER_MS", "1")),
            # Monitoring
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "metrics_enabled": os.getenv("METRICS_ENABLED", "true").lower() == "true",
            "health_check_interval": os.getenv("HEALTH_CHECK_INTERVAL", "30s"),
        }

    def _parse_score_weights(self) -> Dict[str, float]:
        """Parse score weights from environment variable."""
        weights_str = os.getenv("SCORE_WEIGHTS", "{}")
        try:
            return json.loads(weights_str)
        except json.JSONDecodeError:
            # Default weights if parsing fails
            return {
                "adv_ratio": 0.2,
                "up_dn_vol_ratio": 0.2,
                "pct_above_ma20": 0.2,
                "pct_above_ma50": 0.2,
                "mcclellan_osc": 0.15,
                "zbt_flag": 0.05,
            }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def get_spark_config(self) -> Dict[str, str]:
        """Get Spark configuration."""
        return {
            "spark.master": self.get("spark_master"),
            "spark.executor.memory": self.get("spark_executor_memory"),
            "spark.executor.cores": str(self.get("spark_executor_cores")),
            "spark.driver.memory": self.get("spark_driver_memory"),
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.streaming.checkpointLocation": ".checkpoints",
        }

    def get_kafka_config(self) -> Dict[str, str]:
        """Get Kafka configuration."""
        return {
            "bootstrap.servers": self.get("kafka_brokers"),
            "acks": "all",
            "retries": "3",
            "batch.size": str(self.get("kafka_batch_size")),
            "linger.ms": str(self.get("kafka_linger_ms")),
            "buffer.memory": "33554432",
        }

    def get_minio_config(self) -> Dict[str, str]:
        """Get MinIO configuration."""
        return {
            "endpoint": self.get("minio_endpoint"),
            "access_key": self.get("minio_access_key"),
            "secret_key": self.get("minio_secret_key"),
            "bucket": self.get("minio_bucket"),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()


# Global configuration instance
_config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return _config


def get_spark_config() -> Dict[str, str]:
    """Get Spark configuration."""
    return _config.get_spark_config()


def get_kafka_config() -> Dict[str, str]:
    """Get Kafka configuration."""
    return _config.get_kafka_config()


def get_minio_config() -> Dict[str, str]:
    """Get MinIO configuration."""
    return _config.get_minio_config()
