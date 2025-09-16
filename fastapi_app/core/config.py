from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://pipeline:pipeline123@postgres:5432/breadthflow"

    # API
    api_title: str = "BreadthFlow API v2.0"
    api_version: str = "2.0.0"
    api_description: str = "Modern FastAPI backend for BreadthFlow dashboard"

    # Security
    secret_key: str = "breadthflow-secret-key-2024"
    access_token_expire_minutes: int = 30

    # External Services
    spark_command_server_url: str = "http://spark-master:8081"
    redis_url: str = "redis://redis:6379"

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3005",  # React dev server
        "http://localhost:8083",  # Legacy dashboard
        "http://localhost:80",  # Production frontend
    ]

    # App Settings
    debug: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
