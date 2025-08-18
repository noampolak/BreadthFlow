#!/usr/bin/env python3
"""
Infrastructure startup and health check script for Breadth/Thrust Signals POC.

This script starts the infrastructure and performs health checks.
"""

import subprocess
import time
import requests
import json
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)
    
    return result


def check_service_health(url: str, service_name: str) -> bool:
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"❌ {service_name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {service_name} is not accessible: {e}")
        return False


def main():
    """Main function to start infrastructure and perform health checks."""
    print("🚀 Starting Breadth/Thrust Signals Infrastructure")
    print("=" * 50)
    
    # Check if Docker Compose file exists
    docker_compose_file = Path("infra/docker-compose.yml")
    if not docker_compose_file.exists():
        print("❌ Docker Compose file not found!")
        print("Please ensure infra/docker-compose.yml exists")
        return
    
    # Start infrastructure
    print("\n📦 Starting Docker services...")
    try:
        run_command("docker compose -f infra/docker-compose.yml up -d")
        print("✅ Docker services started successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to start Docker services")
        return
    
    # Wait for services to start
    print("\n⏳ Waiting for services to start...")
    time.sleep(30)
    
    # Health checks
    print("\n🏥 Performing health checks...")
    services = [
        ("http://localhost:8080", "Spark UI"),
        ("http://localhost:9000/minio/health/live", "MinIO"),
        ("http://localhost:9200/_cluster/health", "Elasticsearch"),
        ("http://localhost:5601/api/status", "Kibana"),
    ]
    
    healthy_services = 0
    for url, name in services:
        if check_service_health(url, name):
            healthy_services += 1
    
    print(f"\n📊 Health Check Summary: {healthy_services}/{len(services)} services healthy")
    
    if healthy_services == len(services):
        print("\n🎉 All services are healthy! Infrastructure is ready.")
        print("\n📋 Service URLs:")
        print("  • Spark UI: http://localhost:8080")
        print("  • MinIO Console: http://localhost:9001 (admin/admin)")
        print("  • Kibana: http://localhost:5601")
        print("  • Elasticsearch: http://localhost:9200")
        print("  • Kafka: localhost:9092")
        
        print("\n🚀 Next steps:")
        print("  1. Install Python dependencies: pip install -r requirements.txt")
        print("  2. Copy environment config: cp env.example .env")
        print("  3. Run data fetching: python -m cli.bf fetch")
        print("  4. Start streaming: python -m cli.bf replay")
    else:
        print("\n⚠️  Some services are not healthy. Please check the logs:")
        print("  docker compose -f infra/docker-compose.yml logs")


if __name__ == "__main__":
    main()
