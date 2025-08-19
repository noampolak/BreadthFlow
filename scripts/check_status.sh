#!/bin/bash
# BreadthFlow Status Check Script
# Comprehensive health check for all services

set -e  # Exit on any error

echo "🔍 BreadthFlow Infrastructure Status"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Please run this script from the infra/ directory"
    exit 1
fi

echo ""
echo "🐳 Docker Containers Status:"
echo "============================"
docker-compose ps

echo ""
echo "🌐 Service Health Checks:"
echo "========================"

# Check PostgreSQL
if docker exec breadthflow-postgres pg_isready -U pipeline > /dev/null 2>&1; then
    print_success "PostgreSQL: Healthy"
else
    print_error "PostgreSQL: Unhealthy"
fi

# Check Elasticsearch
if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
    print_success "Elasticsearch: Healthy"
else
    print_error "Elasticsearch: Unhealthy"
fi

# Check Kibana
if curl -s http://localhost:5601/api/status > /dev/null 2>&1; then
    print_success "Kibana: Healthy"
else
    print_error "Kibana: Unhealthy"
fi

# Check MinIO
if curl -s http://localhost:9001/minio/health/live > /dev/null 2>&1; then
    print_success "MinIO: Healthy"
else
    print_warning "MinIO: May be starting up (this is normal)"
fi

# Check Kafka
if docker exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092 > /dev/null 2>&1; then
    print_success "Kafka: Healthy"
else
    print_error "Kafka: Unhealthy"
fi

# Check Kafdrop
if curl -s http://localhost:9002 > /dev/null 2>&1; then
    print_success "Kafdrop: Healthy"
else
    print_error "Kafdrop: Unhealthy"
fi

# Check Web Dashboard
if curl -s http://localhost:8083 > /dev/null 2>&1; then
    print_success "Web Dashboard: Healthy"
else
    print_error "Web Dashboard: Unhealthy"
fi

echo ""
echo "📊 Kafka Topics:"
echo "================"
if docker exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092 2>/dev/null; then
    print_success "Kafka topics listed successfully"
else
    print_error "Could not list Kafka topics"
fi

echo ""
echo "📈 Pipeline Data Summary:"
echo "========================"
if docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary 2>/dev/null; then
    print_success "Data summary retrieved"
else
    print_warning "Could not retrieve data summary (may be no data yet)"
fi

echo ""
echo "🌐 Service URLs:"
echo "================"
echo "   • 🎯 Web Dashboard: http://localhost:8083"
echo "   • 📊 Kibana Analytics: http://localhost:5601"
echo "   • 🎨 Kafka UI (Kafdrop): http://localhost:9002"
echo "   • 🗄️ MinIO Storage: http://localhost:9001 (minioadmin/minioadmin)"
echo "   • ⚡ Spark UI: http://localhost:8080"
echo ""
echo "💡 Quick Actions:"
echo "   • Start demo: ./scripts/run_demo.sh"
echo "   • Restart services: ./scripts/restart_infrastructure.sh"
echo "   • Stop services: ./scripts/stop_infrastructure.sh"
echo ""
