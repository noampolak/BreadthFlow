#!/bin/bash
# BreadthFlow Infrastructure Startup Script
# Complete setup including Kafka topics, Kibana dashboards, and data initialization

set -e  # Exit on any error

echo "🚀 BreadthFlow Infrastructure Startup"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

print_status "Starting BreadthFlow infrastructure..."

# Step 1: Start all containers
print_status "Step 1/6: Starting Docker containers..."
docker-compose up -d

# Step 2: Wait for services to be ready
print_status "Step 2/6: Waiting for services to be ready..."
sleep 30

# Step 3: Verify all containers are running
print_status "Step 3/6: Verifying container status..."
docker-compose ps

# Step 4: Create Kafka topics
print_status "Step 4/6: Setting up Kafka topics..."
sleep 10  # Wait for Kafka to be fully ready

# Create financial data topics
docker exec kafka kafka-topics.sh --create --topic market-data --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 --if-not-exists
docker exec kafka kafka-topics.sh --create --topic trading-signals --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 --if-not-exists
docker exec kafka kafka-topics.sh --create --topic pipeline-events --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 --if-not-exists
docker exec kafka kafka-topics.sh --create --topic backtest-results --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 --if-not-exists

print_success "Created Kafka topics: market-data, trading-signals, pipeline-events, backtest-results"

# Step 5: Initialize Kibana dashboards
print_status "Step 5/6: Setting up Kibana dashboards..."
sleep 10  # Wait for Kibana to be ready

# Run Kibana setup script
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py setup-kibana

# Step 6: Generate initial data and logs
print_status "Step 6/6: Generating initial data and logs..."
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

print_success "Infrastructure startup completed!"
echo ""
echo "🌐 Access your services:"
echo "   • 🎯 Web Dashboard: http://localhost:8083"
echo "   • 📊 Kibana Analytics: http://localhost:5601"
echo "   • 🎨 Kafka UI (Kafdrop): http://localhost:9002"
echo "   • 🗄️ MinIO Storage: http://localhost:9001 (minioadmin/minioadmin)"
echo "   • ⚡ Spark UI: http://localhost:8080"
echo ""
echo "📋 Next steps:"
echo "   • Run demo: ./scripts/run_demo.sh"
echo "   • Check status: ./scripts/check_status.sh"
echo "   • Stop services: ./scripts/stop_infrastructure.sh"
echo ""
