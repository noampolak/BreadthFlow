#!/bin/bash
# BreadthFlow Infrastructure Restart Script
# Stops and restarts all services

set -e  # Exit on any error

echo "🔄 BreadthFlow Infrastructure Restart"
echo "====================================="

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

print_status "Restarting BreadthFlow infrastructure..."

# Stop services
print_status "Step 1/2: Stopping services..."
docker-compose down

# Start services
print_status "Step 2/2: Starting services..."
docker-compose up -d

print_success "Infrastructure restarted successfully!"

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 30

echo ""
echo "🌐 Access your services:"
echo "   • 🎯 Web Dashboard: http://localhost:8083"
echo "   • 📊 Kibana Analytics: http://localhost:5601"
echo "   • 🎨 Kafka UI (Kafdrop): http://localhost:9002"
echo "   • 🗄️ MinIO Storage: http://localhost:9001 (minioadmin/minioadmin)"
echo "   • ⚡ Spark UI: http://localhost:8080"
echo ""
echo "📋 Next steps:"
echo "   • Check status: ./scripts/check_status.sh"
echo "   • Run demo: ./scripts/run_demo.sh"
echo ""
