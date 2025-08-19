#!/bin/bash
# BreadthFlow Infrastructure Stop Script
# Safely stops all services and cleans up

set -e  # Exit on any error

echo "🛑 BreadthFlow Infrastructure Stop"
echo "=================================="

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

# Check if containers are running
if ! docker ps | grep -q "spark-master"; then
    print_warning "No BreadthFlow containers are currently running"
    exit 0
fi

print_status "Stopping BreadthFlow infrastructure..."

# Stop all containers
print_status "Stopping Docker containers..."
docker-compose down

print_success "All services stopped successfully!"

echo ""
echo "💡 To restart services:"
echo "   • ./scripts/start_infrastructure.sh"
echo ""
echo "💡 To clean up completely (removes volumes):"
echo "   • docker-compose down -v"
echo "   • docker system prune -f"
echo ""
