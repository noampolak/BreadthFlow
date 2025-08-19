#!/bin/bash
# BreadthFlow Demo Script
# Demonstrates the complete pipeline: data fetching, signal generation, backtesting

set -e  # Exit on any error

echo "🎬 BreadthFlow Complete Demo"
echo "============================"

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

# Check if infrastructure is running
print_status "Checking infrastructure status..."
if ! docker ps | grep -q "spark-master"; then
    print_error "Infrastructure not running. Please run ./scripts/start_infrastructure.sh first"
    exit 1
fi

print_success "Infrastructure is running!"

echo ""
echo "🎯 Demo Pipeline Steps:"
echo "======================="

# Step 1: Data Summary
print_status "Step 1/5: Checking current data..."
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data summary

# Step 2: Fetch Market Data
print_status "Step 2/5: Fetching market data for demo symbols..."
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT,GOOGL --start-date 2024-08-15 --end-date 2024-08-16

# Step 3: Generate Trading Signals
print_status "Step 3/5: Generating trading signals..."
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_minio.py signals generate --symbols AAPL,MSFT,GOOGL --start-date 2024-08-15 --end-date 2024-08-16

# Step 4: Run Backtesting
print_status "Step 4/5: Running backtesting simulation..."
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_minio.py backtest run --symbols AAPL,MSFT,GOOGL --from-date 2024-08-15 --to-date 2024-08-16 --initial-capital 100000

# Step 5: Show Results
print_status "Step 5/5: Displaying results summary..."
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/bf_minio.py signals summary

print_success "Demo completed successfully!"
echo ""
echo "📊 View Results:"
echo "   • 🎯 Web Dashboard: http://localhost:8083 (Real-time pipeline monitoring)"
echo "   • 📊 Kibana Analytics: http://localhost:5601 (Detailed log analysis)"
echo "   • 🎨 Kafka UI: http://localhost:9002 (Streaming data monitoring)"
echo "   • 🗄️ MinIO Storage: http://localhost:9001 (Data storage)"
echo ""
echo "💡 Demo Highlights:"
echo "   • ✅ Data fetched from Yahoo Finance"
echo "   • ✅ Signals generated using technical analysis"
echo "   • ✅ Backtesting performed with realistic simulation"
echo "   • ✅ All results logged to PostgreSQL and Elasticsearch"
echo "   • ✅ Real-time monitoring available in dashboard"
echo ""
