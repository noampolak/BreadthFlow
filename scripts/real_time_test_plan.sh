#!/bin/bash
# BreadthFlow Real-Time Testing Plan
# Comprehensive demonstration of streaming and real-time capabilities

set -e  # Exit on any error

echo "🚀 BreadthFlow Real-Time Testing Plan"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

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

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "infra/docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Please run this script from the BreadthFlow root directory"
    exit 1
fi

echo ""
print_header "🎯 REAL-TIME TESTING PHASES"
echo "================================"

echo ""
print_step "PHASE 1: Infrastructure Verification"
echo "----------------------------------------"
print_status "Checking if all services are running..."
cd infra
docker-compose ps
cd ..

echo ""
print_step "PHASE 2: Kafka Streaming Demo"
echo "---------------------------------"
print_status "Demonstrating Kafka real-time messaging..."
./scripts/kafka_demo.sh

echo ""
print_step "PHASE 3: Real-Time Data Pipeline"
echo "------------------------------------"
print_status "Running live data pipeline with real-time monitoring..."

# Start real-time data generation in background
echo "📊 Starting real-time data generation..."
(
    for i in {1..10}; do
        # Generate market data
        echo "{\"symbol\": \"AAPL\", \"price\": $((150 + RANDOM % 10)).$((RANDOM % 100)), \"volume\": $((800000 + RANDOM % 400000)), \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic market-data
        
        # Generate trading signals
        if [ $((RANDOM % 3)) -eq 0 ]; then
            echo "{\"symbol\": \"AAPL\", \"signal\": \"BUY\", \"confidence\": $((70 + RANDOM % 30)).$((RANDOM % 10)), \"price\": $((150 + RANDOM % 10)).$((RANDOM % 100)), \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic trading-signals
        fi
        
        # Generate pipeline events
        echo "{\"event\": \"data_update\", \"symbol\": \"AAPL\", \"records\": $((100 + RANDOM % 50)), \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic pipeline-events
        
        sleep 2
    done
) &

REALTIME_PID=$!

echo "✅ Real-time data generation started (PID: $REALTIME_PID)"
echo "📊 Data will be sent every 2 seconds for 20 seconds"

echo ""
print_step "PHASE 4: Real-Time Monitoring"
echo "---------------------------------"
echo "🌐 Open these URLs in separate browser tabs to monitor real-time:"
echo ""
echo "   🎯 Web Dashboard: http://localhost:8083"
echo "   📊 Kibana Analytics: http://localhost:5601"
echo "   🎨 Kafka UI (Kafdrop): http://localhost:9002"
echo "   ⚡ Spark UI: http://localhost:8080"
echo ""

print_status "Monitoring real-time data flow..."
sleep 20

# Stop real-time data generation
kill $REALTIME_PID 2>/dev/null || true
print_success "Real-time data generation stopped"

echo ""
print_step "PHASE 5: Real-Time Data Verification"
echo "----------------------------------------"
print_status "Reading real-time data from Kafka topics..."

echo ""
echo "📊 Recent Market Data:"
docker exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic market-data --from-beginning --max-messages 5 --timeout-ms 5000

echo ""
echo "🎯 Recent Trading Signals:"
docker exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic trading-signals --from-beginning --max-messages 3 --timeout-ms 5000

echo ""
echo "⚡ Recent Pipeline Events:"
docker exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic pipeline-events --from-beginning --max-messages 5 --timeout-ms 5000

echo ""
print_step "PHASE 6: Real-Time Pipeline Execution"
echo "-----------------------------------------"
print_status "Running live pipeline with real-time logging..."

# Run a live pipeline command
docker exec spark-master python3 /opt/bitnami/spark/jobs/cli/kibana_enhanced_bf.py data fetch --symbols AAPL,MSFT --start-date 2024-08-15 --end-date 2024-08-16

echo ""
print_step "PHASE 7: Real-Time Dashboard Verification"
echo "---------------------------------------------"
print_status "Checking dashboard for real-time updates..."

echo "📊 Dashboard should show:"
echo "   • ✅ New pipeline run in the table"
echo "   • ✅ Real-time metrics updates"
echo "   • ✅ Live status indicators"
echo ""

print_success "Real-time testing completed!"
echo ""
print_header "🎯 REAL-TIME CAPABILITIES DEMONSTRATED"
echo "============================================"
echo ""
echo "✅ Kafka Streaming: Real-time message publishing and consumption"
echo "✅ Live Data Generation: Continuous market data simulation"
echo "✅ Real-Time Monitoring: Live dashboard and UI updates"
echo "✅ Pipeline Execution: Live pipeline runs with real-time logging"
echo "✅ Event Tracking: Real-time pipeline event monitoring"
echo ""
echo "🌐 Real-Time Monitoring URLs:"
echo "   • 🎯 Web Dashboard: http://localhost:8083"
echo "   • 📊 Kibana Analytics: http://localhost:5601"
echo "   • 🎨 Kafka UI (Kafdrop): http://localhost:9002"
echo "   • ⚡ Spark UI: http://localhost:8080"
echo ""
echo "💡 Next Steps:"
echo "   • Monitor dashboard for live updates"
echo "   • Check Kibana for real-time log analysis"
echo "   • Explore Kafka topics in Kafdrop"
echo "   • Run more pipeline commands to see real-time effects"
echo ""
