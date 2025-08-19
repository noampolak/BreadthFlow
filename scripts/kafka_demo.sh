#!/bin/bash
# BreadthFlow Kafka Demo Script
# Demonstrates Kafka streaming capabilities with financial data

set -e  # Exit on any error

echo "🎨 BreadthFlow Kafka Streaming Demo"
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

# Check if infrastructure is running
print_status "Checking infrastructure status..."
if ! docker ps | grep -q "kafka"; then
    print_error "Kafka not running. Please run ./scripts/start_infrastructure.sh first"
    exit 1
fi

print_success "Kafka is running!"

echo ""
echo "🔄 Kafka Flow in BreadthFlow:"
echo "============================="
echo ""
echo "📊 Current Topics:"
docker exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092

echo ""
echo "🎯 Kafka Use Cases in Financial Pipeline:"
echo "========================================="
echo "1. 📈 Live Market Data Streaming"
echo "2. 🔄 Historical Data Replay for Backtesting"
echo "3. ⚡ Real-time Signal Notifications"
echo "4. 🔄 Event-driven Pipeline Orchestration"
echo ""

print_status "Step 1/4: Sending market data to Kafka..."
# Send sample market data to Kafka
echo '{"symbol": "AAPL", "price": 150.25, "volume": 1000000, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic market-data
echo '{"symbol": "MSFT", "price": 320.50, "volume": 800000, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic market-data
echo '{"symbol": "GOOGL", "price": 2800.75, "volume": 500000, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic market-data

print_success "Sent 3 market data messages to Kafka"

print_status "Step 2/4: Sending trading signals to Kafka..."
# Send sample trading signals
echo '{"symbol": "AAPL", "signal": "BUY", "confidence": 85.5, "price": 150.25, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic trading-signals
echo '{"symbol": "MSFT", "signal": "HOLD", "confidence": 65.2, "price": 320.50, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic trading-signals

print_success "Sent 2 trading signals to Kafka"

print_status "Step 3/4: Sending pipeline events to Kafka..."
# Send pipeline events
echo '{"event": "data_fetch_completed", "symbols": ["AAPL", "MSFT", "GOOGL"], "records": 1500, "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic pipeline-events
echo '{"event": "signal_generation_started", "symbols": ["AAPL", "MSFT"], "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' | docker exec -i kafka kafka-console-producer.sh --bootstrap-server localhost:9092 --topic pipeline-events

print_success "Sent 2 pipeline events to Kafka"

print_status "Step 4/4: Reading messages from Kafka topics..."
echo ""
echo "📊 Market Data Messages:"
docker exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic market-data --from-beginning --max-messages 3 --timeout-ms 5000

echo ""
echo "🎯 Trading Signals:"
docker exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic trading-signals --from-beginning --max-messages 2 --timeout-ms 5000

echo ""
echo "⚡ Pipeline Events:"
docker exec kafka kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic pipeline-events --from-beginning --max-messages 2 --timeout-ms 5000

print_success "Kafka demo completed!"
echo ""
echo "🌐 View Kafka UI:"
echo "   • 🎨 Kafdrop: http://localhost:9002"
echo "   • 📊 Topics: market-data, trading-signals, pipeline-events, backtest-results"
echo ""
echo "💡 Kafka Flow Explanation:"
echo "   • 📈 Market data flows from data sources → Kafka → Spark processing"
echo "   • 🎯 Trading signals are generated and sent to Kafka for real-time distribution"
echo "   • ⚡ Pipeline events track the entire workflow through Kafka"
echo "   • 🔄 Historical replay uses Kafka to simulate real-time data streams"
echo ""
echo "🚀 Next Steps:"
echo "   • Run full demo: ./scripts/run_demo.sh"
echo "   • Check dashboard: http://localhost:8083"
echo "   • Monitor in Kibana: http://localhost:5601"
echo ""
