#!/usr/bin/env python3
"""
BreadthFlow Kafka Demo Script (Python Version)
Demonstrates Kafka streaming capabilities with financial data
"""

import os
import sys
import json
import time
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

def main():
    print("🎨 BreadthFlow Kafka Streaming Demo")
    print("===================================")
    
    # Kafka configuration
    kafka_config = {
        'bootstrap_servers': 'kafka:9092',
        'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
        'key_serializer': lambda k: k.encode('utf-8') if k else None
    }
    
    try:
        # Create producer
        producer = KafkaProducer(**kafka_config)
        print("✅ Connected to Kafka")
        
        print("\n📊 Current Topics:")
        # List topics (simplified - in real implementation you'd use admin client)
        topics = ['market-data', 'trading-signals', 'pipeline-events', 'backtest-results']
        for topic in topics:
            print(f"   • {topic}")
        
        print("\n🎯 Kafka Use Cases in Financial Pipeline:")
        print("=========================================")
        print("1. 📈 Live Market Data Streaming")
        print("2. 🔄 Historical Data Replay for Backtesting")
        print("3. ⚡ Real-time Signal Notifications")
        print("4. 🔄 Event-driven Pipeline Orchestration")
        
        print("\n📤 Step 1/4: Sending market data to Kafka...")
        
        # Send sample market data
        market_data = [
            {"symbol": "AAPL", "price": 150.25, "volume": 1000000, "timestamp": datetime.utcnow().isoformat()},
            {"symbol": "MSFT", "price": 320.50, "volume": 800000, "timestamp": datetime.utcnow().isoformat()},
            {"symbol": "GOOGL", "price": 2800.75, "volume": 500000, "timestamp": datetime.utcnow().isoformat()}
        ]
        
        for data in market_data:
            producer.send('market-data', key=data['symbol'], value=data)
            print(f"   ✅ Sent: {data['symbol']} at ${data['price']}")
        
        producer.flush()
        print("✅ Sent 3 market data messages to Kafka")
        
        print("\n📤 Step 2/4: Sending trading signals to Kafka...")
        
        # Send sample trading signals
        signals = [
            {"symbol": "AAPL", "signal": "BUY", "confidence": 85.5, "price": 150.25, "timestamp": datetime.utcnow().isoformat()},
            {"symbol": "MSFT", "signal": "HOLD", "confidence": 65.2, "price": 320.50, "timestamp": datetime.utcnow().isoformat()}
        ]
        
        for signal in signals:
            producer.send('trading-signals', key=signal['symbol'], value=signal)
            print(f"   ✅ Sent: {signal['symbol']} {signal['signal']} signal")
        
        producer.flush()
        print("✅ Sent 2 trading signals to Kafka")
        
        print("\n📤 Step 3/4: Sending pipeline events to Kafka...")
        
        # Send pipeline events
        events = [
            {"event": "data_fetch_completed", "symbols": ["AAPL", "MSFT", "GOOGL"], "records": 1500, "timestamp": datetime.utcnow().isoformat()},
            {"event": "signal_generation_started", "symbols": ["AAPL", "MSFT"], "timestamp": datetime.utcnow().isoformat()}
        ]
        
        for event in events:
            producer.send('pipeline-events', value=event)
            print(f"   ✅ Sent: {event['event']}")
        
        producer.flush()
        print("✅ Sent 2 pipeline events to Kafka")
        
        print("\n📥 Step 4/4: Reading messages from Kafka topics...")
        
        # Read messages (simplified consumer)
        consumer_config = {
            'bootstrap_servers': 'kafka:9092',
            'value_deserializer': lambda v: json.loads(v.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'auto_offset_reset': 'earliest',
            'group_id': 'demo-consumer'
        }
        
        print("\n📊 Market Data Messages:")
        consumer = KafkaConsumer('market-data', **consumer_config)
        for message in consumer:
            print(f"   {message.value}")
            break  # Just show one message
        
        consumer.close()
        
        print("\n🎯 Trading Signals:")
        consumer = KafkaConsumer('trading-signals', **consumer_config)
        for message in consumer:
            print(f"   {message.value}")
            break  # Just show one message
        
        consumer.close()
        
        print("\n⚡ Pipeline Events:")
        consumer = KafkaConsumer('pipeline-events', **consumer_config)
        for message in consumer:
            print(f"   {message.value}")
            break  # Just show one message
        
        consumer.close()
        
        print("\n✅ Kafka demo completed!")
        print("\n🌐 View Kafka UI:")
        print("   • 🎨 Kafdrop: http://localhost:9002")
        print("   • 📊 Topics: market-data, trading-signals, pipeline-events, backtest-results")
        print("\n💡 Kafka Flow Explanation:")
        print("   • 📈 Market data flows from data sources → Kafka → Spark processing")
        print("   • 🎯 Trading signals are generated and sent to Kafka for real-time distribution")
        print("   • ⚡ Pipeline events track the entire workflow through Kafka")
        print("   • 🔄 Historical replay uses Kafka to simulate real-time data streams")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    finally:
        try:
            producer.close()
        except:
            pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
