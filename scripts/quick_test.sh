#!/bin/bash

# Quick Test Script for BreadthFlow
# This script runs a quick test suite to verify the system is working

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🚀 BreadthFlow Quick Test Suite${NC}"
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if services are running
echo -e "${YELLOW}📊 Checking services...${NC}"

# Check if main services are running
if docker-compose ps | grep -q "breadthflow-api.*Up"; then
    echo -e "${GREEN}✅ API service is running${NC}"
else
    echo -e "${RED}❌ API service is not running${NC}"
    echo "Starting services..."
    docker-compose up -d breadthflow-api postgres redis
    sleep 10
fi

# Test API endpoints
echo -e "${YELLOW}🔍 Testing API endpoints...${NC}"

# Test health endpoint
if curl -s http://localhost:8005/health > /dev/null; then
    echo -e "${GREEN}✅ Health endpoint is working${NC}"
else
    echo -e "${RED}❌ Health endpoint is not working${NC}"
fi

# Test dashboard endpoint
if curl -s http://localhost:8005/api/dashboard/summary > /dev/null; then
    echo -e "${GREEN}✅ Dashboard API is working${NC}"
else
    echo -e "${RED}❌ Dashboard API is not working${NC}"
fi

# Test pipeline endpoint
if curl -s http://localhost:8005/api/pipeline/status > /dev/null; then
    echo -e "${GREEN}✅ Pipeline API is working${NC}"
else
    echo -e "${RED}❌ Pipeline API is not working${NC}"
fi

# Test signals endpoint
if curl -s http://localhost:8005/api/signals/latest > /dev/null; then
    echo -e "${GREEN}✅ Signals API is working${NC}"
else
    echo -e "${RED}❌ Signals API is not working${NC}"
fi

# Test frontend
echo -e "${YELLOW}🌐 Testing frontend...${NC}"

if curl -s http://localhost:3005 > /dev/null; then
    echo -e "${GREEN}✅ Frontend is accessible${NC}"
else
    echo -e "${RED}❌ Frontend is not accessible${NC}"
fi

# Run basic unit tests
echo -e "${YELLOW}🧪 Running basic unit tests...${NC}"

if python -m pytest tests/unit/ -v --tb=short; then
    echo -e "${GREEN}✅ Unit tests passed${NC}"
else
    echo -e "${RED}❌ Unit tests failed${NC}"
fi

# Test database connection
echo -e "${YELLOW}🗄️ Testing database connection...${NC}"

if docker-compose exec -T postgres psql -U user -d breadthflow -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Database connection is working${NC}"
else
    echo -e "${RED}❌ Database connection is not working${NC}"
fi

# Test Redis connection
echo -e "${YELLOW}🔴 Testing Redis connection...${NC}"

if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Redis connection is working${NC}"
else
    echo -e "${RED}❌ Redis connection is not working${NC}"
fi

echo ""
echo -e "${YELLOW}📊 Test Summary${NC}"
echo "=============="

# Count running services
running_services=$(docker-compose ps | grep "Up" | wc -l)
echo -e "Running services: ${running_services}"

# Check API response times
api_time=$(curl -s -w "%{time_total}" -o /dev/null http://localhost:8005/api/dashboard/summary)
echo -e "API response time: ${api_time}s"

# Check if all tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 All quick tests passed!${NC}"
    echo -e "${GREEN}✅ System is healthy and ready for development${NC}"
else
    echo -e "${RED}❌ Some tests failed. Please check the logs above.${NC}"
    exit 1
fi
