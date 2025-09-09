#!/bin/bash

# BreadthFlow Test Runner Script
# This script runs different types of tests for the BreadthFlow project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
VERBOSE=false
COVERAGE=true
PARALLEL=false
CLEANUP=true

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Test type: unit, integration, e2e, all (default: all)"
    echo "  -v, --verbose          Verbose output"
    echo "  -c, --no-coverage      Disable coverage reporting"
    echo "  -p, --parallel         Run tests in parallel"
    echo "  -n, --no-cleanup       Don't cleanup test environment"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run all tests"
    echo "  $0 -t unit             # Run only unit tests"
    echo "  $0 -t integration -v   # Run integration tests with verbose output"
    echo "  $0 -t e2e -p           # Run E2E tests in parallel"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -n|--no-cleanup)
            CLEANUP=false
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate test type
if [[ ! "$TEST_TYPE" =~ ^(unit|integration|e2e|all)$ ]]; then
    print_error "Invalid test type: $TEST_TYPE"
    show_usage
    exit 1
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
}

# Function to setup test environment
setup_test_environment() {
    print_status "Setting up test environment..."
    
    # Create test directories
    mkdir -p test-results
    mkdir -p logs
    
    # Start test services
    print_status "Starting test services..."
    docker-compose -f docker-compose.test.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check if services are healthy
    print_status "Checking service health..."
    
    # Check PostgreSQL
    if ! docker-compose -f docker-compose.test.yml exec -T postgres-test pg_isready -U test_user -d breadthflow_test > /dev/null 2>&1; then
        print_error "PostgreSQL test service is not ready"
        exit 1
    fi
    
    # Check Redis
    if ! docker-compose -f docker-compose.test.yml exec -T redis-test redis-cli ping > /dev/null 2>&1; then
        print_error "Redis test service is not ready"
        exit 1
    fi
    
    # Check MinIO
    if ! docker-compose -f docker-compose.test.yml exec -T minio-test curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        print_error "MinIO test service is not ready"
        exit 1
    fi
    
    print_success "Test environment is ready"
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    
    local pytest_args="tests/unit/ -v"
    
    if [ "$VERBOSE" = true ]; then
        pytest_args="$pytest_args -s"
    fi
    
    if [ "$COVERAGE" = true ]; then
        pytest_args="$pytest_args --cov=. --cov-report=html --cov-report=xml --cov-report=term-missing"
    fi
    
    if [ "$PARALLEL" = true ]; then
        pytest_args="$pytest_args -n auto"
    fi
    
    if pytest $pytest_args; then
        print_success "Unit tests passed"
        return 0
    else
        print_error "Unit tests failed"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    local pytest_args="tests/integration/ -v"
    
    if [ "$VERBOSE" = true ]; then
        pytest_args="$pytest_args -s"
    fi
    
    if [ "$COVERAGE" = true ]; then
        pytest_args="$pytest_args --cov=. --cov-report=html --cov-report=xml --cov-report=term-missing"
    fi
    
    if [ "$PARALLEL" = true ]; then
        pytest_args="$pytest_args -n auto"
    fi
    
    if pytest $pytest_args; then
        print_success "Integration tests passed"
        return 0
    else
        print_error "Integration tests failed"
        return 1
    fi
}

# Function to run E2E tests
run_e2e_tests() {
    print_status "Running E2E tests..."
    
    local pytest_args="tests/e2e/ -v"
    
    if [ "$VERBOSE" = true ]; then
        pytest_args="$pytest_args -s"
    fi
    
    if [ "$COVERAGE" = true ]; then
        pytest_args="$pytest_args --cov=. --cov-report=html --cov-report=xml --cov-report=term-missing"
    fi
    
    if pytest $pytest_args; then
        print_success "E2E tests passed"
        return 0
    else
        print_error "E2E tests failed"
        return 1
    fi
}

# Function to cleanup test environment
cleanup_test_environment() {
    if [ "$CLEANUP" = true ]; then
        print_status "Cleaning up test environment..."
        docker-compose -f docker-compose.test.yml down -v
        print_success "Test environment cleaned up"
    else
        print_warning "Skipping cleanup (use -n flag to disable)"
    fi
}

# Function to generate test report
generate_test_report() {
    print_status "Generating test report..."
    
    if [ -f "test-results/junit.xml" ]; then
        print_success "JUnit XML report generated: test-results/junit.xml"
    fi
    
    if [ -f "htmlcov/index.html" ]; then
        print_success "Coverage HTML report generated: htmlcov/index.html"
    fi
    
    if [ -f "coverage.xml" ]; then
        print_success "Coverage XML report generated: coverage.xml"
    fi
}

# Main execution
main() {
    print_status "Starting BreadthFlow test suite..."
    print_status "Test type: $TEST_TYPE"
    print_status "Verbose: $VERBOSE"
    print_status "Coverage: $COVERAGE"
    print_status "Parallel: $PARALLEL"
    print_status "Cleanup: $CLEANUP"
    echo ""
    
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Setup test environment
    setup_test_environment
    
    # Run tests based on type
    local test_result=0
    
    case $TEST_TYPE in
        unit)
            run_unit_tests || test_result=1
            ;;
        integration)
            run_integration_tests || test_result=1
            ;;
        e2e)
            run_e2e_tests || test_result=1
            ;;
        all)
            run_unit_tests || test_result=1
            run_integration_tests || test_result=1
            run_e2e_tests || test_result=1
            ;;
    esac
    
    # Generate test report
    generate_test_report
    
    # Cleanup
    cleanup_test_environment
    
    # Final result
    if [ $test_result -eq 0 ]; then
        print_success "All tests completed successfully!"
        exit 0
    else
        print_error "Some tests failed!"
        exit 1
    fi
}

# Run main function
main "$@"
