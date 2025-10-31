#!/bin/bash
# Run Playwright MCP Integration Tests
# Tests both production and development modes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-both}"  # production, development, or both
TIMEOUT=300  # 5 minutes timeout for backend startup

echo "=========================================="
echo "Playwright MCP Integration Test Suite"
echo "=========================================="
echo "Mode: $MODE"
echo ""

# Function to check if server is running
check_server() {
    local url="$1"
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for server at $url..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f -o /dev/null "$url/health" 2>/dev/null; then
            echo "✓ Server is ready"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo "✗ Server not ready after $max_attempts attempts"
    return 1
}

# Function to start server in background
start_server() {
    local mode="$1"
    local script="start_${mode}.sh"
    
    if [ ! -f "$script" ]; then
        echo "Error: $script not found"
        return 1
    fi
    
    echo "Starting server in $mode mode..."
    chmod +x "$script"
    "$script" > "logs/server_${mode}.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    # Wait for server to be ready
    sleep 5
    if ! check_server "http://localhost:8000/v1"; then
        echo "Failed to start server"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
    
    return 0
}

# Function to stop server
stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

# Cleanup on exit
trap stop_server EXIT

# Run tests based on mode
if [ "$MODE" = "production" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "=========================================="
    echo "Testing PRODUCTION Mode (Neo4j + Qdrant)"
    echo "=========================================="
    
    # Check if Neo4j and Qdrant are running
    if ! docker ps | grep -q "neo4j\|qdrant"; then
        echo "⚠ Warning: Neo4j/Qdrant containers may not be running"
        echo "  Start them with: docker-compose up -d neo4j qdrant"
    fi
    
    start_server "production"
    
    echo "Running Playwright MCP tests for production mode..."
    python test_playwright_integration.py --mode production --output "test_outputs/playwright_production_report.json"
    
    stop_server
    sleep 5
fi

if [ "$MODE" = "development" ] || [ "$MODE" = "both" ]; then
    echo ""
    echo "=========================================="
    echo "Testing DEVELOPMENT Mode (FAISS + NetworkX)"
    echo "=========================================="
    
    start_server "development"
    
    echo "Running Playwright MCP tests for development mode..."
    python test_playwright_integration.py --mode development --output "test_outputs/playwright_development_report.json"
    
    stop_server
    sleep 5
fi

if [ "$MODE" = "both" ]; then
    echo ""
    echo "=========================================="
    echo "Generating Cross-Environment Comparison"
    echo "=========================================="
    
    # Compare results if both modes were tested
    if [ -f "test_outputs/playwright_production_report.json" ] && \
       [ -f "test_outputs/playwright_development_report.json" ]; then
        echo "Comparison reports available:"
        echo "  - test_outputs/playwright_production_report.json"
        echo "  - test_outputs/playwright_development_report.json"
    fi
fi

echo ""
echo "=========================================="
echo "Test Suite Complete"
echo "=========================================="

