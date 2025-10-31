#!/bin/bash
# Start ResearcherAI in Production Mode (Neo4j + Qdrant)

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for production mode
export CUDA_VISIBLE_DEVICES=''
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-AIzaSyByzzNJIDjTdhoQJs4H2fBMoBbDW3XLA0A}"

# Production backend configuration
export VECTOR_DB_TYPE="qdrant"
export GRAPH_DB_TYPE="neo4j"
export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-research_password}"
export QDRANT_HOST="${QDRANT_HOST:-localhost}"
export QDRANT_PORT="${QDRANT_PORT:-6333}"

echo "🚀 Starting ResearcherAI in PRODUCTION mode..."
echo "   Backend: Neo4j + Qdrant"
echo "   API: http://localhost:8000"
echo ""

python api_gateway.py

