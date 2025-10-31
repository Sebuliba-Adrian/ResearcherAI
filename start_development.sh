#!/bin/bash
# Start ResearcherAI in Development Mode (FAISS + NetworkX)

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set environment variables for development mode
export CUDA_VISIBLE_DEVICES=''
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-AIzaSyByzzNJIDjTdhoQJs4H2fBMoBbDW3XLA0A}"

# Development backend configuration (no external dependencies)
export VECTOR_DB_TYPE="faiss"
export GRAPH_DB_TYPE="networkx"

echo "🚀 Starting ResearcherAI in DEVELOPMENT mode..."
echo "   Backend: FAISS + NetworkX"
echo "   API: http://localhost:8000"
echo ""

python api_gateway.py

