#!/bin/bash
# Setup script for Self-Improving RAG System

echo "🚀 Setting up Self-Improving Agentic RAG System..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Download spaCy model
echo "📚 Downloading spaCy language model..."
python -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the system:"
echo "  source venv/bin/activate"
echo "  python self_improving_rag.py sample_knowledge.txt"
echo ""
