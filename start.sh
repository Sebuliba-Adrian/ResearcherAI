#!/bin/bash

# ResearcherAI Startup Script
# Replace YOUR_GEMINI_API_KEY with your actual key from https://aistudio.google.com/app/apikey

cd /home/adrian/Desktop/Projects/ResearcherAI

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=''  # Use CPU (no GPU issues)
export GOOGLE_API_KEY="AIzaSyByzzNJIDjTdhoQJs4H2fBMoBbDW3XLA0A"  # ← PUT YOUR KEY HERE

# Check if API key is set
if [ "$GOOGLE_API_KEY" = "YOUR_GEMINI_API_KEY" ]; then
    echo "❌ ERROR: Please edit start.sh and add your Gemini API key"
    echo "Get a free key at: https://aistudio.google.com/app/apikey"
    exit 1
fi

# Start the server
echo "🚀 Starting ResearcherAI..."
python api_gateway.py
