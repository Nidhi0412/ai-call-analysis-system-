#!/bin/bash

# Call Recordings AI - Local Startup Script (Inside Call_recordings_AI folder)
echo "🎤 Starting Call Recordings AI - Local Mode"
echo "=============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not found. Installing..."
    pip3 install fastapi uvicorn python-multipart
fi

# Set environment variables
export NODE_ENV=Development
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p uploads cache results logs
echo "✅ Directories created"

# Run the local application
echo "🚀 Starting application..."
python3 run_local.py
