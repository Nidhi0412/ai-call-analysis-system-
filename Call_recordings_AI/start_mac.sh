#!/bin/bash

# Call Recordings AI - Mac Startup Script
echo "🍎 Starting Call Recordings AI - Mac Mode"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    echo "💡 Install via: brew install python3"
    exit 1
fi

# Check if FFmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is not installed. Installing..."
    if command -v brew &> /dev/null; then
        brew install ffmpeg
    else
        echo "❌ Homebrew not found. Please install FFmpeg manually."
        echo "💡 Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
fi

# Check if required packages are installed
echo "📦 Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, whisper, pydub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not found. Installing..."
    pip3 install fastapi uvicorn python-multipart openai-whisper pydub numpy scipy librosa
fi

# Set environment variables for Mac
export NODE_ENV=Development
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create necessary directories
mkdir -p uploads cache results logs
echo "✅ Directories created"

# Run the local application
echo "🚀 Starting application on Mac..."
echo "🌐 Access UI at: http://localhost:8000/innex/transcribe-ui"
python3 run_local.py
