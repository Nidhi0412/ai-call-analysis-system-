#!/bin/bash

# Mac Dependencies Installation Script
echo "🍎 Installing Mac Dependencies for Call Recordings AI"
echo "====================================================="

# Install Homebrew if not installed
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install FFmpeg
echo "🎵 Installing FFmpeg..."
brew install ffmpeg

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install fastapi uvicorn python-multipart openai-whisper pydub numpy scipy librosa

# Verify installations
echo "✅ Verifying installations..."
python3 -c "import whisper; print('✅ Whisper installed')"
python3 -c "import ffmpeg; print('✅ FFmpeg available')" 2>/dev/null || echo "⚠️ FFmpeg not in Python path (but should work)"

echo "🎉 Mac setup complete!"
echo "Run: python3 run_local.py"
