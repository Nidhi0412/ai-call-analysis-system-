#!/usr/bin/env python3
"""
Start Local Models Application
=============================
Simple script to run the local models application with correct Python path
"""

import os
import sys
import uvicorn

# Add the parent directory to Python path so we can import from Call_recordings_AI
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set environment variables for local models
os.environ['TRANSCRIPTION_PROVIDER'] = 'faster_whisper'
os.environ['LLM_PROVIDER'] = 'ollama'

if __name__ == "__main__":
    print("🎯 Starting Local Models Application")
    print("=" * 50)
    print("🏠 Host: 0.0.0.0")
    print("🔌 Port: 8001")
    print("\n📝 Available endpoints:")
    print("   GET  / - Root endpoint")
    print("   GET  /health - Health check")
    print("   GET  /local-models-status - Local models status")
    print("   GET  /transcribe-ui - Local models UI")
    print("   POST /transcribe - Transcribe audio")
    print("   POST /innex/analyze - Analyze transcription")
    print("   POST /innex/translate - Translate text")
    print("\n📚 API docs: http://localhost:8001/docs")
    print("🌐 Local models UI: http://localhost:8001/transcribe-ui")
    
    # Import and run the application
    uvicorn.run("app_simple_local_only:app", host="0.0.0.0", port=8001, reload=True, access_log=True, log_level="info") 