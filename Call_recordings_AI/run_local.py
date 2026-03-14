#!/usr/bin/env python3
"""
Standalone Local Application Runner
Run the Call Recordings AI application locally within Call_recordings_AI folder
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set environment variables for local development
os.environ['NODE_ENV'] = 'Development'

# Import the main web UI router
try:
    from web_ui import router as web_ui_router
    print("✅ Web UI router imported successfully")
except ImportError as e:
    print(f"❌ Failed to import web UI router: {e}")
    web_ui_router = None

# Create FastAPI app
from fastapi import FastAPI
app = FastAPI(
    title="Call Recordings AI - Local",
    description="Audio Transcription with Speaker Identification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include the web UI router
if web_ui_router:
    app.include_router(web_ui_router, prefix="/innex")

# Add health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Call Recordings AI - Local Mode",
        "status": "running",
        "docs": "/docs",
        "transcription_ui": "/innex/transcribe-ui"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "mode": "local"}

def main():
    """Run the application locally"""
    print("🚀 Starting Call Recordings AI - Local Mode")
    print("=" * 50)
    print("✅ FastAPI application configured")
    print("📱 Available endpoints:")
    print("   • Main UI: http://localhost:8000/innex/transcribe-ui")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    # Run the application
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main()
