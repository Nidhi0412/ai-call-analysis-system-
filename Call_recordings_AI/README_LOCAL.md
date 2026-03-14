# Call Recordings AI - Local Development

## 🎤 Audio Transcription with Speaker Identification

This application provides audio transcription with speaker identification capabilities, designed to run locally within the Call_recordings_AI folder.

## 🚀 Quick Start

### Method 1: Using the Startup Script (Recommended)
```bash
cd Call_recordings_AI
./start_local.sh
```

### Method 2: Manual Start
```bash
cd Call_recordings_AI
python3 run_local.py
```

## 📱 Available Endpoints

Once running, you can access:

- **Main UI**: http://localhost:8000/innex/transcribe-ui
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

## 📁 Directory Structure (Within Call_recordings_AI/)

```
Call_recordings_AI/
├── run_local.py              # Main local application runner
├── start_local.sh            # Startup script
├── local_config.py           # Local configuration
├── web_ui.py                 # Main web interface
├── transcription_with_speakers.py
├── unified_audio_processor.py
├── speaker_diarization.py
├── call_analysis.py
├── analyze_preprocessed_audio.py
├── hindi_debug_helper.py
├── optimized_transcription_service.py
├── whisper_provider_config.py
├── mock_pylogger.py
├── templates/                # HTML templates
├── Local_model_recordings/   # Local model implementations
├── uploads/                  # Uploaded files (auto-created)
├── cache/                    # Cache files (auto-created)
├── results/                  # Analysis results (auto-created)
└── logs/                     # Application logs (auto-created)
```

## 🔧 Features

- **Audio Transcription**: Convert audio to text
- **Speaker Identification**: Identify different speakers in conversations
- **Hindi Language Support**: Enhanced Hindi transcription
- **Call Analysis**: Business insights from call recordings
- **Real-time Processing**: Fast audio processing
- **Local Processing**: No external API dependencies

## 📋 Requirements

- Python 3.8+
- Required packages: `fastapi`, `uvicorn`, `python-multipart`

## 🛠️ Installation

1. Navigate to the Call_recordings_AI folder:
```bash
cd Call_recordings_AI
```

2. Install Python dependencies:
```bash
pip3 install fastapi uvicorn python-multipart
```

3. Run the application:
```bash
./start_local.sh
```

## 🎯 Usage

1. Open your browser and go to: http://localhost:8000/innex/transcribe-ui
2. Upload an audio file
3. Configure transcription options:
   - Enable Hindi enhancements
   - Run debug analysis
   - Use advanced speaker diarization
   - Save preprocessed audio
4. Click "Transcribe Audio" to start processing
5. View results with speaker identification and call analysis

## 🔍 Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the Call_recordings_AI directory:
```bash
cd Call_recordings_AI
python3 run_local.py
```

### Port Already in Use
If port 8000 is busy, modify `run_local.py` and change the port:
```python
uvicorn.run(app, host="0.0.0.0", port=8001, ...)
```

### Missing Dependencies
Install required packages:
```bash
pip3 install fastapi uvicorn python-multipart
```

## 📊 Performance

- **Small files (< 5MB)**: ~30-60 seconds
- **Medium files (5-20MB)**: ~1-3 minutes  
- **Large files (20-50MB)**: ~3-8 minutes
- **Very large files (> 50MB)**: ~8+ minutes

Processing time depends on file size, audio quality, and selected options.

## 🔒 Security

- All processing happens locally
- No data sent to external services
- Files are automatically cleaned up after processing
- Cache files are stored locally

## 📝 Logs

Application logs are stored in the `logs/` directory. Check these files for debugging information.

## 🤝 Support

For issues or questions, check the logs in the `logs/` directory or review the API documentation at http://localhost:8000/docs when the application is running.

## 🎯 Key Differences from Full App

This local version:
- Runs entirely within the Call_recordings_AI folder
- Uses relative imports (no absolute paths)
- Creates local directories automatically
- Provides a simplified FastAPI interface
- No dependency on external app.py or router configurations
