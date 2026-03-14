# Call Recordings AI - Mac Setup Guide

## 🍎 Mac-Specific Installation & Setup

### Prerequisites
- **macOS** (tested on macOS 10.15+)
- **Python 3.8+** 
- **Homebrew** (for package management)

### Quick Setup

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Dependencies**:
   ```bash
   chmod +x install_mac_dependencies.sh
   ./install_mac_dependencies.sh
   ```

3. **Start Application**:
   ```bash
   chmod +x start_mac.sh
   ./start_mac.sh
   ```

### Manual Installation

If you prefer manual setup:

1. **Install FFmpeg**:
   ```bash
   brew install ffmpeg
   ```

2. **Install Python Dependencies**:
   ```bash
   pip3 install fastapi uvicorn python-multipart openai-whisper pydub numpy scipy librosa
   ```

3. **Run Application**:
   ```bash
   python3 run_local.py
   ```

### Mac-Specific Notes

#### 🔧 **Path Differences**
- **Linux**: `/home/saas/logs/innex`
- **Mac**: Uses `mock_pylogger` (console logging)
- **No changes needed** - the code automatically handles this

#### 🔑 **API Key Setup**
- **OpenAI API Key**: Hardcoded in `call_analysis.py` and `optimized_transcription_service.py`
- **No config.py needed** - API key is embedded directly in the code
- **Simplified setup** - no environment variables or config files required

#### 🎵 **Audio Processing**
- **FFmpeg**: Installed via Homebrew
- **Audio formats**: Same support as Linux
- **Performance**: May be slower on older Macs

#### 🚀 **Performance Considerations**
- **M1/M2 Macs**: Excellent performance with local Whisper
- **Intel Macs**: Good performance, may be slower for large files
- **RAM**: Recommend 8GB+ for `large-v2` model

#### 🌐 **Network Access**
- **Local**: `http://localhost:8000`
- **Network**: `http://[your-mac-ip]:8000` (if needed)

### Troubleshooting

#### FFmpeg Issues
```bash
# Check FFmpeg installation
ffmpeg -version

# Reinstall if needed
brew uninstall ffmpeg
brew install ffmpeg
```

#### Python Package Issues
```bash
# Update pip
pip3 install --upgrade pip

# Reinstall packages
pip3 uninstall openai-whisper
pip3 install openai-whisper
```

#### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 [PID]
```

### File Structure
```
Call_recordings_AI/
├── run_local.py              # Main application
├── start_mac.sh              # Mac startup script
├── install_mac_dependencies.sh # Mac dependency installer
├── templates/                # Web UI templates
├── uploads/                  # Audio file uploads
├── cache/                    # Processing cache
├── results/                  # Analysis results
└── logs/                     # Application logs
```

### API Endpoints (Mac)
- **Main UI**: `http://localhost:8000/innex/transcribe-ui`
- **API Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### Model Download
On first run, Whisper will download the `large-v2` model (~3GB):
- **Location**: `~/.cache/whisper/`
- **Time**: 5-10 minutes depending on internet speed
- **Space**: ~3GB required

### Support
- **macOS Version**: 10.15+ (Catalina+)
- **Python**: 3.8+
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space
