# Local Development Configuration
# This file contains settings for running the application locally within Call_recordings_AI folder

# Local Development Settings
LOCAL_CONFIG = {
    'Development': {
        'APP_HOST': '0.0.0.0',
        'APP_PORT': 8000,
        'WORKERS': 1,
        'RELOAD': True,
        'LOG_LEVEL': 'info',
        'DEBUG': True
    }
}

# Local file paths (relative to Call_recordings_AI folder)
LOCAL_PATHS = {
    'UPLOAD_DIR': './uploads',
    'CACHE_DIR': './cache',
    'RESULTS_DIR': './results',
    'LOGS_DIR': './logs'
}

# Create necessary directories
import os
for path in LOCAL_PATHS.values():
    os.makedirs(path, exist_ok=True)
    print(f"✅ Created directory: {path}")

print("✅ Local configuration loaded successfully")
